"""
Live grasp classifier GUI for the Universal Arm capacitive-sensor setup.

This is a SEPARATE program from arduino_signal.py -- that recorder is left
completely untouched so the CSV recordings it produces (and any future
recordings you make with it) stay in the exact same format the model was
trained on. This script opens its OWN serial connection to the same ESP32
and reuses the identical packet-parsing rules (see parse_packet() below,
copied from arduino_signal.py on purpose) so the raw numbers it sees are
exactly the numbers arduino_signal.py would have written to a CSV.

Workflow:
  1. Pick the ESP32's serial port and click Connect.
  2. Hold/present an object to the arm, click "Start Grasp".
  3. When the grasp is done, click "Stop && Classify".
  4. A pop-up window shows the predicted object (Ball / Square box /
     Triangular) with its picture from assets/, plus the model's confidence.

The classification itself (preprocessing, feature extraction, the trained
model) is 100% delegated to evaluate.py / feature_extraction.py in this same
folder -- this file adds no classification logic of its own, so training
and live inference can never drift apart.
"""
import os
import sys
import time
import csv
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import serial
import serial.tools.list_ports
from PIL import Image, ImageTk

from evaluate import predict_from_raw

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "assets")
LOG_PATH = os.path.join(HERE, "live_classification_log.csv")

BAUD_RATE = 115200
SELECTED_CHANNELS = [0, 1, 2]   # same as arduino_signal.py
MIN_SAMPLES_TO_CLASSIFY = 8      # extract_features() itself requires >= 8

# object name (as the model outputs it) -> image file in assets/
# matched by keyword so small label-string variations don't break it
IMAGE_KEYWORDS = [
    ("ball", "ball.jpeg"),
    ("square", "square.jpg"),
    ("triangl", "triangle.jpeg"),
]


def image_path_for_label(label):
    norm = label.lower()
    for keyword, filename in IMAGE_KEYWORDS:
        if keyword in norm:
            path = os.path.join(ASSETS_DIR, filename)
            if os.path.exists(path):
                return path
    return None


# ============================================================
# PACKET PARSER -- identical logic to arduino_signal.py's parse_packet(),
# duplicated here (not imported) so arduino_signal.py never has to change.
# If you ever change the ESP32 firmware's packet format, update BOTH copies.
# ============================================================

def parse_packet(line, selected_channels=SELECTED_CHANNELS):
    if not line:
        return None, None

    payload = line[1:] if line.startswith("$") else line
    parts = [x.strip() for x in payload.split(",")]

    try:
        numbers = [float(x) for x in parts]
    except ValueError:
        return None, None

    if len(numbers) == 16:
        values = [numbers[selected_channels[0]], numbers[selected_channels[1]], numbers[selected_channels[2]]]
        fmt = "16-channel"
    elif len(numbers) == 4:
        values = [numbers[1], numbers[2], numbers[3]]
        fmt = "timestamp + 3 sensors"
    elif len(numbers) == 3:
        values = [numbers[0], numbers[1], numbers[2]]
        fmt = "3 sensors"
    else:
        return None, None

    return values, fmt


# ============================================================
# BACKGROUND SERIAL READER
# ============================================================

class SerialReader(threading.Thread):
    """Reads lines from the serial port on a background thread and pushes
    (program_time, s1, s2, s3) tuples onto a thread-safe queue. Keeping this
    off the Tk main thread keeps the GUI responsive."""

    def __init__(self, port, baud, out_queue):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self._stop_flag = threading.Event()
        self.ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(2)
        self.ser.reset_input_buffer()
        self._t0 = time.perf_counter()
        self.last_error = None

    def stop(self):
        self._stop_flag.set()

    def run(self):
        while not self._stop_flag.is_set():
            try:
                raw = self.ser.readline()
            except Exception as e:
                self.last_error = str(e)
                break
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            values, fmt = parse_packet(line)
            if values is None:
                continue
            t = time.perf_counter() - self._t0
            self.out_queue.put((t, values[0], values[1], values[2], fmt))
        try:
            self.ser.close()
        except Exception:
            pass


# ============================================================
# GRASP BUFFER + CLASSIFICATION
# ============================================================

class GraspBuffer:
    """Collects samples for one grasp trial (mirrors what arduino_signal.py
    writes to a CSV, just kept in memory instead)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t, self.s1, self.s2, self.s3 = [], [], [], []
        self._t0 = None

    def add(self, program_time, s1, s2, s3):
        if self._t0 is None:
            self._t0 = program_time
        self.t.append(program_time - self._t0)
        self.s1.append(s1)
        self.s2.append(s2)
        self.s3.append(s3)

    def __len__(self):
        return len(self.t)


def classify_grasp(buf: GraspBuffer):
    """Runs the SAME preprocessing + feature extraction + model used in
    training (via evaluate.predict_from_raw). Returns (label, proba_dict)."""
    if len(buf) < MIN_SAMPLES_TO_CLASSIFY:
        raise ValueError(
            f"Only {len(buf)} samples captured (need at least {MIN_SAMPLES_TO_CLASSIFY}). "
            f"Hold the grasp a little longer before clicking Stop && Classify."
        )
    label, proba = predict_from_raw(buf.t, buf.s1, buf.s2, buf.s3, return_proba=True)
    return label, proba


def log_prediction(label, proba, n_samples, duration_s):
    """Small append-only audit trail of live predictions (not raw signal) --
    handy for spotting a model that's drifting without re-opening the GUI."""
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(["timestamp", "predicted_label", "confidence", "n_samples", "duration_s"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), label, f"{proba.get(label, 0):.4f}", n_samples, f"{duration_s:.3f}"])


# ============================================================
# GUI
# ============================================================

class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Arm - Live Grasp Classifier")
        self.root.geometry("480x420")
        self.root.resizable(False, False)

        self.reader = None
        self.rx_queue = queue.Queue()
        self.buffer = GraspBuffer()
        self.capturing = False
        self._last_sample = (0.0, 0.0, 0.0)

        self._build_widgets()
        self._poll_queue()

    # ---------- layout ----------
    def _build_widgets(self):
        pad = {"padx": 10, "pady": 6}

        conn = ttk.LabelFrame(self.root, text="Connection")
        conn.pack(fill="x", **pad)

        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var, state="readonly", width=28)
        self.port_combo.grid(row=0, column=0, padx=6, pady=6)
        self._refresh_ports()

        ttk.Button(conn, text="Refresh", command=self._refresh_ports).grid(row=0, column=1, padx=4)
        self.connect_btn = ttk.Button(conn, text="Connect", command=self._toggle_connect)
        self.connect_btn.grid(row=0, column=2, padx=4)

        live = ttk.LabelFrame(self.root, text="Live sensor readout (pF)")
        live.pack(fill="x", **pad)
        self.live_labels = []
        for i in range(3):
            lbl = ttk.Label(live, text=f"Sensor {i+1}: --", font=("TkDefaultFont", 12))
            lbl.grid(row=0, column=i, padx=16, pady=8)
            self.live_labels.append(lbl)

        ctrl = ttk.LabelFrame(self.root, text="Grasp capture")
        ctrl.pack(fill="x", **pad)
        self.start_btn = ttk.Button(ctrl, text="Start Grasp", command=self._start_grasp, state="disabled")
        self.start_btn.grid(row=0, column=0, padx=8, pady=8)
        self.stop_btn = ttk.Button(ctrl, text="Stop && Classify", command=self._stop_and_classify, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Not connected.")
        ttk.Label(self.root, textvariable=self.status_var, foreground="gray").pack(fill="x", **pad)

        result = ttk.LabelFrame(self.root, text="Last result")
        result.pack(fill="both", expand=True, **pad)
        self.result_image_label = ttk.Label(result)
        self.result_image_label.pack(pady=6)
        self.result_text_var = tk.StringVar(value="No grasp classified yet.")
        ttk.Label(result, textvariable=self.result_text_var, font=("TkDefaultFont", 12, "bold")).pack()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    # ---------- connection ----------
    def _toggle_connect(self):
        if self.reader is None:
            self._connect()
        else:
            self._disconnect()

    def _connect(self):
        port = self.port_var.get()
        if not port:
            messagebox.showerror("No port selected", "Pick a serial port first (click Refresh if the list is empty).")
            return
        try:
            self.reader = SerialReader(port, BAUD_RATE, self.rx_queue)
        except Exception as e:
            messagebox.showerror("Connection failed", f"Could not open {port}:\n{e}\n\nClose the Arduino Serial Monitor / Plotter if it's open.")
            self.reader = None
            return
        self.reader.start()
        self.connect_btn.config(text="Disconnect")
        self.start_btn.config(state="normal")
        self.status_var.set(f"Connected on {port}. Waiting for data...")

    def _disconnect(self):
        if self.capturing:
            self._stop_and_classify(silent=True)
        if self.reader:
            self.reader.stop()
            self.reader = None
        self.connect_btn.config(text="Connect")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Disconnected.")

    # ---------- grasp capture ----------
    def _start_grasp(self):
        self.buffer.reset()
        self.capturing = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Recording grasp... click Stop && Classify when done.")

    def _stop_and_classify(self, silent=False):
        self.capturing = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        try:
            label, proba = classify_grasp(self.buffer)
        except ValueError as e:
            if not silent:
                messagebox.showwarning("Grasp too short", str(e))
            self.status_var.set("Grasp too short -- try again.")
            return
        except Exception as e:
            if not silent:
                messagebox.showerror("Classification failed", str(e))
            self.status_var.set("Classification failed -- see error.")
            return

        duration = self.buffer.t[-1] - self.buffer.t[0] if len(self.buffer) > 1 else 0.0
        log_prediction(label, proba, len(self.buffer), duration)

        confidence = proba.get(label, 0.0)
        self.status_var.set(f"Classified {len(self.buffer)} samples ({duration:.1f}s) as '{label}' ({confidence:.0%}).")
        self._show_result(label, proba)
        if not silent:
            self._show_popup(label, confidence)

    def _show_result(self, label, proba):
        proba_str = "\n".join(f"{c}: {p:.1%}" for c, p in sorted(proba.items(), key=lambda kv: -kv[1]))
        self.result_text_var.set(f"Predicted: {label}\n\n{proba_str}")
        img = self._load_thumbnail(label, size=(160, 160))
        if img is not None:
            self.result_image_label.config(image=img, text="")
            self.result_image_label.image = img  # keep a reference, else Tk garbage-collects it
        else:
            self.result_image_label.config(image="", text="(no image found for this label)")

    def _load_thumbnail(self, label, size):
        path = image_path_for_label(label)
        if path is None:
            return None
        im = Image.open(path)
        im.thumbnail(size)
        return ImageTk.PhotoImage(im)

    def _show_popup(self, label, confidence):
        top = tk.Toplevel(self.root)
        top.title("Grasp classified")
        top.geometry("320x360")
        top.resizable(False, False)
        img = self._load_thumbnail(label, size=(240, 240))
        if img is not None:
            img_lbl = ttk.Label(top, image=img)
            img_lbl.image = img
            img_lbl.pack(pady=12)
        ttk.Label(top, text=label, font=("TkDefaultFont", 18, "bold")).pack()
        ttk.Label(top, text=f"confidence: {confidence:.0%}", foreground="gray").pack(pady=(2, 12))
        ttk.Button(top, text="OK", command=top.destroy).pack(pady=4)
        top.transient(self.root)
        top.grab_set()

    # ---------- queue polling (runs on the Tk main thread) ----------
    def _poll_queue(self):
        latest = None
        drained = 0
        while True:
            try:
                item = self.rx_queue.get_nowait()
            except queue.Empty:
                break
            t, s1, s2, s3, fmt = item
            latest = (s1, s2, s3)
            if self.capturing:
                self.buffer.add(t, s1, s2, s3)
            drained += 1

        if latest is not None:
            self.live_labels[0].config(text=f"Sensor 1: {latest[0]:.3f} pF")
            self.live_labels[1].config(text=f"Sensor 2: {latest[1]:.3f} pF")
            self.live_labels[2].config(text=f"Sensor 3: {latest[2]:.3f} pF")

        if self.reader is not None and self.reader.last_error and self.reader.is_alive() is False:
            messagebox.showerror("Serial error", self.reader.last_error)
            self._disconnect()

        if self.capturing:
            self.status_var.set(f"Recording grasp... {len(self.buffer)} samples captured.")

        self.root.after(50, self._poll_queue)

    def _on_close(self):
        self._disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    ClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
