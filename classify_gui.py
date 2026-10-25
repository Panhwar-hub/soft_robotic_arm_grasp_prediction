"""
Live grasp classifier GUI for the Universal Arm capacitive-sensor setup.

Visual style matches arduino_signal.py on purpose (same 3-channel live plot,
same right-hand object panel, same button layout) so it feels like part of
the same toolkit -- just for classifying instead of recording.

This is a SEPARATE program from arduino_signal.py -- that recorder is left
completely untouched, so the CSV recordings it produces (and any future
recordings you make with it) stay in exactly the format the model was
trained on. This script opens its OWN serial connection to the same ESP32
and reuses the identical packet-parsing rule (parse_packet() below, copied
from arduino_signal.py on purpose -- if you ever change the ESP32 firmware's
packet format, update BOTH copies) so the raw numbers it sees are exactly
the numbers arduino_signal.py would have written to a CSV.

Workflow:
  1. Run the script, pick the ESP32's serial port when prompted.
  2. Click "START GRASP", present/hold the object to the arm.
  3. Click "STOP & CLASSIFY" when the grasp is done.
  4. The matching object panel on the right lights up green, the prediction
     + confidence are printed next to it, and a pop-up window shows the
     object's photo full-size.

The classification itself (preprocessing, feature extraction, the trained
model) is 100% delegated to evaluate.py / feature_extraction.py in this same
folder -- this file adds no classification logic of its own, so training
and live inference can never drift apart.
"""
import os
import sys
import time
import csv
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image, ImageOps

from evaluate import predict_from_raw

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "assets")
LOG_PATH = os.path.join(HERE, "live_classification_log.csv")

# ============================================================
# USER SETTINGS -- kept the same as arduino_signal.py
# ============================================================

BAUD_RATE = 115200
SELECTED_CHANNELS = [0, 1, 2]
WINDOW_SIZE = 300          # samples kept on screen for the live plot
GRAPH_INTERVAL_MS = 50
MIN_SAMPLES_TO_CLASSIFY = 8  # extract_features() itself requires >= 8

ITEMS = ["Square box", "Triangular", "Ball"]  # same objects/order as arduino_signal.py

IMAGE_FILES = {
    "Square box": "square.jpg",
    "Triangular": "triangle.jpeg",
    "Ball": "ball.jpeg",
}
# fallback keyword matcher, in case the model's exact class strings ever
# drift from the ITEMS list above (e.g. after retraining on relabeled data)
IMAGE_KEYWORDS = [("ball", "ball.jpeg"), ("square", "square.jpg"), ("triang", "triangle.jpeg")]


def image_path_for_label(label):
    fname = IMAGE_FILES.get(label)
    if fname and os.path.exists(os.path.join(ASSETS_DIR, fname)):
        return os.path.join(ASSETS_DIR, fname)
    norm = label.lower()
    for kw, fname in IMAGE_KEYWORDS:
        if kw in norm:
            path = os.path.join(ASSETS_DIR, fname)
            if os.path.exists(path):
                return path
    return None


_thumb_cache = {}


def load_thumbnail(label, size=(220, 220)):
    """Loads + center-crops an object photo to a square so the panels line
    up neatly regardless of the source photo's aspect ratio. Cached so the
    live 20fps redraw loop never touches disk."""
    key = (label, size)
    if key in _thumb_cache:
        return _thumb_cache[key]
    path = image_path_for_label(label)
    if path is None:
        _thumb_cache[key] = None
        return None
    im = Image.open(path).convert("RGB")
    im = ImageOps.fit(im, size, Image.LANCZOS)
    arr = np.asarray(im)
    _thumb_cache[key] = arr
    return arr


# ============================================================
# PACKET PARSER -- identical logic to arduino_signal.py's parse_packet()
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


def log_prediction(label, proba, n_samples, duration_s):
    """Small append-only audit trail of live predictions (not raw signal)."""
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(["timestamp", "predicted_label", "confidence", "n_samples", "duration_s"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), label, f"{proba.get(label, 0):.4f}", n_samples, f"{duration_s:.3f}"])


# ============================================================
# FIGURE BUILDING -- pure functions, no serial/animation involved, so they
# can be exercised head-to-head with a plain script (see test_gui_layout.py)
# ============================================================

def build_figure():
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    plt.subplots_adjust(left=0.08, right=0.74, top=0.90, bottom=0.10, hspace=0.33)
    fig.suptitle("Universal Arm - Live Grasp Classifier", fontsize=18)

    lines, value_texts = [], []
    for i in range(3):
        line, = axes[i].plot([], [], linewidth=2)
        lines.append(line)
        axes[i].set_title(f"Sensor {i + 1} - CH{SELECTED_CHANNELS[i]}")
        axes[i].set_ylabel("Capacitance (pF)")
        axes[i].grid(True)
        vt = axes[i].text(0.02, 0.90, "Waiting...", transform=axes[i].transAxes, fontsize=11)
        value_texts.append(vt)
    axes[2].set_xlabel("Program Time (s)")

    # object panels, same coordinates arduino_signal.py used for its
    # vector-drawn shapes -- now showing the actual object photos instead
    coords = {
        "Square box": [0.77, 0.73, 0.20, 0.16],
        "Triangular": [0.77, 0.55, 0.20, 0.16],
        "Ball": [0.77, 0.37, 0.20, 0.16],
    }
    image_axes = {}
    for label, rect in coords.items():
        ax = fig.add_axes(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_edgecolor("gray")
        img = load_thumbnail(label, size=(200, 200))
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "(no image)", ha="center", va="center", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        image_axes[label] = ax

    pred_text = fig.text(0.77, 0.335, "Prediction: --\n(no grasp classified yet)", fontsize=10, va="top")

    start_ax = fig.add_axes([0.77, 0.15, 0.20, 0.06])
    start_button = Button(start_ax, "START GRASP")

    stop_ax = fig.add_axes([0.77, 0.07, 0.20, 0.06])
    stop_button = Button(stop_ax, "STOP && CLASSIFY")

    status_text = fig.text(0.77, 0.012, "", fontsize=9, va="bottom")

    return {
        "fig": fig, "axes": axes, "lines": lines, "value_texts": value_texts,
        "image_axes": image_axes, "pred_text": pred_text,
        "start_button": start_button, "stop_button": stop_button,
        "status_text": status_text,
    }


def update_live_plot(handles, time_buffer, sensor_buffers):
    axes, lines, value_texts = handles["axes"], handles["lines"], handles["value_texts"]
    if len(time_buffer) == 0:
        return
    x = list(time_buffer)
    for i in range(3):
        y = list(sensor_buffers[i])
        lines[i].set_data(x, y)
        if y:
            value_texts[i].set_text(f"{y[-1]:.4f} pF")
        if len(x) > 1:
            if x[-1] <= 30:
                axes[i].set_xlim(0, 30)
            else:
                axes[i].set_xlim(x[-1] - 30, x[-1])
        if len(y) > 2:
            ymin, ymax = min(y), max(y)
            diff = max(ymax - ymin, 0.01)
            margin = diff * 0.20
            axes[i].set_ylim(ymin - margin, ymax + margin)


def highlight_prediction(handles, predicted_label):
    for label, ax in handles["image_axes"].items():
        for spine in ax.spines.values():
            if label == predicted_label:
                spine.set_edgecolor("limegreen")
                spine.set_linewidth(4.0)
            else:
                spine.set_edgecolor("gray")
                spine.set_linewidth(2.0)


def set_prediction_text(handles, message=None, label=None, proba=None):
    if message is not None:
        handles["pred_text"].set_text(message)
        return
    lines_ = [f"Prediction: {label}", ""]
    for c, p in sorted(proba.items(), key=lambda kv: -kv[1]):
        lines_.append(f"  {c}: {p:.0%}")
    handles["pred_text"].set_text("\n".join(lines_))


def build_popup_figure(label, confidence):
    img = load_thumbnail(label, size=(320, 320))
    fig2 = plt.figure(figsize=(4.4, 5.2))
    fig2.suptitle("Grasp classified", fontsize=14)
    ax = fig2.add_axes([0.08, 0.26, 0.84, 0.64])
    ax.set_xticks([])
    ax.set_yticks([])
    if img is not None:
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "(no image found)", ha="center", va="center")
    for spine in ax.spines.values():
        spine.set_edgecolor("limegreen")
        spine.set_linewidth(3.0)
    fig2.text(0.5, 0.18, label, ha="center", fontsize=18, fontweight="bold")
    fig2.text(0.5, 0.11, f"confidence: {confidence:.0%}", ha="center", fontsize=11, color="dimgray")
    close_ax = fig2.add_axes([0.32, 0.03, 0.36, 0.06])
    close_button = Button(close_ax, "Close")
    close_button.on_clicked(lambda evt: plt.close(fig2))
    fig2._close_button_ref = close_button  # keep a reference alive
    return fig2


# ============================================================
# CLASSIFICATION GLUE
# ============================================================

def classify_grasp(t, s1, s2, s3):
    if len(t) < MIN_SAMPLES_TO_CLASSIFY:
        raise ValueError(
            f"Only {len(t)} samples captured (need at least {MIN_SAMPLES_TO_CLASSIFY}). "
            f"Hold the grasp a little longer before clicking STOP & CLASSIFY."
        )
    return predict_from_raw(t, s1, s2, s3, return_proba=True)


# ============================================================
# LIVE APP (real serial hardware + animation loop) -- only runs when this
# file is executed directly, so the pure functions above stay importable
# and testable without a serial port attached.
# ============================================================

def _select_port():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    print()
    print("=" * 72)
    print(" UNIVERSAL ARM - LIVE GRASP CLASSIFIER")
    print("=" * 72)
    if not ports:
        print("No COM port detected.")
        input("Press Enter to exit...")
        raise SystemExit
    print()
    print("AVAILABLE COM PORTS")
    print("-" * 72)
    for i, port in enumerate(ports):
        print(f"[{i}] {port.device:<8} | {port.description}")
    print("-" * 72)
    if len(ports) == 1:
        return ports[0].device
    while True:
        try:
            selection = int(input("\nSelect ESP32 port number: "))
            if 0 <= selection < len(ports):
                return ports[selection].device
            print("Please select a valid number.")
        except ValueError:
            print("Please enter a number.")


def _connect(port):
    import serial
    print()
    print("Connecting...")
    print("PORT :", port)
    print("BAUD :", BAUD_RATE)
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.01)
        time.sleep(2)
        ser.reset_input_buffer()
    except Exception as e:
        print()
        print("Serial connection failed:")
        print(e)
        print()
        print("Close the Arduino Serial Monitor / Serial Plotter.")
        input("Press Enter to exit...")
        raise SystemExit
    print()
    print("Connected successfully.")
    print("Waiting for ESP32 data...")
    print()
    return ser


def main():
    from matplotlib.animation import FuncAnimation

    port = _select_port()
    ser = _connect(port)

    handles = build_figure()

    capturing = False
    last_packet_format = "No packet yet"
    valid_packet_count = 0
    invalid_packet_count = 0
    program_start_time = time.perf_counter()

    time_buffer = deque(maxlen=WINDOW_SIZE)
    sensor_buffers = [deque(maxlen=WINDOW_SIZE) for _ in range(3)]

    grasp_t, grasp_s1, grasp_s2, grasp_s3 = [], [], [], []
    grasp_start_time = None

    def read_serial_packets():
        nonlocal valid_packet_count, invalid_packet_count, last_packet_format
        nonlocal grasp_start_time
        samples = []
        for _ in range(100):
            try:
                raw = ser.readline()
            except Exception as e:
                print("SERIAL READ ERROR:", e)
                break
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            values, fmt = parse_packet(line)
            if values is None:
                invalid_packet_count += 1
                continue
            last_packet_format = fmt
            valid_packet_count += 1
            program_time = time.perf_counter() - program_start_time
            samples.append((program_time, values[0], values[1], values[2]))

            if capturing:
                if grasp_start_time is None:
                    grasp_start_time = program_time
                grasp_t.append(program_time - grasp_start_time)
                grasp_s1.append(values[0])
                grasp_s2.append(values[1])
                grasp_s3.append(values[2])
        return samples

    def start_grasp(event=None):
        nonlocal capturing, grasp_start_time
        if capturing:
            return
        grasp_t.clear(); grasp_s1.clear(); grasp_s2.clear(); grasp_s3.clear()
        grasp_start_time = None
        capturing = True
        set_prediction_text(handles, message="Recording grasp...\nclick STOP & CLASSIFY when done.")
        handles["fig"].canvas.draw_idle()

    def stop_and_classify(event=None):
        nonlocal capturing
        if not capturing:
            return
        capturing = False
        try:
            label, proba = classify_grasp(grasp_t, grasp_s1, grasp_s2, grasp_s3)
        except ValueError as e:
            set_prediction_text(handles, message=f"Grasp too short:\n{e}")
            handles["fig"].canvas.draw_idle()
            return
        except Exception as e:
            set_prediction_text(handles, message=f"Classification failed:\n{e}")
            handles["fig"].canvas.draw_idle()
            return

        duration = grasp_t[-1] if grasp_t else 0.0
        log_prediction(label, proba, len(grasp_t), duration)

        highlight_prediction(handles, label)
        set_prediction_text(handles, label=label, proba=proba)
        handles["fig"].canvas.draw_idle()

        popup = build_popup_figure(label, proba.get(label, 0.0))
        plt.show(block=False)

    handles["start_button"].on_clicked(start_grasp)
    handles["stop_button"].on_clicked(stop_and_classify)

    def update(frame):
        samples = read_serial_packets()
        for t, s1, s2, s3 in samples:
            time_buffer.append(t)
            sensor_buffers[0].append(s1)
            sensor_buffers[1].append(s2)
            sensor_buffers[2].append(s3)
        update_live_plot(handles, time_buffer, sensor_buffers)
        state = "RECORDING GRASP" if capturing else "LIVE"
        handles["status_text"].set_text(
            f"STATUS: {state}\n"
            f"Grasp samples: {len(grasp_t)}\n"
            f"Valid packets: {valid_packet_count}  Invalid: {invalid_packet_count}\n"
            f"Packet: {last_packet_format}"
        )
        return handles["lines"] + handles["value_texts"]

    ani = FuncAnimation(handles["fig"], update, interval=GRAPH_INTERVAL_MS, blit=False, cache_frame_data=False)

    try:
        plt.show()
    finally:
        if ser.is_open:
            ser.close()
        print()
        print("=" * 72)
        print("PROGRAM CLOSED")
        print("Valid packets  :", valid_packet_count)
        print("Invalid packets:", invalid_packet_count)
        print("=" * 72)


if __name__ == "__main__":
    main()
