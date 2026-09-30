import serial
import serial.tools.list_ports
import time
import csv
import os
from datetime import datetime
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import Rectangle, Polygon, Circle


# ============================================================
# USER SETTINGS
# ============================================================

BAUD_RATE = 115200

# Used only when the ESP32 sends all 16 channels
TOTAL_CHANNELS = 16
SELECTED_CHANNELS = [0, 1, 2]

ITEMS = [
    "Square box",
    "Triangular",
    "Ball",
]

CURRENT_ITEM = "Square box"

WINDOW_SIZE = 300
DATASET_FOLDER = "grasp_classification_dataset"
GRAPH_INTERVAL_MS = 50


# ============================================================
# HELPERS
# ============================================================

def safe_name(text):
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


os.makedirs(DATASET_FOLDER, exist_ok=True)


# ============================================================
# FIND COM PORTS
# ============================================================

ports = list(serial.tools.list_ports.comports())

print()
print("=" * 72)
print(" UNIVERSAL ARM - CAPACITIVE GRASP CLASSIFICATION RECORDER")
print("=" * 72)

if len(ports) == 0:
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
    PORT = ports[0].device
else:
    while True:
        try:
            selection = int(input("\nSelect ESP32 port number: "))
            if 0 <= selection < len(ports):
                PORT = ports[selection].device
                break
            print("Please select a valid number.")
        except ValueError:
            print("Please enter a number.")


# ============================================================
# SERIAL CONNECTION
# ============================================================

print()
print("Connecting...")
print("PORT :", PORT)
print("BAUD :", BAUD_RATE)

try:
    ser = serial.Serial(
        PORT,
        BAUD_RATE,
        timeout=0.01
    )

    time.sleep(2)
    ser.reset_input_buffer()

except serial.SerialException as e:
    print()
    print("Serial connection failed:")
    print(e)
    print()
    print("Close Arduino Serial Monitor / Serial Plotter.")
    input("Press Enter to exit...")
    raise SystemExit

print()
print("Connected successfully.")
print("Waiting for ESP32 data...")
print()


# ============================================================
# STATE
# ============================================================

current_item = CURRENT_ITEM
recording = False

csv_file = None
csv_writer = None
recording_id = None
recording_file_path = None

trial_start_time = None
trial_sample_index = 0

valid_packet_count = 0
invalid_packet_count = 0
last_packet_format = "No packet yet"

program_start_time = time.perf_counter()

time_buffer = deque(maxlen=WINDOW_SIZE)

sensor_buffers = [
    deque(maxlen=WINDOW_SIZE)
    for _ in range(3)
]


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14, 9),
    sharex=True
)

plt.subplots_adjust(
    left=0.08,
    right=0.74,
    top=0.90,
    bottom=0.10,
    hspace=0.33
)

fig.suptitle(
    "Universal Arm - Capacitive Grasp Classification",
    fontsize=18
)

lines = []
value_texts = []

for i in range(3):
    line, = axes[i].plot([], [], linewidth=2)
    lines.append(line)

    axes[i].set_title(
        f"Sensor {i + 1} - CH{SELECTED_CHANNELS[i]}"
    )

    axes[i].set_ylabel("Capacitance (pF)")
    axes[i].grid(True)

    value_text = axes[i].text(
        0.02,
        0.90,
        "Waiting...",
        transform=axes[i].transAxes,
        fontsize=11
    )
    value_texts.append(value_text)

axes[2].set_xlabel("Program Time (s)")


# ============================================================
# OBJECT DRAWINGS
# ============================================================

def setup_object_axis(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("gray")

    ax.text(
        0.5,
        0.08,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )


square_ax = fig.add_axes([0.77, 0.73, 0.20, 0.14])
setup_object_axis(square_ax, "Square box")
square_ax.add_patch(
    Rectangle(
        (0.28, 0.30),
        0.32,
        0.32,
        fill=False,
        linewidth=2.5
    )
)
square_ax.add_patch(
    Polygon(
        [
            [0.60, 0.30],
            [0.73, 0.40],
            [0.73, 0.72],
            [0.60, 0.62]
        ],
        closed=True,
        fill=False,
        linewidth=2.0
    )
)
square_ax.add_patch(
    Polygon(
        [
            [0.28, 0.62],
            [0.41, 0.72],
            [0.73, 0.72],
            [0.60, 0.62]
        ],
        closed=True,
        fill=False,
        linewidth=2.0
    )
)


tri_ax = fig.add_axes([0.77, 0.56, 0.20, 0.14])
setup_object_axis(tri_ax, "Triangular")
tri_ax.add_patch(
    Polygon(
        [
            [0.20, 0.28],
            [0.52, 0.72],
            [0.84, 0.28]
        ],
        closed=True,
        fill=False,
        linewidth=2.5
    )
)


ball_ax = fig.add_axes([0.77, 0.39, 0.20, 0.14])
setup_object_axis(ball_ax, "Ball")
ball_ax.add_patch(
    Circle(
        (0.50, 0.48),
        0.20,
        fill=False,
        linewidth=2.5
    )
)


# ============================================================
# CONTROLS
# ============================================================

radio_ax = fig.add_axes([0.77, 0.24, 0.20, 0.11])

radio = RadioButtons(
    radio_ax,
    ITEMS,
    active=ITEMS.index(CURRENT_ITEM)
)

radio_ax.set_title(
    "Select object class",
    fontsize=11
)


start_ax = fig.add_axes([0.77, 0.15, 0.20, 0.06])

start_button = Button(
    start_ax,
    "START RECORDING"
)


stop_ax = fig.add_axes([0.77, 0.07, 0.20, 0.06])

stop_button = Button(
    stop_ax,
    "STOP & SAVE"
)


status_text = fig.text(
    0.77,
    0.012,
    "",
    fontsize=9,
    va="bottom"
)


# ============================================================
# STATUS / HIGHLIGHT
# ============================================================

def update_object_highlight():

    mapping = {
        "Square box": square_ax,
        "Triangular": tri_ax,
        "Ball": ball_ax,
    }

    for name, ax in mapping.items():

        for spine in ax.spines.values():

            if name == current_item:
                spine.set_edgecolor("green")
                spine.set_linewidth(3.0)

            else:
                spine.set_edgecolor("gray")
                spine.set_linewidth(1.5)


def update_status():

    if recording:
        state = "RECORDING / SAVING"
    else:
        state = "LIVE ONLY"

    status_text.set_text(
        f"STATUS: {state}\n"
        f"Selected: {current_item}\n"
        f"Trial samples: {trial_sample_index}\n"
        f"Valid packets: {valid_packet_count}\n"
        f"Packet: {last_packet_format}"
    )


# ============================================================
# CLASS SELECTION
# ============================================================

def select_item(label):

    global current_item

    if recording:
        print("Stop recording before changing class.")
        radio.set_active(ITEMS.index(current_item))
        return

    current_item = label

    update_object_highlight()
    update_status()
    fig.canvas.draw_idle()


radio.on_clicked(select_item)


# ============================================================
# START RECORDING
# ============================================================

def start_recording(event=None):

    global recording
    global csv_file
    global csv_writer
    global recording_id
    global recording_file_path
    global trial_start_time
    global trial_sample_index

    if recording:
        print("Recording is already running.")
        return

    item_folder = os.path.join(
        DATASET_FOLDER,
        safe_name(current_item)
    )

    os.makedirs(item_folder, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    recording_id = (
        f"{safe_name(current_item)}_{timestamp}"
    )

    recording_file_path = os.path.join(
        item_folder,
        recording_id + ".csv"
    )

    csv_file = open(
        recording_file_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "Class",
            "Recording_ID",
            "Sample_Index",
            "Trial_Time_s",
            "PC_Timestamp",
            "Sensor1_pF",
            "Sensor2_pF",
            "Sensor3_pF",
            "Packet_Format",
            "Raw_Packet"
        ]
    )

    csv_file.flush()

    trial_sample_index = 0
    trial_start_time = time.perf_counter()
    recording = True

    print()
    print("=" * 72)
    print(" RECORDING STARTED")
    print("=" * 72)
    print("Class :", current_item)
    print("File  :", os.path.abspath(recording_file_path))
    print("=" * 72)

    update_status()


start_button.on_clicked(start_recording)


# ============================================================
# STOP / SAVE
# ============================================================

def stop_recording(event=None):

    global recording
    global csv_file
    global csv_writer

    if not recording:
        print("No active recording.")
        return

    recording = False

    if csv_file is not None:
        csv_file.flush()
        csv_file.close()

    csv_file = None
    csv_writer = None

    print()
    print("=" * 72)
    print(" RECORDING SAVED")
    print("=" * 72)
    print("Class   :", current_item)
    print("Samples :", trial_sample_index)
    print("File    :", os.path.abspath(recording_file_path))
    print("=" * 72)

    update_status()


stop_button.on_clicked(stop_recording)


# ============================================================
# UNIVERSAL PACKET PARSER
# ============================================================

def parse_packet(line):

    global invalid_packet_count
    global last_packet_format

    if not line:
        return None

    # Accept packets with or without "$"
    if line.startswith("$"):
        payload = line[1:]
    else:
        payload = line

    parts = [
        x.strip()
        for x in payload.split(",")
    ]

    try:
        numbers = [
            float(x)
            for x in parts
        ]
    except ValueError:
        invalid_packet_count += 1
        return None


    # --------------------------------------------------------
    # FORMAT A:
    # Original ESP32 program:
    # $CH0,CH1,...,CH15
    # --------------------------------------------------------

    if len(numbers) == 16:

        values = [
            numbers[SELECTED_CHANNELS[0]],
            numbers[SELECTED_CHANNELS[1]],
            numbers[SELECTED_CHANNELS[2]]
        ]

        last_packet_format = "16-channel"


    # --------------------------------------------------------
    # FORMAT B:
    # $time_ms,sensor1,sensor2,sensor3
    # --------------------------------------------------------

    elif len(numbers) == 4:

        values = [
            numbers[1],
            numbers[2],
            numbers[3]
        ]

        last_packet_format = "timestamp + 3 sensors"


    # --------------------------------------------------------
    # FORMAT C:
    # $sensor1,sensor2,sensor3
    # --------------------------------------------------------

    elif len(numbers) == 3:

        values = [
            numbers[0],
            numbers[1],
            numbers[2]
        ]

        last_packet_format = "3 sensors"


    else:

        invalid_packet_count += 1

        print(
            f"IGNORED PACKET: got {len(numbers)} values"
        )
        print("RAW:", line)

        return None


    return values


# ============================================================
# READ SERIAL
# ============================================================

def read_serial_packets():

    global valid_packet_count
    global trial_sample_index

    samples = []


    # Do not rely only on ser.in_waiting.
    # readline() has a short timeout and works well with ESP32 USB CDC.
    max_reads = 100

    for _ in range(max_reads):

        try:
            raw = ser.readline()

        except Exception as e:
            print("SERIAL READ ERROR:", e)
            break


        if not raw:
            break


        line = raw.decode(
            "utf-8",
            errors="ignore"
        ).strip()


        if not line:
            continue


        values = parse_packet(line)


        if values is None:
            continue


        valid_packet_count += 1

        program_time = (
            time.perf_counter()
            -
            program_start_time
        )

        samples.append(
            (
                program_time,
                values[0],
                values[1],
                values[2]
            )
        )


        # ----------------------------------------------------
        # SAVE ONLY DURING A RECORDING
        # ----------------------------------------------------

        if recording:

            trial_time = (
                time.perf_counter()
                -
                trial_start_time
            )

            trial_sample_index += 1

            timestamp = datetime.now().isoformat(
                timespec="milliseconds"
            )

            csv_writer.writerow(
                [
                    current_item,
                    recording_id,
                    trial_sample_index,
                    f"{trial_time:.6f}",
                    timestamp,
                    f"{values[0]:.6f}",
                    f"{values[1]:.6f}",
                    f"{values[2]:.6f}",
                    last_packet_format,
                    line
                ]
            )

            csv_file.flush()


    return samples


# ============================================================
# GRAPH UPDATE
# ============================================================

def update(frame):

    samples = read_serial_packets()


    for t, s1, s2, s3 in samples:

        time_buffer.append(t)

        sensor_buffers[0].append(s1)
        sensor_buffers[1].append(s2)
        sensor_buffers[2].append(s3)


    if len(time_buffer) > 0:

        x = list(time_buffer)

        for i in range(3):

            y = list(sensor_buffers[i])

            lines[i].set_data(x, y)

            if len(y) > 0:
                value_texts[i].set_text(
                    f"{y[-1]:.4f} pF"
                )


            if len(x) > 1:

                if x[-1] <= 30:
                    axes[i].set_xlim(0, 30)

                else:
                    axes[i].set_xlim(
                        x[-1] - 30,
                        x[-1]
                    )


            if len(y) > 2:

                ymin = min(y)
                ymax = max(y)

                diff = ymax - ymin

                if diff < 0.01:
                    diff = 0.01

                margin = diff * 0.20

                axes[i].set_ylim(
                    ymin - margin,
                    ymax + margin
                )


    update_status()

    return lines + value_texts


# ============================================================
# SHORTCUTS
# ============================================================

def keyboard_control(event):

    if event.key == "s":
        start_recording()

    elif event.key == "e":
        stop_recording()


fig.canvas.mpl_connect(
    "key_press_event",
    keyboard_control
)


update_object_highlight()
update_status()


# ============================================================
# ANIMATION
# ============================================================

ani = FuncAnimation(
    fig,
    update,
    interval=GRAPH_INTERVAL_MS,
    blit=False,
    cache_frame_data=False
)


# ============================================================
# RUN
# ============================================================

try:

    plt.show()


finally:

    if recording:

        recording = False

        if csv_file is not None:
            csv_file.flush()
            csv_file.close()


    if ser.is_open:
        ser.close()


    print()
    print("=" * 72)
    print("PROGRAM CLOSED")
    print("Valid packets  :", valid_packet_count)
    print("Invalid packets:", invalid_packet_count)
    print("=" * 72)