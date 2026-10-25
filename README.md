# Soft Robotic Arm Object Classifier

This repository classifies a grasped object from a three-channel capacitive
sensor recording. It extracts recording-level signal features and trains a
balanced logistic-regression pipeline for the `Ball`, `Square_Box`, and
`Triangular` classes.

## Contents

- `build_dataset.py` — converts local labelled recordings into feature tables.
- `train_model.py` — trains and evaluates the classifier, then writes a local
  deployable artifact.
- `evaluate.py` — command-line and importable inference helpers.
- `classify_gui.py` — live classification interface.
- `arduino_signal.py` and `FDC2214_grasp_GUI_AUTO_PACKET.py` — data-capture
  utilities for the sensor hardware.
- `assets/` — class illustrations used by the GUI.

## Setup

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

The raw labelled recordings live outside this repository, in sibling class
folders named `Ball`, `Square_Box`, and `Triangular`. From this directory:

```bash
python build_dataset.py
python train_model.py
```

Generated feature tables, fitted model artifacts, diagnostics, logs, and search
results are intentionally excluded from Git. Keep a separately versioned data
archive if you need to reproduce a particular training result.

## Predict from a recording

```bash
python evaluate.py path/to/recording.csv --proba
```

The classifier artifact must first be generated with `train_model.py`. For the
accepted raw CSV column names, see `feature_extraction.py`.

## Development

The repository uses a 100-character line limit. With Ruff and Black installed,
run:

```bash
ruff check .
black .
```
