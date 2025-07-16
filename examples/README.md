# Examples

This directory contains toy examples demonstrating how to use the DY SDR Encoder.

## Setup

The examples are **self-contained** and don't require installing the package. They automatically add the source directory to Python's path.

For the QT visualization example, install PySide6:
```bash
pip install -r examples/requirements.txt
```

Alternatively, if you want to install the package system-wide:
```bash
pip install -e .  # Install in development mode from project root
```

## Examples

### 1. Embedded Training (`example1_embedded_training.py`)
Train a small encoder with vocabulary and text embedded directly in the source code. This is great for quick experimentation and understanding the basic API.

**Usage:**
```bash
cd examples
python example1_embedded_training.py
```

### 2. File-based Training (`example2_file_training.py`)
Train an encoder using separate vocabulary and corpus files. This demonstrates the typical workflow when working with larger datasets.

**Usage:**
```bash
cd examples
python example2_file_training.py
```

### 3. Interactive QT Visualization (`example3_qt_visualization.py`)
A graphical application that loads a trained encoder and visualizes word representations as 2D grids. Type a word at the bottom and see its sparse distributed representation at the top.

**Usage:**
```bash
cd examples
python example3_qt_visualization.py
```

## Data Files

- `../data/train.txt` - Sample corpus file for file-based training
- `../data/train.vocab` - Sample vocabulary file for file-based training
- `models/` - Directory where trained encoders are saved 