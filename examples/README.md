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

### 1. Expanded Vocabulary Training (`example1_embedded_training.py`)
Train an encoder using external vocabulary, corpus, and test pairs files. This example uses a curated dataset with rich semantic relationships perfect for understanding the core concepts and seeing clear training progress.

**Data files used:**
- `data/vocab.txt` - Expanded vocabulary (264 words) 
- `data/corpus.txt` - Rich corpus with semantic relationships (374 lines)
- `data/test_pairs.txt` - Diverse test word pairs for overlap measurement (79 pairs)

**Usage:**
```bash
cd examples
python example1_embedded_training.py
```

### 2. Large File-based Training (`example2_file_training.py`)
Train an encoder using large vocabulary and corpus files. This demonstrates the typical workflow when working with real-world datasets.

**Data files used:**
- `../data/train.vocab` - Large vocabulary file
- `../data/train.txt` - Large corpus file  
- `data/test_pairs_large.txt` - Test word pairs suitable for large vocabulary

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

## Utility Modules

### `training_utils.py`
Common functionality for loading data files, testing word overlaps, and progress tracking:
- `load_vocab_file()` - Load vocabulary from text file
- `load_test_pairs()` - Load test word pairs from CSV-like file
- `get_corpus_info()` - Get corpus statistics
- `test_overlaps()` - Test overlaps for word pairs
- `show_progress_summary()` - Show training progress summary
- Functions for printing encoder info and saving models

### `logging_utils.py`
Structured logging for training experiments:
- `TrainingLogger` class for comprehensive experiment logging
- JSON-based log format with encoder config, training progress, and statistics
- Export functions for CSV and markdown table formats
- Automatic timestamp tracking and experiment naming

## Data Files Structure

```
examples/
├── data/                          # Example data
│   ├── vocab.txt                  # Expanded vocabulary (264 words)
│   ├── corpus.txt                 # Rich corpus with relationships (374 lines)
│   ├── test_pairs.txt             # Test pairs for expanded vocab (79 pairs)
│   └── test_pairs_large.txt       # Test pairs for large vocab
├── models/                        # Saved trained encoders
├── logs/                          # Training logs (auto-created)
├── training_utils.py              # Common training utilities
├── logging_utils.py               # Logging and export utilities  
└── example*.py                    # Example scripts
```

Large corpus data:
```
data/                              # Large datasets (in project root)
├── train.txt                      # Large corpus file (404MB)
└── train.vocab                    # Large vocabulary (1MB)
```

## Features

- **Clean separation**: Data, utilities, and examples are cleanly separated
- **External data files**: All vocabulary, corpus, and test data loaded from files
- **Streamlined code**: Examples focus on workflow, utilities handle complexity
- **Comprehensive logging**: Optional structured logging with export capabilities
- **Flexible test pairs**: Easy to modify test word pairs via external files
- **Self-contained**: No package installation required, examples work out-of-the-box 