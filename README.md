# DY SDR Encoder

A pip-installable Python package for training dynamic Sparse Distributed Representations (SDRs) using Hebbian learning with bit-swap algorithm.

## Overview

This package implements a novel approach to creating semantic representations by:
1. Initializing random 2-D SDRs for vocabulary words
2. Training SDRs using Hebbian learning + bit migration
3. Producing representations where semantically related words have greater bit overlap

## Installation

```bash
pip install dy-sdr-encoder
```

For development:
```bash
git clone <repository-url>
cd dy_sdr_encoder
pip install -e ".[dev]"
```

## Quick Start

### Library API

```python
from dy_sdr_encoder import Encoder, train
from pathlib import Path

# Create encoder with 128x128 grid, 2% sparsity
enc = Encoder(grid=(128, 128), sparsity=0.02, rng=42)

# Initialize vocabulary with random SDRs
vocab = ["cat", "dog", "animal", "table", "chair"]
enc.init_vocab(vocab)

# Get 2D representation
vec2d = enc["cat"]  # shape (128, 128)

# Get flattened view (O(1) operation)
vec1d = enc.flat("cat")  # shape (16384,)

# Train on corpus
train(
    encoder=enc,
    corpus_path=Path("data/corpus.txt"),
    window_size=5,
    epochs=3,
    swap_per_epoch=3,
    neighbourhood_d=2,
)

# Save and load
enc.save("encoder.npz")
enc2 = Encoder.load("encoder.npz")
```

### Command Line Interface

```bash
# Train with default configuration
dy-sdr-train --config configs/default.yaml

# Custom parameters
dy-sdr-train --corpus-path data/my_corpus.txt --epochs 5
```

## Algorithm

The training uses a Hebbian learning rule with bit migration:

1. **Context Collection**: For each word, collect surrounding words within a window
2. **Hebbian Updates**: Increment counters for missed opportunities, decrement for lonely bits
3. **Bit Migration**: At epoch end, swap low-scoring active bits with high-scoring inactive bits in the neighborhood

This creates representations where words appearing in similar contexts develop overlapping bit patterns.

## Configuration

Example `configs/default.yaml`:

```yaml
corpus_path: ../data/corpus.txt
vocab_path: ../data/vocab.txt

grid:
  H: 128
  W: 128

sparsity: 0.02
window_size: 5
epochs: 3
swap_per_epoch: 3
neighbourhood_d: 2
seed: 42
```

## Data Format

- **Corpus**: Plain text file, one sentence per line
- **Vocabulary**: Newline-separated words (optional, can be extracted from corpus)

## Testing

```bash
pytest
```

## License

MIT License - see LICENSE file for details.

## Dependencies

- numpy >= 1.23
- typer >= 0.9
- pyyaml
- pytest >= 7 (dev) 