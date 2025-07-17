"""DY SDR Encoder - Dynamic Sparse Distributed Representations with Hebbian Learning."""

from __future__ import annotations

from .encoder import Encoder
from .train_loop import train

# Dataset utilities
from .dataset import (
    load_vocab_file,
    load_test_pairs, 
    get_corpus_info,
    validate_corpus_vocabulary
)

# Testing utilities  
from .testing import (
    calculate_overlap,
    show_overlap,
    test_overlaps,
    analyze_semantic_relationships,
    compare_encoders,
    track_training_progress
)

__version__ = "0.1.0"
__all__ = [
    # Core classes and functions
    'Encoder',
    'train',
    
    # Dataset utilities
    'load_vocab_file',
    'load_test_pairs',
    'get_corpus_info', 
    'validate_corpus_vocabulary',
    
    # Testing utilities
    'calculate_overlap',
    'show_overlap', 
    'test_overlaps',
    'analyze_semantic_relationships',
    'compare_encoders',
    'track_training_progress'
]
