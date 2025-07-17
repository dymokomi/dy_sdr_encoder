#!/usr/bin/env python3
"""Training utilities for DY SDR Encoder examples.

This module provides example-specific functionality for display and progress tracking.
Core dataset and testing functions have been moved to dy_sdr_encoder.dataset and dy_sdr_encoder.testing.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add the source directory to Python path for examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import from the main package
from dy_sdr_encoder.dataset import load_vocab_file, load_test_pairs, get_corpus_info
from dy_sdr_encoder.testing import calculate_overlap, show_overlap, test_overlaps


def show_progress_summary(test_pairs: List[Tuple[str, str]], epoch_overlaps: List[Tuple[str, Dict]], encoder):
    """Show a comprehensive progress summary across all epochs."""
    print("=== Training Complete - Full Progress Summary ===")
    print("Overlap progression:")
    
    for word1, word2 in test_pairs:
        print(f"\n{word1} ↔ {word2}:")
        for epoch_name, overlaps in epoch_overlaps:
            overlap_val = overlaps.get((word1, word2), 0)
            if overlap_val > 0:
                percentage = (overlap_val / encoder.active_bits) * 100
                print(f"  {epoch_name}: {overlap_val}/{encoder.active_bits} bits ({percentage:.1f}%)")
    print()


def show_epoch_progress(test_pairs: List[Tuple[str, str]], initial_overlaps: Dict, current_overlaps: Dict):
    """Show progress for the current epoch compared to initial state."""
    print("Progress since start:")
    for word1, word2 in test_pairs:
        initial = initial_overlaps.get((word1, word2), 0)
        current = current_overlaps.get((word1, word2), 0)
        change = current - initial
        if initial > 0 or current > 0:  # Only show if at least one measurement exists
            print(f"  {word1} ↔ {word2}: {initial} → {current} ({change:+d})")
    print()


def print_encoder_info(encoder):
    """Print comprehensive encoder information."""
    print(f"Encoder initialized with {encoder.vocab_size} words")
    print(f"Grid size: {encoder.height}x{encoder.width} = {encoder.total_bits} bits")
    print(f"Active bits per word: {encoder.active_bits}")
    print(f"Sparsity: {encoder.sparsity:.3f}")
    print()


def print_data_info(vocab: List[str], corpus_info: Dict[str, Any]):
    """Print information about loaded data."""
    print("Vocabulary:")
    print(f"  Size: {len(vocab)} words")
    print(f"  Sample: {vocab[:10]}")
    if len(vocab) > 10:
        print(f"  ... and {len(vocab) - 10} more words")
    print()
    
    print("Corpus:")
    print(f"  Total lines: {corpus_info['total_lines']}")
    print(f"  Total tokens: {corpus_info['total_tokens']}")
    print("  Sample lines:")
    for i, line in enumerate(corpus_info['sample_lines'], 1):
        print(f"    {i}. {line}")
    print()


def save_encoder_with_info(encoder, save_path: Path, example_name: str):
    """Save encoder and print information about the saved model."""
    save_path.parent.mkdir(exist_ok=True)
    encoder.save(save_path)
    print(f"Encoder saved to: {save_path}")
    
    print(f"\nEncoder statistics:")
    print(f"  Vocabulary size: {encoder.vocab_size}")
    print(f"  Grid dimensions: {encoder.height} × {encoder.width}")
    print(f"  Total bits: {encoder.total_bits}")
    print(f"  Active bits per word: {encoder.active_bits}")
    print(f"  Sparsity: {encoder.sparsity:.3f}")
    
    print(f"\n=== {example_name} Complete ===")
    print("The encoder has learned to associate semantically related words!")
    print("Related words now share more bits in their sparse representations.") 