#!/usr/bin/env python3
"""Training utilities for DY SDR Encoder examples.

This module provides common functionality for loading data files,
testing word overlaps, and logging training progress.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any


def load_vocab_file(vocab_path: str) -> List[str]:
    """Load vocabulary from a text file (one word per line)."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    return vocab


def load_test_pairs(test_pairs_path: str) -> List[Tuple[str, str]]:
    """Load test word pairs from a CSV-like file.
    
    Format: word1,word2,description
    Lines starting with # are treated as comments.
    """
    pairs = []
    with open(test_pairs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    return pairs


def get_corpus_info(corpus_path: str) -> Dict[str, Any]:
    """Get basic information about a corpus file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_tokens = sum(len(line.split()) for line in lines)
    
    return {
        'total_lines': len(lines),
        'total_tokens': total_tokens,
        'sample_lines': [line.strip() for line in lines[:5]]
    }


def calculate_overlap(encoder, word1: str, word2: str) -> Tuple[int, float]:
    """Calculate overlap between two words in the encoder.
    
    Returns:
        tuple: (overlap_count, overlap_percentage)
    """
    if word1 not in encoder or word2 not in encoder:
        return 0, 0.0
    
    overlap = np.count_nonzero(encoder.flat(word1) & encoder.flat(word2))
    percentage = (overlap / encoder.active_bits) * 100
    return overlap, percentage


def show_overlap(encoder, word1: str, word2: str, prefix: str = "  ") -> int:
    """Display overlap between two words and return overlap count."""
    overlap, percentage = calculate_overlap(encoder, word1, word2)
    
    if word1 not in encoder or word2 not in encoder:
        missing = [w for w in [word1, word2] if w not in encoder]
        print(f"{prefix}{word1} ↔ {word2}: word(s) not in vocabulary: {missing}")
        return 0
    
    print(f"{prefix}{word1} ↔ {word2}: {overlap}/{encoder.active_bits} bits ({percentage:.1f}%)")
    return overlap


def test_overlaps(encoder, test_pairs: List[Tuple[str, str]], title: str) -> Dict[Tuple[str, str], int]:
    """Test overlaps for all word pairs and return results."""
    print(f"{title}:")
    overlaps = {}
    for word1, word2 in test_pairs:
        overlaps[(word1, word2)] = show_overlap(encoder, word1, word2)
    print()
    return overlaps


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