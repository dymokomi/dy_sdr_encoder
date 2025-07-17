#!/usr/bin/env python3
"""Testing utilities for DY SDR Encoder.

This module provides functions for testing and evaluating encoder performance,
including overlap calculations and word relationship analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from .encoder import Encoder


def calculate_overlap(encoder: Encoder, word1: str, word2: str) -> Tuple[int, float]:
    """Calculate overlap between two words in the encoder.
    
    Args:
        encoder: The encoder instance
        word1: First word
        word2: Second word
        
    Returns:
        Tuple of (overlap_count, overlap_percentage)
        Returns (0, 0.0) if either word not in vocabulary
    """
    if word1 not in encoder or word2 not in encoder:
        return 0, 0.0
    
    overlap = int(np.count_nonzero(encoder.flat(word1) & encoder.flat(word2)))
    percentage = (overlap / encoder.active_bits) * 100
    return overlap, percentage


def show_overlap(encoder: Encoder, word1: str, word2: str, 
                prefix: str = "  ", verbose: bool = True) -> int:
    """Display overlap between two words and return overlap count.
    
    Args:
        encoder: The encoder instance
        word1: First word
        word2: Second word
        prefix: Prefix string for output formatting
        verbose: Whether to print the result
        
    Returns:
        Number of overlapping bits (0 if words not in vocabulary)
    """
    overlap, percentage = calculate_overlap(encoder, word1, word2)
    
    if word1 not in encoder or word2 not in encoder:
        missing = [w for w in [word1, word2] if w not in encoder]
        if verbose:
            print(f"{prefix}{word1} ↔ {word2}: word(s) not in vocabulary: {missing}")
        return 0
    
    if verbose:
        print(f"{prefix}{word1} ↔ {word2}: {overlap}/{encoder.active_bits} bits ({percentage:.1f}%)")
    return overlap


def test_overlaps(encoder: Encoder, test_pairs: List[Tuple[str, str]], 
                 title: str = "Overlap Results", verbose: bool = True) -> Dict[Tuple[str, str], int]:
    """Test overlaps for all word pairs and return results.
    
    Args:
        encoder: The encoder instance
        test_pairs: List of word pairs to test
        title: Title for the output section
        verbose: Whether to print results
        
    Returns:
        Dictionary mapping (word1, word2) tuples to overlap counts
    """
    if verbose:
        print(f"{title}:")
    
    overlaps = {}
    for word1, word2 in test_pairs:
        overlaps[(word1, word2)] = show_overlap(encoder, word1, word2, verbose=verbose)
    
    if verbose:
        print()
    
    return overlaps


def analyze_semantic_relationships(encoder: Encoder, test_pairs: List[Tuple[str, str]], 
                                 categories: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> Dict[str, Any]:
    """Analyze semantic relationships across different categories of word pairs.
    
    Args:
        encoder: The encoder instance
        test_pairs: List of word pairs to analyze
        categories: Optional dictionary mapping category names to lists of word pairs
        
    Returns:
        Dictionary with analysis results including overlap statistics by category
    """
    if categories is None:
        # Default categorization based on overlap levels
        overlaps = test_overlaps(encoder, test_pairs, verbose=False)
        categories = {
            'high_overlap': [(w1, w2) for (w1, w2), overlap in overlaps.items() if overlap > encoder.active_bits * 0.3],
            'medium_overlap': [(w1, w2) for (w1, w2), overlap in overlaps.items() if encoder.active_bits * 0.1 < overlap <= encoder.active_bits * 0.3],
            'low_overlap': [(w1, w2) for (w1, w2), overlap in overlaps.items() if overlap <= encoder.active_bits * 0.1]
        }
    
    results = {}
    for category, pairs in categories.items():
        if not pairs:
            continue
            
        overlaps = test_overlaps(encoder, pairs, verbose=False)
        overlap_values = list(overlaps.values())
        
        results[category] = {
            'pair_count': len(pairs),
            'mean_overlap': np.mean(overlap_values) if overlap_values else 0,
            'std_overlap': np.std(overlap_values) if overlap_values else 0,
            'min_overlap': min(overlap_values) if overlap_values else 0,
            'max_overlap': max(overlap_values) if overlap_values else 0,
            'overlap_percentage': np.mean(overlap_values) / encoder.active_bits * 100 if overlap_values else 0
        }
    
    return results


def compare_encoders(encoders: Dict[str, Encoder], test_pairs: List[Tuple[str, str]]) -> Dict[str, Dict]:
    """Compare overlap patterns across multiple encoders.
    
    Args:
        encoders: Dictionary mapping encoder names to encoder instances
        test_pairs: List of word pairs to test
        
    Returns:
        Dictionary with comparison results for each encoder
    """
    results = {}
    
    for name, encoder in encoders.items():
        overlaps = test_overlaps(encoder, test_pairs, verbose=False)
        overlap_values = [overlap for overlap in overlaps.values() if overlap > 0]
        
        results[name] = {
            'total_pairs': len(test_pairs),
            'valid_pairs': len(overlap_values),
            'mean_overlap': np.mean(overlap_values) if overlap_values else 0,
            'overlap_percentage': np.mean(overlap_values) / encoder.active_bits * 100 if overlap_values else 0,
            'overlaps': overlaps
        }
    
    return results


def track_training_progress(overlap_history: List[Dict[Tuple[str, str], int]], 
                          test_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Track and analyze training progress over epochs.
    
    Args:
        overlap_history: List of overlap dictionaries, one per epoch
        test_pairs: List of word pairs being tracked
        
    Returns:
        Dictionary with progress analysis including trends and convergence info
    """
    if not overlap_history:
        return {}
    
    progress = {}
    
    for word1, word2 in test_pairs:
        pair_key = f"{word1}↔{word2}"
        values = [epoch_overlaps.get((word1, word2), 0) for epoch_overlaps in overlap_history]
        
        if any(v > 0 for v in values):  # Only analyze pairs with some overlap
            progress[pair_key] = {
                'initial': values[0],
                'final': values[-1],
                'change': values[-1] - values[0],
                'max_value': max(values),
                'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable',
                'values': values
            }
    
    # Overall statistics
    if progress:
        changes = [data['change'] for data in progress.values()]
        progress['_summary'] = {
            'total_pairs_tracked': len(progress) - 1,  # Exclude summary itself
            'pairs_improved': sum(1 for change in changes if change > 0),
            'pairs_degraded': sum(1 for change in changes if change < 0),
            'mean_change': np.mean(changes),
            'total_epochs': len(overlap_history)
        }
    
    return progress 