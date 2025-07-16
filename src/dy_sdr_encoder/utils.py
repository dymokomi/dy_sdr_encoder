"""Utility functions for SDR manipulation and corpus processing."""

from __future__ import annotations

from typing import Iterator, List, Tuple
import numpy as np


def generate_sparse_2d(
    H: int, W: int, sparsity: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a random 2D sparse boolean array.
    
    Args:
        H: Height of the grid
        W: Width of the grid
        sparsity: Fraction of bits that should be True (0.0 to 1.0)
        rng: Random number generator
        
    Returns:
        Boolean array of shape (H, W) with approximately sparsity fraction of True bits
    """
    n_total = H * W
    n_active = int(sparsity * n_total)
    
    # Create flat array with correct number of active bits
    flat = np.zeros(n_total, dtype=bool)
    active_indices = rng.choice(n_total, size=n_active, replace=False)
    flat[active_indices] = True
    
    # Reshape to 2D
    return flat.reshape(H, W)


def manhattan_neighbors(
    H: int, W: int, center_i: int, center_j: int, max_distance: int
) -> List[Tuple[int, int]]:
    """Get all positions within Manhattan distance of center.
    
    Args:
        H: Grid height
        W: Grid width
        center_i: Center row
        center_j: Center column
        max_distance: Maximum Manhattan distance
        
    Returns:
        List of (i, j) coordinate tuples within the distance
    """
    neighbors = []
    for i in range(max(0, center_i - max_distance), 
                   min(H, center_i + max_distance + 1)):
        for j in range(max(0, center_j - max_distance), 
                       min(W, center_j + max_distance + 1)):
            distance = abs(i - center_i) + abs(j - center_j)
            if distance <= max_distance:
                neighbors.append((i, j))
    return neighbors


def extract_windows(
    tokens: List[str], window_size: int
) -> Iterator[Tuple[str, List[str]]]:
    """Extract context windows from a sequence of tokens.
    
    Args:
        tokens: List of tokens/words
        window_size: Radius of context window (±window_size tokens)
        
    Yields:
        Tuples of (center_word, context_words) where context_words
        are the tokens within ±window_size of center_word
    """
    for i, center_word in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        # Extract context, excluding the center word
        context = tokens[start:i] + tokens[i+1:end]
        
        yield center_word, context


def parse_corpus_file(corpus_path: str) -> Iterator[List[str]]:
    """Parse a corpus file into tokenized sentences.
    
    Args:
        corpus_path: Path to corpus file (one sentence per line)
        
    Yields:
        Lists of tokens for each sentence
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Simple whitespace tokenization
                tokens = line.split()
                if tokens:  # Skip empty token lists
                    yield tokens


def load_vocab_file(vocab_path: str) -> List[str]:
    """Load vocabulary from a file.
    
    Args:
        vocab_path: Path to vocabulary file (format: word [whitespace] [frequency])
        
    Returns:
        List of vocabulary words (without frequency counts)
    """
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Split on whitespace and take only the first part (the word)
                word = line.split()[0]
                vocab.append(word)
    return vocab


def extract_vocab_from_corpus(corpus_path: str) -> List[str]:
    """Extract unique vocabulary from a corpus file.
    
    Args:
        corpus_path: Path to corpus file
        
    Returns:
        Sorted list of unique words found in corpus
    """
    vocab_set = set()
    for tokens in parse_corpus_file(corpus_path):
        vocab_set.update(tokens)
    return sorted(vocab_set) 