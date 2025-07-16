"""Training loop implementation for SDR Hebbian learning."""

from __future__ import annotations

from typing import List, Tuple, Iterator, Union
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .encoder import Encoder
from .utils import parse_corpus_file, extract_windows, extract_vocab_from_corpus


def fit_one_epoch(
    encoder: Encoder,
    token_windows: List[Tuple[str, List[str]]],
    window_size: int,
    K_swap: int,
    d: int,
) -> None:
    """Fit one epoch of Hebbian learning.
    
    Args:
        encoder: The Encoder instance to train
        token_windows: List of (center_word, context_words) tuples
        window_size: Context window radius (not used in this function but kept for API compatibility)
        K_swap: Number of bits to swap per word at epoch end
        d: Manhattan distance for neighborhood search
    """
    # Track which words were seen this epoch for bit migration
    seen_words = set()
    
    # Process each token window with progress bar
    for center_word, context_words in tqdm(token_windows, desc="  Training", leave=False):
        if center_word not in encoder:
            continue  # Skip words not in vocabulary
            
        seen_words.add(center_word)
        
        # Get center word SDR
        S_v = encoder[center_word]
        
        # Compute context union U
        U = np.zeros_like(S_v, dtype=bool)
        for context_word in context_words:
            if context_word in encoder:
                U |= encoder[context_word]
        
        # Apply Hebbian updates
        # Missed opportunity: S_v[i,j]==0 and U[i,j]==1 → increment
        missed_mask = (~S_v) & U
        
        # Lonely bit: S_v[i,j]==1 and U[i,j]==0 → decrement  
        lonely_mask = S_v & (~U)
        
        # Update counters
        encoder.update_counter(center_word, missed_mask, lonely_mask)
    
    # Perform bit migration for all seen words
    for word in tqdm(seen_words, desc="  Migrating bits", leave=False):
        encoder.migrate_bits(word, K_swap, d)


def train(
    encoder: Encoder,
    corpus_path: Union[str, Path],
    window_size: int = 5,
    epochs: int = 3,
    swap_per_epoch: int = 3,
    neighbourhood_d: int = 2,
    verbose: bool = True,
) -> None:
    """Train the encoder on a corpus using Hebbian learning with bit migration.
    
    Args:
        encoder: The Encoder instance to train
        corpus_path: Path to the corpus file (one sentence per line)
        window_size: Context window radius (±window_size tokens)
        epochs: Number of training epochs
        swap_per_epoch: Number of bits to swap per word each epoch
        neighbourhood_d: Manhattan distance for neighborhood search
        verbose: Whether to print progress information
    """
    corpus_path = Path(corpus_path)
    
    if verbose:
        print(f"Training encoder on {corpus_path}")
        print(f"Vocabulary size: {encoder.vocab_size}")
        print(f"Grid dimensions: {encoder.height}x{encoder.width}")
        print(f"Sparsity: {encoder.sparsity:.3f} ({encoder.active_bits} active bits)")
        print(f"Window size: ±{window_size}")
        print(f"Epochs: {epochs}")
        print(f"Swap per epoch: {swap_per_epoch}")
        print(f"Neighborhood distance: {neighbourhood_d}")
        print()
    
    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")
        
        # Determine K_swap for this epoch (optional decay after epoch 1)
        if epoch == 0:
            K_swap = swap_per_epoch
        else:
            # Optionally decay K_swap after first epoch
            K_swap = max(1, swap_per_epoch // 2)
        
        # Extract all token windows from corpus
        if verbose:
            print(f"  Extracting token windows from corpus...")
        
        token_windows = []
        sentences = list(parse_corpus_file(str(corpus_path)))
        
        for sentence_tokens in tqdm(sentences, desc="  Processing sentences", leave=False, disable=not verbose):
            for center_word, context_words in extract_windows(sentence_tokens, window_size):
                token_windows.append((center_word, context_words))
        
        if verbose:
            print(f"  Processing {len(token_windows)} token windows")
            print(f"  K_swap = {K_swap}")
        
        # Run one epoch
        fit_one_epoch(
            encoder=encoder,
            token_windows=token_windows,
            window_size=window_size,
            K_swap=K_swap,
            d=neighbourhood_d
        )
        
        if verbose:
            print(f"  Epoch {epoch + 1} completed")
    
    if verbose:
        print("\nTraining completed!")


def create_encoder_from_corpus(
    corpus_path: Union[str, Path],
    grid: Tuple[int, int] = (128, 128),
    sparsity: float = 0.02,
    seed: int = 42,
) -> Encoder:
    """Create an encoder with vocabulary extracted from corpus.
    
    Args:
        corpus_path: Path to the corpus file
        grid: SDR grid dimensions (H, W)
        sparsity: Fraction of active bits per SDR
        seed: Random seed for reproducibility
        
    Returns:
        Encoder instance with initialized vocabulary
    """
    corpus_path = Path(corpus_path)
    
    # Extract vocabulary from corpus
    vocab = extract_vocab_from_corpus(str(corpus_path))
    
    # Create encoder
    encoder = Encoder(grid=grid, sparsity=sparsity, rng=seed)
    encoder.init_vocab(vocab)
    
    return encoder 