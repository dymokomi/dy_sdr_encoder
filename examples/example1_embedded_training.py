#!/usr/bin/env python3
"""Example 1: Train an encoder with external vocabulary and text files.

This example demonstrates basic encoder usage with rich semantic data.
Perfect for understanding the core concepts with clean, streamlined code.
"""

import sys
from pathlib import Path

# Add the source directory to Python path so we can import dy_sdr_encoder
# This makes the example self-contained without requiring package installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dy_sdr_encoder import Encoder, train
from dy_sdr_encoder import load_vocab_file, load_test_pairs, get_corpus_info, test_overlaps
from training_utils import (
    show_progress_summary, show_epoch_progress,
    print_encoder_info, print_data_info, save_encoder_with_info
)


def main():
    print("=== DY SDR Encoder - Example 1: Expanded Vocabulary Training ===\n")
    
    # File paths
    data_dir = Path(__file__).parent / "data"
    vocab_path = data_dir / "vocab.txt"
    corpus_path = data_dir / "corpus.txt"
    test_pairs_path = data_dir / "test_pairs.txt"
    
    # Check if files exist
    for file_path in [vocab_path, corpus_path, test_pairs_path]:
        if not file_path.exists():
            print(f"Error: Required file not found: {file_path}")
            print("Please ensure all data files exist before running this example.")
            return
    
    # Load data from files
    print("Loading data from files...")
    vocab = load_vocab_file(str(vocab_path))
    test_pairs = load_test_pairs(str(test_pairs_path))
    corpus_info = get_corpus_info(str(corpus_path))
    
    # Display data information
    print_data_info(vocab, corpus_info)
    
    # Create encoder with appropriately sized grid for vocabulary
    print("Creating encoder...")
    encoder = Encoder(
        grid=(64, 64),    # Larger grid to accommodate 5x more words (200 vs 40)
        sparsity=0.02,    # 2% sparsity (about 128 active bits)
        rng=42            # Fixed seed for reproducibility
    )
    
    # Initialize vocabulary
    encoder.init_vocab(vocab)
    print_encoder_info(encoder)
    
    # Test initial overlaps
    epoch_overlaps = []
    initial_overlaps = test_overlaps(encoder, test_pairs, "Initial overlaps (before training)")
    epoch_overlaps.append(("Initial", initial_overlaps))
    
    # Train the encoder epoch by epoch to monitor progress
    print("Training encoder...")
    print(f"Corpus: {corpus_info['total_tokens']} total tokens")
    print("Note: Monitoring overlap changes after each epoch...\n")
    
    epochs = 90
    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1}/{epochs} ===")
        print("Training on corpus...")
        
        # Train for one epoch (quietly)
        train(
            encoder=encoder,
            corpus_path=corpus_path,
            window_size=2,      # ±2 word context window
            epochs=1,           # Train one epoch at a time
            swap_per_epoch=4,   # Conservative bit swapping
            neighbourhood_d=3,  # Allow wider neighborhood search
            verbose=False       # Quiet training
        )
        
        # Test overlaps after this epoch
        current_overlaps = test_overlaps(encoder, test_pairs, f"After epoch {epoch + 1}")
        epoch_overlaps.append((f"Epoch {epoch + 1}", current_overlaps))
        
        # Show progress compared to initial (compact format)
        show_epoch_progress(test_pairs, initial_overlaps, current_overlaps)
    
    # Show final summary of all epochs
    show_progress_summary(test_pairs, epoch_overlaps, encoder)
    
    # Save the trained encoder
    encoder_path = Path("models") / "embedded_trained_encoder.npz"
    save_encoder_with_info(encoder, encoder_path, "Example 1")
    
    # Demonstrate loading
    print("\nTesting save/load...")
    loaded_encoder = Encoder.load(encoder_path)
    print(f"Successfully loaded encoder with {loaded_encoder.vocab_size} words")
    
    # Verify loaded encoder works
    if test_pairs:
        word1, word2 = test_pairs[0]
        final_overlap = epoch_overlaps[-1][1].get((word1, word2), 0)
        from training_utils import calculate_overlap
        test_overlap, _ = calculate_overlap(loaded_encoder, word1, word2)
        print(f"Loaded encoder {word1}↔{word2} overlap: {test_overlap} (matches: {test_overlap == final_overlap})")


if __name__ == "__main__":
    main() 