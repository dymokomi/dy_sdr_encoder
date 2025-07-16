#!/usr/bin/env python3
"""Example 2: Train encoder using large vocabulary and corpus files.

This example demonstrates the typical workflow when working with
large vocabulary and corpus files, as is common in real applications.
"""

import sys
from pathlib import Path

# Add the source directory to Python path so we can import dy_sdr_encoder
# This makes the example self-contained without requiring package installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dy_sdr_encoder import Encoder, train
from training_utils import (
    load_vocab_file, load_test_pairs, get_corpus_info,
    test_overlaps, show_progress_summary, show_epoch_progress,
    print_encoder_info, print_data_info, save_encoder_with_info
)


def main():
    print("=== DY SDR Encoder - Example 2: Large File-based Training ===\n")
    
    # File paths
    vocab_path = Path("data/train.vocab")
    corpus_path = Path("data/train.txt")
    test_pairs_path = Path("examples/data/test_pairs_large.txt")
    
    # Check if files exist
    for file_path in [vocab_path, corpus_path, test_pairs_path]:
        if not file_path.exists():
            print(f"Error: Required file not found: {file_path}")
            print("Please ensure all data files exist before running this example.")
            return
    
    print(f"Vocabulary file: {vocab_path}")
    print(f"Corpus file: {corpus_path}")
    print(f"Test pairs file: {test_pairs_path}")
    print()
    
    # Load data from files
    print("Loading data from files...")
    vocab = load_vocab_file(str(vocab_path))
    test_pairs = load_test_pairs(str(test_pairs_path))
    corpus_info = get_corpus_info(str(corpus_path))
    
    # Display data information
    print_data_info(vocab, corpus_info)
    
    # Create encoder - larger grid for more vocabulary
    print("Creating encoder...")
    encoder = Encoder(
        grid=(64, 64),    # Larger grid to accommodate more words
        sparsity=0.02,    # 2% sparsity 
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
    print("Note: This may take a while with the full corpus...")
    print()
    
    epochs = 3
    for epoch in range(epochs):
        print(f"=== Starting Epoch {epoch + 1}/{epochs} ===")
        
        # Train for one epoch
        train(
            encoder=encoder,
            corpus_path=corpus_path,
            window_size=3,      # ±3 word context window
            epochs=1,           # Train one epoch at a time
            swap_per_epoch=2,   # Conservative bit swapping
            neighbourhood_d=2,  # Standard neighborhood distance
            verbose=True
        )
        
        # Test overlaps after this epoch
        current_overlaps = test_overlaps(encoder, test_pairs, f"Overlaps after epoch {epoch + 1}")
        epoch_overlaps.append((f"Epoch {epoch + 1}", current_overlaps))
        
        # Show progress compared to initial
        show_epoch_progress(test_pairs, initial_overlaps, current_overlaps)
    
    # Show final summary of all epochs
    show_progress_summary(test_pairs, epoch_overlaps, encoder)
    
    # Save the trained encoder
    encoder_path = Path("models") / "file_trained_encoder.npz"
    save_encoder_with_info(encoder, encoder_path, "Example 2")
    
    print(f"Use the saved model at {encoder_path} for visualization or further analysis.")


if __name__ == "__main__":
    main() 