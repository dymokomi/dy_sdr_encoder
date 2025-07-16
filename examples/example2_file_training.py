#!/usr/bin/env python3
"""Example 2: Train encoder using vocabulary and corpus files.

This example demonstrates the typical workflow when working with
separate vocabulary and corpus files, as is common in real applications.
"""

import sys
import numpy as np
from pathlib import Path

# Add the source directory to Python path so we can import dy_sdr_encoder
# This makes the example self-contained without requiring package installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dy_sdr_encoder import Encoder, train
from dy_sdr_encoder.utils import load_vocab_file


def main():
    print("=== DY SDR Encoder - Example 2: File-based Training ===\n")
    
    # File paths (as specified by user)
    vocab_path = Path("data/train.vocab")
    corpus_path = Path("data/train.txt")
    
    # Check if files exist
    if not vocab_path.exists():
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print("Please ensure the file exists before running this example.")
        return
    
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        print("Please ensure the file exists before running this example.")
        return
    
    print(f"Vocabulary file: {vocab_path}")
    print(f"Corpus file: {corpus_path}")
    print()
    
    # Load vocabulary from file
    print("Loading vocabulary from file...")
    vocab = load_vocab_file(str(vocab_path))
    print(f"Loaded {len(vocab)} words from vocabulary file")
    
    # Show first few words
    print("First 10 words in vocabulary:")
    for i, word in enumerate(vocab[:10]):
        print(f"  {i+1:2d}. {word}")
    if len(vocab) > 10:
        print(f"  ... and {len(vocab) - 10} more words")
    print()
    
    # Get corpus info
    print("Corpus information:")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"  Total lines: {len(lines)}")
    
    # Show first few lines
    print("First 5 lines of corpus:")
    for i, line in enumerate(lines[:5]):
        print(f"  {i+1}. {line.strip()}")
    if len(lines) > 5:
        print(f"  ... and {len(lines) - 5} more lines")
    print()
    
    # Create encoder - larger grid for more vocabulary
    print("Creating encoder...")
    encoder = Encoder(
        grid=(64, 64),    # Larger grid to accommodate more words
        sparsity=0.02,    # 2% sparsity 
        rng=42            # Fixed seed for reproducibility  
    )
    
    # Initialize vocabulary
    encoder.init_vocab(vocab)
    print(f"Encoder initialized with {encoder.vocab_size} words")
    print(f"Grid size: {encoder.height}x{encoder.width} = {encoder.total_bits} bits")
    print(f"Active bits per word: {encoder.active_bits}")
    print()
    
    # Helper function to show overlaps between word pairs
    def show_overlap(word1, word2):
        if word1 in encoder and word2 in encoder:
            overlap = np.count_nonzero(encoder.flat(word1) & encoder.flat(word2))
            total_possible = encoder.active_bits
            percentage = (overlap / total_possible) * 100
            print(f"  {word1} ↔ {word2}: {overlap}/{total_possible} bits ({percentage:.1f}%)")
            return overlap
        else:
            missing = [w for w in [word1, word2] if w not in encoder]
            print(f"  {word1} ↔ {word2}: word(s) not in vocabulary: {missing}")
            return 0
    
    # Helper function to test all word pairs and return results
    def test_overlaps(title):
        print(f"{title}:")
        overlaps = {}
        for word1, word2 in test_pairs:
            overlaps[(word1, word2)] = show_overlap(word1, word2)
        print()
        return overlaps
    
    # Test some word pairs if they exist in vocabulary
    test_pairs = [
        ("cat", "dog"), ("table", "chair"), ("house", "room"),
        ("animal", "cat"), ("people", "person"), ("the", "a")
    ]
    
    # Test initial overlaps
    epoch_overlaps = []
    initial_overlaps = test_overlaps("Initial overlaps (before training)")
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
        current_overlaps = test_overlaps(f"Overlaps after epoch {epoch + 1}")
        epoch_overlaps.append((f"Epoch {epoch + 1}", current_overlaps))
        
        # Show progress compared to initial
        print("Progress since start:")
        for word1, word2 in test_pairs:
            initial = initial_overlaps.get((word1, word2), 0)
            current = current_overlaps.get((word1, word2), 0)
            change = current - initial
            if initial > 0 or current > 0:  # Only show if at least one measurement exists
                print(f"  {word1} ↔ {word2}: {initial} → {current} ({change:+d})")
        print()
    
    # Show final summary of all epochs
    print("=== Training Complete - Full Progress Summary ===")
    print("Overlap progression:")
    for word1, word2 in test_pairs:
        if any(overlaps.get((word1, word2), 0) > 0 for _, overlaps in epoch_overlaps):
            print(f"\n{word1} ↔ {word2}:")
            for epoch_name, overlaps in epoch_overlaps:
                overlap_val = overlaps.get((word1, word2), 0)
                if overlap_val > 0:
                    percentage = (overlap_val / encoder.active_bits) * 100
                    print(f"  {epoch_name}: {overlap_val}/{encoder.active_bits} bits ({percentage:.1f}%)")
    print()
    
    # Save the trained encoder
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    encoder_path = models_dir / "file_trained_encoder.npz"
    encoder.save(encoder_path)
    print(f"Encoder saved to: {encoder_path}")
    
    # Show some encoder statistics
    print("\nEncoder statistics:")
    print(f"  Vocabulary size: {encoder.vocab_size}")
    print(f"  Grid dimensions: {encoder.height} × {encoder.width}")
    print(f"  Total bits: {encoder.total_bits}")
    print(f"  Active bits per word: {encoder.active_bits}")
    print(f"  Sparsity: {encoder.sparsity:.3f}")
    
    print("\n=== Example 2 Complete ===")
    print("The encoder has been trained on the provided vocabulary and corpus files!")
    print(f"Use the saved model at {encoder_path} for visualization or further analysis.")


if __name__ == "__main__":
    main() 