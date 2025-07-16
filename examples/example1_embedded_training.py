#!/usr/bin/env python3
"""Example 1: Train a small encoder with embedded vocabulary and text.

This example demonstrates basic encoder usage with small, embedded data.
Perfect for understanding the core concepts without external files.
"""

import sys
import numpy as np
import tempfile
from pathlib import Path

# Add the source directory to Python path so we can import dy_sdr_encoder
# This makes the example self-contained without requiring package installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dy_sdr_encoder import Encoder, train


def main():
    print("=== DY SDR Encoder - Example 1: Embedded Training ===\n")
    
    # Embedded vocabulary - expanded set of related words (4x larger)
    vocab = [
        # Animals
        "cat", "dog", "bird", "fish", "horse", "cow", "pig", "sheep", "chicken", "duck",
        "animal", "pet", "farm", "wild", "zoo",
        # Home/Building
        "house", "room", "kitchen", "bedroom", "bathroom", "living", "door", "window", "wall", "roof",
        # Furniture 
        "table", "chair", "bed", "sofa", "desk", "lamp", "shelf",
        # Actions
        "sit", "sleep", "eat", "drink", "walk", "run", "fly", "swim"
    ]
    
    # Embedded corpus - expanded sentences demonstrating word relationships (4x larger)
    corpus_text = """cat sleep house
dog sleep house
cat animal
dog animal
animal sleep
house room
room table
room chair
sit chair
sit table
bird fly wild
bird animal
fish swim wild
fish animal
horse run farm
horse animal
cow farm animal
pig farm animal
sheep farm animal
chicken farm animal
duck farm animal
pet animal house
zoo animal wild
kitchen house room
bedroom house room
bathroom house room
living house room
door house
window house
wall house
roof house
table kitchen
chair kitchen
bed bedroom
sofa living
desk bedroom
lamp bedroom
shelf living
sit chair
sit sofa
sleep bed
eat kitchen
drink kitchen
walk house
run wild
fly bird
swim fish
cat pet
dog pet
bird pet
fish pet"""
    
    print("Vocabulary:")
    print(f"  {vocab}")
    print(f"  Size: {len(vocab)} words")
    print()
    
    print("Corpus:")
    for line in corpus_text.strip().split('\n'):
        print(f"  {line}")
    print()
    
    # Create encoder with appropriately sized grid for expanded vocabulary
    print("Creating encoder...")
    encoder = Encoder(
        grid=(48, 48),    # Larger grid to accommodate 4x more words
        sparsity=0.02,    # 2% sparsity (about 46 active bits)
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
        overlap = np.count_nonzero(encoder.flat(word1) & encoder.flat(word2))
        total_possible = encoder.active_bits
        percentage = (overlap / total_possible) * 100
        print(f"  {word1} ↔ {word2}: {overlap}/{total_possible} bits ({percentage:.1f}%)")
        return overlap
    
    # Helper function to test all word pairs and return results
    def test_overlaps(title):
        print(f"{title}:")
        overlaps = {}
        for word1, word2 in test_pairs:
            overlaps[(word1, word2)] = show_overlap(word1, word2)
        print()
        return overlaps
    
    # Test word pairs - mix of related and unrelated
    test_pairs = [
        ("cat", "dog"),           # related animals
        ("cat", "animal"),        # cat is animal
        ("table", "chair"),       # furniture
        ("bird", "fly"),          # bird action
        ("fish", "swim"),         # fish action
        ("kitchen", "house"),     # room in house
        ("bed", "sleep"),         # bed for sleeping
        ("farm", "cow"),          # farm animal
        ("cat", "table"),         # unrelated
        ("bird", "kitchen")       # unrelated
    ]
    
    # Test initial overlaps
    epoch_overlaps = []
    initial_overlaps = test_overlaps("Initial overlaps (before training)")
    epoch_overlaps.append(("Initial", initial_overlaps))
    
    # Write corpus to temporary file for training
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(corpus_text)
        corpus_path = f.name
    
    try:
        # Train the encoder epoch by epoch to monitor progress
        print("Training encoder...")
        print(f"Corpus: {len(corpus_text.strip().split())} total tokens")
        print("Note: Monitoring overlap changes after each epoch...\n")
        
        epochs = 10
        for epoch in range(epochs):
            print(f"=== Epoch {epoch + 1}/{epochs} ===")
            print("Training on corpus...")
            
            # Train for one epoch (quietly)
            train(
                encoder=encoder,
                corpus_path=corpus_path,
                window_size=2,      # ±2 word context window
                epochs=1,           # Train one epoch at a time
                swap_per_epoch=2,   # Conservative bit swapping
                neighbourhood_d=3,  # Allow wider neighborhood search
                verbose=False       # Quiet training
            )
            
            # Test overlaps after this epoch
            current_overlaps = test_overlaps(f"After epoch {epoch + 1}")
            epoch_overlaps.append((f"Epoch {epoch + 1}", current_overlaps))
            
            # Show progress compared to initial (compact format)
            print("Progress since start:")
            for word1, word2 in test_pairs:
                initial = initial_overlaps.get((word1, word2), 0)
                current = current_overlaps.get((word1, word2), 0)
                change = current - initial
                print(f"  {word1} ↔ {word2}: {initial} → {current} ({change:+d})")
            print()
        
        # Show final summary of all epochs
        print("=== Training Complete - Full Progress Summary ===")
        print("Overlap progression:")
        for word1, word2 in test_pairs:
            print(f"\n{word1} ↔ {word2}:")
            for epoch_name, overlaps in epoch_overlaps:
                overlap_val = overlaps.get((word1, word2), 0)
                percentage = (overlap_val / encoder.active_bits) * 100
                print(f"  {epoch_name}: {overlap_val}/{encoder.active_bits} bits ({percentage:.1f}%)")
        print()
        
        # Save the trained encoder
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        encoder_path = models_dir / "embedded_trained_encoder.npz"
        encoder.save(encoder_path)
        print(f"Encoder saved to: {encoder_path}")
        
        # Demonstrate loading
        print("\nTesting save/load...")
        loaded_encoder = Encoder.load(encoder_path)
        print(f"Successfully loaded encoder with {loaded_encoder.vocab_size} words")
        
        # Verify loaded encoder works
        final_cat_dog = epoch_overlaps[-1][1].get(("cat", "dog"), 0)
        test_overlap = np.count_nonzero(loaded_encoder.flat("cat") & loaded_encoder.flat("dog"))
        print(f"Loaded encoder cat↔dog overlap: {test_overlap} (matches: {test_overlap == final_cat_dog})")
        
    finally:
        # Clean up temporary file
        Path(corpus_path).unlink()
    
    print("\n=== Example 1 Complete ===")
    print("The encoder has learned to associate semantically related words!")
    print("Related words now share more bits in their sparse representations.")


if __name__ == "__main__":
    main() 