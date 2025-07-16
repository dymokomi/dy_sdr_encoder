#!/usr/bin/env python3
"""Demo script showing logging utilities usage."""

import sys
from pathlib import Path

# Add the source directory to Python path so we can import dy_sdr_encoder
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dy_sdr_encoder import Encoder
from training_utils import load_vocab_file, load_test_pairs, get_corpus_info
from logging_utils import TrainingLogger


def demo_logging():
    """Demonstrate the logging utilities."""
    print("=== Logging Utilities Demo ===\n")
    
    # Initialize logger
    logger = TrainingLogger(experiment_name="demo_experiment")
    print(f"Initialized logger: {logger.experiment_name}")
    print(f"Log file: {logger.log_file}\n")
    
    # Load some example data
    data_dir = Path(__file__).parent / "data"
    vocab = load_vocab_file(str(data_dir / "vocab.txt"))
    test_pairs = load_test_pairs(str(data_dir / "test_pairs.txt"))
    corpus_info = get_corpus_info(str(data_dir / "corpus.txt"))
    
    # Create a small encoder for demo
    encoder = Encoder(grid=(16, 16), sparsity=0.05, rng=42)
    encoder.init_vocab(vocab[:10])  # Use only first 10 words for quick demo
    
    # Log configuration
    logger.log_encoder_config(encoder)
    logger.log_training_config(window_size=2, epochs=3, swap_per_epoch=2)
    logger.log_data_info(len(vocab), corpus_info)
    
    # Simulate a few training epochs with dummy overlap data
    print("Simulating training epochs...")
    for epoch in range(1, 4):
        # Create dummy overlap data (normally from actual training)
        dummy_overlaps = {}
        for i, (word1, word2) in enumerate(test_pairs[:3]):  # Just first 3 pairs
            # Simulate increasing overlaps
            dummy_overlaps[(word1, word2)] = i + epoch
        
        logger.log_epoch(epoch, dummy_overlaps, training_time=1.5)
        print(f"  Logged epoch {epoch}")
    
    # Log final stats
    logger.log_final_stats("demo_model.npz", total_training_time=4.5)
    
    # Show summary
    logger.print_summary()
    
    print(f"\nDemo complete! Check the log file at: {logger.log_file}")


if __name__ == "__main__":
    demo_logging() 