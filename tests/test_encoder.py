"""Tests for the Encoder class functionality."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from dy_sdr_encoder.encoder import Encoder


class TestEncoder:
    """Test cases for the Encoder class."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization with different parameters."""
        enc = Encoder(grid=(64, 64), sparsity=0.03, rng=42)
        assert enc.height == 64
        assert enc.width == 64
        assert enc.sparsity == 0.03
        assert enc.total_bits == 64 * 64
        assert enc.active_bits == int(0.03 * 64 * 64)
        assert len(enc.vocab) == 0
    
    def test_vocab_initialization(self):
        """Test vocabulary initialization."""
        enc = Encoder(grid=(32, 32), sparsity=0.05, rng=0)
        vocab = ["cat", "dog", "house", "car"]
        enc.init_vocab(vocab)
        
        assert len(enc.vocab) == 4
        assert enc.vocab_size == 4
        assert "cat" in enc
        assert "dog" in enc
        assert "house" in enc
        assert "car" in enc
        assert "unknown" not in enc
        
        # Check SDR properties
        for word in vocab:
            sdr = enc[word]
            assert sdr.shape == (32, 32)
            assert sdr.dtype == bool
            # Check sparsity is approximately correct
            active_count = np.sum(sdr)
            expected_active = int(0.05 * 32 * 32)
            assert abs(active_count - expected_active) <= 1
    
    def test_getitem_and_flat(self):
        """Test __getitem__ and flat methods."""
        enc = Encoder(grid=(16, 16), sparsity=0.1, rng=42)
        enc.init_vocab(["test"])
        
        # Test 2D access
        sdr_2d = enc["test"]
        assert sdr_2d.shape == (16, 16)
        assert sdr_2d.dtype == bool
        
        # Test flat access (should be a view)
        sdr_flat = enc.flat("test")
        assert sdr_flat.shape == (256,)
        assert sdr_flat.dtype == bool
        
        # Verify it's a view (changes should propagate)
        original_value = sdr_2d[0, 0]
        sdr_2d[0, 0] = not original_value
        assert sdr_flat[0] == sdr_2d[0, 0]
        
        # Test KeyError for unknown word
        with pytest.raises(KeyError):
            enc["unknown"]
        
        with pytest.raises(KeyError):
            enc.flat("unknown")
    
    def test_counter_operations(self):
        """Test Hebbian counter operations."""
        enc = Encoder(grid=(8, 8), sparsity=0.25, rng=0)
        enc.init_vocab(["word"])
        
        # Initial counters should be zero
        counter = enc.get_counter("word")
        assert counter.shape == (8, 8)
        assert np.all(counter == 0)
        
        # Test counter updates
        inc_mask = np.zeros((8, 8), dtype=bool)
        dec_mask = np.zeros((8, 8), dtype=bool)
        inc_mask[0, 0] = True
        dec_mask[1, 1] = True
        
        enc.update_counter("word", inc_mask, dec_mask)
        
        updated_counter = enc.get_counter("word")
        assert updated_counter[0, 0] == 1
        assert updated_counter[1, 1] == -1
        assert np.sum(np.abs(updated_counter)) == 2
        
        # Test KeyError for unknown word
        with pytest.raises(KeyError):
            enc.get_counter("unknown")
    
    def test_bit_migration(self):
        """Test bit migration functionality."""
        enc = Encoder(grid=(8, 8), sparsity=0.25, rng=0)
        enc.init_vocab(["word"])
        
        original_sdr = enc["word"].copy()
        original_active_count = np.sum(original_sdr)
        
        # Set up counters to encourage migration
        counter = enc.get_counter("word")
        # Make active bits have low scores
        counter[original_sdr] = -5
        # Make some inactive bits have high scores
        inactive_positions = np.where(~original_sdr)
        if len(inactive_positions[0]) > 0:
            counter[inactive_positions[0][0], inactive_positions[1][0]] = 10
        
        enc.migrate_bits("word", k_swap=1, neighbourhood_d=3)
        
        # Check that total active bits remains the same
        new_sdr = enc["word"]
        new_active_count = np.sum(new_sdr)
        assert new_active_count == original_active_count
        
        # Check that some migration occurred (SDR should be different)
        assert not np.array_equal(original_sdr, new_sdr)
    
    def test_save_and_load(self):
        """Test saving and loading encoder."""
        enc = Encoder(grid=(16, 16), sparsity=0.0625, rng=123)
        vocab = ["apple", "banana", "cherry"]
        enc.init_vocab(vocab)
        
        # Modify some counters
        inc_mask = np.zeros((16, 16), dtype=bool)
        dec_mask = np.zeros((16, 16), dtype=bool)
        inc_mask[0, 0] = True
        inc_mask[1, 1] = True
        dec_mask[0, 1] = True
        dec_mask[1, 0] = True
        enc.update_counter("apple", inc_mask, dec_mask)
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_encoder.npz"
            enc.save(save_path)
            
            # Load encoder
            enc_loaded = Encoder.load(save_path)
            
            # Verify properties
            assert enc_loaded.height == enc.height
            assert enc_loaded.width == enc.width
            assert enc_loaded.sparsity == enc.sparsity
            assert enc_loaded.vocab == enc.vocab
            assert enc_loaded.vocab_size == enc.vocab_size
            
            # Verify SDRs
            for word in vocab:
                assert np.array_equal(enc[word], enc_loaded[word])
                assert np.array_equal(enc.get_counter(word), enc_loaded.get_counter(word))
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        vocab = ["test1", "test2", "test3"]
        
        enc1 = Encoder(grid=(16, 16), sparsity=0.1, rng=42)
        enc1.init_vocab(vocab)
        
        enc2 = Encoder(grid=(16, 16), sparsity=0.1, rng=42)
        enc2.init_vocab(vocab)
        
        # Should produce identical SDRs
        for word in vocab:
            assert np.array_equal(enc1[word], enc2[word])
    
    def test_duplicate_vocab_initialization(self):
        """Test that duplicate words in vocab are handled correctly."""
        enc = Encoder(grid=(8, 8), sparsity=0.25, rng=0)
        vocab = ["word", "word", "other", "word"]  # Duplicates
        enc.init_vocab(vocab)
        
        # Should only have unique words
        assert len(enc.vocab) == 2
        assert "word" in enc
        assert "other" in enc 