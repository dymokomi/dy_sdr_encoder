"""Tests for training functionality and utilities."""

import tempfile
from pathlib import Path
import numpy as np

from dy_sdr_encoder.encoder import Encoder
from dy_sdr_encoder.train_loop import train, create_encoder_from_corpus
from dy_sdr_encoder.utils import (
    extract_windows, parse_corpus_file, extract_vocab_from_corpus,
    manhattan_neighbors, generate_sparse_2d
)


class TestUtils:
    """Test utility functions."""
    
    def test_generate_sparse_2d(self):
        """Test sparse 2D array generation."""
        rng = np.random.default_rng(42)
        H, W = 10, 8
        sparsity = 0.25
        
        sdr = generate_sparse_2d(H, W, sparsity, rng)
        
        assert sdr.shape == (H, W)
        assert sdr.dtype == bool
        
        # Check sparsity
        expected_active = int(sparsity * H * W)
        actual_active = np.sum(sdr)
        assert actual_active == expected_active
    
    def test_manhattan_neighbors(self):
        """Test Manhattan distance neighbor finding."""
        H, W = 5, 5
        center_i, center_j = 2, 2
        
        # Distance 0 (just center)
        neighbors_0 = manhattan_neighbors(H, W, center_i, center_j, 0)
        assert neighbors_0 == [(2, 2)]
        
        # Distance 1
        neighbors_1 = manhattan_neighbors(H, W, center_i, center_j, 1)
        expected_1 = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
        assert sorted(neighbors_1) == sorted(expected_1)
        
        # Test edge cases (center at corner)
        neighbors_corner = manhattan_neighbors(H, W, 0, 0, 1)
        expected_corner = [(0, 0), (0, 1), (1, 0)]
        assert sorted(neighbors_corner) == sorted(expected_corner)
    
    def test_extract_windows(self):
        """Test context window extraction."""
        tokens = ["the", "cat", "sat", "on", "mat"]
        
        # Window size 1
        windows_1 = list(extract_windows(tokens, 1))
        expected_1 = [
            ("the", ["cat"]),
            ("cat", ["the", "sat"]),
            ("sat", ["cat", "on"]),
            ("on", ["sat", "mat"]),
            ("mat", ["on"])
        ]
        assert windows_1 == expected_1
        
        # Window size 2
        windows_2 = list(extract_windows(tokens, 2))
        expected_2 = [
            ("the", ["cat", "sat"]),
            ("cat", ["the", "sat", "on"]),
            ("sat", ["the", "cat", "on", "mat"]),
            ("on", ["cat", "sat", "mat"]),
            ("mat", ["sat", "on"])
        ]
        assert windows_2 == expected_2
    
    def test_corpus_file_parsing(self):
        """Test corpus file parsing."""
        corpus_content = """the cat sat on the mat
the dog ran in the park

cats and dogs are animals
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(corpus_content)
            corpus_path = f.name
        
        try:
            sentences = list(parse_corpus_file(corpus_path))
            expected = [
                ["the", "cat", "sat", "on", "the", "mat"],
                ["the", "dog", "ran", "in", "the", "park"],
                ["cats", "and", "dogs", "are", "animals"]
            ]
            assert sentences == expected
        finally:
            Path(corpus_path).unlink()
    
    def test_vocab_extraction(self):
        """Test vocabulary extraction from corpus."""
        corpus_content = """the cat sat
the dog ran
cats are animals"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(corpus_content)
            corpus_path = f.name
        
        try:
            vocab = extract_vocab_from_corpus(corpus_path)
            expected = sorted(["the", "cat", "sat", "dog", "ran", "cats", "are", "animals"])
            assert vocab == expected
        finally:
            Path(corpus_path).unlink()


class TestTraining:
    """Test training functionality."""
    
    def test_create_encoder_from_corpus(self):
        """Test encoder creation from corpus."""
        corpus_content = """cat dog animal
table chair furniture
house room building"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(corpus_content)
            corpus_path = f.name
        
        try:
            encoder = create_encoder_from_corpus(
                corpus_path, 
                grid=(16, 16), 
                sparsity=0.1, 
                seed=42
            )
            
            expected_vocab = sorted(["cat", "dog", "animal", "table", "chair", 
                                   "furniture", "house", "room", "building"])
            assert encoder.vocab == expected_vocab
            assert encoder.height == 16
            assert encoder.width == 16
            assert encoder.sparsity == 0.1
            
        finally:
            Path(corpus_path).unlink()
    
    def test_training_integration(self):
        """Test full training pipeline."""
        # Create a small corpus
        corpus_content = """cat dog
dog cat
cat animal
dog animal
animal pet
pet animal"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(corpus_content)
            corpus_path = f.name
        
        try:
            # Create encoder
            encoder = create_encoder_from_corpus(
                corpus_path,
                grid=(16, 16),
                sparsity=0.1,
                seed=42
            )
            
            # Record initial overlaps
            initial_cat_dog = np.count_nonzero(encoder.flat("cat") & encoder.flat("dog"))
            initial_cat_animal = np.count_nonzero(encoder.flat("cat") & encoder.flat("animal"))
            
            # Train
            train(
                encoder=encoder,
                corpus_path=corpus_path,
                window_size=1,
                epochs=2,
                swap_per_epoch=2,
                neighbourhood_d=2,
                verbose=False
            )
            
            # Check that related words have more overlap
            final_cat_dog = np.count_nonzero(encoder.flat("cat") & encoder.flat("dog"))
            final_cat_animal = np.count_nonzero(encoder.flat("cat") & encoder.flat("animal"))
            
            # At least one pair should have increased overlap
            assert (final_cat_dog >= initial_cat_dog or 
                    final_cat_animal >= initial_cat_animal)
            
        finally:
            Path(corpus_path).unlink()
    
    def test_training_preserves_sparsity(self):
        """Test that training preserves sparsity of SDRs."""
        corpus_content = """word1 word2
word2 word3
word3 word1"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(corpus_content)
            corpus_path = f.name
        
        try:
            encoder = create_encoder_from_corpus(
                corpus_path,
                grid=(8, 8),
                sparsity=0.25,
                seed=42
            )
            
            # Check initial sparsity
            for word in encoder.vocab:
                assert np.sum(encoder[word]) == encoder.w
            
            # Train
            train(
                encoder=encoder,
                corpus_path=corpus_path,
                window_size=1,
                epochs=1,
                swap_per_epoch=1,
                neighbourhood_d=1,
                verbose=False
            )
            
            # Check sparsity is preserved
            for word in encoder.vocab:
                assert np.sum(encoder[word]) == encoder.w
                
        finally:
            Path(corpus_path).unlink() 