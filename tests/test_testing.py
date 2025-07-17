#!/usr/bin/env python3
"""Tests for dy_sdr_encoder.testing module."""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dy_sdr_encoder import Encoder
from dy_sdr_encoder.testing import (
    calculate_overlap,
    show_overlap, 
    test_overlaps,
    analyze_semantic_relationships,
    compare_encoders,
    track_training_progress
)


class TestCalculateOverlap:
    """Test overlap calculation function."""
    
    def setup_method(self):
        """Set up test encoder."""
        self.encoder = Encoder(grid=(16, 16), sparsity=0.1, rng=42)
        self.encoder.init_vocab(["cat", "dog", "bird"])
    
    def test_calculate_overlap_valid_words(self):
        """Test overlap calculation with valid words."""
        overlap, percentage = calculate_overlap(self.encoder, "cat", "dog")
        assert isinstance(overlap, int)
        assert isinstance(percentage, float)
        assert overlap >= 0
        assert percentage >= 0.0
    
    def test_calculate_overlap_invalid_word(self):
        """Test overlap calculation with invalid word."""
        overlap, percentage = calculate_overlap(self.encoder, "cat", "elephant")
        assert overlap == 0
        assert percentage == 0.0
    
    def test_calculate_overlap_both_invalid(self):
        """Test overlap calculation with both words invalid."""
        overlap, percentage = calculate_overlap(self.encoder, "elephant", "tiger")
        assert overlap == 0
        assert percentage == 0.0


class TestShowOverlap:
    """Test overlap display function."""
    
    def setup_method(self):
        """Set up test encoder."""
        self.encoder = Encoder(grid=(16, 16), sparsity=0.1, rng=42)
        self.encoder.init_vocab(["cat", "dog", "bird"])
    
    def test_show_overlap_valid_words(self, capsys):
        """Test overlap display with valid words."""
        overlap = show_overlap(self.encoder, "cat", "dog", verbose=True)
        captured = capsys.readouterr()
        assert isinstance(overlap, int)
        assert overlap >= 0
        assert "cat ↔ dog" in captured.out
        assert "bits" in captured.out
    
    def test_show_overlap_invalid_word(self, capsys):
        """Test overlap display with invalid word."""
        overlap = show_overlap(self.encoder, "cat", "elephant", verbose=True)
        captured = capsys.readouterr()
        assert overlap == 0
        assert "not in vocabulary" in captured.out
    
    def test_show_overlap_quiet(self, capsys):
        """Test overlap display with verbose=False."""
        overlap = show_overlap(self.encoder, "cat", "dog", verbose=False)
        captured = capsys.readouterr()
        assert isinstance(overlap, int)
        assert captured.out == ""


class TestTestOverlaps:
    """Test multiple overlaps testing function."""
    
    def setup_method(self):
        """Set up test encoder."""
        self.encoder = Encoder(grid=(16, 16), sparsity=0.1, rng=42)
        self.encoder.init_vocab(["cat", "dog", "bird"])
        self.test_pairs = [("cat", "dog"), ("cat", "bird"), ("dog", "bird")]
    
    def test_test_overlaps_verbose(self, capsys):
        """Test overlaps testing with verbose output."""
        overlaps = test_overlaps(self.encoder, self.test_pairs, verbose=True)
        captured = capsys.readouterr()
        
        assert len(overlaps) == 3
        assert all(isinstance(overlap, int) for overlap in overlaps.values())
        assert "Overlap Results:" in captured.out
    
    def test_test_overlaps_quiet(self, capsys):
        """Test overlaps testing without verbose output."""
        overlaps = test_overlaps(self.encoder, self.test_pairs, verbose=False)
        captured = capsys.readouterr()
        
        assert len(overlaps) == 3
        assert captured.out == ""


class TestAnalyzeSemanticRelationships:
    """Test semantic relationship analysis."""
    
    def setup_method(self):
        """Set up test encoder."""
        self.encoder = Encoder(grid=(16, 16), sparsity=0.1, rng=42)
        self.encoder.init_vocab(["cat", "dog", "bird"])
        self.test_pairs = [("cat", "dog"), ("cat", "bird"), ("dog", "bird")]
    
    def test_analyze_semantic_relationships_default(self):
        """Test semantic analysis with default categorization."""
        analysis = analyze_semantic_relationships(self.encoder, self.test_pairs)
        
        assert isinstance(analysis, dict)
        # Should have at least one category
        assert len(analysis) > 0
        
        # Each category should have proper structure
        for category, data in analysis.items():
            assert 'pair_count' in data
            assert 'mean_overlap' in data
            assert 'overlap_percentage' in data


class TestCompareEncoders:
    """Test encoder comparison function."""
    
    def setup_method(self):
        """Set up test encoders."""
        self.encoder1 = Encoder(grid=(16, 16), sparsity=0.1, rng=42)
        self.encoder1.init_vocab(["cat", "dog", "bird"])
        
        self.encoder2 = Encoder(grid=(16, 16), sparsity=0.1, rng=24)
        self.encoder2.init_vocab(["cat", "dog", "bird"])
        
        self.encoders = {"encoder1": self.encoder1, "encoder2": self.encoder2}
        self.test_pairs = [("cat", "dog"), ("cat", "bird")]
    
    def test_compare_encoders(self):
        """Test encoder comparison."""
        comparison = compare_encoders(self.encoders, self.test_pairs)
        
        assert len(comparison) == 2
        assert "encoder1" in comparison
        assert "encoder2" in comparison
        
        for name, data in comparison.items():
            assert 'total_pairs' in data
            assert 'valid_pairs' in data
            assert 'mean_overlap' in data
            assert 'overlaps' in data


class TestTrackTrainingProgress:
    """Test training progress tracking."""
    
    def test_track_training_progress_empty(self):
        """Test with empty overlap history."""
        progress = track_training_progress([], [])
        assert progress == {}
    
    def test_track_training_progress_with_data(self):
        """Test with actual overlap data."""
        test_pairs = [("cat", "dog"), ("cat", "bird")]
        overlap_history = [
            {("cat", "dog"): 1, ("cat", "bird"): 0},
            {("cat", "dog"): 3, ("cat", "bird"): 1},
            {("cat", "dog"): 5, ("cat", "bird"): 2}
        ]
        
        progress = track_training_progress(overlap_history, test_pairs)
        
        assert "_summary" in progress
        summary = progress["_summary"]
        assert summary["total_epochs"] == 3
        assert summary["pairs_improved"] >= 0
        
        # Check individual pair tracking
        cat_dog_key = "cat↔dog"
        if cat_dog_key in progress:
            data = progress[cat_dog_key]
            assert data["initial"] == 1
            assert data["final"] == 5
            assert data["change"] == 4
            assert data["trend"] == "increasing" 