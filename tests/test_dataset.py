#!/usr/bin/env python3
"""Tests for dy_sdr_encoder.dataset module."""

import tempfile
import pytest
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dy_sdr_encoder.dataset import (
    load_vocab_file, 
    load_test_pairs, 
    get_corpus_info,
    validate_corpus_vocabulary
)


class TestLoadVocabFile:
    """Test vocabulary file loading."""
    
    def test_load_vocab_file_success(self):
        """Test successful vocabulary loading."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat\ndog\nbird\n")
            vocab_path = f.name
        
        try:
            vocab = load_vocab_file(vocab_path)
            assert vocab == ["cat", "dog", "bird"]
        finally:
            Path(vocab_path).unlink()
    
    def test_load_vocab_file_with_empty_lines(self):
        """Test vocabulary loading with empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat\n\ndog\n  \nbird\n")
            vocab_path = f.name
        
        try:
            vocab = load_vocab_file(vocab_path)
            assert vocab == ["cat", "dog", "bird"]
        finally:
            Path(vocab_path).unlink()
    
    def test_load_vocab_file_not_found(self):
        """Test error when vocabulary file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_vocab_file("nonexistent_file.txt")
    
    def test_load_vocab_file_empty(self):
        """Test error when vocabulary file is empty."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("")
            vocab_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_vocab_file(vocab_path)
        finally:
            Path(vocab_path).unlink()

    def test_load_vocab_file_with_tab_frequency(self):
        """Test vocabulary loading with tab-separated frequency format."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat\t100\ndog\t50\nbird\t25\n")
            vocab_path = f.name
        
        try:
            vocab = load_vocab_file(vocab_path)
            assert vocab == ["cat", "dog", "bird"]
        finally:
            Path(vocab_path).unlink()
    
    def test_load_vocab_file_with_space_frequency(self):
        """Test vocabulary loading with space-separated frequency format."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat 100\ndog 50\nbird 25\n")
            vocab_path = f.name
        
        try:
            vocab = load_vocab_file(vocab_path)
            assert vocab == ["cat", "dog", "bird"]
        finally:
            Path(vocab_path).unlink()
    
    def test_load_vocab_file_mixed_formats(self):
        """Test vocabulary loading with mixed single words and frequency formats."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat\t100\ndog\nbird 25\nhorse\t75\n")
            vocab_path = f.name
        
        try:
            vocab = load_vocab_file(vocab_path)
            assert vocab == ["cat", "dog", "bird", "horse"]
        finally:
            Path(vocab_path).unlink()
    
    def test_load_vocab_file_with_extra_whitespace(self):
        """Test vocabulary loading with extra whitespace around words and frequencies."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("  cat  \t  100  \n  dog  50  \n  bird  \n")
            vocab_path = f.name
        
        try:
            vocab = load_vocab_file(vocab_path)
            assert vocab == ["cat", "dog", "bird"]
        finally:
            Path(vocab_path).unlink()


class TestLoadTestPairs:
    """Test test pairs file loading."""
    
    def test_load_test_pairs_success(self):
        """Test successful test pairs loading."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("# Comment\ncat,dog,related\nhappy,sad,opposite\n")
            pairs_path = f.name
        
        try:
            pairs = load_test_pairs(pairs_path)
            assert pairs == [("cat", "dog"), ("happy", "sad")]
        finally:
            Path(pairs_path).unlink()
    
    def test_load_test_pairs_with_spaces(self):
        """Test test pairs loading with spaces around words."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat , dog , related\n  happy,  sad  , opposite\n")
            pairs_path = f.name
        
        try:
            pairs = load_test_pairs(pairs_path)
            assert pairs == [("cat", "dog"), ("happy", "sad")]
        finally:
            Path(pairs_path).unlink()
    
    def test_load_test_pairs_not_found(self):
        """Test error when test pairs file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_test_pairs("nonexistent_file.txt")


class TestGetCorpusInfo:
    """Test corpus information extraction."""
    
    def test_get_corpus_info_success(self):
        """Test successful corpus info extraction."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat sleep house\ndog run park\nbird fly sky\n")
            corpus_path = f.name
        
        try:
            info = get_corpus_info(corpus_path)
            assert info['total_lines'] == 3
            assert info['total_tokens'] == 9
            assert len(info['sample_lines']) == 3
            assert info['sample_lines'][0] == "cat sleep house"
        finally:
            Path(corpus_path).unlink()
    
    def test_get_corpus_info_not_found(self):
        """Test error when corpus file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            get_corpus_info("nonexistent_file.txt")


class TestValidateCorpusVocabulary:
    """Test corpus vocabulary validation."""
    
    def test_validate_corpus_vocabulary_valid(self):
        """Test validation with valid corpus."""
        vocab = ["cat", "dog", "sleep", "run"]
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat sleep\ndog run\n")
            corpus_path = f.name
        
        try:
            result = validate_corpus_vocabulary(corpus_path, vocab)
            assert result['valid'] == True
            assert len(result['missing_words']) == 0
            assert result['coverage'] == 100.0
        finally:
            Path(corpus_path).unlink()
    
    def test_validate_corpus_vocabulary_invalid(self):
        """Test validation with invalid corpus."""
        vocab = ["cat", "dog"]
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cat sleep\ndog run bird\n")
            corpus_path = f.name
        
        try:
            result = validate_corpus_vocabulary(corpus_path, vocab)
            assert result['valid'] == False
            assert "sleep" in result['missing_words']
            assert "run" in result['missing_words'] 
            assert "bird" in result['missing_words']
            assert result['coverage'] < 100.0
        finally:
            Path(corpus_path).unlink() 