#!/usr/bin/env python3
"""Dataset utilities for DY SDR Encoder.

This module provides functions for loading and processing datasets,
including vocabulary files, corpus files, and test pairs.
"""

from typing import List, Tuple, Dict, Any
from pathlib import Path


def load_vocab_file(vocab_path: str) -> List[str]:
    """Load vocabulary from a text file.
    
    Supports two formats:
    - One word per line: "word"
    - Word with frequency: "word\tfrequency" or "word frequency"
    
    When frequency information is present, it is ignored and only 
    the word is extracted.
    
    Args:
        vocab_path: Path to vocabulary file
        
    Returns:
        List of vocabulary words
        
    Raises:
        FileNotFoundError: If vocabulary file doesn't exist
        ValueError: If vocabulary file is empty
    """
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Split on tab first (preferred format: word\tfrequency)
                if '\t' in line:
                    word = line.split('\t')[0].strip()
                else:
                    # Fallback to general whitespace splitting
                    word = line.split()[0]
                
                if word:  # Only add non-empty words
                    vocab.append(word)
    
    if not vocab:
        raise ValueError(f"Vocabulary file is empty: {vocab_path}")
    
    return vocab


def load_test_pairs(test_pairs_path: str) -> List[Tuple[str, str]]:
    """Load test word pairs from a CSV-like file.
    
    File format: word1,word2,description
    Lines starting with # are treated as comments.
    
    Args:
        test_pairs_path: Path to test pairs file
        
    Returns:
        List of (word1, word2) tuples
        
    Raises:
        FileNotFoundError: If test pairs file doesn't exist
    """
    test_pairs_path = Path(test_pairs_path)
    if not test_pairs_path.exists():
        raise FileNotFoundError(f"Test pairs file not found: {test_pairs_path}")
    
    pairs = []
    with open(test_pairs_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    pairs.append((parts[0].strip(), parts[1].strip()))
                else:
                    print(f"Warning: Skipping malformed line {line_num} in {test_pairs_path}")
    
    return pairs


def get_corpus_info(corpus_path: str) -> Dict[str, Any]:
    """Get basic information about a corpus file.
    
    Args:
        corpus_path: Path to corpus file
        
    Returns:
        Dictionary with corpus statistics:
        - total_lines: Number of lines in corpus
        - total_tokens: Total number of tokens
        - sample_lines: First 5 lines as examples
        
    Raises:
        FileNotFoundError: If corpus file doesn't exist
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_tokens = sum(len(line.split()) for line in lines)
    
    return {
        'total_lines': len(lines),
        'total_tokens': total_tokens,
        'sample_lines': [line.strip() for line in lines[:5]]
    }


def validate_corpus_vocabulary(corpus_path: str, vocabulary: List[str]) -> Dict[str, Any]:
    """Validate that all words in corpus exist in vocabulary.
    
    Args:
        corpus_path: Path to corpus file
        vocabulary: List of vocabulary words
        
    Returns:
        Dictionary with validation results:
        - valid: Whether all corpus words are in vocabulary
        - missing_words: Set of words in corpus but not in vocabulary
        - coverage: Percentage of corpus tokens that are in vocabulary
    """
    vocab_set = set(vocabulary)
    missing_words = set()
    total_tokens = 0
    valid_tokens = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            total_tokens += len(tokens)
            for token in tokens:
                if token in vocab_set:
                    valid_tokens += 1
                else:
                    missing_words.add(token)
    
    coverage = (valid_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    return {
        'valid': len(missing_words) == 0,
        'missing_words': missing_words,
        'coverage': coverage,
        'total_tokens': total_tokens,
        'valid_tokens': valid_tokens
    } 