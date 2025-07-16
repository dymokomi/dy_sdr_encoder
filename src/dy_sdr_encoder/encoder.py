"""Core Encoder class for managing SDR vocabularies."""

from __future__ import annotations

from typing import Dict, Iterable, Union
from pathlib import Path
import numpy as np

from .utils import generate_sparse_2d


class Encoder:
    """Encoder for Sparse Distributed Representations.
    
    Manages a vocabulary of words, each represented by a 2D sparse boolean array.
    Supports training via Hebbian learning and bit migration for dynamic adaptation.
    
    The encoder maintains:
    - A dictionary of Sparse Distributed Representations (SDRs) for each word
    - Hebbian counters that track co-activation patterns for adaptive learning
    - Methods for bit migration based on learned statistics
    """
    
    def __init__(
        self, 
        grid: tuple[int, int], 
        sparsity: float, 
        rng: Union[int, np.random.Generator] = None
    ):
        """Initialize the encoder with specified grid dimensions and sparsity.
        
        Args:
            grid: Tuple of (height, width) for SDR grid dimensions
            sparsity: Fraction of bits that should be active (0.0 to 1.0)
                     Higher sparsity means more bits are active
            rng: Random seed (int) or Generator for reproducibility
                 If None, uses system randomness
        """
        self.height, self.width = grid
        self.sparsity = sparsity
        self.total_bits = self.height * self.width
        self.active_bits = int(sparsity * self.total_bits)  # Number of active bits per SDR
        
        # Set up random number generator for reproducible results
        if isinstance(rng, int) or rng is None:
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng
            
        # Storage for SDRs and Hebbian counters
        # _sdrs: Maps word -> 2D boolean array representing the SDR
        # _counters: Maps word -> 2D integer array tracking Hebbian learning
        self._sdrs: Dict[str, np.ndarray] = {}
        self._counters: Dict[str, np.ndarray] = {}
        
    def init_vocab(self, vocab_iterable: Iterable[str]) -> None:
        """Initialize vocabulary with random SDRs and zero counters.
        
        Creates a random sparse distributed representation for each word
        in the vocabulary, along with corresponding Hebbian counters for
        tracking co-activation patterns during training.
        
        Args:
            vocab_iterable: Iterable of vocabulary words (strings)
        """
        for word in vocab_iterable:
            if word not in self._sdrs:
                # Generate random sparse SDR with specified sparsity
                sdr = generate_sparse_2d(self.height, self.width, self.sparsity, self.rng)
                self._sdrs[word] = sdr
                
                # Initialize Hebbian counters to zero for tracking co-activations
                self._counters[word] = np.zeros((self.height, self.width), dtype=np.int32)
    
    def __getitem__(self, word: str) -> np.ndarray:
        """Get 2D SDR for a word.
        
        Args:
            word: Vocabulary word to retrieve
            
        Returns:
            2D boolean array of shape (height, width) representing the SDR
            
        Raises:
            KeyError: If word not in vocabulary
        """
        if word not in self._sdrs:
            raise KeyError(f"Word '{word}' not in vocabulary")
        return self._sdrs[word]
    
    def flat(self, word: str) -> np.ndarray:
        """Get flattened view of SDR (O(1) operation).
        
        Returns a 1D view of the 2D SDR, useful for operations that
        require flat arrays like dot products or distance calculations.
        
        Args:
            word: Vocabulary word to retrieve
            
        Returns:
            1D boolean array view of shape (height*width,)
            
        Raises:
            KeyError: If word not in vocabulary
        """
        return self[word].ravel()  # ravel() returns a view when possible
    
    def get_counter(self, word: str) -> np.ndarray:
        """Get Hebbian counter array for a word.
        
        The counter array tracks how often each bit position has been
        involved in co-activations during training. Higher values indicate
        bits that frequently co-activate with other patterns.
        
        Args:
            word: Vocabulary word to retrieve counter for
            
        Returns:
            2D integer array of shape (height, width) with counter values
            
        Raises:
            KeyError: If word not in vocabulary
        """
        if word not in self._counters:
            raise KeyError(f"Word '{word}' not in vocabulary")
        return self._counters[word]
    
    def update_counter(self, word: str, increment_mask: np.ndarray, 
                      decrement_mask: np.ndarray) -> None:
        """Update Hebbian counters for a word based on co-activation patterns.
        
        This is the core learning mechanism that adjusts counter values
        based on which bits should be strengthened (incremented) or
        weakened (decremented) based on training feedback.
        
        Args:
            word: Vocabulary word to update
            increment_mask: Boolean mask for positions to increment (strengthen)
            decrement_mask: Boolean mask for positions to decrement (weaken)
        """
        if word not in self._counters:
            raise KeyError(f"Word '{word}' not in vocabulary")
            
        counter = self._counters[word]
        counter[increment_mask] += 1
        counter[decrement_mask] -= 1
    
    def migrate_bits(self, word: str, bits_to_swap: int, neighborhood_distance: int) -> None:
        """Perform bit migration for a word based on Hebbian counters.
        
        This adaptive mechanism moves poorly performing active bits to
        better locations based on accumulated Hebbian statistics. Bits
        with low counter values (poor co-activation) are moved to nearby
        inactive positions with higher counter values.
        
        Args:
            word: Vocabulary word to perform migration on
            bits_to_swap: Number of worst-performing bits to migrate
            neighborhood_distance: Manhattan distance for neighbor search
                                  Larger values allow migration to more distant positions
        """
        if word not in self._sdrs or word not in self._counters:
            raise KeyError(f"Word '{word}' not in vocabulary")
            
        sdr = self._sdrs[word]
        counter = self._counters[word]
        
        # Find current active positions and their performance scores
        active_positions = np.where(sdr)
        active_scores = counter[active_positions]
        
        # Sort active positions by counter values (ascending - worst performers first)
        sorted_indices = np.argsort(active_scores)
        worst_active_indices = sorted_indices[:bits_to_swap]
        
        # Get coordinates of worst performing active bits
        worst_row_coords = active_positions[0][worst_active_indices]
        worst_col_coords = active_positions[1][worst_active_indices]
        
        # For each worst active bit, find the best inactive neighbor within distance
        new_positions = []
        for row_idx, col_idx in zip(worst_row_coords, worst_col_coords):
            # Search for neighbors within Manhattan distance
            neighbor_candidates = []
            for neighbor_row in range(max(0, row_idx - neighborhood_distance), 
                                    min(self.height, row_idx + neighborhood_distance + 1)):
                for neighbor_col in range(max(0, col_idx - neighborhood_distance), 
                                        min(self.width, col_idx + neighborhood_distance + 1)):
                    manhattan_distance = abs(neighbor_row - row_idx) + abs(neighbor_col - col_idx)
                    # Only consider inactive neighbors within the specified distance
                    if (manhattan_distance <= neighborhood_distance and 
                        not sdr[neighbor_row, neighbor_col]):
                        neighbor_candidates.append((neighbor_row, neighbor_col))
            
            if neighbor_candidates:
                # Find neighbor with highest counter value (best performance potential)
                neighbor_scores = [counter[neighbor_row, neighbor_col] 
                                 for neighbor_row, neighbor_col in neighbor_candidates]
                best_neighbor_idx = np.argmax(neighbor_scores)
                new_positions.append(neighbor_candidates[best_neighbor_idx])
            else:
                # No valid neighbors found, keep the original position
                new_positions.append((row_idx, col_idx))
        
        # Perform the bit migrations (swaps)
        for old_position, new_position in zip(zip(worst_row_coords, worst_col_coords), new_positions):
            old_row, old_col = old_position
            new_row, new_col = new_position
            
            if (old_row, old_col) != (new_row, new_col):  # Only migrate if position actually changes
                # Deactivate old bit, activate new bit
                sdr[old_row, old_col] = False
                sdr[new_row, new_col] = True
                
                # Reset counters for migrated positions to start fresh learning
                counter[old_row, old_col] = 0
                counter[new_row, new_col] = 0
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save encoder to compressed numpy archive.
        
        Saves all encoder state including SDRs, counters, and metadata
        in a compressed format for efficient storage and loading.
        
        Args:
            filepath: Path to save the encoder (.npz extension recommended)
        """
        filepath = Path(filepath)
        
        # Prepare metadata and structure for saving
        save_data = {
            'height': self.height,
            'width': self.width,
            'sparsity': self.sparsity,
            'vocab': list(self._sdrs.keys()),
        }
        
        # Pack SDRs efficiently using bit packing to reduce file size
        for word, sdr in self._sdrs.items():
            packed_bits = np.packbits(sdr.ravel())
            save_data[f'sdr_{word}'] = packed_bits
            
        # Save Hebbian counters (these remain as full integers)
        for word, counter in self._counters.items():
            save_data[f'counter_{word}'] = counter
        
        np.savez_compressed(filepath, **save_data)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Encoder:
        """Load encoder from saved file.
        
        Reconstructs a complete Encoder instance from a previously
        saved .npz file, restoring all SDRs, counters, and metadata.
        
        Args:
            filepath: Path to the saved encoder file
            
        Returns:
            Loaded Encoder instance with all state restored
        """
        filepath = Path(filepath)
        data = np.load(filepath, allow_pickle=True)
        
        # Reconstruct encoder with original parameters
        height = int(data['height'])
        width = int(data['width'])
        sparsity = float(data['sparsity'])
        vocabulary = data['vocab'].tolist()
        
        encoder = cls(grid=(height, width), sparsity=sparsity)
        
        # Restore SDRs from packed bit representation
        for word in vocabulary:
            packed_bits = data[f'sdr_{word}']
            unpacked_bits = np.unpackbits(packed_bits)
            # Trim to correct size and reshape to original 2D grid
            sdr = unpacked_bits[:height*width].reshape(height, width).astype(bool)
            encoder._sdrs[word] = sdr
            
        # Restore Hebbian counters
        for word in vocabulary:
            counter = data[f'counter_{word}']
            encoder._counters[word] = counter
            
        return encoder
    
    @property
    def vocab(self) -> list[str]:
        """Get list of vocabulary words."""
        return list(self._sdrs.keys())
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (number of words in the encoder)."""
        return len(self._sdrs)
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is in vocabulary, False otherwise
        """
        return word in self._sdrs
    
    def __len__(self) -> int:
        """Get vocabulary size (number of words in the encoder)."""
        return len(self._sdrs) 