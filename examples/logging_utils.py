#!/usr/bin/env python3
"""Logging utilities for DY SDR Encoder examples.

This module provides structured logging functionality for training progress,
overlap measurements, and model statistics.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime


class TrainingLogger:
    """Logger for training progress and overlap measurements."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        """Initialize the training logger.
        
        Args:
            log_dir: Directory to save log files
            experiment_name: Name for this training experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.json"
        
        # Initialize log structure
        self.log_data = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "encoder_config": {},
            "training_config": {},
            "data_info": {},
            "epochs": [],
            "final_stats": {}
        }
    
    def log_encoder_config(self, encoder):
        """Log encoder configuration."""
        self.log_data["encoder_config"] = {
            "grid_height": encoder.height,
            "grid_width": encoder.width,
            "total_bits": encoder.total_bits,
            "active_bits": encoder.active_bits,
            "sparsity": encoder.sparsity,
            "vocab_size": encoder.vocab_size
        }
        self._save_log()
    
    def log_training_config(self, **kwargs):
        """Log training configuration parameters."""
        self.log_data["training_config"].update(kwargs)
        self._save_log()
    
    def log_data_info(self, vocab_size: int, corpus_info: Dict[str, Any]):
        """Log information about training data."""
        self.log_data["data_info"] = {
            "vocab_size": vocab_size,
            "corpus_lines": corpus_info.get("total_lines", 0),
            "corpus_tokens": corpus_info.get("total_tokens", 0)
        }
        self._save_log()
    
    def log_epoch(self, epoch: int, overlaps: Dict[Tuple[str, str], int], 
                  training_time: float = None, additional_metrics: Dict = None):
        """Log results for a single epoch."""
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "overlaps": {f"{w1}_{w2}": overlap for (w1, w2), overlap in overlaps.items()},
            "training_time_seconds": training_time
        }
        
        if additional_metrics:
            epoch_data.update(additional_metrics)
        
        self.log_data["epochs"].append(epoch_data)
        self._save_log()
    
    def log_final_stats(self, save_path: str, total_training_time: float = None):
        """Log final training statistics."""
        self.log_data["final_stats"] = {
            "end_time": datetime.now().isoformat(),
            "total_training_time_seconds": total_training_time,
            "model_save_path": str(save_path),
            "total_epochs": len(self.log_data["epochs"])
        }
        self._save_log()
    
    def _save_log(self):
        """Save log data to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def print_summary(self):
        """Print a summary of the training session."""
        print(f"\n=== Training Summary ===")
        print(f"Experiment: {self.experiment_name}")
        print(f"Log file: {self.log_file}")
        
        if self.log_data["encoder_config"]:
            config = self.log_data["encoder_config"]
            print(f"Encoder: {config['grid_height']}x{config['grid_width']} grid, "
                  f"{config['active_bits']} active bits, {config['vocab_size']} words")
        
        if self.log_data["epochs"]:
            print(f"Training: {len(self.log_data['epochs'])} epochs completed")
            
            # Show overlap progression for first test pair
            if self.log_data["epochs"]:
                first_epoch = self.log_data["epochs"][0]
                last_epoch = self.log_data["epochs"][-1]
                
                if first_epoch["overlaps"] and last_epoch["overlaps"]:
                    first_pair = next(iter(first_epoch["overlaps"]))
                    initial_overlap = first_epoch["overlaps"][first_pair]
                    final_overlap = last_epoch["overlaps"][first_pair]
                    
                    pair_name = first_pair.replace("_", " ↔ ")
                    print(f"Sample progress ({pair_name}): {initial_overlap} → {final_overlap}")
        
        if self.log_data["final_stats"].get("total_training_time_seconds"):
            time_sec = self.log_data["final_stats"]["total_training_time_seconds"]
            print(f"Total time: {time_sec:.1f}s")


def create_training_log_table(log_file: str) -> str:
    """Create a markdown table from training log for reports.
    
    Args:
        log_file: Path to JSON log file
        
    Returns:
        Markdown formatted table string
    """
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    if not log_data["epochs"]:
        return "No epoch data available."
    
    # Get all unique word pairs
    all_pairs = set()
    for epoch in log_data["epochs"]:
        all_pairs.update(epoch["overlaps"].keys())
    
    all_pairs = sorted(list(all_pairs))
    
    # Create header
    header = "| Epoch | " + " | ".join(pair.replace("_", "↔") for pair in all_pairs) + " |\n"
    separator = "|-------|" + "------|" * len(all_pairs) + "\n"
    
    # Create rows
    rows = []
    for epoch in log_data["epochs"]:
        epoch_num = epoch["epoch"]
        values = []
        for pair in all_pairs:
            overlap = epoch["overlaps"].get(pair, 0)
            values.append(str(overlap))
        
        row = f"| {epoch_num} | " + " | ".join(values) + " |\n"
        rows.append(row)
    
    return header + separator + "".join(rows)


def export_training_csv(log_file: str, output_file: str):
    """Export training log to CSV format.
    
    Args:
        log_file: Path to JSON log file
        output_file: Path for output CSV file
    """
    import csv
    
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    if not log_data["epochs"]:
        return
    
    # Get all unique word pairs
    all_pairs = set()
    for epoch in log_data["epochs"]:
        all_pairs.update(epoch["overlaps"].keys())
    
    all_pairs = sorted(list(all_pairs))
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ["epoch"] + [pair.replace("_", "_to_") for pair in all_pairs]
        writer.writerow(header)
        
        # Data rows
        for epoch in log_data["epochs"]:
            row = [epoch["epoch"]]
            for pair in all_pairs:
                overlap = epoch["overlaps"].get(pair, 0)
                row.append(overlap)
            writer.writerow(row) 