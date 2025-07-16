"""Command-line interface for DY SDR Encoder training."""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import yaml

import typer

from .encoder import Encoder  
from .train_loop import train, create_encoder_from_corpus
from .utils import load_vocab_file


app = typer.Typer(help="Train dynamic Sparse Distributed Representations using Hebbian learning")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@app.command()
def main(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file"
    ),
    corpus_path: Optional[Path] = typer.Option(
        None,
        "--corpus-path",
        help="Path to corpus file (one sentence per line)"
    ),
    vocab_path: Optional[Path] = typer.Option(
        None,
        "--vocab-path", 
        help="Path to vocabulary file (one word per line)"
    ),
    output: Path = typer.Option(
        "encoder.npz",
        "--output",
        "-o",
        help="Output path for trained encoder"
    ),
    grid_h: Optional[int] = typer.Option(
        None,
        "--grid-h",
        help="Grid height dimension"
    ),
    grid_w: Optional[int] = typer.Option(
        None,
        "--grid-w", 
        help="Grid width dimension"
    ),
    sparsity: Optional[float] = typer.Option(
        None,
        "--sparsity",
        help="Sparsity fraction (0.0 to 1.0)"
    ),
    window_size: Optional[int] = typer.Option(
        None,
        "--window-size",
        help="Context window radius"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Number of training epochs"
    ),
    swap_per_epoch: Optional[int] = typer.Option(
        None,
        "--swap-per-epoch",
        help="Number of bits to swap per word each epoch"
    ),
    neighbourhood_d: Optional[int] = typer.Option(
        None,
        "--neighbourhood-d",
        help="Manhattan distance for neighborhood search"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        help="Enable verbose output"
    ),
) -> None:
    """Train a DY SDR Encoder on a text corpus."""
    
    # Load configuration if provided
    if config is not None:
        if not config.exists():
            typer.echo(f"Error: Configuration file {config} does not exist", err=True)
            raise typer.Exit(1)
            
        try:
            config_dict = load_config(config)
        except Exception as e:
            typer.echo(f"Error loading configuration: {e}", err=True)
            raise typer.Exit(1)
    else:
        config_dict = {}
    
    # Override config with command line arguments
    if corpus_path is not None:
        config_dict['corpus_path'] = str(corpus_path)
    if vocab_path is not None:
        config_dict['vocab_path'] = str(vocab_path)
    if grid_h is not None or grid_w is not None:
        grid_dict = config_dict.get('grid', {})
        if grid_h is not None:
            grid_dict['H'] = grid_h
        if grid_w is not None:
            grid_dict['W'] = grid_w
        config_dict['grid'] = grid_dict
    if sparsity is not None:
        config_dict['sparsity'] = sparsity
    if window_size is not None:
        config_dict['window_size'] = window_size
    if epochs is not None:
        config_dict['epochs'] = epochs
    if swap_per_epoch is not None:
        config_dict['swap_per_epoch'] = swap_per_epoch
    if neighbourhood_d is not None:
        config_dict['neighbourhood_d'] = neighbourhood_d
    if seed is not None:
        config_dict['seed'] = seed
    
    # Validate required parameters
    if 'corpus_path' not in config_dict:
        typer.echo("Error: corpus_path must be specified in config or via --corpus-path", err=True)
        raise typer.Exit(1)
    
    corpus_path_str = config_dict['corpus_path']
    corpus_path_obj = Path(corpus_path_str)
    
    if not corpus_path_obj.exists():
        typer.echo(f"Error: Corpus file {corpus_path_obj} does not exist", err=True)
        raise typer.Exit(1)
    
    # Set defaults
    grid_config = config_dict.get('grid', {})
    H = grid_config.get('H', 128)
    W = grid_config.get('W', 128)
    sparsity_val = config_dict.get('sparsity', 0.02)
    window_size_val = config_dict.get('window_size', 5)
    epochs_val = config_dict.get('epochs', 3)
    swap_per_epoch_val = config_dict.get('swap_per_epoch', 3)
    neighbourhood_d_val = config_dict.get('neighbourhood_d', 2)
    seed_val = config_dict.get('seed', 42)
    
    try:
        # Create encoder
        if 'vocab_path' in config_dict and config_dict['vocab_path']:
            vocab_path_obj = Path(config_dict['vocab_path'])
            if not vocab_path_obj.exists():
                typer.echo(f"Error: Vocabulary file {vocab_path_obj} does not exist", err=True)
                raise typer.Exit(1)
            
            if verbose:
                typer.echo("Loading vocabulary from file...")
            vocab = load_vocab_file(str(vocab_path_obj))
            encoder = Encoder(grid=(H, W), sparsity=sparsity_val, rng=seed_val)
            encoder.init_vocab(vocab)
        else:
            if verbose:
                typer.echo("Extracting vocabulary from corpus...")
            encoder = create_encoder_from_corpus(
                corpus_path_obj,
                grid=(H, W),
                sparsity=sparsity_val,
                seed=seed_val
            )
        
        # Train encoder
        train(
            encoder=encoder,
            corpus_path=corpus_path_obj,
            window_size=window_size_val,
            epochs=epochs_val,
            swap_per_epoch=swap_per_epoch_val,
            neighbourhood_d=neighbourhood_d_val,
            verbose=verbose
        )
        
        # Save encoder
        if verbose:
            typer.echo(f"Saving encoder to {output}")
        encoder.save(output)
        
        if verbose:
            typer.echo("Training completed successfully!")
            
    except Exception as e:
        typer.echo(f"Error during training: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 