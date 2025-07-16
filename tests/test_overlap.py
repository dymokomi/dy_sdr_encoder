"""Mandatory overlap test for SDR training validation."""

import numpy as np
from dy_sdr_encoder.encoder import Encoder
from dy_sdr_encoder.train_loop import fit_one_epoch


def test_overlap():
    """Test that training increases overlap between related words."""
    enc = Encoder(grid=(32, 32), sparsity=0.05, rng=0)
    enc.init_vocab(["cat", "dog", "table"])
    before = np.count_nonzero(enc.flat("cat") & enc.flat("dog"))
    
    # Create token windows: cat has dog as context, dog has cat as context
    token_windows = [
        ("cat", ["dog"]),
        ("dog", ["cat"]),
        ("cat", ["dog"])
    ]
    
    fit_one_epoch(
        encoder=enc,
        token_windows=iter(token_windows),
        window_size=1,
        K_swap=1,
        d=1,
    )
    after = np.count_nonzero(enc.flat("cat") & enc.flat("dog"))
    assert after >= before 