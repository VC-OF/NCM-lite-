import pytest
import torch
import torch.nn.functional as F
from ncm.quantization.rvq import MultiscaleResidualVectorQuantizer


def test_rvq_output_shapes():
    B, L, D = 2, 256, 64
    levels = [64, 128, 256]
    rvq = MultiscaleResidualVectorQuantizer(
        num_levels=3,
        levels=levels,
        codebook_size=256,
        embed_dim=D,
        nested_drop_prob=0.0,
        target_len=L
    )

    z = torch.randn(B, L, D)
    z_hat, code_indices, loss_dict = rvq(z)

    assert z_hat.shape == (B, L, D), f"Expected z_hat shape {(B, L, D)}, got {z_hat.shape}"
    assert len(code_indices) == 3, f"Expected 3 code index tensors, got {len(code_indices)}"
    for k, r_k in enumerate(code_indices):
        assert r_k.shape == (B, levels[k]), f"Level {k} shape mismatch: expected {(B, levels[k])}, got {r_k.shape}"
        assert r_k.dtype == torch.int64, f"Level {k} dtype must be int64, got {r_k.dtype}"
        assert (r_k >= 0).all() and (r_k < 256).all(), f"Level {k} codes out of range [0, 255]"
    assert "vq_loss" in loss_dict
    assert torch.isfinite(loss_dict["vq_loss"])


def test_rvq_ste_gradient_flow():
    """Verify that the Straight-Through Estimator passes gradients back to input continuous latents."""
    B, L, D = 2, 64, 32
    rvq = MultiscaleResidualVectorQuantizer(
        num_levels=2,
        levels=[32, 64],
        codebook_size=128,
        embed_dim=D,
        nested_drop_prob=0.0,
        target_len=L
    )

    z = torch.randn(B, L, D, requires_grad=True)
    z_hat, _, _ = rvq(z)

    loss = z_hat.sum()
    loss.backward()

    assert z.grad is not None, "Gradients failed to flow back to input z"
    assert not torch.isnan(z.grad).any(), "NaN found in input gradient"
    assert (z.grad != 0).any(), "Gradient to input z is all zeros"


def test_rvq_idx_to_z_q_dequantization():
    """Verify that discrete codes can be dequantized back into a continuous tensor without grad."""
    B, L, D = 2, 256, 64
    levels = [64, 128, 256]
    rvq = MultiscaleResidualVectorQuantizer(
        num_levels=3,
        levels=levels,
        codebook_size=256,
        embed_dim=D,
        target_len=L
    )

    dummy_codes = [
        torch.randint(0, 256, (B, levels[0])),
        torch.randint(0, 256, (B, levels[1])),
        torch.randint(0, 256, (B, levels[2]))
    ]

    z_hat = rvq.idx_to_z_q(dummy_codes)
    assert z_hat.shape == (B, L, D)
    assert not z_hat.requires_grad


def test_get_next_autoregressive_input():
    """Verify get_next_autoregressive_input signature and output shapes match models/quant.py."""
    B, L, D = 2, 256, 64
    levels = [64, 128, 256]
    rvq = MultiscaleResidualVectorQuantizer(
        num_levels=3,
        levels=levels,
        embed_dim=D,
        target_len=L
    )

    z_hat = torch.zeros(B, L, D)
    z_q_0 = torch.randn(B, levels[0], D)

    updated_z_hat, next_input = rvq.get_next_autoregressive_input(
        level_idx=0,
        z_hat=z_hat,
        z_q_level=z_q_0
    )

    assert updated_z_hat.shape == (B, L, D)
    # Next input for level 1 should be resized to level 1's resolution: levels[1] = 128
    assert next_input.shape == (B, levels[1], D)
