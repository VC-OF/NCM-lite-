import pytest
import torch
from ncm.tokenizer.model import TokenizerModel


def test_tokenizer_model_forward():
    B, L = 2, 256
    model = TokenizerModel(
        vocab_size=256,
        embed_dim=64,
        max_seq_len=L,
        num_levels=3,
        levels=[64, 128, 256],
        codebook_size=256,
        nested_drop_prob=0.0
    )

    x = torch.randint(0, 256, (B, L))
    logits, code_indices, losses = model(x)

    assert logits.shape == (B, L, 256)
    assert len(code_indices) == 3
    assert "recon_loss" in losses
    assert "vq_loss" in losses
    assert "total_loss" in losses
    assert torch.isfinite(losses["total_loss"])


def test_tokenizer_encode_decode_roundtrip():
    B, L = 2, 256
    model = TokenizerModel(
        vocab_size=256,
        embed_dim=64,
        max_seq_len=L,
        num_levels=3,
        levels=[64, 128, 256],
        codebook_size=256
    )

    x = torch.randint(0, 256, (B, L))
    code_indices = model.encode(x)

    assert len(code_indices) == 3
    for k, r_k in enumerate(code_indices):
        assert r_k.shape == (B, model.levels[k])
        assert r_k.dtype == torch.int64

    decoded_logits = model.decode(code_indices)
    assert decoded_logits.shape == (B, L, 256)


def test_tokenizer_overfit_smoke():
    """Verify that 20 training steps on a single synthetic batch significantly reduce loss."""
    torch.manual_seed(42)
    B, L = 1, 64
    model = TokenizerModel(
        vocab_size=256,
        embed_dim=32,
        max_seq_len=L,
        num_levels=2,
        levels=[32, 64],
        codebook_size=64,
        nested_drop_prob=0.0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    x = torch.randint(0, 256, (B, L))

    # Initial loss
    _, _, initial_losses = model(x)
    initial_loss = initial_losses["total_loss"].item()

    # Train for 20 steps
    for _ in range(20):
        optimizer.zero_grad()
        _, _, step_losses = model(x)
        loss = step_losses["total_loss"]
        loss.backward()
        optimizer.step()

    _, _, final_losses = model(x)
    final_loss = final_losses["total_loss"].item()

    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
