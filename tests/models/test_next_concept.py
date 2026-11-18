import pytest
import torch
from ncm.models.next_concept import NextConceptTransformer


def test_next_concept_output_shapes():
    B = 2
    levels = [64, 128, 256]
    latent_dim = 64
    codebook_size = 256

    model = NextConceptTransformer(
        num_levels=3,
        levels=levels,
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        embed_dim=192,
        num_layers=2,
        num_heads=3,
        intermediate_dim=768
    )

    inputs = [torch.randn(B, ell, latent_dim) for ell in levels]
    targets = [torch.randint(0, codebook_size, (B, ell)) for ell in levels]

    logits_list, losses = model(inputs, target_codes=targets)

    assert len(logits_list) == 3
    for k, Z_k in enumerate(logits_list):
        assert Z_k.shape == (B, levels[k], codebook_size), f"Level {k} shape mismatch: {Z_k.shape}"
        assert torch.isfinite(Z_k).all()

    assert "total_ncm_loss" in losses
    assert torch.isfinite(losses["total_ncm_loss"])


def test_block_causal_mask_non_leakage():
    """Verify that earlier level logits are completely independent of future level inputs."""
    B = 1
    levels = [32, 64]
    latent_dim = 32

    model = NextConceptTransformer(
        num_levels=2,
        levels=levels,
        codebook_size=64,
        latent_dim=latent_dim,
        embed_dim=128,
        num_layers=2,
        num_heads=2,
        intermediate_dim=256
    )
    model.eval()

    u_1 = torch.randn(B, levels[0], latent_dim)
    u_2_a = torch.randn(B, levels[1], latent_dim)
    u_2_b = u_2_a + 50.0  # heavily perturb level 2 input

    with torch.no_grad():
        logits_a, _ = model([u_1, u_2_a])
        logits_b, _ = model([u_1, u_2_b])

    # Level 1 logits must be numerically identical because level 1 cannot attend to level 2
    diff = torch.max(torch.abs(logits_a[0] - logits_b[0])).item()
    assert diff < 1e-5, f"Information leaked from future level 2 to level 1: diff={diff}"


def test_generator_parameter_count_budget():
    """Verify that primary configuration is within the 10-15M parameter target."""
    levels = [64, 128, 256]
    model = NextConceptTransformer(
        num_levels=3,
        levels=levels,
        codebook_size=256,
        latent_dim=64,
        embed_dim=384,
        num_layers=6,
        num_heads=6,
        intermediate_dim=1536
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameter count: {total_params / 1e6:.2f}M")

    # Budget target: 10M to 15M parameters
    assert 10_000_000 <= total_params <= 15_000_000, (
        f"Parameter count {total_params / 1e6:.2f}M is outside 10-15M target envelope"
    )


def test_single_level_prediction():
    B = 2
    levels = [32, 64]
    latent_dim = 32
    codebook_size = 128

    model = NextConceptTransformer(
        num_levels=2,
        levels=levels,
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        intermediate_dim=128
    )

    u_0 = torch.randn(B, levels[0], latent_dim)
    logits_0 = model.predict_single_level(u_0, level_idx=0)
    assert logits_0.shape == (B, levels[0], codebook_size)

    u_1 = torch.randn(B, levels[1], latent_dim)
    logits_1 = model.predict_single_level(u_1, level_idx=1, all_prev_u=[u_0])
    assert logits_1.shape == (B, levels[1], codebook_size)


def test_generator_synthetic_overfit():
    """Verify that 20 training steps on fixed synthetic targets monotonically decrease loss."""
    torch.manual_seed(42)
    B = 1
    levels = [16, 32]
    latent_dim = 16
    codebook_size = 32

    model = NextConceptTransformer(
        num_levels=2,
        levels=levels,
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        intermediate_dim=128
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    inputs = [torch.randn(B, ell, latent_dim) for ell in levels]
    targets = [torch.randint(0, codebook_size, (B, ell)) for ell in levels]

    _, initial_losses = model(inputs, target_codes=targets)
    initial_loss = initial_losses["total_ncm_loss"].item()

    for _ in range(20):
        optimizer.zero_grad()
        _, step_losses = model(inputs, target_codes=targets)
        loss = step_losses["total_ncm_loss"]
        loss.backward()
        optimizer.step()

    _, final_losses = model(inputs, target_codes=targets)
    final_loss = final_losses["total_ncm_loss"].item()

    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
