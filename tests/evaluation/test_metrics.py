import pytest
import torch
from torch.utils.data import DataLoader

from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.data.dataset import SyntheticTextDataset
from ncm.evaluation.metrics import evaluate_reconstruction, evaluate_generation


@pytest.fixture
def test_setup():
    torch.manual_seed(42)
    tokenizer = TokenizerModel(
        vocab_size=256,
        embed_dim=64,
        max_seq_len=256,
        num_levels=3,
        levels=[64, 128, 256],
        codebook_size=256,
        commitment_cost=0.25,
        nested_drop_prob=0.0,
    )

    generator = NextConceptTransformer(
        num_levels=3,
        levels=[64, 128, 256],
        codebook_size=256,
        latent_dim=64,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        intermediate_dim=128,
        loss_alpha=0.5,
    )

    dataset = SyntheticTextDataset(num_samples=8, seq_len=256, seed=42)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    return tokenizer, generator, dataloader


def test_evaluate_reconstruction(test_setup):
    """
    Verifies that evaluate_reconstruction computes correct metrics bounds.
    """
    tokenizer, _, dataloader = test_setup

    metrics = evaluate_reconstruction(tokenizer, dataloader)

    assert "reconstruction_accuracy" in metrics
    assert 0.0 <= metrics["reconstruction_accuracy"] <= 1.0
    assert "reconstruction_loss" in metrics
    assert metrics["reconstruction_loss"] >= 0.0

    for k in range(1, 4):
        util_key = f"codebook_utilization_level_{k}"
        perp_key = f"codebook_perplexity_level_{k}"
        assert util_key in metrics
        assert 0.0 <= metrics[util_key] <= 1.0
        assert perp_key in metrics
        assert metrics[perp_key] >= 1.0

    assert "mean_codebook_utilization" in metrics
    assert 0.0 <= metrics["mean_codebook_utilization"] <= 1.0


def test_evaluate_generation(test_setup):
    """
    Verifies that evaluate_generation captures latency, memory, and samples.
    """
    tokenizer, generator, _ = test_setup

    results = evaluate_generation(
        generator=generator,
        tokenizer=tokenizer,
        num_samples=2,
        temperature=1.0,
    )

    assert results["num_samples"] == 2
    assert results["sequence_length"] == 256
    assert results["num_passes"] == 3
    assert results["total_time_seconds"] > 0.0
    assert results["latency_per_sample_seconds"] > 0.0
    assert results["bytes_per_second"] > 0.0
    assert results["peak_memory_mb"] >= 0.0
    assert len(results["samples"]) == 2
    for s in results["samples"]:
        assert isinstance(s, str)
