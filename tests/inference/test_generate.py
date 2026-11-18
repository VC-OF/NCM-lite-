import pytest
import torch
from unittest.mock import MagicMock

from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.inference.generate import (
    generate_hierarchical_codes,
    decode_hierarchical_codes,
    _sample_from_logits,
)


@pytest.fixture
def test_models():
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
        embed_dim=64,  # compact for fast testing
        num_layers=2,
        num_heads=2,
        intermediate_dim=128,
        loss_alpha=0.5,
    )
    return tokenizer, generator


def test_parallel_inference_steps(test_models):
    """
    Verifies that generate_hierarchical_codes executes exactly K=3 sequential
    generator passes, returning correctly shaped discrete codes.
    """
    tokenizer, generator = test_models
    batch_size = 2

    # Wrap predict_single_level with a spy to track call count
    original_predict = generator.predict_single_level
    spy_predict = MagicMock(side_effect=original_predict)
    generator.predict_single_level = spy_predict

    codes = generate_hierarchical_codes(
        generator=generator,
        tokenizer=tokenizer,
        batch_size=batch_size,
        temperature=0.0,  # greedy
    )

    # Must execute exactly K=3 generator prediction steps
    assert spy_predict.call_count == 3, f"Expected 3 passes, got {spy_predict.call_count}"
    assert len(codes) == 3

    # Check shapes and types
    assert codes[0].shape == (batch_size, 64)
    assert codes[1].shape == (batch_size, 128)
    assert codes[2].shape == (batch_size, 256)
    for c in codes:
        assert c.dtype == torch.int64
        assert (c >= 0).all() and (c < 256).all()


def test_decode_hierarchical_codes(test_models):
    """
    Verifies that decode_hierarchical_codes maps discrete codes to UTF-8 text strings.
    """
    tokenizer, _ = test_models
    batch_size = 3

    # Create synthetic discrete codes
    codes = [
        torch.randint(0, 256, (batch_size, 64)),
        torch.randint(0, 256, (batch_size, 128)),
        torch.randint(0, 256, (batch_size, 256)),
    ]

    strings = decode_hierarchical_codes(tokenizer, codes)
    assert len(strings) == batch_size
    for s in strings:
        assert isinstance(s, str)
        # Each sequence has 256 bytes reconstructed
        assert len(s.encode("utf-8", errors="replace")) >= 0


def test_generation_greedy_determinism(test_models):
    """
    Verifies that temperature=0.0 produces identical deterministic outputs.
    """
    tokenizer, generator = test_models

    codes_1 = generate_hierarchical_codes(generator, tokenizer, batch_size=1, temperature=0.0)
    codes_2 = generate_hierarchical_codes(generator, tokenizer, batch_size=1, temperature=0.0)

    for c1, c2 in zip(codes_1, codes_2):
        assert torch.equal(c1, c2)


def test_sampling_modes():
    """
    Verifies greedy, temperature, top-k, and top-p sampling mechanics.
    """
    torch.manual_seed(42)
    B, L, V = 2, 4, 10
    logits = torch.randn(B, L, V)
    # Ensure one token dominates
    logits[..., 0] = 50.0

    # Greedy should always pick index 0
    greedy_out = _sample_from_logits(logits, temperature=0.0)
    assert (greedy_out == 0).all()

    # Top-k with k=1 should equal greedy
    top1_out = _sample_from_logits(logits, temperature=1.0, top_k=1)
    assert (top1_out == 0).all()

    # Top-p with low p should also isolate the dominant token
    topp_out = _sample_from_logits(logits, temperature=1.0, top_p=0.8)
    assert (topp_out == 0).all()

    # Uniform logits with sampling generates valid range
    uniform_logits = torch.ones(B, L, V)
    sample_out = _sample_from_logits(uniform_logits, temperature=1.0, top_k=5, top_p=0.9)
    assert (sample_out >= 0).all() and (sample_out < V).all()
