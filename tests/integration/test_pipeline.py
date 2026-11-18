import time
import tracemalloc
import pytest
import torch
from torch.utils.data import DataLoader

from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.data.dataset import SyntheticTextDataset
from ncm.training.trainer import TokenizerTrainer, GeneratorTrainer
from ncm.training.checkpoint import CheckpointManager
from ncm.inference.generate import generate_hierarchical_codes, decode_hierarchical_codes
from ncm.evaluation.metrics import evaluate_reconstruction, evaluate_generation


def test_end_to_end_smoke_pipeline(tmp_path):
    """
    Comprehensive end-to-end integration test:
    Stage 1 training -> Freeze -> Stage 2 training -> Save/Load -> Generate -> Decode.
    Verifies CPU latency ceiling (< 2.0s per sample) and memory ceiling (< 1.5 GB).
    """
    torch.manual_seed(42)

    # 1. Configuration
    tokenizer = TokenizerModel(
        vocab_size=256,
        embed_dim=64,
        max_seq_len=256,
        num_levels=3,
        levels=[64, 128, 256],
        codebook_size=256,
        commitment_cost=0.25,
        nested_drop_prob=0.5,
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

    # 2. Data pipeline
    dataset = SyntheticTextDataset(num_samples=16, seq_len=256, seed=42)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))

    # 3. Stage 1 Optimization
    tok_opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
    tok_trainer = TokenizerTrainer(tokenizer, tok_opt)
    tok_metrics = tok_trainer.train_step(batch)
    assert "total_loss" in tok_metrics
    assert "recon_loss" in tok_metrics

    # 4. Stage 2 Optimization (with strictly frozen tokenizer)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=1e-3)
    gen_trainer = GeneratorTrainer(generator, tokenizer, gen_opt)
    gen_metrics = gen_trainer.train_step(batch)
    assert "total_ncm_loss" in gen_metrics

    # Verify tokenizer remains frozen
    for param in tokenizer.parameters():
        assert param.grad is None or not param.requires_grad

    # 5. Checkpoint roundtrip
    ckpt_path = str(tmp_path / "ncm_pipeline_test.pt")
    CheckpointManager.save(
        filepath=ckpt_path,
        stage=2,
        step=1,
        model=generator,
        optimizer=gen_opt,
        loss=gen_metrics["total_ncm_loss"],
    )

    loaded = CheckpointManager.load(ckpt_path, model=generator, optimizer=gen_opt)
    assert loaded["step"] == 1
    assert loaded["stage"] == 2

    # 6. Evaluation of Reconstruction
    recon_metrics = evaluate_reconstruction(tokenizer, dataloader)
    assert recon_metrics["reconstruction_loss"] >= 0.0

    # 7. Generation and Memory/Latency Profiling
    tracemalloc.start()
    t0 = time.perf_counter()

    codes = generate_hierarchical_codes(
        generator=generator,
        tokenizer=tokenizer,
        batch_size=2,
        temperature=1.0,
    )
    samples = decode_hierarchical_codes(tokenizer, codes)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    latency_per_sample = elapsed / 2.0

    assert len(samples) == 2
    assert latency_per_sample < 2.0, f"Latency {latency_per_sample:.3f}s exceeds 2.0s ceiling"
    assert peak_mb < 1500.0, f"Peak memory {peak_mb:.1f} MB exceeds 1.5 GB ceiling"
