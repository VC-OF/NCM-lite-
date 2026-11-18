import os
import tempfile
import pytest
import torch
from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.training.trainer import TokenizerTrainer, GeneratorTrainer
from ncm.training.checkpoint import CheckpointManager


def test_tokenizer_trainer_step():
    B, L = 2, 64
    model = TokenizerModel(
        vocab_size=256,
        embed_dim=32,
        max_seq_len=L,
        num_levels=2,
        levels=[32, 64],
        codebook_size=64,
        nested_drop_prob=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = TokenizerTrainer(model, optimizer)

    batch = torch.randint(0, 256, (B, L))
    metrics = trainer.train_step(batch)

    assert "total_loss" in metrics
    assert "recon_loss" in metrics
    assert "vq_loss" in metrics
    assert trainer.step_count == 1


def test_generator_trainer_frozen_tokenizer():
    """Verify that Stage 2 training strictly freezes tokenizer parameters (grad is None)."""
    B, L = 2, 64
    tokenizer = TokenizerModel(
        vocab_size=256,
        embed_dim=32,
        max_seq_len=L,
        num_levels=2,
        levels=[32, 64],
        codebook_size=64
    )
    generator = NextConceptTransformer(
        num_levels=2,
        levels=[32, 64],
        codebook_size=64,
        latent_dim=32,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        intermediate_dim=128
    )

    optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-3)
    trainer = GeneratorTrainer(generator, tokenizer, optimizer)

    batch = torch.randint(0, 256, (B, L))
    metrics = trainer.train_step(batch)

    assert "total_ncm_loss" in metrics
    assert trainer.step_count == 1

    # Verify that all tokenizer parameters have grad is None
    for name, param in tokenizer.named_parameters():
        assert param.grad is None, f"Tokenizer parameter {name} accumulated gradient during Stage 2!"


def test_checkpoint_manager_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_checkpoint.pt")

        model1 = TokenizerModel(vocab_size=256, embed_dim=16, max_seq_len=32, num_levels=2, levels=[16, 32], codebook_size=32)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

        # Save checkpoint
        CheckpointManager.save(
            filepath=ckpt_path,
            stage=1,
            step=42,
            model=model1,
            optimizer=optimizer1,
            loss=1.234
        )
        assert os.path.exists(ckpt_path)

        # Load checkpoint into new model
        model2 = TokenizerModel(vocab_size=256, embed_dim=16, max_seq_len=32, num_levels=2, levels=[16, 32], codebook_size=32)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        ckpt_data = CheckpointManager.load(ckpt_path, model=model2, optimizer=optimizer2)

        assert ckpt_data["step"] == 42
        assert ckpt_data["stage"] == 1
        assert ckpt_data["loss"] == 1.234

        # Verify parameter equality
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model parameters did not match after checkpoint load"
