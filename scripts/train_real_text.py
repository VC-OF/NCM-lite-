"""
NCM-Lite Real Text Training Script.
Trains Stage 1 Multiscale RVQ Tokenizer and Stage 2 Next-Concept Transformer
on a real text corpus (TinyStories-style text dataset) using CPU.
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

from ncm.data.dataset import ByteTextDataset
from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.training.trainer import TokenizerTrainer, GeneratorTrainer
from ncm.training.checkpoint import CheckpointManager
from ncm.inference.generate import generate_hierarchical_codes, decode_hierarchical_codes
from ncm.evaluation.metrics import evaluate_reconstruction, evaluate_generation


def train_on_real_text():
    print("=" * 72)
    print("NCM-LITE: Real Text Training on TinyStories Corpus (CPU-First)")
    print("=" * 72)

    device = "cpu"
    torch.manual_seed(42)

    # 1. Dataset loading
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "tinystories_sample.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Corpus not found at {data_path}")

    seq_len = 256
    stride = 64  # sliding window data augmentation
    dataset = ByteTextDataset.from_file(data_path, seq_len=seq_len, stride=stride)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"\n[1/5] Loaded text corpus from: {data_path}")
    print(f"  • Total training sequences: {len(dataset)} chunks of {seq_len} bytes (stride={stride})")

    # 2. Model Initialization
    print("\n[2/5] Initializing Two-Stage Models...")
    levels = [64, 128, 256]

    tokenizer = TokenizerModel(
        vocab_size=256,
        embed_dim=64,
        max_seq_len=seq_len,
        num_levels=3,
        levels=levels,
        codebook_size=256,
        commitment_cost=0.25,
        decay=0.95,
        nested_drop_prob=0.5
    ).to(device)

    generator = NextConceptTransformer(
        num_levels=3,
        levels=levels,
        codebook_size=256,
        latent_dim=64,
        embed_dim=384,
        num_layers=6,
        num_heads=6,
        intermediate_dim=1536,
        dropout=0.1,
        loss_alpha=0.5
    ).to(device)

    print(f"  • Stage 1 Tokenizer parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")
    print(f"  • Stage 2 Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")

    # 3. Stage 1 Optimization (RVQ Tokenizer on Real Text)
    stage1_steps = 60
    print(f"\n[3/5] Stage 1: Training RVQ Tokenizer for {stage1_steps} steps...")
    tok_opt = torch.optim.Adam(tokenizer.parameters(), lr=2e-3)
    tok_trainer = TokenizerTrainer(tokenizer, tok_opt, device=device)

    t0 = time.perf_counter()
    data_iter = iter(dataloader)
    for step in range(1, stage1_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        metrics = tok_trainer.train_step(batch)
        if step % 10 == 0 or step == 1:
            print(f"  • Step {step:02d}/{stage1_steps} | Total Loss: {metrics['total_loss']:.4f} "
                  f"(Recon CE: {metrics['recon_loss']:.4f}, VQ: {metrics['vq_loss']:.4f})")

    stage1_time = time.perf_counter() - t0
    print(f"  • Stage 1 finished in {stage1_time:.2f}s ({stage1_steps / stage1_time:.1f} steps/s)")

    # Measure real text reconstruction
    recon_eval = evaluate_reconstruction(tokenizer, dataloader)
    print(f"  • Real Text Byte Accuracy: {recon_eval['reconstruction_accuracy'] * 100:.2f}%")
    print(f"  • Codebook Utilization:    {recon_eval['mean_codebook_utilization'] * 100:.2f}%")

    # 4. Stage 2 Optimization (Generator with strictly frozen Tokenizer)
    stage2_steps = 60
    print(f"\n[4/5] Stage 2: Training Next-Concept Transformer for {stage2_steps} steps (corruption_rate=0.15)...")
    gen_opt = torch.optim.AdamW(generator.parameters(), lr=5e-4, weight_decay=0.01)
    gen_trainer = GeneratorTrainer(
        generator=generator,
        tokenizer=tokenizer,
        optimizer=gen_opt,
        corruption_rate=0.15,
        device=device
    )

    t0 = time.perf_counter()
    data_iter = iter(dataloader)
    for step in range(1, stage2_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        metrics = gen_trainer.train_step(batch)
        if step % 10 == 0 or step == 1:
            print(f"  • Step {step:02d}/{stage2_steps} | NCM Loss: {metrics['total_ncm_loss']:.4f} "
                  f"(L1: {metrics.get('loss_level_1', 0.0):.4f}, "
                  f"L2: {metrics.get('loss_level_2', 0.0):.4f}, "
                  f"L3: {metrics.get('loss_level_3', 0.0):.4f})")

    stage2_time = time.perf_counter() - t0
    print(f"  • Stage 2 finished in {stage2_time:.2f}s ({stage2_steps / stage2_time:.1f} steps/s)")

    # 5. Save Checkpoint
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ncm_real_text_model.pt")
    CheckpointManager.save(
        filepath=ckpt_path,
        stage=2,
        step=stage2_steps,
        model=generator,
        optimizer=gen_opt,
        loss=metrics["total_ncm_loss"],
        config={"levels": levels, "embed_dim": 384, "seq_len": seq_len}
    )
    # Also save tokenizer weights alongside
    tok_ckpt_path = os.path.join(ckpt_dir, "ncm_real_text_tokenizer.pt")
    CheckpointManager.save(
        filepath=tok_ckpt_path,
        stage=1,
        step=stage1_steps,
        model=tokenizer,
        optimizer=tok_opt,
        loss=recon_eval["reconstruction_loss"]
    )
    print(f"\n[5/5] Saved model checkpoints to:")
    print(f"  • Generator: {ckpt_path}")
    print(f"  • Tokenizer: {tok_ckpt_path}")

    # Generate initial sample
    print("\nGenerating initial text sample (K=3 parallel passes, greedy)...")
    codes = generate_hierarchical_codes(
        generator=generator,
        tokenizer=tokenizer,
        batch_size=1,
        temperature=0.0
    )
    sample_text = decode_hierarchical_codes(tokenizer, codes)[0]
    preview = repr(sample_text[:180])
    print(f"  Result: {preview}")

    print("\n" + "=" * 72)
    print("REAL TEXT TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    train_on_real_text()
