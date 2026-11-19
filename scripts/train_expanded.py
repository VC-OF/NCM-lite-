"""
NCM-Lite: Expanded Real Text Training Pipeline (CPU-Optimized).
Trains Stage 1 Multiscale RVQ Tokenizer and Stage 2 Next-Concept Transformer
on the expanded 512 KB TinyStories dataset (~101k words, 8,174 chunks).
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from ncm.data.dataset import ByteTextDataset
from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.training.trainer import TokenizerTrainer, GeneratorTrainer
from ncm.training.checkpoint import CheckpointManager
from ncm.inference.generate import generate_hierarchical_codes, decode_hierarchical_codes
from ncm.evaluation.metrics import evaluate_reconstruction


def train_expanded():
    print("=" * 76)
    print("NCM-LITE: EXPANDED REAL TEXT TRAINING PIPELINE (CPU-OPTIMIZED)")
    print("=" * 76)

    # Maximize CPU thread efficiency
    num_cpus = os.cpu_count() or 4
    threads = min(8, num_cpus)
    torch.set_num_threads(threads)
    print(f"[+] Initialized CPU execution with {threads} worker threads.")

    device = "cpu"
    torch.manual_seed(42)

    # 1. Dataset loading
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "tinystories_expanded.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expanded corpus not found at {data_path}. Run prepare_expanded_data.py first.")

    seq_len = 256
    stride = 64
    dataset = ByteTextDataset.from_file(data_path, seq_len=seq_len, stride=stride)
    print(f"\n[1/5] Loaded expanded dataset: {data_path}")
    print(f"  • Total training sequences: {len(dataset):,} chunks of {seq_len} bytes (stride={stride})")

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

    tok_params = sum(p.numel() for p in tokenizer.parameters())
    gen_params = sum(p.numel() for p in generator.parameters())
    print(f"  • Stage 1 Tokenizer parameters: {tok_params:,}")
    print(f"  • Stage 2 Generator parameters: {gen_params:,} (Total: {tok_params + gen_params:,})")

    # 3. Stage 1 Optimization: RVQ Tokenizer
    stage1_steps = 120
    tok_batch_size = 8
    tok_dataloader = DataLoader(dataset, batch_size=tok_batch_size, shuffle=True)
    print(f"\n[3/5] Stage 1: Training RVQ Tokenizer for {stage1_steps} steps (batch_size={tok_batch_size})...")

    tok_opt = torch.optim.Adam(tokenizer.parameters(), lr=2e-3)
    tok_trainer = TokenizerTrainer(tokenizer, tok_opt, device=device)

    t0 = time.perf_counter()
    data_iter = iter(tok_dataloader)
    for step in range(1, stage1_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(tok_dataloader)
            batch = next(data_iter)

        metrics = tok_trainer.train_step(batch)
        if step % 20 == 0 or step == 1 or step == stage1_steps:
            print(f"  • Step {step:03d}/{stage1_steps} | Total Loss: {metrics['total_loss']:.4f} "
                  f"(Recon CE: {metrics['recon_loss']:.4f}, VQ: {metrics['vq_loss']:.4f})")

    stage1_time = time.perf_counter() - t0
    print(f"  • Stage 1 finished in {stage1_time:.2f}s ({stage1_steps / stage1_time:.1f} steps/s)")

    # Evaluate reconstruction accuracy
    eval_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    recon_eval = evaluate_reconstruction(tokenizer, eval_loader)
    print(f"  • Real Text Byte Reconstruction Accuracy: {recon_eval['reconstruction_accuracy'] * 100:.2f}%")
    print(f"  • Codebook Utilization:                   {recon_eval['mean_codebook_utilization'] * 100:.2f}%")

    # 4. Stage 2 Optimization: Generator with strictly frozen Tokenizer
    stage2_steps = 150
    gen_batch_size = 4
    gen_dataloader = DataLoader(dataset, batch_size=gen_batch_size, shuffle=True)
    print(f"\n[4/5] Stage 2: Training Next-Concept Transformer for {stage2_steps} steps (corruption=0.15)...")

    gen_opt = torch.optim.AdamW(generator.parameters(), lr=6e-4, weight_decay=0.01)
    gen_scheduler = CosineAnnealingLR(gen_opt, T_max=stage2_steps, eta_min=1e-5)
    gen_trainer = GeneratorTrainer(
        generator=generator,
        tokenizer=tokenizer,
        optimizer=gen_opt,
        corruption_rate=0.15,
        device=device
    )

    t0 = time.perf_counter()
    data_iter = iter(gen_dataloader)
    for step in range(1, stage2_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(gen_dataloader)
            batch = next(data_iter)

        metrics = gen_trainer.train_step(batch)
        gen_scheduler.step()

        if step % 15 == 0 or step == 1 or step == stage2_steps:
            current_lr = gen_opt.param_groups[0]["lr"]
            print(f"  • Step {step:03d}/{stage2_steps} | NCM Loss: {metrics['total_ncm_loss']:.4f} "
                  f"(L1: {metrics.get('loss_level_1', 0.0):.4f}, "
                  f"L2: {metrics.get('loss_level_2', 0.0):.4f}, "
                  f"L3: {metrics.get('loss_level_3', 0.0):.4f}) | lr: {current_lr:.1e}")

    stage2_time = time.perf_counter() - t0
    print(f"  • Stage 2 finished in {stage2_time:.2f}s ({stage2_steps / stage2_time:.2f} steps/s)")

    # 5. Checkpoint Saving
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

    tok_ckpt_path = os.path.join(ckpt_dir, "ncm_real_text_tokenizer.pt")
    CheckpointManager.save(
        filepath=tok_ckpt_path,
        stage=1,
        step=stage1_steps,
        model=tokenizer,
        optimizer=tok_opt,
        loss=recon_eval["reconstruction_loss"],
        config={"levels": levels, "embed_dim": 64, "seq_len": seq_len}
    )

    print(f"\n[5/5] Saved updated model checkpoints:")
    print(f"  • Generator: {ckpt_path}")
    print(f"  • Tokenizer: {tok_ckpt_path}")

    # 6. Sample Generation
    print("\n" + "=" * 76)
    print("SAMPLE GENERATION TEST (K=3 Coarse-to-Fine Parallel Passes):")
    print("=" * 76)
    generator.eval()
    tokenizer.eval()

    sample_codes = generate_hierarchical_codes(generator, tokenizer, batch_size=1, temperature=0.7, top_k=30, top_p=0.9)
    sample_text = decode_hierarchical_codes(tokenizer, sample_codes)[0]
    print(f"\n[Coherent Preset Sample (T=0.7, top_k=30, top_p=0.9)]:\n\"{sample_text[:220]}\"")
    print("\n" + "=" * 76)
    print("EXPANDED CONVERGENCE TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 76)


if __name__ == "__main__":
    train_expanded()
