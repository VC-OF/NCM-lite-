"""
NCM-Lite Prompt-Conditioned Text Continuation Script.

Demonstrates prefix-conditioned hierarchical text generation (Algorithm 2).
Given a prompt prefix, prompt codes are clamped across all K=3 scales while the
generator predicts the remaining continuation tokens in parallel.
"""

import os
import sys
import time
import argparse
import torch

from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.training.checkpoint import CheckpointManager
from ncm.inference.generate import generate_prompt_continuation


def run_prompt_generation(
    prompt: str,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.92,
    num_samples: int = 1
):
    print("=" * 76)
    print("NCM-LITE: PROMPT-CONDITIONED TEXT CONTINUATION")
    print("Parallel Hierarchical Completion Across K=3 Scales (Algorithm 2)")
    print("=" * 76)

    device = "cpu"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    levels = [64, 128, 256]
    seq_len = 256

    # 1. Initialize models
    tokenizer = TokenizerModel(
        vocab_size=256,
        embed_dim=64,
        max_seq_len=seq_len,
        num_levels=3,
        levels=levels,
        codebook_size=256,
        commitment_cost=0.25,
        decay=0.95,
        nested_drop_prob=0.0
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
        dropout=0.0,
        loss_alpha=0.5
    ).to(device)

    # 2. Load Checkpoint
    gen_ckpt = os.path.join(base_dir, "checkpoints", "ncm_real_text_model.pt")
    tok_ckpt = os.path.join(base_dir, "checkpoints", "ncm_real_text_tokenizer.pt")

    if os.path.exists(gen_ckpt):
        print(f"[+] Loaded Generator from: {gen_ckpt}")
        CheckpointManager.load(gen_ckpt, model=generator, device=device)
    else:
        print("[!] Generator checkpoint not found. Using initialized weights.")

    if os.path.exists(tok_ckpt):
        print(f"[+] Loaded Tokenizer from: {tok_ckpt}")
        CheckpointManager.load(tok_ckpt, model=tokenizer, device=device)
    else:
        print("[!] Tokenizer checkpoint not found. Using initialized weights.")

    generator.eval()
    tokenizer.eval()

    print(f"\n[Prompt Input]: \"{prompt}\"")
    print(f"[Sampling Params]: T={temperature}, top_k={top_k}, top_p={top_p}")

    t0 = time.perf_counter()
    continuations = generate_prompt_continuation(
        generator=generator,
        tokenizer=tokenizer,
        prompt=prompt,
        batch_size=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device
    )
    elapsed = time.perf_counter() - t0

    print(f"[Generation Time]: {elapsed:.3f}s ({seq_len / elapsed:.1f} bytes/sec on CPU in 3 passes)")
    print("\n" + "-" * 76)
    print("COMPLETED TEXT OUTPUT:")
    print("-" * 76)

    for i, completed in enumerate(continuations, 1):
        clean_text = completed.replace("\n", " ").strip()
        print(f"\nSample #{i}:")
        print(f"\"{clean_text}\"")

    print("\n" + "=" * 76)


def main():
    parser = argparse.ArgumentParser(description="NCM-Lite Prompt Continuation")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to continue")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--top_p", type=float, default=0.92, help="Top-p nucleus threshold")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples")
    args = parser.parse_args()

    default_prompts = [
        "Once upon a time, Lily found a ",
        "Tim and his dog Spot ran outside to ",
        "The little bird wanted to build a "
    ]

    if args.prompt:
        run_prompt_generation(
            prompt=args.prompt,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples=args.samples
        )
    else:
        for p in default_prompts:
            run_prompt_generation(
                prompt=p,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                num_samples=1
            )
            print("\n")


if __name__ == "__main__":
    main()
