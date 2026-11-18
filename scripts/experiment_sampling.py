"""
NCM-Lite Sampling Exploration Script.
Compares decoding strategies across temperature, top-k, and top-p (nucleus)
sampling on the trained Next-Concept Modeling prototype.
"""

import os
import sys
import time
import math
import collections
from typing import Dict, List, Tuple
import torch

from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.training.checkpoint import CheckpointManager
from ncm.inference.generate import generate_hierarchical_codes, decode_hierarchical_codes


def compute_text_entropy(text: str) -> float:
    """Computes Shannon entropy (in bits) over character frequencies."""
    if not text:
        return 0.0
    counts = collections.Counter(text)
    total = len(text)
    entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())
    return float(entropy)


def count_unique_words(text: str) -> int:
    """Counts unique alphanumeric tokens in the generated text."""
    words = [w.strip(".,!?;:\"'()[]{} \t\n") for w in text.split()]
    return len(set(w.lower() for w in words if w))


def run_sampling_experiments():
    print("=" * 76)
    print("NCM-LITE: SAMPLING EXPERIMENTS (GREEDY vs. TOP-K vs. TOP-P / NUCLEUS)")
    print("Sequential K=3 Parallel Passes Coarse-to-Fine Generation (Algorithm 2)")
    print("=" * 76)

    device = "cpu"
    torch.manual_seed(42)

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

    # 2. Load trained checkpoint if available
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gen_ckpt = os.path.join(base_dir, "checkpoints", "ncm_real_text_model.pt")
    tok_ckpt = os.path.join(base_dir, "checkpoints", "ncm_real_text_tokenizer.pt")

    if os.path.exists(gen_ckpt):
        print(f"\n[+] Loading Generator checkpoint from: {gen_ckpt}")
        CheckpointManager.load(gen_ckpt, model=generator, device=device)
    else:
        print("\n[!] Generator checkpoint not found. Using initialized weights.")

    if os.path.exists(tok_ckpt):
        print(f"[+] Loading Tokenizer checkpoint from: {tok_ckpt}")
        CheckpointManager.load(tok_ckpt, model=tokenizer, device=device)
    else:
        print("[!] Tokenizer checkpoint not found. Using initialized weights.")

    generator.eval()
    tokenizer.eval()

    # 3. Define sampling experiment regimes
    experiment_configs = [
        {
            "name": "1. Pure Greedy (Argmax)",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "description": "Deterministic mode; selects max-probability code at each of the K=3 levels."
        },
        {
            "name": "2. Low Temperature",
            "temperature": 0.3,
            "top_k": 0,
            "top_p": 1.0,
            "description": "Sharpened distribution; conservative sampling concentrating on dominant codes."
        },
        {
            "name": "3. Top-k Filtering",
            "temperature": 0.8,
            "top_k": 20,
            "top_p": 1.0,
            "description": "Truncates logits to the top 20 candidate codes before softmax."
        },
        {
            "name": "4. Top-p (Nucleus) Filtering",
            "temperature": 0.8,
            "top_k": 0,
            "top_p": 0.90,
            "description": "Dynamically selects smallest set of codes with cumulative prob >= 90%."
        },
        {
            "name": "5. Combined Top-k + Top-p",
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.92,
            "description": "Standard modern LLM generation regime balancing diversity and coherence."
        },
        {
            "name": "6. High Temperature / Creative",
            "temperature": 1.2,
            "top_k": 0,
            "top_p": 0.95,
            "description": "Flattened distribution allowing tail exploration with wide nucleus cutoff."
        },
        {
            "name": "7. Unconstrained High Temperature",
            "temperature": 1.5,
            "top_k": 0,
            "top_p": 1.0,
            "description": "High entropy regime with no filtering; demonstrates maximum diversity."
        }
    ]

    print("\nRunning systematic generation benchmarks across 7 sampling regimes...\n")

    results_table = []

    for cfg in experiment_configs:
        t0 = time.perf_counter()
        codes = generate_hierarchical_codes(
            generator=generator,
            tokenizer=tokenizer,
            batch_size=1,
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
            device=device
        )
        elapsed = time.perf_counter() - t0

        decoded_text = decode_hierarchical_codes(tokenizer, codes)[0]
        entropy = compute_text_entropy(decoded_text)
        uniq_words = count_unique_words(decoded_text)

        results_table.append({
            "name": cfg["name"],
            "temp": cfg["temperature"],
            "top_k": cfg["top_k"],
            "top_p": cfg["top_p"],
            "elapsed": elapsed,
            "entropy": entropy,
            "uniq_words": uniq_words,
            "text": decoded_text,
            "description": cfg["description"]
        })

    # 4. Display Comparison Table
    print("-" * 76)
    print(f"{'Sampling Strategy':<30} | {'Temp':<4} | {'Top-k':<5} | {'Top-p':<5} | {'Entropy':<7} | {'Words':<5} | {'Time (s)':<8}")
    print("-" * 76)
    for r in results_table:
        print(f"{r['name']:<30} | {r['temp']:<4.1f} | {r['top_k']:<5} | {r['top_p']:<5.2f} | {r['entropy']:<7.2f} | {r['uniq_words']:<5} | {r['elapsed']:<8.3f}")
    print("-" * 76)

    # 5. Display Detailed Text Outputs
    print("\n" + "=" * 76)
    print("DECODED TEXT OUTPUT SAMPLES BY SAMPLING STRATEGY")
    print("=" * 76)

    for i, r in enumerate(results_table, 1):
        print(f"\n[{r['name']}]")
        print(f"  Configuration: T={r['temp']}, top_k={r['top_k']}, top_p={r['top_p']}")
        print(f"  Behavior: {r['description']}")
        print(f"  Metrics: Shannon Entropy = {r['entropy']:.2f} bits | Unique Words = {r['uniq_words']} | Latency = {r['elapsed']:.3f}s")
        print("  Decoded Text (first 200 characters):")
        # Clean preview showing printable characters
        clean_preview = r['text'][:200].replace("\n", " ").replace("\r", " ")
        print(f"    \"{clean_preview}\"")

    print("\n" + "=" * 76)
    print("SAMPLING EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 76)


if __name__ == "__main__":
    run_sampling_experiments()
