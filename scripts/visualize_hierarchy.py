"""
NCM-Lite Coarse-to-Fine Hierarchy Visualization Script.

Demonstrates how Next-Concept Modeling progressively resolves text from coarse
semantic layout (Level 1, 64 tokens) -> intermediate structure (Level 2, 128 tokens)
-> fine byte details (Level 3, 256 tokens) across exactly K=3 parallel generator passes.

Saves an interactive HTML report to runs/hierarchy_visualization.html.
"""

import os
import sys
import time
import math
import collections
from typing import Dict, List
import torch

from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.training.checkpoint import CheckpointManager
from ncm.inference.generate import generate_hierarchical_progression


def compute_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = collections.Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def generate_html_report(
    levels: List[int],
    level_texts: List[str],
    level_latents: List[torch.Tensor],
    output_html_path: str,
    sampling_info: Dict[str, str]
):
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    cards_html = ""
    for k, (ell, text, latent) in enumerate(zip(levels, level_texts, level_latents), 1):
        latent_norm = latent.norm().item()
        entropy = compute_entropy(text)
        preview_escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        cards_html += f"""
        <div class="level-card level-{k}">
            <div class="card-header">
                <h3>Level {k} (Scale: {ell} tokens)</h3>
                <span class="badge">Pass {k} of 3</span>
            </div>
            <div class="stats-row">
                <div class="stat"><span class="label">Tokens Predicted:</span> <strong>{ell}</strong></div>
                <div class="stat"><span class="label">Latent Norm ||z_hat||:</span> <strong>{latent_norm:.2f}</strong></div>
                <div class="stat"><span class="label">Shannon Entropy:</span> <strong>{entropy:.2f} bits</strong></div>
                <div class="stat"><span class="label">Sequence Length:</span> <strong>256 bytes</strong></div>
            </div>
            <div class="text-box">
                <pre>{preview_escaped}</pre>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NCM-Lite: Coarse-to-Fine Progression</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            color: #38bdf8;
            font-size: 28px;
            margin-bottom: 8px;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 16px;
            margin-bottom: 24px;
        }}
        .meta-box {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .level-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .level-1 {{ border-left: 6px solid #fbbf24; }}
        .level-2 {{ border-left: 6px solid #38bdf8; }}
        .level-3 {{ border-left: 6px solid #4ade80; }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }}
        .card-header h3 {{
            margin: 0;
            font-size: 20px;
        }}
        .badge {{
            background: #334155;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .stats-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 14px;
            font-size: 13px;
            color: #cbd5e1;
            flex-wrap: wrap;
        }}
        .text-box {{
            background: #090d16;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 14px;
            overflow-x: auto;
        }}
        pre {{
            margin: 0;
            font-family: "Courier New", Courier, monospace;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-all;
            color: #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NCM-Lite: Hierarchical Coarse-to-Fine Progression</h1>
        <div class="subtitle">Next-Concept Modeling Resolution Dynamics Across K=3 Parallel Passes</div>
        
        <div class="meta-box">
            <strong>Architecture:</strong> Multiscale RVQ (0.16M) + Next-Concept Transformer (11.14M)<br>
            <strong>Sampling Configuration:</strong> {sampling_info['config']}<br>
            <strong>Total Inference Latency:</strong> {sampling_info['latency']}<br>
            <strong>Paper Citation:</strong> Kumbong et al., arXiv:2610.14162 (Stanford / Together AI)
        </div>

        {cards_html}
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def run_visualization():
    print("=" * 76)
    print("NCM-LITE: COARSE-TO-FINE PROGRESSION VISUALIZATION")
    print("Tracking Progressive Sharpness Across K=3 Parallel Generator Passes")
    print("=" * 76)

    device = "cpu"
    torch.manual_seed(42)

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

    # 3. Generate progression
    print("\nExecuting Algorithm 2 Coarse-to-Fine Generation (T=0.8, top_k=40, top_p=0.92)...")
    t0 = time.perf_counter()
    codes, latents, texts = generate_hierarchical_progression(
        generator=generator,
        tokenizer=tokenizer,
        batch_size=1,
        temperature=0.8,
        top_k=40,
        top_p=0.92,
        device=device
    )
    total_time = time.perf_counter() - t0
    print(f"  • Generation completed in: {total_time:.3f}s on CPU")

    level_texts = [texts[k][0] for k in range(len(levels))]
    level_latents = [latents[k][0] for k in range(len(levels))]

    # 4. Display Terminal Side-by-Side Progression
    print("\n" + "-" * 76)
    print("STEP-BY-STEP COARSE-TO-FINE PROGRESSION ACROSS SCALES")
    print("-" * 76)

    stage_names = [
        "LEVEL 1: Coarse Semantic Sketch (64 tokens)",
        "LEVEL 2: Intermediate Syntactic Refinement (128 tokens)",
        "LEVEL 3: Fine Byte-Level Rendering (256 tokens)"
    ]

    for k in range(len(levels)):
        ell = levels[k]
        text_k = level_texts[k]
        latent_norm = level_latents[k].norm().item()
        entropy = compute_entropy(text_k)
        preview = text_k[:180].replace("\n", " ")

        print(f"\n[Pass {k+1}/3] {stage_names[k]}")
        print(f"  Scale: ell_{k+1}={ell} | Cumulative Latent Norm: {latent_norm:.2f} | Entropy: {entropy:.2f} bits")
        print(f"  Rendered Snapshot: \"{preview}...\"")

    # 5. Export HTML Report
    html_out = os.path.join(base_dir, "runs", "hierarchy_visualization.html")
    generate_html_report(
        levels=levels,
        level_texts=level_texts,
        level_latents=level_latents,
        output_html_path=html_out,
        sampling_info={
            "config": "T=0.8, top_k=40, top_p=0.92 (Combined Nucleus + Top-k)",
            "latency": f"{total_time:.3f} seconds ({seq_len / total_time:.1f} bytes/s)"
        }
    )
    print("\n" + "-" * 76)
    print(f"[✓] Interactive HTML visualization generated at:")
    print(f"    {html_out}")
    print("=" * 76)


if __name__ == "__main__":
    run_visualization()
