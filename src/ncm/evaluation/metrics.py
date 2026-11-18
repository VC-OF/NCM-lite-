"""
Evaluation metrics and diagnostics for NCM-Lite.
Provides reconstruction quality, codebook utilization, and generation benchmarks.
"""

import time
import tracemalloc
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ncm.models.next_concept import NextConceptTransformer
from ncm.tokenizer.model import TokenizerModel
from ncm.inference.generate import generate_hierarchical_codes, decode_hierarchical_codes


@torch.no_grad()
def evaluate_reconstruction(
    tokenizer: TokenizerModel,
    dataloader: DataLoader,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, float]:
    """
    Evaluates Stage 1 Tokenizer reconstruction fidelity and codebook dynamics.
    
    Args:
        tokenizer: TokenizerModel to evaluate.
        dataloader: DataLoader delivering batches of byte index tensors [B, L].
        device: Evaluation device.
    Returns:
        metrics: Dictionary containing:
            - 'reconstruction_accuracy': Fraction of bytes reconstructed exactly.
            - 'reconstruction_loss': Average cross-entropy reconstruction loss.
            - 'codebook_utilization_level_{k}': Fraction of active codes used at level k.
            - 'codebook_perplexity_level_{k}': Exponent of entropy of code usage at level k.
            - 'mean_codebook_utilization': Mean codebook utilization across all levels.
    """
    tokenizer.eval()
    if device is None:
        device = next(tokenizer.parameters()).device

    total_correct = 0
    total_tokens = 0
    total_recon_loss = 0.0
    num_batches = 0

    num_levels = tokenizer.vq.num_levels
    codebook_size = tokenizer.vq.codebook_size

    # Code occurrence counters for each level
    level_counts = [torch.zeros(codebook_size, dtype=torch.long) for _ in range(num_levels)]

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)
        B, L = batch.shape

        logits, code_indices, losses = tokenizer(batch)

        preds = torch.argmax(logits, dim=-1)
        correct = (preds == batch).sum().item()
        total_correct += correct
        total_tokens += (B * L)

        total_recon_loss += losses["recon_loss"].item()
        num_batches += 1

        for k in range(num_levels):
            codes_k = code_indices[k].cpu().view(-1)
            bincount = torch.bincount(codes_k, minlength=codebook_size)
            level_counts[k] += bincount

    avg_acc = total_correct / max(1, total_tokens)
    avg_loss = total_recon_loss / max(1, num_batches)

    metrics: Dict[str, float] = {
        "reconstruction_accuracy": float(avg_acc),
        "reconstruction_loss": float(avg_loss)
    }

    utilizations: List[float] = []
    for k in range(num_levels):
        counts = level_counts[k].float()
        active = (counts > 0).sum().item()
        util = active / codebook_size
        utilizations.append(util)
        metrics[f"codebook_utilization_level_{k + 1}"] = float(util)

        # Perplexity
        total_codes = counts.sum().item()
        if total_codes > 0:
            probs = counts / total_codes
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log(probs)).item()
            import math
            perplexity = math.exp(entropy)
        else:
            perplexity = 0.0
        metrics[f"codebook_perplexity_level_{k + 1}"] = float(perplexity)

    metrics["mean_codebook_utilization"] = float(sum(utilizations) / max(1, len(utilizations)))
    return metrics


@torch.no_grad()
def evaluate_generation(
    generator: NextConceptTransformer,
    tokenizer: TokenizerModel,
    num_samples: int = 5,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Evaluates Stage 2 generation throughput, peak memory consumption, and samples.
    
    Args:
        generator: Trained NextConceptTransformer model.
        tokenizer: Frozen TokenizerModel.
        num_samples: Number of sequences to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering threshold.
        top_p: Top-p nucleus filtering threshold.
        device: Target execution device.
    Returns:
        results: Dictionary containing latency metrics, peak memory, and text samples.
    """
    generator.eval()
    tokenizer.eval()

    if device is None:
        device = next(generator.parameters()).device

    L = tokenizer.vq.target_len

    # Start memory and latency tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    # Generate codes
    generated_codes = generate_hierarchical_codes(
        generator=generator,
        tokenizer=tokenizer,
        batch_size=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device
    )

    # Decode into text strings
    samples = decode_hierarchical_codes(tokenizer, generated_codes)

    elapsed = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_bytes_generated = num_samples * L
    throughput = total_bytes_generated / max(1e-6, elapsed)
    latency_per_sample = elapsed / max(1, num_samples)

    return {
        "num_samples": num_samples,
        "sequence_length": L,
        "num_passes": generator.num_levels,
        "total_time_seconds": float(elapsed),
        "latency_per_sample_seconds": float(latency_per_sample),
        "bytes_per_second": float(throughput),
        "peak_memory_mb": float(peak_bytes / (1024 * 1024)),
        "samples": samples
    }
