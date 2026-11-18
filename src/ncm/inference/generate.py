"""
Hierarchical code generation and decoding for NCM-Lite.
Follows Algorithm 2 from Kumbong et al., 2024 ("Next-Concept Modeling").
"""

from typing import List, Optional, Union
import torch
import torch.nn.functional as F

from ncm.models.next_concept import NextConceptTransformer
from ncm.tokenizer.model import TokenizerModel


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0
) -> torch.Tensor:
    """
    Samples discrete token indices from unnormalized logits.
    
    Args:
        logits: [B, L, V] unnormalized logit tensor.
        temperature: Temperature for sampling (0.0 or <= 1e-5 for greedy argmax).
        top_k: If > 0, restrict sampling to top k highest-probability tokens.
    Returns:
        indices: [B, L] discrete indices as torch.int64.
    """
    if temperature <= 1e-5 or top_k == 1:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k > 1 and top_k < logits.shape[-1]:
        val, _ = torch.topk(logits, top_k, dim=-1)
        min_val = val[..., -1, None]
        logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

    probs = F.softmax(logits, dim=-1)
    B, L, V = probs.shape
    # Flatten across B and L for multinomial sampling
    probs_flat = probs.view(-1, V)
    samples = torch.multinomial(probs_flat, num_samples=1)
    return samples.view(B, L)


@torch.no_grad()
def generate_hierarchical_codes(
    generator: NextConceptTransformer,
    tokenizer: TokenizerModel,
    batch_size: int = 1,
    temperature: float = 1.0,
    top_k: int = 0,
    device: Optional[Union[str, torch.device]] = None
) -> List[torch.Tensor]:
    """
    Generates hierarchical discrete concept codes in exactly K sequential passes
    (Algorithm 2 in paper). Within each pass, all ell_k codes are predicted in parallel.
    
    Args:
        generator: Trained NextConceptTransformer generator model.
        tokenizer: Frozen TokenizerModel containing multiscale RVQ codebooks.
        batch_size: Number of parallel sequences to generate.
        temperature: Sampling temperature (1.0 = standard, 0.0 = greedy).
        top_k: Top-k filtering threshold.
        device: Target execution device.
    Returns:
        generated_codes: List of K tensors [r_1, ..., r_K], where r_k has shape [B, ell_k].
    """
    generator.eval()
    tokenizer.eval()

    if device is None:
        device = next(generator.parameters()).device

    L = tokenizer.vq.target_len
    num_levels = generator.num_levels
    embed_dim = tokenizer.vq.embed_dim

    # Cumulative continuous reconstruction: initially all zeros
    z_hat = torch.zeros(batch_size, L, embed_dim, device=device)

    generated_codes: List[torch.Tensor] = []
    prev_u_inputs: List[torch.Tensor] = []

    # Exactly K sequential passes (one per scale)
    for k in range(num_levels):
        ell_k = generator.levels[k]

        # 1. Downsample current cumulative reconstruction to resolution ell_k
        u_k = tokenizer.vq._down_to_level(z_hat, ell_k)

        # 2. Predict level k logits conditioned on earlier levels
        logits_k = generator.predict_single_level(
            u_k=u_k,
            level_idx=k,
            all_prev_u=prev_u_inputs
        )  # [B, ell_k, codebook_size]

        # 3. Sample discrete codes for all ell_k positions simultaneously
        r_k = _sample_from_logits(logits_k, temperature=temperature, top_k=top_k)  # [B, ell_k]
        generated_codes.append(r_k)

        # 4. Update cumulative reconstruction z_hat with new code vectors
        q_k = tokenizer.vq.embeddings[k](r_k)  # [B, ell_k, embed_dim]
        r_full = tokenizer.vq._up_to_full(q_k, L)  # [B, L, embed_dim]
        z_hat = z_hat + r_full

        # Save u_k for conditioning subsequent levels
        prev_u_inputs.append(u_k)

    return generated_codes


@torch.no_grad()
def decode_hierarchical_codes(
    tokenizer: TokenizerModel,
    generated_codes: List[torch.Tensor]
) -> List[str]:
    """
    Decodes hierarchical discrete code indices into UTF-8 text strings
    via the frozen tokenizer decoder.
    
    Args:
        tokenizer: Frozen TokenizerModel containing RVQ codebooks and decoder.
        generated_codes: List of K tensors [r_1, ..., r_K], where r_k has shape [B, ell_k].
    Returns:
        decoded_strings: List of length B containing decoded UTF-8 strings.
    """
    tokenizer.eval()
    device = generated_codes[0].device

    # Reconstruct token logits from code indices
    token_logits = tokenizer.decode(generated_codes)  # [B, L, vocab_size=256]
    pred_bytes = torch.argmax(token_logits, dim=-1).cpu().numpy()  # [B, L]

    decoded_strings: List[str] = []
    for row in pred_bytes:
        raw_bytes = bytes(row.tolist())
        # Decode as utf-8, replacing invalid sequences cleanly
        text = raw_bytes.decode("utf-8", errors="replace")
        decoded_strings.append(text)

    return decoded_strings
