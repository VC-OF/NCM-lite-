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
    top_k: int = 0,
    top_p: float = 1.0
) -> torch.Tensor:
    """
    Samples discrete token indices from unnormalized logits.
    
    Supports:
      - Greedy argmax (temperature <= 1e-5 or top_k == 1)
      - Temperature scaling
      - Top-k filtering (truncating to top k logits)
      - Top-p / nucleus filtering (truncating to smallest set with cumulative prob >= p)
      - Combined top-k and top-p sampling
    
    Args:
        logits: [B, L, V] unnormalized logit tensor.
        temperature: Temperature for sampling (0.0 or <= 1e-5 for greedy argmax).
        top_k: If > 0, restrict sampling to top k highest-probability tokens.
        top_p: If < 1.0, restrict sampling to cumulative probability mass <= top_p.
    Returns:
        indices: [B, L] discrete indices as torch.int64.
    """
    if temperature <= 1e-5 or top_k == 1:
        return torch.argmax(logits, dim=-1)

    # 1. Apply temperature scaling
    scaled_logits = logits / max(temperature, 1e-5)

    # 2. Top-k filtering
    if 0 < top_k < scaled_logits.shape[-1]:
        top_k_val, _ = torch.topk(scaled_logits, top_k, dim=-1)
        min_val = top_k_val[..., -1, None]
        scaled_logits = torch.where(
            scaled_logits < min_val,
            torch.full_like(scaled_logits, float("-inf")),
            scaled_logits
        )

    # 3. Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right so that the first token exceeding threshold is preserved
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter mask back to original logit positions
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

    # 4. Softmax probability distribution
    probs = F.softmax(scaled_logits, dim=-1)

    # Safeguard against all -inf
    if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
        return torch.argmax(logits, dim=-1)

    B, L, V = probs.shape
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
    top_p: float = 1.0,
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
        top_k: Top-k filtering threshold (0 = disabled).
        top_p: Top-p nucleus filtering threshold (1.0 = disabled).
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
        r_k = _sample_from_logits(
            logits_k,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )  # [B, ell_k]
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
