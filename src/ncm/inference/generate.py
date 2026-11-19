"""
Hierarchical code generation and decoding for NCM-Lite.
Follows Algorithm 2 from Kumbong et al., 2024 ("Next-Concept Modeling").
"""

from typing import List, Optional, Tuple, Union
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
def decode_latent_to_text(
    tokenizer: TokenizerModel,
    z_hat: torch.Tensor
) -> List[str]:
    """
    Decodes a continuous cumulative latent tensor z_hat of shape [B, L, D]
    directly into UTF-8 text strings via the multiscale decoder.
    
    Args:
        tokenizer: Frozen TokenizerModel containing the decoder.
        z_hat: [B, L, embed_dim] continuous latent tensor.
    Returns:
        decoded_strings: List of length B of decoded text strings.
    """
    tokenizer.eval()
    token_logits = tokenizer.decoder(z_hat)  # [B, L, vocab_size=256]
    pred_bytes = torch.argmax(token_logits, dim=-1).cpu().numpy()  # [B, L]

    decoded_strings: List[str] = []
    for row in pred_bytes:
        raw_bytes = bytes(row.tolist())
        text = raw_bytes.decode("utf-8", errors="replace")
        decoded_strings.append(text)
    return decoded_strings


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
def generate_hierarchical_progression(
    generator: NextConceptTransformer,
    tokenizer: TokenizerModel,
    batch_size: int = 1,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    """
    Generates hierarchical discrete concept codes in K sequential passes (Algorithm 2)
    and captures the progressive cumulative continuous reconstruction z_hat^(k) and
    decoded text representation at each level k in {1, ..., K}.
    
    Demonstrates the coarse-to-fine sharpening dynamics of Next-Concept Modeling.
    
    Returns:
        generated_codes: List of K tensors [r_1, ..., r_K].
        intermediate_latents: List of K tensors [z_hat^(1), ..., z_hat^(K)], each [B, L, D].
        intermediate_texts: List of K lists of strings, where intermediate_texts[k] has length B.
    """
    generator.eval()
    tokenizer.eval()

    if device is None:
        device = next(generator.parameters()).device

    L = tokenizer.vq.target_len
    num_levels = generator.num_levels
    embed_dim = tokenizer.vq.embed_dim

    z_hat = torch.zeros(batch_size, L, embed_dim, device=device)

    generated_codes: List[torch.Tensor] = []
    intermediate_latents: List[torch.Tensor] = []
    intermediate_texts: List[List[str]] = []
    prev_u_inputs: List[torch.Tensor] = []

    for k in range(num_levels):
        ell_k = generator.levels[k]

        # 1. Downsample current cumulative reconstruction to resolution ell_k
        u_k = tokenizer.vq._down_to_level(z_hat, ell_k)

        # 2. Predict level k logits conditioned on earlier levels
        logits_k = generator.predict_single_level(
            u_k=u_k,
            level_idx=k,
            all_prev_u=prev_u_inputs
        )

        # 3. Sample discrete codes
        r_k = _sample_from_logits(
            logits_k,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        generated_codes.append(r_k)

        # 4. Accumulate into continuous representation
        q_k = tokenizer.vq.embeddings[k](r_k)
        r_full = tokenizer.vq._up_to_full(q_k, L)
        z_hat = z_hat + r_full

        # Capture cumulative continuous latent and decode text at this scale
        cumulative_snapshot = z_hat.clone()
        intermediate_latents.append(cumulative_snapshot)
        text_snapshot = decode_latent_to_text(tokenizer, cumulative_snapshot)
        intermediate_texts.append(text_snapshot)

        prev_u_inputs.append(u_k)

    return generated_codes, intermediate_latents, intermediate_texts


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



@torch.no_grad()
def generate_prompt_continuation(
    generator: NextConceptTransformer,
    tokenizer: TokenizerModel,
    prompt: str,
    batch_size: int = 1,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.92,
    device: Optional[Union[str, torch.device]] = None
) -> List[str]:
    """
    Generates text continuations conditioned on a user-provided prompt prefix.
    
    Encodes the prompt prefix through the frozen tokenizer into multi-scale
    prompt codes, fixes the prompt prefix across all K levels, and generates
    the remaining tokens in K parallel passes.
    
    Args:
        generator: NextConceptTransformer generator.
        tokenizer: Frozen TokenizerModel.
        prompt: Prefix text string to continue.
        batch_size: Number of parallel continuations.
        temperature: Sampling temperature.
        top_k: Top-k filtering cutoff.
        top_p: Top-p nucleus filtering cutoff.
        device: Target execution device.
    Returns:
        continued_texts: List of length B of completed text strings.
    """
    generator.eval()
    tokenizer.eval()

    if device is None:
        device = next(generator.parameters()).device

    L = tokenizer.vq.target_len
    num_levels = generator.num_levels
    embed_dim = tokenizer.vq.embed_dim

    # 1. Prepare prompt byte tensor
    prompt_bytes = list(prompt.encode("utf-8", errors="replace"))
    P = len(prompt_bytes)
    if P >= L:
        # If prompt is as long as L, truncate so there is at least room to continue
        P = L - 16
        prompt_bytes = prompt_bytes[:P]

    padded_prompt = prompt_bytes + [0] * (L - P)
    x_prompt = torch.tensor([padded_prompt] * batch_size, dtype=torch.long, device=device)

    # 2. Extract multi-scale prompt codes using frozen tokenizer
    prompt_codes = tokenizer.encode(x_prompt)  # List of K tensors [B, ell_k]

    # 3. Progressive K-pass generation with prefix clamping
    z_hat = torch.zeros(batch_size, L, embed_dim, device=device)
    generated_codes: List[torch.Tensor] = []
    prev_u_inputs: List[torch.Tensor] = []

    for k in range(num_levels):
        ell_k = generator.levels[k]

        # Calculate prefix length at resolution ell_k
        P_k = max(1, min(ell_k - 1, round(P * (ell_k / L))))

        u_k = tokenizer.vq._down_to_level(z_hat, ell_k)

        logits_k = generator.predict_single_level(
            u_k=u_k,
            level_idx=k,
            all_prev_u=prev_u_inputs
        )

        r_k_pred = _sample_from_logits(
            logits_k,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Clamp the prefix positions to the ground-truth prompt codes
        r_k = torch.cat([prompt_codes[k][:, :P_k], r_k_pred[:, P_k:]], dim=1)
        generated_codes.append(r_k)

        # Update cumulative continuous representation
        q_k = tokenizer.vq.embeddings[k](r_k)
        r_full = tokenizer.vq._up_to_full(q_k, L)
        z_hat = z_hat + r_full

        prev_u_inputs.append(u_k)

    # 4. Decode full generated codes and guarantee exact prompt prefix
    token_logits = tokenizer.decode(generated_codes)
    pred_bytes = torch.argmax(token_logits, dim=-1).cpu().numpy()

    completed_texts: List[str] = []
    for row in pred_bytes:
        raw_bytes = bytes(row.tolist())
        continuation_bytes = raw_bytes[P:]
        continuation_text = continuation_bytes.decode("utf-8", errors="replace")
        completed_texts.append(prompt + continuation_text)

    return completed_texts


@torch.no_grad()
def generate_chained_sequence(
    generator: NextConceptTransformer,
    tokenizer: TokenizerModel,
    prompt: str = "",
    target_bytes: int = 512,
    overlap_bytes: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.92,
    device: Optional[Union[str, torch.device]] = None
) -> str:
    """
    Generates multi-chunk, long-form text sequences using sliding-window
    autoregressive chaining across sequential 256-byte NCM generation blocks.

    For each 256-byte window, generation runs in exactly K=3 parallel passes.
    The suffix of each generated chunk serves as the prompt prefix for the
    subsequent window, enabling seamless multi-paragraph story generation.

    Args:
        generator: NextConceptTransformer generator model.
        tokenizer: Frozen TokenizerModel.
        prompt: Initial opening prompt text.
        target_bytes: Desired total length in bytes (e.g. 512, 768, 1024).
        overlap_bytes: Length of context window passed to next chunk (default: 64).
        temperature: Sampling temperature.
        top_k: Top-k filtering threshold.
        top_p: Top-p nucleus filtering threshold.
        device: Target execution device.

    Returns:
        full_story: The seamlessly concatenated multi-chunk generated text string.
    """
    generator.eval()
    tokenizer.eval()

    L = tokenizer.vq.target_len  # 256
    overlap = min(overlap_bytes, L // 2)

    # 1. First chunk generation
    initial_prompt = prompt if prompt else "Once upon a time, "
    first_chunk = generate_prompt_continuation(
        generator=generator,
        tokenizer=tokenizer,
        prompt=initial_prompt,
        batch_size=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device
    )[0]

    accumulated_text = first_chunk
    curr_bytes = len(accumulated_text.encode("utf-8", errors="replace"))

    # 2. Sequential sliding-window chaining
    while curr_bytes < target_bytes:
        # Extract the trailing context to prime the next chunk
        raw_accum = accumulated_text.encode("utf-8", errors="replace")
        prefix_bytes = raw_accum[-overlap:]
        prefix_text = prefix_bytes.decode("utf-8", errors="ignore")

        # Generate continuation from prefix
        next_chunk = generate_prompt_continuation(
            generator=generator,
            tokenizer=tokenizer,
            prompt=prefix_text,
            batch_size=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device
        )[0]

        # Append only the newly generated suffix beyond the overlap prefix
        new_text = next_chunk[len(prefix_text):]
        if not new_text:
            break

        accumulated_text += new_text
        curr_bytes = len(accumulated_text.encode("utf-8", errors="replace"))

    # Trim to exact target_bytes boundary if specified
    encoded = accumulated_text.encode("utf-8", errors="replace")[:target_bytes]
    return encoded.decode("utf-8", errors="ignore")

