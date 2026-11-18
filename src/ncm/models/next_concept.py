"""
Hierarchical Next-Concept Model (Stage 2 Generator) for NCM-Lite.

Implements the coarse-to-fine Next-Concept Transformer predictor:
- Block-causal attention pattern (bidirectional within level, causal across levels).
- Cumulative continuous latent conditioning (Down_k(z_hat^(k-1)) projected to n_embd).
- Level and positional embeddings.
- Single-pass training over all K levels (Algorithm 1).
- Parallel within-level sampling (Algorithm 2).

Grounding: Section 3.4 of arXiv:2610.14162 and HazyResearch/ncm/models/next_concept.py.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm.models.embeddings import HierarchicalCodeEmbedding


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with support for arbitrary 2D/4D attention masks.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, SeqLen, embed_dim]
            attn_mask: [1, 1, SeqLen, SeqLen] with 0.0 for allow and -inf for mask.
        Returns:
            out: [B, SeqLen, embed_dim]
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # [B, T, 3 * C]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        y = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class NextConceptTransformer(nn.Module):
    """
    Hierarchical Next-Concept Predictor.
    
    Generates text coarse-to-fine by predicting discrete code targets across K levels.
    All codes within a given level are predicted in parallel.
    """

    def __init__(
        self,
        num_levels: int = 3,
        levels: Optional[List[int]] = None,
        codebook_size: int = 256,
        latent_dim: int = 64,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        intermediate_dim: int = 1536,
        dropout: float = 0.1,
        loss_alpha: float = 0.0
    ):
        super().__init__()
        # Allow passing config dictionary as single positional or keyword argument
        if isinstance(num_levels, dict):
            cfg = num_levels
            num_levels = cfg.get("num_levels", 3)
            levels = cfg.get("levels", [64, 128, 256])
            codebook_size = cfg.get("codebook_size", 256)
            latent_dim = cfg.get("latent_dim", cfg.get("code_dim", 64))
            embed_dim = cfg.get("embed_dim", cfg.get("hidden_dim", 384))
            num_layers = cfg.get("num_layers", 6)
            num_heads = cfg.get("num_heads", 6)
            intermediate_dim = cfg.get("intermediate_dim", 1536)
            dropout = cfg.get("dropout", 0.1)
            loss_alpha = cfg.get("loss_alpha", 0.0)

        self.num_levels = num_levels
        self.levels = levels or [64, 128, 256]
        self.total_seq_len = sum(self.levels)
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.loss_alpha = loss_alpha

        # Input Embedding: Continuous projection + Level + Positional
        self.embedding = HierarchicalCodeEmbedding(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_levels=num_levels,
            max_seq_len=self.total_seq_len
        )

        # Transformer Backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

        # Per-level prediction heads
        self.lm_heads = nn.ModuleList([
            nn.Linear(embed_dim, codebook_size) for _ in range(num_levels)
        ])

        # Precompute block-causal attention mask
        self.register_buffer(
            "block_causal_mask",
            self._build_block_causal_mask(self.levels),
            persistent=False
        )

    def _build_block_causal_mask(self, levels: List[int]) -> torch.Tensor:
        """
        Builds the block-causal attention mask:
        - Positions within the same level attend bidirectionally (0.0).
        - Positions at level k can attend to positions at earlier levels < k (0.0).
        - Positions at level k cannot attend to positions at future levels > k (-inf).
        """
        L = sum(levels)
        mask = torch.full((L, L), float("-inf"))

        # Map each position index to its level index
        level_map = []
        for k, l_k in enumerate(levels):
            level_map.extend([k] * l_k)
        level_map = torch.tensor(level_map, dtype=torch.long)

        # Allow attention if level_j <= level_i
        for i in range(L):
            for j in range(L):
                if level_map[j] <= level_map[i]:
                    mask[i, j] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

    def forward(
        self,
        level_inputs: List[torch.Tensor],
        target_codes: Optional[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Single forward pass predicting all K levels in parallel (Algorithm 1).
        Args:
            level_inputs: List of K tensors, where level_inputs[k] has shape [B, ell_k, latent_dim].
                          Represents Down_k(z_hat^(k-1)) for each level.
            target_codes: Optional list of K ground truth discrete codes [B, ell_k].
        Returns:
            logits_list: List of K tensors [Z_1, ..., Z_K], where Z_k is [B, ell_k, codebook_size].
            losses: Dictionary containing per-level and total cross-entropy loss.
        """
        B = level_inputs[0].shape[0]
        device = level_inputs[0].device

        # Embed each level and concatenate across sequence dimension
        embedded_levels: List[torch.Tensor] = []
        offset = 0
        for k, u_k in enumerate(level_inputs):
            h_k = self.embedding(u_k, level_idx=k, position_offset=offset)
            embedded_levels.append(h_k)
            offset += self.levels[k]

        x = torch.cat(embedded_levels, dim=1)  # [B, L_total, embed_dim]

        # Pass through transformer with block-causal mask
        mask = self.block_causal_mask.to(device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.ln_f(x)

        # Slice hidden representations and apply corresponding level prediction heads
        logits_list: List[torch.Tensor] = []
        level_losses: List[torch.Tensor] = []
        offset = 0

        for k in range(self.num_levels):
            ell_k = self.levels[k]
            h_slice = x[:, offset:offset + ell_k, :]
            logits_k = self.lm_heads[k](h_slice)  # [B, ell_k, codebook_size]
            logits_list.append(logits_k)

            if target_codes is not None:
                targets_k = target_codes[k]
                loss_k = F.cross_entropy(logits_k.view(B * ell_k, -1), targets_k.view(B * ell_k))
                level_losses.append(loss_k)

            offset += ell_k

        losses: Dict[str, torch.Tensor] = {}
        if target_codes is not None:
            # Paper weighting: w_k = ell_k^(-alpha) / sum(ell_j^(1-alpha))
            weights = [ell ** (-self.loss_alpha) for ell in self.levels]
            norm_factor = sum(ell ** (1.0 - self.loss_alpha) for ell in self.levels)
            norm_weights = [w / norm_factor for w in weights]

            total_loss = sum(w * l for w, l in zip(norm_weights, level_losses))
            losses["total_ncm_loss"] = total_loss
            for k, l_k in enumerate(level_losses):
                losses[f"loss_level_{k + 1}"] = l_k

        return logits_list, losses

    @torch.no_grad()
    def predict_single_level(
        self,
        u_k: torch.Tensor,
        level_idx: int,
        all_prev_u: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass predicting a single level k in parallel,
        conditioned on previous levels.
        """
        device = u_k.device
        if all_prev_u is None or len(all_prev_u) == 0:
            full_inputs = [u_k]
        else:
            full_inputs = list(all_prev_u) + [u_k]

        embedded_levels: List[torch.Tensor] = []
        offset = 0
        for k, u in enumerate(full_inputs):
            h_k = self.embedding(u, level_idx=k, position_offset=offset)
            embedded_levels.append(h_k)
            offset += u.shape[1]

        x = torch.cat(embedded_levels, dim=1)

        # Build partial mask for current lengths
        curr_len = x.shape[1]
        mask = self.block_causal_mask[:, :, :curr_len, :curr_len].to(device)

        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.ln_f(x)

        # Slice the last level (target level)
        ell_k = u_k.shape[1]
        h_target = x[:, -ell_k:, :]
        logits = self.lm_heads[level_idx](h_target)
        return logits
