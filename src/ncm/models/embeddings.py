"""
Hierarchical Concept and Level Embeddings for NCM-Lite.

Implements scale-aware input projections, level embeddings (wle),
and positional embeddings (wpe) matching HazyResearch/ncm/models/next_concept.py.
"""

from typing import Optional
import torch
import torch.nn as nn


class HierarchicalCodeEmbedding(nn.Module):
    """
    Embeds cumulative continuous latents with level indicators and position signals.
    
    In NCM (Algorithm 1 and models/next_concept.py):
    At level k, input is the resized cumulative reconstruction Down_k(z_hat^(k-1)),
    projected through a linear layer wte, plus level embedding wle(k) and
    positional embedding wpe(pos).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        embed_dim: int = 384,
        num_levels: int = 3,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.max_seq_len = max_seq_len

        # Linear projection from continuous latent space (D_tok) to generator space (D_gen)
        self.wte = nn.Linear(latent_dim, embed_dim)

        # Word-Level-Embedding: identifies the hierarchical scale index (0 to K-1)
        self.wle = nn.Embedding(num_levels, embed_dim)

        # Positional embedding for sequence positions
        self.wpe = nn.Embedding(max_seq_len, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        continuous_input: torch.Tensor,
        level_idx: int,
        position_offset: int = 0
    ) -> torch.Tensor:
        """
        Embeds a single level's continuous input.
        Args:
            continuous_input: [B, ell_k, latent_dim] continuous latent vectors.
            level_idx: integer level index in [0, num_levels - 1].
            position_offset: starting position offset for wpe.
        Returns:
            h: [B, ell_k, embed_dim] embedded transformer representations.
        """
        B, ell_k, _ = continuous_input.shape
        device = continuous_input.device

        # Linear projection
        proj = self.wte(continuous_input)  # [B, ell_k, embed_dim]

        # Level embedding
        lvl_tensor = torch.tensor(level_idx, device=device, dtype=torch.long)
        lvl_emb = self.wle(lvl_tensor).unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]

        # Positional embedding
        positions = torch.arange(position_offset, position_offset + ell_k, device=device).unsqueeze(0)
        pos_emb = self.wpe(positions)  # [1, ell_k, embed_dim]

        h = proj + lvl_emb + pos_emb
        return self.layer_norm(h)
