"""
Multiscale Residual Vector Quantization (RVQ) Engine for NCM-Lite.

Implements the multiscale residual quantization algorithm described in
Section 3.3 (Equations 7-9) of Kumbong et al., arXiv:2610.14162, and verified
against HazyResearch/ncm/models/quant.py.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleResidualVectorQuantizer(nn.Module):
    """
    Hierarchical multiscale residual vector quantizer.
    
    Residuals across K scales are sequentially quantized and accumulated:
      u_k = Down_k(z^(k-1)) in R^[B, ell_k, D]
      r_k^i = argmin_j ||u_k^i - e_{k, j}||_2^2
      z_hat^(k) = z_hat^(k-1) + R_k(e_k(r_k))
      z^(k) = z - z_hat^(k)
    """

    def __init__(
        self,
        num_levels: int = 3,
        levels: Optional[List[int]] = None,
        codebook_size: int = 256,
        embed_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.95,
        nested_drop_prob: float = 0.5,
        target_len: int = 256
    ):
        super().__init__()
        self.num_levels = num_levels
        self.levels = levels or [64, 128, 256]
        assert len(self.levels) == self.num_levels, (
            f"Number of levels ({self.num_levels}) must match levels schedule length ({len(self.levels)})"
        )
        self.target_len = target_len or self.levels[-1]
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.nested_drop_prob = nested_drop_prob

        # Distinct codebook for each hierarchical level
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim) for _ in range(self.num_levels)
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=1.0 / (embed_dim ** 0.5))

    def _down_to_level(self, z_full: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Down_k operator: Area-pools continuous residual to scale resolution ell_k.
        Shape: [B, L, D] -> [B, ell_k, D]
        """
        if z_full.shape[1] == target_len:
            return z_full
        # F.interpolate operates on [B, Channels, Length]
        z_bcl = z_full.transpose(1, 2)
        pooled_bcl = F.interpolate(z_bcl, size=(target_len,), mode="area")
        return pooled_bcl.transpose(1, 2).contiguous()

    def _up_to_full(self, q_level: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        R_k operator: Linearly interpolates level quantized representation back to length L.
        Shape: [B, ell_k, D] -> [B, L, D]
        """
        if q_level.shape[1] == target_len:
            return q_level
        q_bcl = q_level.transpose(1, 2)
        up_bcl = F.interpolate(q_bcl, size=(target_len,), mode="linear", align_corners=False)
        return up_bcl.transpose(1, 2).contiguous()

    def quantize_level(
        self,
        u_k: torch.Tensor,
        level_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantizes a single level's representation against codebook V_k.
        Args:
            u_k: [B, ell_k, D]
            level_idx: int index in [0, K-1]
        Returns:
            q_k_ste: quantized tensor with Straight-Through Estimator gradients, [B, ell_k, D]
            r_k: discrete code indices in [0, codebook_size - 1], [B, ell_k]
            commit_loss: commitment loss scalar MSE(u_k, sg[q_k])
        """
        emb = self.embeddings[level_idx]
        B, ell_k, D = u_k.shape

        # Pairwise distance: ||u - e||^2 = ||u||^2 + ||e||^2 - 2 u @ e.T
        u_flat = u_k.reshape(-1, D)
        e_weight = emb.weight  # [V, D]

        u_sq = torch.sum(u_flat ** 2, dim=-1, keepdim=True)  # [B*ell_k, 1]
        e_sq = torch.sum(e_weight ** 2, dim=-1)  # [V]
        dists = u_sq + e_sq - 2.0 * torch.matmul(u_flat, e_weight.t())  # [B*ell_k, V]

        code_indices_flat = torch.argmin(dists, dim=-1)
        r_k = code_indices_flat.view(B, ell_k)

        # Lookup quantized vectors
        q_k = emb(r_k)  # [B, ell_k, D]

        # VQ commitment loss
        commit_loss = F.mse_loss(u_k, q_k.detach())
        codebook_loss = F.mse_loss(u_k.detach(), q_k)

        # Straight-Through Estimator: q_k_ste has values of q_k, but gradients flow to u_k
        q_k_ste = u_k + (q_k - u_k).detach()

        total_loss = codebook_loss + self.commitment_cost * commit_loss
        return q_k_ste, r_k, total_loss

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward multiscale residual quantization.
        Args:
            z: Continuous latent from encoder [B, L, D]
        Returns:
            z_hat: Cumulative reconstruction [B, L, D]
            code_indices: List of K discrete index tensors [r_1, ..., r_K]
            loss_dict: Dictionary with commitment, codebook, and total VQ loss
        """
        B, L, D = z.shape
        device = z.device

        z_hat = torch.zeros(B, L, D, device=device)
        residual = z.clone()

        code_indices: List[torch.Tensor] = []
        quant_losses: List[torch.Tensor] = []

        # Nested dropout: during training, optionally sample an early cutoff level
        cutoff_level = self.num_levels
        if self.training and self.nested_drop_prob > 0.0:
            if torch.rand(1).item() < self.nested_drop_prob:
                # Sample cutoff in [1, K-1]
                cutoff_level = torch.randint(1, self.num_levels, (1,)).item()

        for k in range(self.num_levels):
            ell_k = self.levels[k]

            # 1. Downsample residual to level resolution
            u_k = self._down_to_level(residual, ell_k)

            # 2. Quantize against codebook k
            q_k_ste, r_k, vq_loss_k = self.quantize_level(u_k, k)
            code_indices.append(r_k)
            quant_losses.append(vq_loss_k)

            # 3. Reconstruct contribution to full length L
            r_full = self._up_to_full(q_k_ste, L)

            # 4. Update cumulative reconstruction and residual
            z_hat = z_hat + r_full
            residual = z - z_hat

            if (k + 1) == cutoff_level and self.training:
                # Truncate at cutoff level per nested dropout
                break

        total_vq_loss = torch.stack(quant_losses).mean()

        loss_dict = {
            "vq_loss": total_vq_loss,
            "cutoff_level": torch.tensor(cutoff_level, device=device)
        }
        return z_hat, code_indices, loss_dict

    @torch.no_grad()
    def idx_to_z_q(self, code_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Dequantizes a list of discrete code index tensors to the cumulative
        continuous representation z_hat of full length L.
        """
        B = code_indices[0].shape[0]
        device = code_indices[0].device
        L = self.target_len

        z_hat = torch.zeros(B, L, self.embed_dim, device=device)
        for k, r_k in enumerate(code_indices):
            q_k = self.embeddings[k](r_k)
            r_full = self._up_to_full(q_k, L)
            z_hat = z_hat + r_full
        return z_hat

    @torch.no_grad()
    def get_next_autoregressive_input(
        self,
        level_idx: int,
        z_hat: torch.Tensor,
        z_q_level: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cumulative reconstruction z_hat with current level's code vectors,
        and returns updated z_hat and resized input for the next level.
        Matches the method signature in HazyResearch/ncm/models/quant.py.
        """
        L = self.target_len
        upsampled = self._up_to_full(z_q_level, L)
        z_hat = z_hat + upsampled

        next_level = min(level_idx + 1, self.num_levels - 1)
        next_len = self.levels[next_level]
        next_input = self._down_to_level(z_hat, next_len)
        return z_hat, next_input
