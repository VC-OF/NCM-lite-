"""
Stage 1 Tokenizer Architecture for NCM-Lite.

Implements the Multiscale VQ-VAE tokenizer:
- MultiscaleEncoder: maps byte tokens to initial continuous latent z.
- MultiscaleResidualVectorQuantizer: performs multiscale RVQ.
- MultiscaleDecoder: reconstructs byte token logits from cumulative latents.
- TokenizerModel: composed end-to-end autoencoder.

Grounding: Section 3.3 of arXiv:2610.14162 and HazyResearch/ncm/models/vqvae.py.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm.quantization.rvq import MultiscaleResidualVectorQuantizer


class MultiscaleEncoder(nn.Module):
    """
    Encodes byte tokens into an initial continuous latent representation z of length L.
    
    In the paper (models/vqvae.py line 41), the default encoder uses token embeddings
    and positional embeddings with optional lightweight convolution or transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        max_seq_len: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Lightweight 1D convolution block for local contextualization
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=3, padding=1),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L] tensor of byte token IDs in [0, vocab_size - 1].
        Returns:
            z: [B, L, embed_dim] continuous latent representations.
        """
        B, L = x.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds maximum {self.max_seq_len}")

        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.wte(x) + self.wpe(positions)
        h = self.layer_norm(h)

        # Residual convolution
        h_res = self.conv(h.transpose(1, 2)).transpose(1, 2)
        z = self.out_norm(h + h_res)
        return z


class MultiscaleDecoder(nn.Module):
    """
    Decodes the cumulative continuous reconstruction z_hat of length L
    back into byte vocabulary logits.
    
    Grounding: matches models/vqvae.py decoder with lightweight projection.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        vocab_size: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_hat: [B, L, embed_dim] cumulative continuous latent tensor.
        Returns:
            logits: [B, L, vocab_size] unnormalized token predictions.
        """
        return self.net(z_hat)


class TokenizerModel(nn.Module):
    """
    Composed Stage 1 Multiscale RVQ Autoencoder.
    
    Coordinates Encoder, Multiscale RVQ Quantizer, and Decoder.
    Supports joint Stage 1 training and frozen Stage 2 code extraction/decoding.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        max_seq_len: int = 256,
        num_levels: int = 3,
        levels: Optional[List[int]] = None,
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.95,
        nested_drop_prob: float = 0.5
    ):
        super().__init__()
        # Allow passing config dictionary as single positional or keyword argument
        if isinstance(vocab_size, dict):
            cfg = vocab_size
            vocab_size = cfg.get("vocab_size", 256)
            embed_dim = cfg.get("embed_dim", 64)
            max_seq_len = cfg.get("max_seq_len", cfg.get("target_len", 256))
            num_levels = cfg.get("num_levels", 3)
            levels = cfg.get("levels", [64, 128, 256])
            codebook_size = cfg.get("codebook_size", 256)
            commitment_cost = cfg.get("commitment_cost", cfg.get("beta", 0.25))
            decay = cfg.get("decay", 0.95)
            nested_drop_prob = cfg.get("nested_drop_prob", cfg.get("dropout_p", 0.5))

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_levels = num_levels
        self.levels = levels or [64, 128, 256]

        self.encoder = MultiscaleEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len
        )
        self.vq = MultiscaleResidualVectorQuantizer(
            num_levels=num_levels,
            levels=self.levels,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            nested_drop_prob=nested_drop_prob,
            target_len=max_seq_len
        )
        self.decoder = MultiscaleDecoder(
            embed_dim=embed_dim,
            vocab_size=vocab_size
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Full Stage 1 forward step with loss computation.
        Args:
            x: [B, L] input token tensor, dtype torch.int64.
        Returns:
            logits: [B, L, vocab_size] reconstructed token logits.
            code_indices: List of discrete codes [r_1, ..., r_K].
            loss_dict: Dictionary containing recon_loss, vq_loss, and total_loss.
        """
        z = self.encoder(x)
        z_hat, code_indices, vq_losses = self.vq(z)
        logits = self.decoder(z_hat)

        # Cross-entropy reconstruction loss
        B, L, V = logits.shape
        recon_loss = F.cross_entropy(logits.view(B * L, V), x.view(B * L))

        total_loss = recon_loss + vq_losses["vq_loss"]

        losses = {
            "recon_loss": recon_loss,
            "vq_loss": vq_losses["vq_loss"],
            "total_loss": total_loss,
            "cutoff_level": vq_losses.get("cutoff_level", torch.tensor(self.num_levels))
        }
        return logits, code_indices, losses

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes token sequence into the frozen discrete code hierarchy [r_1, ..., r_K].
        Used to generate targets for Stage 2 training.
        """
        self.eval()
        z = self.encoder(x)
        # In eval mode, nested dropout is disabled (all K levels computed)
        _, code_indices, _ = self.vq(z)
        return code_indices

    @torch.no_grad()
    def decode(self, code_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Decodes a list of discrete code indices back to reconstructed token logits.
        """
        self.eval()
        z_hat = self.vq.idx_to_z_q(code_indices)
        logits = self.decoder(z_hat)
        return logits
