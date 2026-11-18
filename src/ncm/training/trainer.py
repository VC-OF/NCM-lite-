"""
Training loops for NCM-Lite.

Implements two-stage training fidelity:
- TokenizerTrainer: Stage 1 multiscale RVQ optimization.
- GeneratorTrainer: Stage 2 coarse-to-fine generator optimization with
  strictly frozen tokenizer in evaluation mode.

Grounding: Section 3.3 and 3.4 of arXiv:2610.14162.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer


class TokenizerTrainer:
    """
    Orchestrates Stage 1 optimization of the Multiscale RVQ Tokenizer.
    """

    def __init__(
        self,
        model: TokenizerModel,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.step_count = 0

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Executes a single optimization step for Stage 1.
        Args:
            batch: [B, L] tensor of byte token indices.
        Returns:
            Dictionary of scalar loss metrics.
        """
        self.model.train()
        batch = batch.to(self.device)

        self.optimizer.zero_grad()
        _, _, losses = self.model(batch)

        total_loss = losses["total_loss"]
        total_loss.backward()

        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.step_count += 1

        return {
            "recon_loss": losses["recon_loss"].item(),
            "vq_loss": losses["vq_loss"].item(),
            "total_loss": total_loss.item()
        }


class GeneratorTrainer:
    """
    Orchestrates Stage 2 optimization of the Next-Concept Transformer.
    
    Research Fidelity:
    The Stage 1 tokenizer is strictly frozen and set to evaluation mode.
    Its parameters never accumulate gradients or update during Stage 2.
    """

    def __init__(
        self,
        generator: NextConceptTransformer,
        tokenizer: TokenizerModel,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.generator = generator.to(device)
        self.tokenizer = tokenizer.to(device)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.step_count = 0

        # Strictly freeze tokenizer
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Executes a single optimization step for Stage 2.
        
        1. Encodes batch into target codes via frozen tokenizer.
        2. Derives cumulative continuous inputs for each level.
        3. Optimizes generator on cross-entropy loss against discrete targets.
        """
        self.generator.train()
        self.tokenizer.eval()
        batch = batch.to(self.device)

        # 1. Extract target discrete codes and cumulative inputs with frozen tokenizer
        with torch.no_grad():
            z = self.tokenizer.encoder(batch)  # [B, L, D]
            B, L, D = z.shape

            # Compute frozen target codes
            _, target_codes, _ = self.tokenizer.vq(z)

            # Build continuous inputs Down_k(z_hat^(k-1)) for each level
            level_inputs: List[torch.Tensor] = []
            z_hat = torch.zeros(B, L, D, device=self.device)

            for k in range(self.generator.num_levels):
                ell_k = self.generator.levels[k]
                u_k = self.tokenizer.vq._down_to_level(z_hat, ell_k)
                level_inputs.append(u_k)

                # Update cumulative reconstruction for the next level
                r_k = target_codes[k]
                q_k = self.tokenizer.vq.embeddings[k](r_k)
                r_full = self.tokenizer.vq._up_to_full(q_k, L)
                z_hat = z_hat + r_full

        # 2. Generator forward and backward step
        self.optimizer.zero_grad()
        _, losses = self.generator(level_inputs, target_codes=target_codes)

        total_loss = losses["total_ncm_loss"]
        total_loss.backward()

        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.step_count += 1

        metric_dict: Dict[str, float] = {"total_ncm_loss": total_loss.item()}
        for k in range(self.generator.num_levels):
            key = f"loss_level_{k + 1}"
            if key in losses:
                metric_dict[key] = losses[key].item()

        return metric_dict
