"""
Atomic checkpoint management for NCM-Lite.

Saves and restores model weights, optimizer states, scheduler states,
loss metadata, and random number generator states atomically.
"""

import os
import tempfile
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages atomic checkpoint serialization and deserialization.
    """

    VERSION = "1.0.0"

    @classmethod
    def save(
        cls,
        filepath: str,
        stage: int,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves a checkpoint atomically using a temporary file.
        """
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "version": cls.VERSION,
            "stage": stage,
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "config": config or {},
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()
            }
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Write to temporary file in the same directory, then rename atomically
        temp_fd, temp_path = tempfile.mkstemp(dir=dir_name if dir_name else ".")
        os.close(temp_fd)

        try:
            torch.save(checkpoint, temp_path)
            # Atomic replace
            os.replace(temp_path, filepath)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    @classmethod
    def load(
        cls,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Loads a checkpoint and restores model and optimizer states.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "rng_state" in checkpoint:
            rng = checkpoint["rng_state"]
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])

        return checkpoint
