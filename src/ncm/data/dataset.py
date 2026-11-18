"""
Dataset implementations for NCM-Lite.

Supports UTF-8 byte chunking and deterministic synthetic datasets.
No external network calls or large dataset downloads.
"""

from typing import List, Optional
import torch
from torch.utils.data import Dataset


class ByteTextDataset(Dataset):
    """
    Datasets representing raw text as fixed-length UTF-8 byte token sequences.
    
    Vocabulary size: 256 (byte values 0 to 255).
    """

    def __init__(self, texts: List[str], seq_len: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.chunks: List[torch.Tensor] = []

        for text in texts:
            # Convert text to UTF-8 bytes
            byte_vals = list(text.encode("utf-8", errors="replace"))
            if len(byte_vals) == 0:
                continue

            # Pad or chunk into fixed lengths of seq_len
            for i in range(0, len(byte_vals), seq_len):
                chunk = byte_vals[i : i + seq_len]
                if len(chunk) < seq_len:
                    # Pad with 0 (null byte)
                    chunk = chunk + [0] * (seq_len - len(chunk))
                tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                self.chunks.append(tensor_chunk)

        if len(self.chunks) == 0:
            # Fallback single zero sequence
            self.chunks.append(torch.zeros(seq_len, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


class SyntheticTextDataset(Dataset):
    """
    Deterministic synthetic dataset for shape, boundary, and memory smoke testing.
    Generates repeatable pseudo-text patterns under a fixed random seed.
    """

    def __init__(self, num_samples: int = 32, seq_len: int = 256, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len

        rng = torch.Generator().manual_seed(seed)
        # Printable ASCII bytes range from 32 to 126
        self.data = torch.randint(32, 127, (num_samples, seq_len), generator=rng, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
