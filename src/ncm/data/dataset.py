"""
Dataset implementations for NCM-Lite.

Supports UTF-8 byte chunking and deterministic synthetic datasets.
No external network calls or large dataset downloads.
"""

import os
from typing import List, Optional
import torch
from torch.utils.data import Dataset


class ByteTextDataset(Dataset):
    """
    Dataset representing raw text as fixed-length UTF-8 byte token sequences.
    
    Vocabulary size: 256 (byte values 0 to 255).
    Supports multi-string input, single-string input with striding,
    and direct text file loading.
    """

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        seq_len: int = 256,
        stride: Optional[int] = None
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.chunks: List[torch.Tensor] = []

        if texts is not None:
            for text in texts:
                self._add_text_chunks(text)

        if len(self.chunks) == 0:
            # Fallback single zero sequence
            self.chunks.append(torch.zeros(seq_len, dtype=torch.long))

    def _add_text_chunks(self, text: str) -> None:
        """Encodes string to UTF-8 bytes and appends strided chunks."""
        byte_vals = list(text.encode("utf-8", errors="replace"))
        if len(byte_vals) == 0:
            return

        if len(byte_vals) <= self.seq_len:
            # Pad to seq_len
            chunk = byte_vals + [0] * (self.seq_len - len(byte_vals))
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))
            return

        for i in range(0, len(byte_vals) - self.seq_len + 1, self.stride):
            chunk = byte_vals[i : i + self.seq_len]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        # Handle any remaining trailing bytes if stride didn't reach the end
        last_start = ((len(byte_vals) - self.seq_len) // self.stride) * self.stride
        if last_start + self.seq_len < len(byte_vals):
            tail = byte_vals[last_start + self.stride :]
            if tail:
                padded = tail + [0] * (self.seq_len - len(tail))
                self.chunks.append(torch.tensor(padded[: self.seq_len], dtype=torch.long))

    @classmethod
    def from_file(
        cls,
        file_path: str,
        seq_len: int = 256,
        stride: Optional[int] = None,
        encoding: str = "utf-8"
    ) -> "ByteTextDataset":
        """
        Loads text from a local file and constructs byte sequences.
        
        Args:
            file_path: Path to raw .txt file.
            seq_len: Sequence length (default 256).
            stride: Step size between consecutive chunks (default seq_len).
            encoding: Text encoding (default utf-8).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text corpus file not found: {file_path}")

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()

        dataset = cls(texts=None, seq_len=seq_len, stride=stride)
        dataset._add_text_chunks(content)
        if len(dataset.chunks) == 0:
            dataset.chunks.append(torch.zeros(seq_len, dtype=torch.long))
        return dataset

    @classmethod
    def from_text(
        cls,
        text: str,
        seq_len: int = 256,
        stride: Optional[int] = None
    ) -> "ByteTextDataset":
        """Constructs dataset directly from a single long text string."""
        dataset = cls(texts=None, seq_len=seq_len, stride=stride)
        dataset._add_text_chunks(text)
        if len(dataset.chunks) == 0:
            dataset.chunks.append(torch.zeros(seq_len, dtype=torch.long))
        return dataset

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
