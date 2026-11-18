import pytest
import torch
from ncm.data.dataset import ByteTextDataset, SyntheticTextDataset


def test_byte_dataset_chunking():
    texts = [
        "Hello world!",
        "Next-Concept Modeling prototype for laptop CPU."
    ]
    dataset = ByteTextDataset(texts, seq_len=16)

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample.shape == (16,)
    assert sample.dtype == torch.long
    assert (sample >= 0).all() and (sample < 256).all()


def test_synthetic_text_dataset_determinism():
    ds1 = SyntheticTextDataset(num_samples=10, seq_len=64, seed=123)
    ds2 = SyntheticTextDataset(num_samples=10, seq_len=64, seed=123)

    assert len(ds1) == 10
    sample1 = ds1[0]
    sample2 = ds2[0]

    assert torch.equal(sample1, sample2), "Synthetic datasets with identical seed produced divergent data"
    assert sample1.shape == (64,)
