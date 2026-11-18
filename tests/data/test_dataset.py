import tempfile
import os
import pytest
import torch
from ncm.data.dataset import ByteTextDataset, SyntheticTextDataset


def test_byte_dataset_chunking():
    texts = [
        "Hello, World!",  # Short string
        "A" * 300,        # > 256 bytes
    ]
    dataset = ByteTextDataset(texts, seq_len=256)

    # First text padded to 256
    sample0 = dataset[0]
    assert sample0.shape == (256,)
    assert sample0.dtype == torch.long
    assert sample0[0].item() == ord("H")
    assert sample0[13].item() == 0  # null padding

    # Second text split into chunks
    sample1 = dataset[1]
    assert sample1.shape == (256,)
    assert (sample1 == ord("A")).all()


def test_byte_dataset_from_file_and_text():
    sample_text = "Once upon a time in a small village, a brave girl discovered an ancient secret. " * 10
    dataset_text = ByteTextDataset.from_text(sample_text, seq_len=128, stride=64)
    assert len(dataset_text) > 1
    assert dataset_text[0].shape == (128,)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as f:
        f.write(sample_text)
        temp_path = f.name

    try:
        dataset_file = ByteTextDataset.from_file(temp_path, seq_len=128, stride=64)
        assert len(dataset_file) == len(dataset_text)
        assert torch.equal(dataset_file[0], dataset_text[0])
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_synthetic_text_dataset_determinism():
    ds1 = SyntheticTextDataset(num_samples=10, seq_len=128, seed=123)
    ds2 = SyntheticTextDataset(num_samples=10, seq_len=128, seed=123)
    ds3 = SyntheticTextDataset(num_samples=10, seq_len=128, seed=456)

    assert torch.equal(ds1[0], ds2[0])
    assert not torch.equal(ds1[0], ds3[0])
    assert (ds1[0] >= 32).all() and (ds1[0] <= 126).all()
