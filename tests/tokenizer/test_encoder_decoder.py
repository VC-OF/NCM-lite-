import pytest
import torch
from ncm.tokenizer.model import MultiscaleEncoder, MultiscaleDecoder


def test_encoder_output_shapes():
    B, L = 2, 256
    embed_dim = 64
    encoder = MultiscaleEncoder(
        vocab_size=256,
        embed_dim=embed_dim,
        max_seq_len=L
    )

    x = torch.randint(0, 256, (B, L))
    z = encoder(x)

    assert z.shape == (B, L, embed_dim)
    assert z.dtype == torch.float32


def test_decoder_output_shapes():
    B, L = 2, 256
    embed_dim = 64
    vocab_size = 256
    decoder = MultiscaleDecoder(
        embed_dim=embed_dim,
        vocab_size=vocab_size
    )

    z_hat = torch.randn(B, L, embed_dim)
    logits = decoder(z_hat)

    assert logits.shape == (B, L, vocab_size)
    assert logits.dtype == torch.float32


def test_encoder_sequence_length_boundary():
    encoder = MultiscaleEncoder(max_seq_len=128)
    x_too_long = torch.randint(0, 256, (2, 256))
    with pytest.raises(ValueError):
        encoder(x_too_long)
