import pytest
import os
import torch

from scripts.serve_web_ui import NCMStudioServer


def test_studio_server_initialization():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    studio = NCMStudioServer(base_dir=base_dir, device="cpu")

    assert studio.generator is not None
    assert studio.tokenizer is not None
    assert studio.seq_len == 256
    assert len(studio.levels) == 3


def test_studio_server_unconditional_generation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    studio = NCMStudioServer(base_dir=base_dir, device="cpu")

    result = studio.generate(
        prompt="",
        temperature=0.8,
        top_k=20,
        top_p=0.9
    )

    assert "latency_seconds" in result
    assert result["latency_seconds"] > 0.0
    assert "level_1_text" in result
    assert "level_2_text" in result
    assert "level_3_text" in result
    assert isinstance(result["level_3_text"], str)


def test_studio_server_prompt_generation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    studio = NCMStudioServer(base_dir=base_dir, device="cpu")

    prompt = "Once upon a time, Lily found a "
    result = studio.generate(
        prompt=prompt,
        temperature=0.8,
        top_k=20,
        top_p=0.9
    )

    assert result["prompt"] == prompt
    assert "level_1_text" in result
    assert "level_2_text" in result
    assert "level_3_text" in result
    assert isinstance(result["level_3_text"], str)
