import sqlite3
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.inputs.apertus_preprocess import (
    ApertusOmniInputPreprocessor,
    is_apertus_model_config,
)


def _make_apertus_preprocessor(monkeypatch):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="apertus", architectures=["ApertusForCausalLM"])
    )
    monkeypatch.setattr(preprocessor, "_tokenize_prompt", lambda prompt_text, tokenization_kwargs=None: [9, 8, 7])
    monkeypatch.setattr(
        preprocessor,
        "_process_apertus_text_with_images",
        lambda *args, **kwargs: {"prompt_token_ids": [4, 5, 6]},
    )
    return preprocessor


def test_is_apertus_model_config_true():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="apertus", architectures=["ApertusForCausalLM"])
    )
    assert is_apertus_model_config(model_config) is True


def test_is_apertus_model_config_false():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen", architectures=["QwenForCausalLM"])
    )
    assert is_apertus_model_config(model_config) is False


def test_apertus_adapter_rejects_unsupported_modalities():
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    with pytest.raises(ValueError, match="text and image inputs only"):
        preprocessor._is_apertus_text_image_input(
            {"image": [object()], "audio": [object()]},
        )


def test_process_text_uses_apertus_adapter_path(monkeypatch):
    preprocessor = _make_apertus_preprocessor(monkeypatch)
    parsed = {
        "prompt": "hello <|image|>",
        "multi_modal_data": {"image": ["fake"]},
        "mm_processor_kwargs": {},
    }

    inputs = ApertusOmniInputPreprocessor._process_text(preprocessor, parsed)
    assert inputs["prompt_token_ids"] == [4, 5, 6]


def test_build_apertus_image_prompt_uses_emu35_format_without_eof():
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace(
        boi_token="<|image start|>",
        img_token="<|image token|>",
        eol_token="<|extra_200|>",
        eoi_token="<|image end|>",
    )

    token_grid = torch.tensor([[5, 6], [7, 8]])
    image_prompt = ApertusOmniInputPreprocessor._build_apertus_image_prompt(
        preprocessor,
        token_grid,
    )

    assert image_prompt == (
        "<|image start|>2*2<|image token|>"
        "<|visual token 5|><|visual token 6|><|extra_200|>"
        "<|visual token 7|><|visual token 8|>"
        "<|image end|>"
    )
    assert "<|img_end_of_frame|>" not in image_prompt


def test_extract_emu35_token_grid_handles_nested_none_tuple():
    token_ids = torch.arange(6, dtype=torch.int64)
    encode_out = (torch.zeros(1), None, (None, None, token_ids))
    image_token_grid = ApertusOmniInputPreprocessor._extract_emu35_token_grid(
        encode_out,
        token_height=2,
        token_width=3,
    )

    assert tuple(image_token_grid.shape) == (2, 3)
    assert image_token_grid.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_process_apertus_text_sets_add_special_tokens_false_by_default(monkeypatch):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        preprocessor,
        "_normalize_apertus_images",
        lambda image_data: [object()],
    )
    monkeypatch.setattr(
        preprocessor,
        "_encode_apertus_images_to_strings",
        lambda images, mm_processor_kwargs: ["<img_prompt>"],
    )
    monkeypatch.setattr(
        preprocessor,
        "_resolve_apertus_image_placeholder",
        lambda prompt_text, mm_processor_kwargs: "<|image|>",
    )
    captured = {}

    def _fake_tokenize(prompt_text, tokenization_kwargs=None):
        captured["prompt_text"] = prompt_text
        captured["tokenization_kwargs"] = tokenization_kwargs
        return [1, 2, 3]

    monkeypatch.setattr(preprocessor, "_tokenize_prompt", _fake_tokenize)

    inputs = ApertusOmniInputPreprocessor._process_apertus_text_with_images(
        preprocessor,
        prompt_text="hello <|image|>",
        multi_modal_data={"image": [object()]},
        mm_processor_kwargs={},
        tokenization_kwargs=None,
    )

    assert inputs["prompt_token_ids"] == [1, 2, 3]
    assert captured["prompt_text"] == "hello <img_prompt>"
    assert captured["tokenization_kwargs"]["add_special_tokens"] is False


def test_encode_apertus_images_reads_prompt_from_sqlite_cache_without_encoder(monkeypatch, tmp_path):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace(
        boi_token="<|image start|>",
        img_token="<|image token|>",
        eol_token="<|extra_200|>",
        eoi_token="<|image end|>",
    )
    image = Image.new("RGB", (16, 16), color=(64, 128, 192))
    mm_processor_kwargs = {
        "apertus_min_pixels": 256,
        "apertus_max_pixels": 256,
        "apertus_image_token_cache": True,
        "apertus_image_token_cache_dir": str(tmp_path / "apertus_cache"),
    }
    cache_db_path = preprocessor._resolve_apertus_image_token_cache_db_path(mm_processor_kwargs)
    assert cache_db_path is not None

    resized_image = preprocessor._smart_resize(
        image,
        area=256,
        ds_factor=preprocessor._APERTUS_EMU35_DS_FACTOR,
    )
    cache_key = preprocessor._build_apertus_image_prompt_cache_key(
        resized_image,
        mm_processor_kwargs=mm_processor_kwargs,
        min_pixels=256,
        max_pixels=256,
    )
    expected_prompt = "<cached-image-prompt>"
    preprocessor._store_apertus_image_prompt_in_cache(cache_db_path, cache_key, expected_prompt)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("vision tokenizer should not be loaded when cache hit exists")

    monkeypatch.setattr(preprocessor, "_get_apertus_vision_components", _fail_if_called)

    image_prompts = preprocessor._encode_apertus_images_to_strings(
        [image],
        mm_processor_kwargs=mm_processor_kwargs,
    )
    assert image_prompts == [expected_prompt]


def test_encode_apertus_images_writes_sqlite_cache_and_reuses_it(monkeypatch, tmp_path):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace(
        boi_token="<|image start|>",
        img_token="<|image token|>",
        eol_token="<|extra_200|>",
        eoi_token="<|image end|>",
    )
    image = Image.new("RGB", (16, 16), color=(32, 64, 96))
    mm_processor_kwargs = {
        "apertus_min_pixels": 256,
        "apertus_max_pixels": 256,
        "apertus_image_token_cache": True,
        "apertus_image_token_cache_dir": str(tmp_path / "apertus_cache"),
    }

    class _FakeVisionTokenizer:
        def __init__(self):
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.encode_calls = 0

        def parameters(self):
            return iter((self._param,))

        def encode(self, *_args, **_kwargs):
            self.encode_calls += 1
            return torch.tensor([[1]], dtype=torch.int64)

    fake_vision_tokenizer = _FakeVisionTokenizer()
    monkeypatch.setattr(
        preprocessor,
        "_get_apertus_vision_components",
        lambda _kwargs: fake_vision_tokenizer,
    )
    monkeypatch.setattr(
        preprocessor,
        "_extract_emu35_token_grid",
        lambda encode_out, token_height, token_width: torch.tensor([[5]], dtype=torch.int64),
    )

    first = preprocessor._encode_apertus_images_to_strings(
        [image],
        mm_processor_kwargs=mm_processor_kwargs,
    )
    assert fake_vision_tokenizer.encode_calls == 1

    cache_db_path = preprocessor._resolve_apertus_image_token_cache_db_path(mm_processor_kwargs)
    assert cache_db_path is not None
    resized_image = preprocessor._smart_resize(
        image,
        area=256,
        ds_factor=preprocessor._APERTUS_EMU35_DS_FACTOR,
    )
    cache_key = preprocessor._build_apertus_image_prompt_cache_key(
        resized_image,
        mm_processor_kwargs=mm_processor_kwargs,
        min_pixels=256,
        max_pixels=256,
    )
    assert cache_db_path.exists()
    with sqlite3.connect(cache_db_path) as conn:
        row = conn.execute(
            f"""
            SELECT image_prompt
            FROM {preprocessor._APERTUS_IMAGE_TOKEN_CACHE_TABLE}
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    assert row is not None
    assert row[0] == first[0]

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("vision tokenizer should not be loaded on second call")

    monkeypatch.setattr(preprocessor, "_get_apertus_vision_components", _fail_if_called)
    second = preprocessor._encode_apertus_images_to_strings(
        [image],
        mm_processor_kwargs=mm_processor_kwargs,
    )
    assert second == first
