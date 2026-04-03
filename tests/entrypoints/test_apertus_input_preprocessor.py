import json
import sqlite3
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import vllm_omni.inputs.apertus.modalities.image as image_mod
from vllm_omni.inputs.apertus.modalities.image import ImageModalityEncoder
from vllm_omni.inputs.apertus.preprocessor import (
    ApertusOmniInputPreprocessor,
    is_apertus_model_config,
)
from vllm_omni.inputs.apertus.types import ModalityContext, StringPiece, TokenPiece
from vllm_omni.inputs.apertus_utils import ensure_local_weights


def _make_apertus_preprocessor(monkeypatch):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="apertus", architectures=["ApertusForCausalLM"])
    )
    preprocessor._modality_encoders = {"image": object()}
    monkeypatch.setattr(preprocessor, "_tokenize_prompt", lambda prompt_text, tokenization_kwargs=None: [9, 8, 7])
    monkeypatch.setattr(
        preprocessor,
        "_process_apertus_prompt",
        lambda *args, **kwargs: {"prompt_token_ids": [4, 5, 6]},
    )
    return preprocessor


class _FakeImageEncoder:
    modality = "image"

    def placeholder_aliases(self, prompt_text, ctx):
        del prompt_text, ctx
        return ["<|image|>"]

    def normalize_inputs(self, raw_input, ctx):
        del ctx
        return list(raw_input or [])

    def encode_many(self, items, ctx):
        del items, ctx
        return [StringPiece("<img_prompt>")]


class _SharedAliasImageEncoder:
    modality = "image"

    def placeholder_aliases(self, prompt_text, ctx):
        del prompt_text, ctx
        return ["<|shared|>"]

    def normalize_inputs(self, raw_input, ctx):
        del raw_input, ctx
        return []

    def encode_many(self, items, ctx):
        del items, ctx
        return []


class _SharedAliasAudioEncoder:
    modality = "audio"

    def placeholder_aliases(self, prompt_text, ctx):
        del prompt_text, ctx
        return ["<|shared|>"]

    def normalize_inputs(self, raw_input, ctx):
        del ctx
        return list(raw_input or [])

    def encode_many(self, items, ctx):
        del items, ctx
        return [TokenPiece([11, 12, 13])]


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
    preprocessor._modality_encoders = {"image": object(), "audio": object()}
    with pytest.raises(ValueError, match="supports only these extra modalities"):
        preprocessor._is_apertus_multimodal_input(
            {"image": [object()], "video": [object()]},
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
    encoder = ImageModalityEncoder()
    tokenizer = SimpleNamespace(
        boi_token="<|image start|>",
        img_token="<|image token|>",
        eol_token="<|extra_200|>",
        eoi_token="<|image end|>",
    )

    token_grid = torch.tensor([[5, 6], [7, 8]])
    image_prompt = encoder._build_apertus_image_prompt(token_grid, tokenizer)

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
    image_token_grid = ImageModalityEncoder._extract_emu35_token_grid(
        encode_out,
        token_height=2,
        token_width=3,
    )

    assert tuple(image_token_grid.shape) == (2, 3)
    assert image_token_grid.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_process_apertus_prompt_sets_add_special_tokens_false_by_default(monkeypatch):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace()
    preprocessor.model_config = None
    preprocessor._modality_encoders = {"image": _FakeImageEncoder()}
    captured = {}

    def _fake_tokenize(prompt_text, tokenization_kwargs=None):
        captured["prompt_text"] = prompt_text
        captured["tokenization_kwargs"] = tokenization_kwargs
        return [1, 2, 3]

    monkeypatch.setattr(preprocessor, "_tokenize_prompt", _fake_tokenize)

    inputs = ApertusOmniInputPreprocessor._process_apertus_prompt(
        preprocessor,
        prompt_text="hello <|image|>",
        multi_modal_data={"image": [object()]},
        mm_processor_kwargs={},
        tokenization_kwargs=None,
    )

    assert inputs["prompt_token_ids"] == [1, 2, 3]
    assert captured["prompt_text"] == "hello <img_prompt>"
    assert captured["tokenization_kwargs"]["add_special_tokens"] is False


def test_process_apertus_prompt_supports_shared_placeholder_for_active_modality(monkeypatch):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace()
    preprocessor.model_config = None
    preprocessor._modality_encoders = {
        "image": _SharedAliasImageEncoder(),
        "audio": _SharedAliasAudioEncoder(),
    }

    def _fake_tokenize(prompt_text, tokenization_kwargs=None):
        assert prompt_text == "listen "
        assert tokenization_kwargs["add_special_tokens"] is False
        return [7]

    monkeypatch.setattr(preprocessor, "_tokenize_prompt", _fake_tokenize)

    inputs = ApertusOmniInputPreprocessor._process_apertus_prompt(
        preprocessor,
        prompt_text="listen <|shared|>",
        multi_modal_data={"audio": [("fake", 16000)]},
        mm_processor_kwargs={"apertus_audio_placeholder": "<|shared|>"},
        tokenization_kwargs=None,
    )

    assert inputs["prompt_token_ids"] == [7, 11, 12, 13]


def test_encode_apertus_images_reads_prompt_from_sqlite_cache_without_encoder(monkeypatch, tmp_path):
    encoder = ImageModalityEncoder()
    tokenizer = SimpleNamespace(
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
    cache_db_path = encoder._resolve_apertus_image_token_cache_db_path(mm_processor_kwargs)
    assert cache_db_path is not None

    resized_image = encoder._smart_resize(
        image,
        area=256,
        ds_factor=encoder._APERTUS_EMU35_DS_FACTOR,
    )
    cache_key = encoder._build_apertus_image_prompt_cache_key(
        resized_image,
        tokenizer=tokenizer,
        mm_processor_kwargs=mm_processor_kwargs,
        min_pixels=256,
        max_pixels=256,
    )
    expected_prompt = "<cached-image-prompt>"
    encoder._store_apertus_image_prompt_in_cache(cache_db_path, cache_key, expected_prompt)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("vision tokenizer should not be loaded when cache hit exists")

    monkeypatch.setattr(encoder, "_get_apertus_vision_components", _fail_if_called)

    ctx = ModalityContext(tokenizer=tokenizer, model_config=None, mm_processor_kwargs=mm_processor_kwargs)
    image_prompts = encoder.encode_many([image], ctx)
    assert image_prompts == [StringPiece(expected_prompt)]


def test_encode_apertus_images_writes_sqlite_cache_and_reuses_it(monkeypatch, tmp_path):
    encoder = ImageModalityEncoder()
    tokenizer = SimpleNamespace(
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
        encoder,
        "_get_apertus_vision_components",
        lambda _kwargs: fake_vision_tokenizer,
    )
    monkeypatch.setattr(
        encoder,
        "_extract_emu35_token_grid",
        lambda encode_out, token_height, token_width: torch.tensor([[5]], dtype=torch.int64),
    )

    ctx = ModalityContext(tokenizer=tokenizer, model_config=None, mm_processor_kwargs=mm_processor_kwargs)
    first = encoder.encode_many([image], ctx)
    assert fake_vision_tokenizer.encode_calls == 1

    cache_db_path = encoder._resolve_apertus_image_token_cache_db_path(mm_processor_kwargs)
    assert cache_db_path is not None
    resized_image = encoder._smart_resize(
        image,
        area=256,
        ds_factor=encoder._APERTUS_EMU35_DS_FACTOR,
    )
    cache_key = encoder._build_apertus_image_prompt_cache_key(
        resized_image,
        tokenizer=tokenizer,
        mm_processor_kwargs=mm_processor_kwargs,
        min_pixels=256,
        max_pixels=256,
    )
    assert cache_db_path.exists()
    with sqlite3.connect(cache_db_path) as conn:
        row = conn.execute(
            f"""
            SELECT image_prompt
            FROM {encoder._APERTUS_IMAGE_TOKEN_CACHE_TABLE}
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    assert row is not None
    assert row[0] == first[0].text

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("vision tokenizer should not be loaded on second call")

    monkeypatch.setattr(encoder, "_get_apertus_vision_components", _fail_if_called)
    second = encoder.encode_many([image], ctx)
    assert second == first


def test_ensure_local_weights_uses_existing_local_checkpoint(tmp_path):
    (tmp_path / "config.yaml").write_text("cfg", encoding="utf-8")
    (tmp_path / "model.ckpt").write_text("ckpt", encoding="utf-8")

    resolved = ensure_local_weights(str(tmp_path), "BAAI/Emu3.5-VisionTokenizer")
    assert resolved == str(tmp_path)


def test_load_vision_tokenizer_uses_build_vision_tokenizer(monkeypatch):
    encoder = ImageModalityEncoder()
    calls = {}

    class _FakeVisionTokenizer:
        def __init__(self):
            self.to_calls = []

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            return self

    fake_vision_tokenizer = _FakeVisionTokenizer()

    def _fake_builder(**kwargs):
        calls.update(kwargs)
        return fake_vision_tokenizer

    monkeypatch.setattr(
        image_mod,
        "build_emu35_vision_tokenizer",
        _fake_builder,
    )

    output = encoder._load_vision_tokenizer(
        vq_hub="BAAI/Emu3.5-VisionTokenizer",
        device="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    assert output is fake_vision_tokenizer
    assert calls["vq_hub"] == "BAAI/Emu3.5-VisionTokenizer"
    assert calls["default_repo"] == "BAAI/Emu3.5-VisionTokenizer"
    assert calls["device"] == "cuda:0"
    assert calls["vq_type"] == "ibq"
    assert calls["trust_remote_code"] is True
    assert calls["dtype"] == torch.bfloat16
    assert "torch_dtype" not in calls
    assert fake_vision_tokenizer.to_calls[-1][1]["dtype"] == torch.bfloat16


def test_get_apertus_vision_components_caches_loader_result(monkeypatch):
    encoder = ImageModalityEncoder()
    fake_vision_tokenizer = object()
    call_count = {"n": 0}

    def _fake_load_vision_tokenizer(**kwargs):
        call_count["n"] += 1
        assert kwargs["vq_hub"] == "BAAI/Emu3.5-VisionTokenizer"
        assert kwargs["type"] == "ibq"
        return fake_vision_tokenizer

    monkeypatch.setattr(encoder, "_load_vision_tokenizer", _fake_load_vision_tokenizer)

    mm_kwargs = {
        "apertus_vq_hub": "BAAI/Emu3.5-VisionTokenizer",
        "apertus_vision_tokenizer_device": "cpu",
        "apertus_vision_tokenizer_dtype": "float32",
    }
    first = encoder._get_apertus_vision_components(mm_kwargs)
    second = encoder._get_apertus_vision_components(mm_kwargs)

    assert first is fake_vision_tokenizer
    assert second is fake_vision_tokenizer
    assert call_count["n"] == 1


def test_process_apertus_prompt_dumps_merged_prompt_and_tokens(monkeypatch, tmp_path):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace()
    preprocessor.model_config = None
    preprocessor._modality_encoders = {"image": _FakeImageEncoder()}

    monkeypatch.setattr(
        preprocessor,
        "_tokenize_prompt",
        lambda prompt_text, tokenization_kwargs=None: [101, 102, 103],
    )

    dump_path = tmp_path / "apertus_prompts.jsonl"
    inputs = ApertusOmniInputPreprocessor._process_apertus_prompt(
        preprocessor,
        prompt_text="hello <|image|>",
        multi_modal_data={"image": [object()]},
        mm_processor_kwargs={"apertus_prompt_dump_path": str(dump_path)},
        tokenization_kwargs=None,
    )

    assert inputs["prompt_token_ids"] == [101, 102, 103]
    lines = dump_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["merged_prompt"] == "hello <img_prompt>"
    assert payload["prompt_token_ids"] == [101, 102, 103]
    assert payload["num_prompt_tokens"] == 3
    assert payload["num_images"] == 1
    assert payload["truncated"] is False
