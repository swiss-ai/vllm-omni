from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torchaudio
from vllm.logger import init_logger

from vllm_omni.inputs.apertus.modalities.base import ApertusModalityEncoder
from vllm_omni.inputs.apertus.types import ModalityContext, PromptPiece, TokenPiece
from vllm_omni.inputs.apertus_utils import build_apertus_wavtokenizer

logger = init_logger(__name__)

try:
    import soundfile as sf
except ImportError:
    sf = None


class AudioModalityEncoder(ApertusModalityEncoder):
    modality = "audio"

    _APERTUS_DEFAULT_AUDIO_PLACEHOLDER = "<|audio|>"
    _APERTUS_DEFAULT_AUDIO_TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/wavtokenizer"
    _APERTUS_DEFAULT_AUDIO_TOKENIZER_TYPE = "wavtokenizer"
    _APERTUS_DEFAULT_AUDIO_TOKENIZER_NAME = "WavTokenizer40"
    _APERTUS_DEFAULT_AUDIO_TOKENIZER_DEVICE = "cuda"
    _APERTUS_DEFAULT_AUDIO_TARGET_SAMPLING_RATE = 24000
    _APERTUS_DEFAULT_AUDIO_DEFAULT_SAMPLING_RATE = 16000
    _APERTUS_DEFAULT_AUDIO_TOKEN_OFFSET = 262344
    _APERTUS_DEFAULT_AUDIO_VOCAB_SIZE = 4096
    _APERTUS_AUDIO_START_TOKEN = "<|audio_start|>"
    _APERTUS_AUDIO_END_TOKEN = "<|audio_end|>"

    def __init__(self) -> None:
        self._audio_tokenizer_cache: dict[tuple[str, str, bool, str, str], Any] = {}
        self._special_token_cache: dict[int, dict[str, int]] = {}

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            value_str = value.strip().lower()
            if value_str in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if value_str in {"0", "false", "f", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _coerce_audio_waveform(audio_obj: Any) -> np.ndarray:
        if isinstance(audio_obj, np.ndarray):
            audio = audio_obj
        elif torch.is_tensor(audio_obj):
            audio = audio_obj.detach().cpu().numpy()
        else:
            audio = np.asarray(audio_obj)

        if audio.ndim == 0:
            raise ValueError("Audio waveform must have at least one dimension.")
        if audio.ndim == 1:
            return audio.astype(np.float32, copy=False)
        if audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            elif audio.shape[1] == 1:
                audio = audio[:, 0]
            elif audio.shape[0] <= audio.shape[1]:
                audio = np.mean(audio, axis=0)
            else:
                audio = np.mean(audio, axis=1)
            return audio.astype(np.float32, copy=False)

        audio = np.reshape(audio, (-1,))
        return audio.astype(np.float32, copy=False)

    @staticmethod
    def _dedupe(items: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @staticmethod
    def _to_audio_tensor(waveform: np.ndarray) -> torch.Tensor:
        audio_tensor = torch.from_numpy(waveform).float()
        if audio_tensor.dim() == 1:
            return audio_tensor.unsqueeze(0)
        if audio_tensor.dim() == 2 and audio_tensor.shape[0] == 1:
            return audio_tensor
        return audio_tensor.reshape(1, -1)

    def _read_audio_from_path(self, path: str) -> tuple[np.ndarray, int]:
        if sf is not None:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            return self._coerce_audio_waveform(audio), int(sr)

        audio_tensor, sr = torchaudio.load(path)
        return self._coerce_audio_waveform(audio_tensor), int(sr)

    def _normalize_one_audio(self, item: Any, ctx: ModalityContext) -> tuple[np.ndarray, int]:
        if isinstance(item, str):
            return self._read_audio_from_path(item)

        if isinstance(item, dict):
            if "array" in item:
                sr = item.get("sampling_rate", item.get("sample_rate"))
                if sr is None:
                    raise ValueError("Decoded audio dict must include `sampling_rate`.")
                return self._coerce_audio_waveform(item["array"]), int(sr)

            if "audio_array" in item:
                sr = item.get(
                    "sr",
                    item.get("sampling_rate", ctx.mm_processor_kwargs.get("apertus_audio_default_sampling_rate")),
                )
                if sr is None:
                    sr = self._APERTUS_DEFAULT_AUDIO_DEFAULT_SAMPLING_RATE
                return self._coerce_audio_waveform(item["audio_array"]), int(sr)

            if "bytes" in item:
                if sf is None:
                    raise ImportError("soundfile is required to decode byte-backed audio inputs.")
                audio_bytes = item["bytes"]
                if not isinstance(audio_bytes, (bytes, bytearray)):
                    raise TypeError(f"Unsupported audio byte payload type: {type(audio_bytes)}")
                audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
                sr = item.get("sampling_rate", item.get("sample_rate", sr))
                return self._coerce_audio_waveform(audio), int(sr)

            if "path" in item:
                return self._read_audio_from_path(str(item["path"]))

            raise TypeError(f"Unsupported audio dict keys for Apertus audio adapter: {sorted(item.keys())}")

        if isinstance(item, tuple) and len(item) == 2:
            return self._coerce_audio_waveform(item[0]), int(item[1])

        if isinstance(item, list) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in item):
            return self._coerce_audio_waveform(item), int(
                ctx.mm_processor_kwargs.get(
                    "apertus_audio_default_sampling_rate",
                    self._APERTUS_DEFAULT_AUDIO_DEFAULT_SAMPLING_RATE,
                )
            )

        if isinstance(item, np.ndarray) or torch.is_tensor(item):
            return self._coerce_audio_waveform(item), int(
                ctx.mm_processor_kwargs.get(
                    "apertus_audio_default_sampling_rate",
                    self._APERTUS_DEFAULT_AUDIO_DEFAULT_SAMPLING_RATE,
                )
            )

        raise TypeError(f"Unsupported audio type for Apertus audio adapter: {type(item)}")

    def _get_audio_special_token_ids(self, tokenizer: Any) -> dict[str, int]:
        cache_key = id(tokenizer)
        cached = self._special_token_cache.get(cache_key)
        if cached is not None:
            return cached

        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            raise AttributeError("Tokenizer must expose convert_tokens_to_ids for Apertus audio prompts.")

        token_ids: dict[str, int] = {}
        for name, token_str in {
            "audio_start": self._APERTUS_AUDIO_START_TOKEN,
            "audio_end": self._APERTUS_AUDIO_END_TOKEN,
        }.items():
            token_id = convert(token_str)
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
                raise ValueError(f"Token {token_str} not found in tokenizer vocabulary.")
            token_ids[name] = int(token_id)

        self._special_token_cache[cache_key] = token_ids
        return token_ids

    def _get_audio_tokenizer(self, ctx: ModalityContext) -> Any:
        mm_processor_kwargs = ctx.mm_processor_kwargs
        tokenizer_path = str(
            mm_processor_kwargs.get(
                "apertus_audio_tokenizer_path",
                self._APERTUS_DEFAULT_AUDIO_TOKENIZER_PATH,
            )
        )
        tokenizer_type = str(
            mm_processor_kwargs.get(
                "apertus_audio_tokenizer_type",
                self._APERTUS_DEFAULT_AUDIO_TOKENIZER_TYPE,
            )
        ).lower()
        tokenizer_name = str(
            mm_processor_kwargs.get(
                "apertus_audio_tokenizer_name",
                self._APERTUS_DEFAULT_AUDIO_TOKENIZER_NAME,
            )
        )
        tokenizer_device = str(
            mm_processor_kwargs.get(
                "apertus_audio_tokenizer_device",
                self._APERTUS_DEFAULT_AUDIO_TOKENIZER_DEVICE,
            )
        )
        tokenizer_compile = self._coerce_bool(
            mm_processor_kwargs.get("apertus_audio_tokenizer_compile"),
            default=False,
        )
        tokenizer_codebase = mm_processor_kwargs.get("apertus_audio_tokenizer_codebase")
        if tokenizer_codebase is not None:
            tokenizer_codebase = str(tokenizer_codebase)

        cache_key = (
            tokenizer_path,
            tokenizer_device,
            tokenizer_compile,
            tokenizer_type,
            tokenizer_name,
            tokenizer_codebase or "",
        )
        if cache_key in self._audio_tokenizer_cache:
            return self._audio_tokenizer_cache[cache_key]

        if tokenizer_type != "wavtokenizer" or tokenizer_name != "WavTokenizer40":
            raise ValueError(
                "Apertus audio adapter currently supports "
                "audio_tokenizer_type=wavtokenizer and audio_tokenizer_name=WavTokenizer40 only."
            )

        audio_tokenizer = build_apertus_wavtokenizer(
            checkpoint_path=tokenizer_path,
            codebase_path=tokenizer_codebase,
            device=tokenizer_device,
            torch_compile=tokenizer_compile,
        )
        self._audio_tokenizer_cache[cache_key] = audio_tokenizer
        return audio_tokenizer

    def placeholder_aliases(self, prompt_text: str, ctx: ModalityContext) -> list[str]:
        del prompt_text
        configured_placeholder = ctx.mm_processor_kwargs.get("apertus_audio_placeholder")
        return self._dedupe(
            [
                configured_placeholder if isinstance(configured_placeholder, str) else "",
                self._APERTUS_DEFAULT_AUDIO_PLACEHOLDER,
            ]
        )

    def normalize_inputs(self, raw_input: Any, ctx: ModalityContext) -> list[Any]:
        if raw_input is None:
            return []

        items = raw_input if isinstance(raw_input, list) else [raw_input]
        if not items:
            return []

        return [self._normalize_one_audio(item, ctx) for item in items]

    def encode_many(self, items: Sequence[Any], ctx: ModalityContext) -> list[PromptPiece]:
        if not items:
            return []

        target_sr = int(
            ctx.mm_processor_kwargs.get(
                "apertus_audio_target_sampling_rate",
                self._APERTUS_DEFAULT_AUDIO_TARGET_SAMPLING_RATE,
            )
        )
        token_offset = int(
            ctx.mm_processor_kwargs.get(
                "apertus_audio_token_offset",
                self._APERTUS_DEFAULT_AUDIO_TOKEN_OFFSET,
            )
        )
        audio_tokenizer = self._get_audio_tokenizer(ctx)
        special_token_ids = self._get_audio_special_token_ids(ctx.tokenizer)

        audio_pieces: list[PromptPiece] = []
        for waveform, sr in items:
            audio_tensor = self._to_audio_tensor(waveform)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                audio_tensor = resampler(audio_tensor)

            with torch.no_grad():
                audio_codes = audio_tokenizer.encode_audio(audio_tensor)

            if audio_codes.dim() == 2:
                audio_codes = audio_codes.squeeze(0)

            shifted_codes = audio_codes.detach().to(device="cpu", dtype=torch.int64) + token_offset
            token_ids = [special_token_ids["audio_start"]]
            token_ids.extend(shifted_codes.tolist())
            token_ids.append(special_token_ids["audio_end"])
            audio_pieces.append(TokenPiece(token_ids))

        return audio_pieces
