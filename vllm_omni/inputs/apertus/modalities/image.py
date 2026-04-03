from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.inputs.apertus.modalities.base import ApertusModalityEncoder
from vllm_omni.inputs.apertus.types import ModalityContext, PromptPiece, StringPiece
from vllm_omni.inputs.apertus_image_token_cache import (
    ApertusImageTokenCacheConfig,
    ApertusImageTokenSQLiteCache,
)
from vllm_omni.inputs.apertus_utils import build_emu35_vision_tokenizer

logger = init_logger(__name__)


class ImageModalityEncoder(ApertusModalityEncoder):
    modality = "image"

    _APERTUS_DEFAULT_VQ_HUB = "BAAI/Emu3.5-VisionTokenizer"
    _APERTUS_DEFAULT_MIN_PIXELS = 256 * 256
    _APERTUS_DEFAULT_MAX_PIXELS = 1400 * 1400
    _APERTUS_DEFAULT_IMAGE_PLACEHOLDER = "<|image|>"
    _APERTUS_VISUAL_TEMPLATE = "<|visual token {token_id}|>"
    _APERTUS_EMU35_DS_FACTOR = 16
    _APERTUS_DEFAULT_BOS_TOKEN = "<|bos|>"
    _APERTUS_DEFAULT_BOI_TOKEN = "<|img_start|>"
    _APERTUS_DEFAULT_IMG_TOKEN = "<|img_token_start|>"
    _APERTUS_DEFAULT_EOL_TOKEN = "<|img_end_of_row|>"
    _APERTUS_DEFAULT_EOF_TOKEN = "<|img_end_of_frame|>"
    _APERTUS_DEFAULT_EOI_TOKEN = "<|img_end|>"
    _APERTUS_IMAGE_TOKEN_CACHE_VERSION = 1
    _APERTUS_IMAGE_TOKEN_CACHE_ENV_VAR = "VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DIR"
    _APERTUS_IMAGE_TOKEN_CACHE_DEFAULT_DIR = "/iopsstor/scratch/cscs/$USER/swissai/cache/image_tokens/"
    _APERTUS_IMAGE_TOKEN_CACHE_DB_FILENAME = "apertus_image_tokens.sqlite3"
    _APERTUS_IMAGE_TOKEN_CACHE_TABLE = "apertus_image_prompt_cache"

    def __init__(self) -> None:
        self._apertus_vision_encoder_cache: dict[tuple[str, str, str, torch.dtype, bool], Any] = {}
        self._apertus_image_token_cache_db_path: Path | None = None
        self._apertus_image_token_cache: ApertusImageTokenSQLiteCache | None = None

    def __del__(self):
        self._close_apertus_image_token_cache_connection()

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_dtype(value: Any) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if value is None:
            return torch.bfloat16

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(str(value).lower().strip(), torch.bfloat16)

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
    def _smart_resize(image: Image.Image, area: int, ds_factor: int) -> Image.Image:
        width, height = image.size
        aspect_ratio = width / height
        new_height = int((area / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor
        new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor
        return image.resize((new_width, new_height), Image.BICUBIC)

    @staticmethod
    def _extract_emu35_token_grid(
        encode_out: Any,
        token_height: int,
        token_width: int,
    ) -> torch.Tensor:
        def _unwrap_token_payload(payload: Any) -> Any:
            token = payload
            if isinstance(token, tuple):
                token = token[2] if len(token) >= 3 else token[-1]

            while isinstance(token, (list, tuple)):
                if not token:
                    raise ValueError("Apertus emu3.5 encoding produced an empty token sequence.")
                non_none = [item for item in token if item is not None]
                if not non_none:
                    raise ValueError("Apertus emu3.5 encoding produced only None token entries.")
                token = non_none[-1]

            if isinstance(token, Mapping):
                for key in ("token_ids", "indices", "codes", "tokens"):
                    value = token.get(key)
                    if value is not None:
                        return _unwrap_token_payload(value)
                non_none_values = [value for value in token.values() if value is not None]
                if not non_none_values:
                    raise ValueError("Apertus emu3.5 encoding produced an empty token mapping.")
                token = non_none_values[-1]

            return token

        token = _unwrap_token_payload(encode_out)
        if not isinstance(token, torch.Tensor):
            token = torch.tensor(token)

        while token.ndim > 2:
            token = token[0] if token.shape[0] == 1 else token[-1]

        if token.ndim == 1:
            expected = token_height * token_width
            if token.numel() != expected:
                raise ValueError(
                    "Apertus emu3.5 token length mismatch: "
                    f"got {token.numel()}, expected {expected}."
                )
            token = token.view(token_height, token_width)
        elif token.ndim == 2:
            if token.shape == (token_height, token_width):
                pass
            elif token.numel() == token_height * token_width:
                token = token.reshape(token_height, token_width)
            else:
                raise ValueError(
                    "Apertus emu3.5 token grid shape mismatch: "
                    f"got {tuple(token.shape)}, expected {(token_height, token_width)}."
                )
        else:
            raise ValueError(f"Unexpected emu3.5 token rank: {token.ndim}.")

        return token.to(dtype=torch.int64)

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

    def _load_vision_tokenizer(self, vq_hub: str, device: str, **kwargs: Any) -> Any:
        if "torch_dtype" in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")

        vq_type = str(kwargs.pop("type", kwargs.pop("vq_type", "ibq")))
        vision_tokenizer = build_emu35_vision_tokenizer(
            vq_hub=vq_hub,
            default_repo=self._APERTUS_DEFAULT_VQ_HUB,
            device=device,
            vq_type=vq_type,
            **kwargs,
        )

        dtype = kwargs.get("dtype")
        if isinstance(dtype, torch.dtype):
            vision_tokenizer = vision_tokenizer.to(dtype=dtype)

        return vision_tokenizer

    def _get_apertus_vision_components(self, mm_processor_kwargs: Mapping[str, Any]) -> Any:
        vq_hub = str(
            mm_processor_kwargs.get(
                "apertus_vq_hub",
                mm_processor_kwargs.get("vq_hub", self._APERTUS_DEFAULT_VQ_HUB),
            )
        )
        vq_type = str(
            mm_processor_kwargs.get(
                "apertus_vq_type",
                mm_processor_kwargs.get("vq_type", "ibq"),
            )
        )
        vision_device = str(mm_processor_kwargs.get("apertus_vision_tokenizer_device", "cuda"))
        vision_dtype = self._coerce_dtype(mm_processor_kwargs.get("apertus_vision_tokenizer_dtype"))
        if vision_device == "cpu" and vision_dtype in (torch.float16, torch.bfloat16):
            vision_dtype = torch.float32
        trust_remote_code = bool(mm_processor_kwargs.get("apertus_vq_trust_remote_code", True))

        cache_key = (vq_hub, vq_type, vision_device, vision_dtype, trust_remote_code)
        if cache_key in self._apertus_vision_encoder_cache:
            return self._apertus_vision_encoder_cache[cache_key]

        vision_tokenizer = self._load_vision_tokenizer(
            vq_hub=vq_hub,
            device=vision_device,
            type=vq_type,
            torch_dtype=vision_dtype,
            trust_remote_code=trust_remote_code,
        )
        self._apertus_vision_encoder_cache[cache_key] = vision_tokenizer
        return vision_tokenizer

    @staticmethod
    def _apertus_special_token(tokenizer: Any, attr_name: str, fallback: str) -> str:
        token = getattr(tokenizer, attr_name, None)
        return token if isinstance(token, str) and token else fallback

    def _resolve_apertus_image_token_cache_dir(self, mm_processor_kwargs: Mapping[str, Any]) -> Path | None:
        cache_enabled = self._coerce_bool(
            mm_processor_kwargs.get(
                "apertus_image_token_cache",
                mm_processor_kwargs.get("apertus_enable_image_token_cache", True),
            ),
            default=True,
        )
        if not cache_enabled:
            return None

        cache_dir_value = mm_processor_kwargs.get(
            "apertus_image_token_cache_dir",
            os.getenv(self._APERTUS_IMAGE_TOKEN_CACHE_ENV_VAR, self._APERTUS_IMAGE_TOKEN_CACHE_DEFAULT_DIR),
        )
        if not isinstance(cache_dir_value, str) or not cache_dir_value.strip():
            return None

        cache_dir = Path(os.path.expandvars(cache_dir_value.strip())).expanduser()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create Apertus image token cache directory %s: %s", cache_dir, exc)
            return None
        return cache_dir

    def _resolve_apertus_image_token_cache_db_path(self, mm_processor_kwargs: Mapping[str, Any]) -> Path | None:
        cache_enabled = self._coerce_bool(
            mm_processor_kwargs.get(
                "apertus_image_token_cache",
                mm_processor_kwargs.get("apertus_enable_image_token_cache", True),
            ),
            default=True,
        )
        if not cache_enabled:
            return None

        configured_db_path = mm_processor_kwargs.get("apertus_image_token_cache_db_path")
        if isinstance(configured_db_path, str) and configured_db_path.strip():
            return Path(os.path.expandvars(configured_db_path.strip())).expanduser()

        cache_dir = self._resolve_apertus_image_token_cache_dir(mm_processor_kwargs)
        if cache_dir is None:
            return None
        return cache_dir / self._APERTUS_IMAGE_TOKEN_CACHE_DB_FILENAME

    def _build_apertus_image_prompt_cache_key(
        self,
        resized_image: Image.Image,
        *,
        tokenizer: Any,
        mm_processor_kwargs: Mapping[str, Any],
        min_pixels: int,
        max_pixels: int,
    ) -> str:
        vq_hub = str(
            mm_processor_kwargs.get(
                "apertus_vq_hub",
                mm_processor_kwargs.get("vq_hub", self._APERTUS_DEFAULT_VQ_HUB),
            )
        )
        vq_type = str(
            mm_processor_kwargs.get(
                "apertus_vq_type",
                mm_processor_kwargs.get("vq_type", "ibq"),
            )
        )
        vision_device = str(mm_processor_kwargs.get("apertus_vision_tokenizer_device", "cpu"))
        vision_dtype = self._coerce_dtype(mm_processor_kwargs.get("apertus_vision_tokenizer_dtype"))
        if vision_device == "cpu" and vision_dtype in (torch.float16, torch.bfloat16):
            vision_dtype = torch.float32
        trust_remote_code = bool(mm_processor_kwargs.get("apertus_vq_trust_remote_code", True))
        hash_payload = {
            "cache_version": self._APERTUS_IMAGE_TOKEN_CACHE_VERSION,
            "vq_hub": vq_hub,
            "vq_type": vq_type,
            "vision_device": "cuda",
            "vision_dtype": str(vision_dtype),
            "trust_remote_code": trust_remote_code,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "ds_factor": self._APERTUS_EMU35_DS_FACTOR,
            "visual_template": self._APERTUS_VISUAL_TEMPLATE,
            "bos_token": self._apertus_special_token(tokenizer, "bos_token", self._APERTUS_DEFAULT_BOS_TOKEN),
            "boi_token": self._apertus_special_token(tokenizer, "boi_token", self._APERTUS_DEFAULT_BOI_TOKEN),
            "img_token": self._apertus_special_token(tokenizer, "img_token", self._APERTUS_DEFAULT_IMG_TOKEN),
            "eol_token": self._apertus_special_token(tokenizer, "eol_token", self._APERTUS_DEFAULT_EOL_TOKEN),
            "eof_token": self._apertus_special_token(tokenizer, "eof_token", self._APERTUS_DEFAULT_EOF_TOKEN),
            "eoi_token": self._apertus_special_token(tokenizer, "eoi_token", self._APERTUS_DEFAULT_EOI_TOKEN),
            "resized_mode": resized_image.mode,
            "resized_size": resized_image.size,
        }
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(
                hash_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        hasher.update(resized_image.tobytes())
        return hasher.hexdigest()

    def _close_apertus_image_token_cache_connection(self) -> None:
        cache_client = getattr(self, "_apertus_image_token_cache", None)
        if cache_client is not None:
            cache_client.close()
        self._apertus_image_token_cache = None
        self._apertus_image_token_cache_db_path = None

    def _get_apertus_image_token_cache(
        self,
        cache_db_path: Path,
        mm_processor_kwargs: Mapping[str, Any] | None = None,
    ) -> ApertusImageTokenSQLiteCache | None:
        cached_path = getattr(self, "_apertus_image_token_cache_db_path", None)
        cached_client = getattr(self, "_apertus_image_token_cache", None)
        if cached_client is not None and cached_path == cache_db_path:
            return cached_client

        if cached_client is not None:
            self._close_apertus_image_token_cache_connection()

        cache_kwargs = mm_processor_kwargs or {}
        try:
            cache_config = ApertusImageTokenCacheConfig.from_mm_processor_kwargs(cache_kwargs)
            cache_client = ApertusImageTokenSQLiteCache(
                cache_db_path=cache_db_path,
                table_name=self._APERTUS_IMAGE_TOKEN_CACHE_TABLE,
                config=cache_config,
            )
        except Exception as exc:
            logger.warning("Failed initializing Apertus image token SQLite cache %s: %s", cache_db_path, exc)
            return None

        self._apertus_image_token_cache_db_path = cache_db_path
        self._apertus_image_token_cache = cache_client
        return cache_client

    def _load_apertus_image_prompt_from_cache(
        self,
        cache_db_path: Path,
        cache_key: str,
        mm_processor_kwargs: Mapping[str, Any] | None = None,
    ) -> str | None:
        cache_client = self._get_apertus_image_token_cache(cache_db_path, mm_processor_kwargs=mm_processor_kwargs)
        if cache_client is None:
            return None
        return cache_client.get(cache_key)

    def _store_apertus_image_prompt_in_cache(
        self,
        cache_db_path: Path,
        cache_key: str,
        image_prompt: str,
        mm_processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        cache_client = self._get_apertus_image_token_cache(cache_db_path, mm_processor_kwargs=mm_processor_kwargs)
        if cache_client is None:
            return
        cache_client.put(cache_key, image_prompt)

    def _build_apertus_image_prompt(self, image_tokens: torch.Tensor, tokenizer: Any) -> str:
        if image_tokens.ndim != 2:
            raise ValueError(f"Apertus image tokens must be 2D, got shape {tuple(image_tokens.shape)}")

        height, width = image_tokens.shape
        rows = [
            "".join(self._APERTUS_VISUAL_TEMPLATE.format(token_id=int(token_id)) for token_id in row)
            for row in image_tokens.detach().to("cpu").tolist()
        ]
        eol_token = self._apertus_special_token(tokenizer, "eol_token", self._APERTUS_DEFAULT_EOL_TOKEN)
        imgstr = eol_token.join(rows)

        boi_token = self._apertus_special_token(tokenizer, "boi_token", self._APERTUS_DEFAULT_BOI_TOKEN)
        img_token = self._apertus_special_token(tokenizer, "img_token", self._APERTUS_DEFAULT_IMG_TOKEN)
        eoi_token = self._apertus_special_token(tokenizer, "eoi_token", self._APERTUS_DEFAULT_EOI_TOKEN)

        return f"{boi_token}{height}*{width}{img_token}{imgstr}{eoi_token}"

    def placeholder_aliases(self, prompt_text: str, ctx: ModalityContext) -> list[str]:
        del prompt_text
        configured_placeholder = ctx.mm_processor_kwargs.get("apertus_image_placeholder")
        tokenizer_placeholder = getattr(ctx.tokenizer, "image_token", None)
        return self._dedupe(
            [
                configured_placeholder if isinstance(configured_placeholder, str) else "",
                tokenizer_placeholder if isinstance(tokenizer_placeholder, str) else "",
                self._APERTUS_DEFAULT_IMAGE_PLACEHOLDER,
                "<image>",
            ]
        )

    def normalize_inputs(self, raw_input: Any, ctx: ModalityContext) -> list[Any]:
        del ctx
        if raw_input is None:
            return []

        images = raw_input if isinstance(raw_input, list) else [raw_input]
        if not images:
            return []

        normalized: list[Image.Image] = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("Apertus image adapter expects PIL images in multi_modal_data['image'].")
            normalized.append(image.convert("RGB"))
        return normalized

    def encode_many(self, items: Sequence[Any], ctx: ModalityContext) -> list[PromptPiece]:
        if not items:
            return []

        mm_processor_kwargs = ctx.mm_processor_kwargs
        tokenizer = ctx.tokenizer
        min_pixels = self._coerce_int(
            mm_processor_kwargs.get(
                "apertus_min_pixels",
                mm_processor_kwargs.get("emu_min_pixels", self._APERTUS_DEFAULT_MIN_PIXELS),
            ),
            default=self._APERTUS_DEFAULT_MIN_PIXELS,
        )
        max_pixels = self._coerce_int(
            mm_processor_kwargs.get(
                "apertus_max_pixels",
                mm_processor_kwargs.get("emu_max_pixels", self._APERTUS_DEFAULT_MAX_PIXELS),
            ),
            default=self._APERTUS_DEFAULT_MAX_PIXELS,
        )
        cache_db_path = self._resolve_apertus_image_token_cache_db_path(mm_processor_kwargs)
        vision_tokenizer = None
        vision_device = None
        vision_dtype = None
        image_pieces: list[PromptPiece] = []

        for image in items:
            width, height = image.size
            current_area = width * height
            target_area = max(min(max_pixels, current_area), min_pixels)
            resized_image = self._smart_resize(image, target_area, self._APERTUS_EMU35_DS_FACTOR)
            resized_w, resized_h = resized_image.size

            cache_key: str | None = None
            if cache_db_path is not None:
                cache_key = self._build_apertus_image_prompt_cache_key(
                    resized_image,
                    tokenizer=tokenizer,
                    mm_processor_kwargs=mm_processor_kwargs,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                cached_prompt = self._load_apertus_image_prompt_from_cache(
                    cache_db_path,
                    cache_key,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                if cached_prompt:
                    image_pieces.append(StringPiece(cached_prompt))
                    continue

            if vision_tokenizer is None:
                vision_tokenizer = self._get_apertus_vision_components(mm_processor_kwargs)
                vision_params = next(vision_tokenizer.parameters())
                vision_device = vision_params.device
                vision_dtype = vision_params.dtype

            image_tensor = torch.tensor((np.array(resized_image) / 127.5 - 1.0)).to(
                device=vision_device,
                dtype=vision_dtype,
            ).permute(2, 0, 1)
            with torch.inference_mode():
                try:
                    encode_out = vision_tokenizer.encode(image_tensor[None])
                except TypeError:
                    try:
                        encode_out = vision_tokenizer.encode(pixel_values=image_tensor[None])
                    except TypeError:
                        encode_out = vision_tokenizer.encode(images=image_tensor[None])

            token_h = resized_h // self._APERTUS_EMU35_DS_FACTOR
            token_w = resized_w // self._APERTUS_EMU35_DS_FACTOR
            image_token_grid = self._extract_emu35_token_grid(encode_out, token_h, token_w)
            image_prompt = self._build_apertus_image_prompt(image_token_grid, tokenizer)
            if cache_db_path is not None and cache_key is not None:
                self._store_apertus_image_prompt_in_cache(
                    cache_db_path,
                    cache_key,
                    image_prompt,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            image_pieces.append(StringPiece(image_prompt))

        return image_pieces
