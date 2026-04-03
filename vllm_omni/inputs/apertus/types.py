from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StringPiece:
    text: str


@dataclass(frozen=True)
class TokenPiece:
    token_ids: list[int]


PromptPiece = StringPiece | TokenPiece


@dataclass(frozen=True)
class TextSegment:
    text: str


@dataclass(frozen=True)
class PlaceholderSegment:
    modality: str
    placeholder: str
    ordinal: int


PromptSegment = TextSegment | PlaceholderSegment


@dataclass(frozen=True)
class ModalityContext:
    tokenizer: Any
    model_config: Any
    mm_processor_kwargs: Mapping[str, Any]
