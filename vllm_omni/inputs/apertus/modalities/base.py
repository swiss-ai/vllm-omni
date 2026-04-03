from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from vllm_omni.inputs.apertus.types import ModalityContext, PromptPiece


class ApertusModalityEncoder(ABC):
    modality: str

    @abstractmethod
    def placeholder_aliases(self, prompt_text: str, ctx: ModalityContext) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def normalize_inputs(self, raw_input: Any, ctx: ModalityContext) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def encode_many(self, items: Sequence[Any], ctx: ModalityContext) -> list[PromptPiece]:
        raise NotImplementedError
