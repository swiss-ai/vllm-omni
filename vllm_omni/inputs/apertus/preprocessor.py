from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from vllm.inputs.data import TextPrompt
from vllm.multimodal.inputs import MultiModalInputs, MultiModalUUIDDict

from vllm_omni.inputs.apertus.assembler import assemble_prompt_token_ids
from vllm_omni.inputs.apertus.parser import parse_prompt_segments
from vllm_omni.inputs.apertus.registry import create_default_apertus_modality_encoders
from vllm_omni.inputs.apertus.types import (
    ModalityContext,
    PlaceholderSegment,
    StringPiece,
    TextSegment,
    TokenPiece,
)
from vllm_omni.inputs.apertus_utils import dump_apertus_prompt_debug
from vllm_omni.inputs.data import OmniTokenInputs, token_inputs_omni
from vllm_omni.inputs.preprocess import OmniInputPreprocessor


def is_apertus_model_config(model_config: Any) -> bool:
    model_arch = getattr(model_config, "model_arch", None)
    if isinstance(model_arch, str) and "ApertusForCausalLM" in model_arch:
        return True

    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return False

    if getattr(hf_config, "model_type", None) == "apertus":
        return True

    architectures = getattr(hf_config, "architectures", None) or []
    return any("ApertusForCausalLM" in arch for arch in architectures)


class ApertusOmniInputPreprocessor(OmniInputPreprocessor):
    """Apertus-specialized multimodal input preprocessor.

    Apertus runs as a token-in/token-out causal LM. This preprocessor turns
    multimodal payloads into a single token stream by:
    - parsing placeholders from the text prompt
    - delegating each modality to its own encoder
    - tokenizing text/image string pieces and splicing in raw audio token ids
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._modality_encoders = create_default_apertus_modality_encoders()

    def remove_system_prompt(self, prompt: str) -> str:
        return prompt

    def _is_apertus_multimodal_input(self, multi_modal_data: Mapping[str, Any]) -> bool:
        supported_modalities = set(self._modality_encoders)
        unsupported_modalities = [k for k, v in multi_modal_data.items() if k not in supported_modalities and v]
        if unsupported_modalities:
            raise ValueError(
                "Apertus Omni adapter currently supports only these extra modalities: "
                f"{sorted(supported_modalities)}. Unsupported modalities: {unsupported_modalities}"
            )

        return any(multi_modal_data.get(modality) for modality in supported_modalities)

    @staticmethod
    def _validate_placeholder_alignment(
        segments: list[Any],
        encoded_pieces_by_modality: Mapping[str, list[Any]],
    ) -> None:
        placeholder_counts = Counter(
            segment.modality for segment in segments if isinstance(segment, PlaceholderSegment)
        )

        for modality, pieces in encoded_pieces_by_modality.items():
            expected = placeholder_counts.get(modality, 0)
            if expected != len(pieces):
                raise ValueError(
                    f"Mismatch for modality '{modality}': found {expected} placeholder(s) in prompt, "
                    f"but received {len(pieces)} input item(s)."
                )

        for modality, count in placeholder_counts.items():
            if modality not in encoded_pieces_by_modality:
                raise ValueError(
                    f"Prompt contains {count} placeholder(s) for modality '{modality}', "
                    "but no corresponding inputs were provided."
                )

    @staticmethod
    def _resolve_placeholders_by_modality(
        *,
        prompt_text: str,
        ctx: ModalityContext,
        normalized_inputs_by_modality: Mapping[str, list[Any]],
        modality_encoders: Mapping[str, Any],
    ) -> dict[str, list[str]]:
        placeholders_by_modality = {
            modality: list(encoder.placeholder_aliases(prompt_text, ctx))
            for modality, encoder in modality_encoders.items()
        }

        alias_to_modalities: dict[str, set[str]] = {}
        for modality, aliases in placeholders_by_modality.items():
            for alias in aliases:
                if alias:
                    alias_to_modalities.setdefault(alias, set()).add(modality)

        active_modalities = {
            modality for modality, items in normalized_inputs_by_modality.items() if items
        }
        if not active_modalities:
            return placeholders_by_modality

        for alias, modalities in alias_to_modalities.items():
            if len(modalities) <= 1:
                continue

            active_claimants = modalities & active_modalities
            if len(active_claimants) != 1:
                continue

            winning_modality = next(iter(active_claimants))
            for modality in modalities:
                if modality == winning_modality:
                    continue
                placeholders_by_modality[modality] = [
                    candidate
                    for candidate in placeholders_by_modality[modality]
                    if candidate != alias
                ]

        return placeholders_by_modality

    @staticmethod
    def _build_debug_merged_prompt(
        *,
        segments: list[Any],
        encoded_pieces_by_modality: Mapping[str, list[Any]],
    ) -> tuple[str, bool]:
        merged_parts: list[str] = []
        truncated = False

        for segment in segments:
            if isinstance(segment, TextSegment):
                merged_parts.append(segment.text)
                continue

            if not isinstance(segment, PlaceholderSegment):
                raise TypeError(f"Unsupported prompt segment type: {type(segment)}")

            piece = encoded_pieces_by_modality[segment.modality][segment.ordinal]
            if isinstance(piece, StringPiece):
                merged_parts.append(piece.text)
            elif isinstance(piece, TokenPiece):
                truncated = True
                merged_parts.append(
                    f"<|{segment.modality}_tokens:{len(piece.token_ids)}|>"
                )
            else:
                raise TypeError(f"Unsupported prompt piece type: {type(piece)}")

        return "".join(merged_parts), truncated

    def _process_apertus_prompt(
        self,
        prompt_text: str,
        multi_modal_data: Mapping[str, Any],
        mm_processor_kwargs: Mapping[str, Any],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> OmniTokenInputs:
        prompt_text = self.remove_system_prompt(prompt_text)
        ctx = ModalityContext(
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        normalized_inputs_by_modality: dict[str, list[Any]] = {}
        for modality, encoder in self._modality_encoders.items():
            normalized_inputs = encoder.normalize_inputs(multi_modal_data.get(modality), ctx)
            if normalized_inputs:
                normalized_inputs_by_modality[modality] = normalized_inputs

        placeholders_by_modality = self._resolve_placeholders_by_modality(
            prompt_text=prompt_text,
            ctx=ctx,
            normalized_inputs_by_modality=normalized_inputs_by_modality,
            modality_encoders=self._modality_encoders,
        )
        segments = parse_prompt_segments(prompt_text, placeholders_by_modality)

        encoded_pieces_by_modality: dict[str, list[Any]] = {}
        for modality, normalized_inputs in normalized_inputs_by_modality.items():
            encoder = self._modality_encoders[modality]
            encoded_pieces_by_modality[modality] = encoder.encode_many(normalized_inputs, ctx)

        self._validate_placeholder_alignment(segments, encoded_pieces_by_modality)

        effective_tokenization_kwargs = dict(tokenization_kwargs or {})
        effective_tokenization_kwargs.setdefault("add_special_tokens", False)
        prompt_token_ids = assemble_prompt_token_ids(
            segments=segments,
            encoded_pieces_by_modality=encoded_pieces_by_modality,
            tokenize_string=lambda text: self._tokenize_prompt(
                text,
                tokenization_kwargs=effective_tokenization_kwargs,
            ),
        )

        merged_prompt, truncated = self._build_debug_merged_prompt(
            segments=segments,
            encoded_pieces_by_modality=encoded_pieces_by_modality,
        )
        dump_apertus_prompt_debug(
            mm_processor_kwargs=mm_processor_kwargs,
            merged_prompt=merged_prompt,
            prompt_token_ids=prompt_token_ids,
            image_count=len(normalized_inputs_by_modality.get("image", [])),
            truncated=truncated,
        )
        return token_inputs_omni(prompt_token_ids)

    def _process_text(
        self,
        parsed_content: TextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> OmniTokenInputs | MultiModalInputs:
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            if isinstance(multi_modal_data, Mapping) and self._is_apertus_multimodal_input(multi_modal_data):
                prompt_text = parsed_content["prompt"]
                mm_processor_kwargs = parsed_content.get("mm_processor_kwargs") or {}
                inputs = self._process_apertus_prompt(
                    prompt_text,
                    multi_modal_data=multi_modal_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                    tokenization_kwargs=tokenization_kwargs,
                )
                prompt_embeds = parsed_content.get("prompt_embeds")
                if prompt_embeds is not None:
                    inputs["prompt_embeds"] = prompt_embeds
                additional_information = parsed_content.get("additional_information")
                if additional_information is not None:
                    inputs["additional_information"] = additional_information
                if cache_salt := parsed_content.get("cache_salt"):
                    inputs["cache_salt"] = cache_salt
                return inputs

        return super()._process_text(
            parsed_content,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
