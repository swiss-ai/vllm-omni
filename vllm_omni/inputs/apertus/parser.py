from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from vllm_omni.inputs.apertus.types import PlaceholderSegment, PromptSegment, TextSegment


def parse_prompt_segments(
    prompt_text: str,
    placeholders_by_modality: Mapping[str, Sequence[str]],
) -> list[PromptSegment]:
    alias_to_modality: dict[str, str] = {}
    for modality, aliases in placeholders_by_modality.items():
        for alias in aliases:
            if not alias:
                continue
            existing = alias_to_modality.get(alias)
            if existing is not None and existing != modality:
                raise ValueError(
                    f"Placeholder '{alias}' is assigned to multiple modalities: "
                    f"{existing} and {modality}."
                )
            alias_to_modality[alias] = modality

    if not alias_to_modality:
        return [TextSegment(prompt_text)]

    pattern = re.compile("|".join(re.escape(alias) for alias in sorted(alias_to_modality, key=len, reverse=True)))
    segments: list[PromptSegment] = []
    ordinals: dict[str, int] = {modality: 0 for modality in placeholders_by_modality}
    cursor = 0

    for match in pattern.finditer(prompt_text):
        start, end = match.span()
        if start > cursor:
            segments.append(TextSegment(prompt_text[cursor:start]))

        placeholder = match.group(0)
        modality = alias_to_modality[placeholder]
        segments.append(
            PlaceholderSegment(
                modality=modality,
                placeholder=placeholder,
                ordinal=ordinals[modality],
            )
        )
        ordinals[modality] += 1
        cursor = end

    if cursor < len(prompt_text):
        segments.append(TextSegment(prompt_text[cursor:]))

    if not segments:
        segments.append(TextSegment(""))

    return segments
