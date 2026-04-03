from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from vllm_omni.inputs.apertus.types import (
    PlaceholderSegment,
    PromptPiece,
    PromptSegment,
    StringPiece,
    TextSegment,
    TokenPiece,
)


def assemble_prompt_token_ids(
    *,
    segments: Sequence[PromptSegment],
    encoded_pieces_by_modality: Mapping[str, Sequence[PromptPiece]],
    tokenize_string: Callable[[str], list[int]],
) -> list[int]:
    prompt_token_ids: list[int] = []
    string_buffer: list[str] = []

    def flush_string_buffer() -> None:
        if not string_buffer:
            return
        buffered = "".join(string_buffer)
        string_buffer.clear()
        if buffered:
            prompt_token_ids.extend(tokenize_string(buffered))

    for segment in segments:
        if isinstance(segment, TextSegment):
            if segment.text:
                string_buffer.append(segment.text)
            continue

        if not isinstance(segment, PlaceholderSegment):
            raise TypeError(f"Unsupported prompt segment type: {type(segment)}")

        pieces = encoded_pieces_by_modality.get(segment.modality)
        if pieces is None or segment.ordinal >= len(pieces):
            raise ValueError(
                f"Prompt references placeholder '{segment.placeholder}' for modality "
                f"{segment.modality} at index {segment.ordinal}, but no encoded payload is available."
            )

        piece = pieces[segment.ordinal]
        if isinstance(piece, StringPiece):
            string_buffer.append(piece.text)
        elif isinstance(piece, TokenPiece):
            flush_string_buffer()
            prompt_token_ids.extend(piece.token_ids)
        else:
            raise TypeError(f"Unsupported prompt piece type: {type(piece)}")

    flush_string_buffer()
    return prompt_token_ids
