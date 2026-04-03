from vllm_omni.inputs.apertus.assembler import assemble_prompt_token_ids
from vllm_omni.inputs.apertus.parser import parse_prompt_segments
from vllm_omni.inputs.apertus.types import StringPiece, TokenPiece


def _fake_tokenize(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_parse_prompt_segments_tracks_modalities_in_prompt_order():
    segments = parse_prompt_segments(
        "A<|image|>B<|audio|>C<|image|>",
        {
            "image": ["<|image|>"],
            "audio": ["<|audio|>"],
        },
    )

    assert [type(segment).__name__ for segment in segments] == [
        "TextSegment",
        "PlaceholderSegment",
        "TextSegment",
        "PlaceholderSegment",
        "TextSegment",
        "PlaceholderSegment",
    ]
    assert segments[1].modality == "image"
    assert segments[1].ordinal == 0
    assert segments[3].modality == "audio"
    assert segments[3].ordinal == 0
    assert segments[5].modality == "image"
    assert segments[5].ordinal == 1


def test_assemble_prompt_token_ids_flushes_strings_before_audio_tokens():
    segments = parse_prompt_segments(
        "A<|image|>B<|audio|>C",
        {
            "image": ["<|image|>"],
            "audio": ["<|audio|>"],
        },
    )

    prompt_token_ids = assemble_prompt_token_ids(
        segments=segments,
        encoded_pieces_by_modality={
            "image": [StringPiece("<img>")],
            "audio": [TokenPiece([9001, 9002])],
        },
        tokenize_string=_fake_tokenize,
    )

    assert prompt_token_ids == [ord(ch) for ch in "A<img>B"] + [9001, 9002] + [ord("C")]
