from __future__ import annotations

from vllm_omni.inputs.apertus.modalities.audio import AudioModalityEncoder
from vllm_omni.inputs.apertus.modalities.base import ApertusModalityEncoder
from vllm_omni.inputs.apertus.modalities.image import ImageModalityEncoder


def create_default_apertus_modality_encoders() -> dict[str, ApertusModalityEncoder]:
    return {
        "image": ImageModalityEncoder(),
        "audio": AudioModalityEncoder(),
    }
