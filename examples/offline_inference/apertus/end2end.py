# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline end-to-end example for Apertus multimodal inference in vLLM-Omni.

Supported scenarios:
1) text only
2) text + image
3) text + audio
4) text + image + audio
5) all of the above in one run
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torchaudio
except ImportError:
    torchaudio = None


IMAGE_PLACEHOLDER = "<|image|>"
AUDIO_PLACEHOLDER = "<|audio|>"
DEFAULT_AUDIO_TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/wavtokenizer"


def _default_stage_config_path() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "vllm_omni" / "model_executor" / "stage_configs" / "apertus.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Apertus offline multimodal inference with vLLM-Omni.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF ID for the Apertus model checkpoint.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=_default_stage_config_path(),
        help="Path to the stage config YAML.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="image",
        choices=["text", "image", "audio", "image_audio", "all"],
        help="Which modality combination to run.",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="Summarize the request in one short sentence.",
        help="Prompt for the text-only scenario.",
    )
    parser.add_argument(
        "--image-prompt",
        type=str,
        default=f"Describe the image briefly: {IMAGE_PLACEHOLDER}",
        help=f"Prompt for the text+image scenario. Include {IMAGE_PLACEHOLDER} where the image should be inserted.",
    )
    parser.add_argument(
        "--audio-prompt",
        type=str,
        default=f"Describe the audio briefly: {AUDIO_PLACEHOLDER}",
        help=f"Prompt for the text+audio scenario. Include {AUDIO_PLACEHOLDER} where the audio should be inserted.",
    )
    parser.add_argument(
        "--image-audio-prompt",
        type=str,
        default=f"Describe the image {IMAGE_PLACEHOLDER} and the audio {AUDIO_PLACEHOLDER}. If they seem related, explain how.",
        help=f"Prompt for the text+image+audio scenario. Include both {IMAGE_PLACEHOLDER} and {AUDIO_PLACEHOLDER}.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Optional path to an input image. A synthetic RGB image is used when omitted.",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Optional path to an input audio file. A synthetic sine wave is used when omitted.",
    )
    parser.add_argument(
        "--audio-duration-seconds",
        type=float,
        default=1.5,
        help="Synthetic audio duration used when --audio-path is omitted.",
    )
    parser.add_argument(
        "--audio-frequency-hz",
        type=float,
        default=440.0,
        help="Synthetic sine-wave frequency used when --audio-path is omitted.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=16000,
        help="Sample rate for synthetic audio.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save structured output JSON.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Enable trust_remote_code for model loading.",
    )
    parser.add_argument(
        "--emu-checkpoint",
        type=str,
        default="BAAI/Emu3.5-VisionTokenizer",
        help="EMU3.5 IBQ vision tokenizer checkpoint.",
    )
    parser.add_argument(
        "--emu-device",
        type=str,
        default="cuda:0",
        help="Device for the EMU vision tokenizer.",
    )
    parser.add_argument(
        "--emu-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype for the EMU vision tokenizer.",
    )
    parser.add_argument(
        "--audio-tokenizer-path",
        type=str,
        default=DEFAULT_AUDIO_TOKENIZER_PATH,
        help="Checkpoint path for the Apertus audio tokenizer.",
    )
    parser.add_argument(
        "--audio-tokenizer-codebase",
        type=str,
        default=None,
        help="Optional benchmark-audio-tokenizer checkout with src/repos/wavtokenizer present.",
    )
    parser.add_argument(
        "--audio-tokenizer-device",
        type=str,
        default="cuda",
        help="Device for the audio tokenizer.",
    )
    parser.add_argument(
        "--audio-tokenizer-compile",
        action="store_true",
        default=False,
        help="Enable torch.compile inside the audio tokenizer.",
    )
    parser.add_argument(
        "--audio-target-sampling-rate",
        type=int,
        default=24000,
        help="Target sampling rate for the audio tokenizer.",
    )
    parser.add_argument(
        "--audio-default-sampling-rate",
        type=int,
        default=16000,
        help="Default sampling rate used when raw audio is missing metadata.",
    )
    parser.add_argument(
        "--audio-token-offset",
        type=int,
        default=262344,
        help="Offset applied to raw WavTokenizer codes.",
    )
    parser.add_argument(
        "--audio-vocab-size",
        type=int,
        default=4096,
        help="Audio vocabulary size.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable Omni orchestrator stats logging.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for stage initialization in seconds.",
    )
    return parser.parse_args()


def _load_or_create_image(image_path: str | None) -> Image.Image:
    if image_path is None:
        return Image.new("RGB", (96, 96), color=(64, 128, 192))

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def _coerce_audio_waveform(audio_obj: Any) -> np.ndarray:
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
    return np.reshape(audio, (-1,)).astype(np.float32, copy=False)


def _load_audio_from_path(audio_path: str) -> tuple[np.ndarray, int]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if sf is not None:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        return _coerce_audio_waveform(audio), int(sr)

    if torchaudio is not None:
        audio, sr = torchaudio.load(path)
        return _coerce_audio_waveform(audio.numpy()), int(sr)

    raise ImportError("Either soundfile or torchaudio is required to load audio files.")


def _create_synthetic_audio(duration_seconds: float, frequency_hz: float, sample_rate: int) -> tuple[np.ndarray, int]:
    num_samples = max(1, int(duration_seconds * sample_rate))
    timestamps = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    waveform = 0.2 * np.sin(2.0 * math.pi * frequency_hz * timestamps)
    return waveform.astype(np.float32), int(sample_rate)


def _load_or_create_audio(args: argparse.Namespace) -> tuple[np.ndarray, int]:
    if args.audio_path is not None:
        return _load_audio_from_path(args.audio_path)
    return _create_synthetic_audio(
        duration_seconds=args.audio_duration_seconds,
        frequency_hz=args.audio_frequency_hz,
        sample_rate=args.audio_sample_rate,
    )


def _ensure_placeholders(prompt: str, placeholders: list[str]) -> str:
    merged_prompt = prompt
    missing = [placeholder for placeholder in placeholders if placeholder not in merged_prompt]
    if not missing:
        return merged_prompt

    if merged_prompt and not merged_prompt.endswith("\n"):
        merged_prompt += "\n"
    merged_prompt += "\n".join(missing)
    return merged_prompt


def _extract_text(outputs) -> str:
    if not outputs:
        raise ValueError("No outputs returned by omni.generate()")

    first = outputs[0]
    if not hasattr(first, "request_output") or not first.request_output:
        raise ValueError("No request_output found in Omni output")

    req_out = first.request_output[0]
    if not hasattr(req_out, "outputs") or not req_out.outputs:
        raise ValueError("No token outputs found in request_output")

    return req_out.outputs[0].text


def _scenario_order(scenario: str) -> list[str]:
    if scenario == "all":
        return ["text", "image", "audio", "image_audio"]
    return [scenario]


def _build_mm_processor_kwargs(args: argparse.Namespace, *, use_image: bool, use_audio: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if use_image:
        kwargs.update(
            {
                "apertus_vq_hub": args.emu_checkpoint,
                "apertus_vision_tokenizer_device": args.emu_device,
                "apertus_vision_tokenizer_dtype": args.emu_dtype,
                "apertus_vq_trust_remote_code": args.trust_remote_code,
                "apertus_image_placeholder": IMAGE_PLACEHOLDER,
            }
        )
    if use_audio:
        kwargs.update(
            {
                "apertus_audio_placeholder": AUDIO_PLACEHOLDER,
                "apertus_audio_tokenizer_path": args.audio_tokenizer_path,
                "apertus_audio_tokenizer_type": "wavtokenizer",
                "apertus_audio_tokenizer_name": "WavTokenizer40",
                "apertus_audio_tokenizer_device": args.audio_tokenizer_device,
                "apertus_audio_tokenizer_compile": args.audio_tokenizer_compile,
                "apertus_audio_target_sampling_rate": args.audio_target_sampling_rate,
                "apertus_audio_default_sampling_rate": args.audio_default_sampling_rate,
                "apertus_audio_token_offset": args.audio_token_offset,
                "apertus_audio_vocab_size": args.audio_vocab_size,
            }
        )
        if args.audio_tokenizer_codebase:
            kwargs["apertus_audio_tokenizer_codebase"] = args.audio_tokenizer_codebase
    return kwargs


def _build_prompt_dict(
    scenario: str,
    args: argparse.Namespace,
    image: Image.Image,
    audio: tuple[np.ndarray, int],
) -> dict[str, Any]:
    if scenario == "text":
        return {"prompt": args.text_prompt}

    if scenario == "image":
        prompt = _ensure_placeholders(args.image_prompt, [IMAGE_PLACEHOLDER])
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": [image]},
            "mm_processor_kwargs": _build_mm_processor_kwargs(args, use_image=True, use_audio=False),
        }

    if scenario == "audio":
        prompt = _ensure_placeholders(args.audio_prompt, [AUDIO_PLACEHOLDER])
        return {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio]},
            "mm_processor_kwargs": _build_mm_processor_kwargs(args, use_image=False, use_audio=True),
        }

    if scenario == "image_audio":
        prompt = _ensure_placeholders(args.image_audio_prompt, [IMAGE_PLACEHOLDER, AUDIO_PLACEHOLDER])
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": [image], "audio": [audio]},
            "mm_processor_kwargs": _build_mm_processor_kwargs(args, use_image=True, use_audio=True),
        }

    raise ValueError(f"Unsupported scenario: {scenario}")


def main() -> None:
    args = parse_args()
    image = _load_or_create_image(args.image_path)
    audio = _load_or_create_audio(args)
    scenarios = _scenario_order(args.scenario)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        trust_remote_code=args.trust_remote_code,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    results: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            prompt_dict = _build_prompt_dict(scenario, args, image, audio)
            started_at = time.time()
            outputs = omni.generate([prompt_dict], [sampling_params])
            elapsed = time.time() - started_at
            text = _extract_text(outputs)

            mm_data = prompt_dict.get("multi_modal_data", {})
            result = {
                "scenario": scenario,
                "prompt": prompt_dict["prompt"],
                "generated_text": text,
                "elapsed_seconds": elapsed,
                "num_images": len(mm_data.get("image", [])),
                "num_audios": len(mm_data.get("audio", [])),
            }
            results.append(result)

            print(f"=== Apertus E2E Result: {scenario} ===")
            print(f"Elapsed: {elapsed:.2f}s")
            print("Prompt:")
            print(prompt_dict["prompt"])
            print("Generated text:")
            print(text)
            print()
    finally:
        omni.close()

    if args.output_json:
        payload = {
            "model": args.model,
            "stage_configs_path": args.stage_configs_path,
            "scenario": args.scenario,
            "results": results,
            "sampling": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved output JSON to: {out_path}")


if __name__ == "__main__":
    main()
