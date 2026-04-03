import importlib
import json
import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from vllm.logger import init_logger

logger = init_logger(__name__)

_EMU35_VISION_TOKENIZER_MODULE = "_vllm_omni_external_emu35_vision_tokenizer"
_EMU35_VQ_REQUIRED_FILES = ("config.yaml", "model.ckpt")
_APERTUS_PROMPT_DUMP_ENV_VAR = "VLLM_OMNI_APERTUS_PROMPT_DUMP_PATH"
_APERTUS_PROMPT_DUMP_KWARG = "apertus_prompt_dump_path"
_APERTUS_AUDIO_TOKENIZER_CODEBASE_ENV_VAR = "VLLM_OMNI_APERTUS_AUDIO_TOKENIZER_CODEBASE"
_APERTUS_AUDIO_TOKENIZER_REPO_NAME = "benchmark-audio-tokenizer"


def get_default_cache_dir() -> Path:
    cache_env = os.getenv("VLLM_OMNI_MODELS_CACHE") or os.getenv("LMMS_EVAL_MODELS_CACHE")
    if cache_env:
        return Path(cache_env).expanduser()
    return Path.home() / ".cache" / "vllm-omni" / "models"


def _has_required_files(path: Path, required_files: Sequence[str]) -> bool:
    return all((path / fname).is_file() for fname in required_files)


def _download_from_hf(
    hf_repo_id: str,
    local_dir: Path,
    required_files: Sequence[str],
) -> None:
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(required_files),
    )


def ensure_local_weights(
    path: str,
    hf_repo_id: str,
    *,
    required_files: Sequence[str] = _EMU35_VQ_REQUIRED_FILES,
    cache_base_dir: str | None = None,
    accelerator: Any | None = None,
) -> str:
    expanded_path = Path(path).expanduser().resolve()
    if expanded_path.exists() and expanded_path.is_dir():
        if not _has_required_files(expanded_path, required_files):
            raise ValueError(
                f"Local checkpoint at {expanded_path} is missing required files: {list(required_files)}."
            )
        logger.info("Found local weights at %s", expanded_path)
        return str(expanded_path)

    cache_dir = Path(cache_base_dir).expanduser() if cache_base_dir else get_default_cache_dir()
    repo_cache_path = cache_dir / hf_repo_id

    if repo_cache_path.exists() and repo_cache_path.is_dir() and _has_required_files(repo_cache_path, required_files):
        logger.info("Found cached weights at %s", repo_cache_path)
        return str(repo_cache_path.resolve())

    should_download_here = True
    if accelerator is not None and getattr(accelerator, "num_processes", 1) > 1:
        should_download_here = bool(getattr(accelerator, "is_main_process", True))

    if should_download_here:
        logger.info("Downloading %s to %s", hf_repo_id, repo_cache_path)
        _download_from_hf(
            hf_repo_id=hf_repo_id,
            local_dir=repo_cache_path,
            required_files=required_files,
        )

    if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
        accelerator.wait_for_everyone()

    if not _has_required_files(repo_cache_path, required_files):
        raise RuntimeError(
            f"Resolved checkpoint at {repo_cache_path} is missing required files: {list(required_files)}."
        )

    return str(repo_cache_path.resolve())


@lru_cache(maxsize=1)
def _load_emu35_build_vision_tokenizer() -> Any:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "external"
        / "Emu3.5"
        / "src"
        / "vision_tokenizer"
        / "__init__.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(f"Unable to locate Emu3.5 vision tokenizer module: {module_path}")

    module = sys.modules.get(_EMU35_VISION_TOKENIZER_MODULE)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            _EMU35_VISION_TOKENIZER_MODULE,
            module_path,
            submodule_search_locations=[str(module_path.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to build import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_EMU35_VISION_TOKENIZER_MODULE] = module
        spec.loader.exec_module(module)

    build_vision_tokenizer = getattr(module, "build_vision_tokenizer", None)
    if build_vision_tokenizer is None:
        raise AttributeError("Emu3.5 vision tokenizer module does not expose build_vision_tokenizer.")
    return build_vision_tokenizer


def build_emu35_vision_tokenizer(
    *,
    vq_hub: str,
    default_repo: str,
    device: str,
    vq_type: str = "ibq",
    cache_base_dir: str | None = None,
    accelerator: Any | None = None,
    **kwargs: Any,
) -> Any:
    local_vq_path = ensure_local_weights(
        vq_hub,
        default_repo,
        required_files=_EMU35_VQ_REQUIRED_FILES,
        cache_base_dir=cache_base_dir,
        accelerator=accelerator,
    )
    build_vision_tokenizer = _load_emu35_build_vision_tokenizer()
    return build_vision_tokenizer(
        type=vq_type,
        model_path=local_vq_path,
        device=device,
        **kwargs,
    ).eval()


def resolve_apertus_prompt_dump_path(
    mm_processor_kwargs: Mapping[str, Any],
) -> Path | None:
    dump_path_value = mm_processor_kwargs.get(
        _APERTUS_PROMPT_DUMP_KWARG,
        os.getenv(_APERTUS_PROMPT_DUMP_ENV_VAR),
    )
    if not isinstance(dump_path_value, str) or not dump_path_value.strip():
        return None
    return Path(os.path.expandvars(dump_path_value.strip())).expanduser()


def dump_apertus_prompt_debug(
    *,
    mm_processor_kwargs: Mapping[str, Any],
    merged_prompt: str,
    prompt_token_ids: Sequence[int],
    image_count: int,
    truncated: bool,
) -> None:
    dump_path = resolve_apertus_prompt_dump_path(mm_processor_kwargs)
    if dump_path is None:
        return

    payload = {
        "merged_prompt": merged_prompt,
        "prompt_token_ids": list(prompt_token_ids),
        "num_prompt_tokens": len(prompt_token_ids),
        "num_images": image_count,
        "truncated": truncated,
    }
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("a", encoding="utf-8") as dump_file:
            dump_file.write(json.dumps(payload, ensure_ascii=False))
            dump_file.write("\n")
    except OSError as exc:
        logger.warning(
            "Failed writing Apertus prompt dump to %s: %s",
            dump_path,
            exc,
        )


def resolve_apertus_audio_tokenizer_codebase(codebase_path: str | None = None) -> Path:
    def _is_valid_wavtokenizer_codebase(candidate: Path) -> bool:
        return all(
            path.is_file()
            for path in (
                candidate / "src" / "audio_tokenizers" / "implementations" / "wavtokenizer.py",
                candidate / "src" / "repos" / "wavtokenizer" / "encoder" / "utils.py",
                candidate / "src" / "repos" / "wavtokenizer" / "decoder" / "pretrained.py",
            )
        )

    candidates: list[Path] = []
    explicit = codebase_path or os.getenv(_APERTUS_AUDIO_TOKENIZER_CODEBASE_ENV_VAR)
    if explicit:
        candidates.append(Path(os.path.expandvars(explicit)).expanduser())

    sibling_repo = Path(__file__).resolve().parents[4] / _APERTUS_AUDIO_TOKENIZER_REPO_NAME
    candidates.append(sibling_repo)

    for candidate in candidates:
        if _is_valid_wavtokenizer_codebase(candidate):
            return candidate

    raise FileNotFoundError(
        "Unable to locate a complete benchmark-audio-tokenizer checkout required for Apertus audio tokenization. "
        "The codebase must include both the wrapper module and the underlying "
        "`src/repos/wavtokenizer` source tree. "
        f"Set {_APERTUS_AUDIO_TOKENIZER_CODEBASE_ENV_VAR} or place a complete repo at {sibling_repo}."
    )


@lru_cache(maxsize=4)
def _load_wavtokenizer40_class(codebase_path: str) -> Any:
    resolved_codebase = resolve_apertus_audio_tokenizer_codebase(codebase_path)
    codebase_str = str(resolved_codebase)
    if codebase_str not in sys.path:
        sys.path.insert(0, codebase_str)

    module = importlib.import_module("src.audio_tokenizers.implementations.wavtokenizer")
    wavtokenizer_cls = getattr(module, "WavTokenizer40", None)
    if wavtokenizer_cls is None:
        raise AttributeError("benchmark-audio-tokenizer does not expose WavTokenizer40.")
    return wavtokenizer_cls


def build_apertus_wavtokenizer(
    *,
    checkpoint_path: str | None = None,
    codebase_path: str | None = None,
    device: str = "cuda",
    torch_compile: bool = False,
) -> Any:
    wavtokenizer_cls = _load_wavtokenizer40_class(
        str(resolve_apertus_audio_tokenizer_codebase(codebase_path))
    )
    kwargs = {
        "device": device,
        "torch_compile": torch_compile,
    }
    if checkpoint_path is not None:
        kwargs["checkpoint"] = checkpoint_path
    return wavtokenizer_cls(**kwargs)
