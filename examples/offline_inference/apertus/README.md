# Apertus Offline End-to-End Examples

This example runs Apertus through the Omni pipeline for all currently supported
input combinations:

- text only
- text + image
- text + audio
- text + image + audio

## Setup

From repo root:

```bash
cd vllm-omni
python3 -m pip install -e . --no-build-isolation --no-deps
```

## Run One Scenario

Text only:

```bash
python3 examples/offline_inference/apertus/end2end.py \
  --model /capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF \
  --stage-configs-path vllm_omni/model_executor/stage_configs/apertus.yaml \
  --scenario text
```

Text + image:

```bash
python3 examples/offline_inference/apertus/end2end.py \
  --model /capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF \
  --stage-configs-path vllm_omni/model_executor/stage_configs/apertus.yaml \
  --scenario image \
  --image-prompt "Describe the image briefly: <|image|>"
```

Text + audio:

```bash
python3 examples/offline_inference/apertus/end2end.py \
  --model /capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF \
  --stage-configs-path vllm_omni/model_executor/stage_configs/apertus.yaml \
  --scenario audio \
  --audio-path /path/to/audio.wav \
  --audio-tokenizer-codebase /path/to/benchmark-audio-tokenizer
```

Text + image + audio:

```bash
python3 examples/offline_inference/apertus/end2end.py \
  --model /capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF \
  --stage-configs-path vllm_omni/model_executor/stage_configs/apertus.yaml \
  --scenario image_audio \
  --image-path /path/to/image.jpg \
  --audio-path /path/to/audio.wav \
  --audio-tokenizer-codebase /path/to/benchmark-audio-tokenizer
```

If `--image-path` or `--audio-path` is omitted, the script uses a synthetic
input for that modality.

## Run All Combinations

```bash
python3 examples/offline_inference/apertus/end2end.py \
  --model /capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF \
  --stage-configs-path vllm_omni/model_executor/stage_configs/apertus.yaml \
  --scenario all \
  --audio-tokenizer-codebase /path/to/benchmark-audio-tokenizer \
  --output-json /tmp/apertus_offline_examples.json
```

## Notes

- The audio checkpoint defaults to `/capstor/store/cscs/swissai/infra01/MLLM/wavtokenizer`.
- Audio examples need a complete `benchmark-audio-tokenizer` checkout with
  `src/repos/wavtokenizer` available, or `VLLM_OMNI_APERTUS_AUDIO_TOKENIZER_CODEBASE`
  set in the environment.
- `vllm_omni/model_executor/stage_configs/apertus.yaml` keeps the same single-stage
  Apertus generation flow.
