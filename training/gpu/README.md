# Qwen3-VL 4B CUDA training

Complete [setup](../../README.md#set-up) and run from the extracted project root. Both profiles use the same 3-epoch BF16 LoRA recipe. `chinese` uses 98 training and 33 validation rows. `bilingual` uses 770 training and 193 validation rows. The exporter checks family and exact image overlap against the formal benchmark and English test split. The training export contains only train and validation conversations and images.

## Export training data

Use Python 3.12. The preparation scripts require Pillow.

```bash
python -m pip install pillow==11.3.0
python training/gpu/prepare_data.py export --profile bilingual --output work/sft-bilingual
```

The export is portable with its `images/` directory and `dataset-manifest.json`.

## Prepare a GPU host

Use Linux x86-64, Python 3.12, and one visible NVIDIA GPU with native BF16 and a CUDA 12.8-compatible driver. Budget 24–32 GB VRAM, 32 GB system RAM and 40 GB free disk as a starting estimate. Peak VRAM varies by GPU, driver, and tokenized inputs. [PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/) · [NVIDIA driver guide](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

```bash
python3.12 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r training/gpu/requirements.txt
hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir work/qwen3vl-4b
```

The launcher verifies the official weight shards, model config, tokenizer and processor files. [Qwen revision](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) · [LlamaFactory recipe](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/examples/train_lora/qwen3vl_lora_sft.yaml)

## Run and resume

```bash
export CUDA_VISIBLE_DEVICES=0
python training/gpu/launch.py train --dataset work/sft-bilingual \
  --model work/qwen3vl-4b --output work/run-bilingual
```

Training evaluates and saves at each epoch. It selects the lowest validation loss and records the selected PEFT files under `trainer/` in `selected-adapter.json`. For the Chinese comparison, export with `--profile chinese --output work/sft-chinese` and train with `--dataset work/sft-chinese --output work/run-chinese`.

For an interrupted full run, keep the run at its original path. Resume from the latest checkpoint with its adapter, optimizer, scheduler, RNG and trainer state intact.

```bash
python training/gpu/launch.py resume --dataset work/sft-bilingual \
  --model work/qwen3vl-4b --output work/run-bilingual
```

Run the held-out evaluation separately after selecting the full-run adapter. [Evaluation guide](eval/README.md)
