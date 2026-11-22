"""Load the pinned Qwen checkpoint and optional PEFT adapter for CUDA inference."""

import copy
from importlib import metadata
import os
import platform
import re
import time
import warnings

from meme_study.backends import OFFICIAL_CHECKPOINTS, _GENERATION
from meme_study.freeze import _safe_file
from meme_study.reference_backends import token_receipt

from materials import MAX_TOKENS, SEED


MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
MODEL_REVISION = OFFICIAL_CHECKPOINTS[MODEL_ID]
VERSIONS = {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128',
            'transformers': '4.57.1', 'peft': '0.18.1', 'Pillow': '11.3.0',
            'accelerate': '1.11.0', 'huggingface-hub': '0.36.0'}
CUBLAS_CONFIG = ':4096:8'
POLICY = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
          'dtype': 'bfloat16', 'attention': 'sdpa', 'seed': SEED,
          'cublas_workspace_config': CUBLAS_CONFIG, 'deterministic_algorithms': True,
          'image_pixels': {'shortest_edge': 262144, 'longest_edge': 1310720},
          'max_new_tokens': MAX_TOKENS, 'generation': _GENERATION, 'versions': VERSIONS,
          'use_model_defaults': False,
          'trust_remote_code': False, 'batch_size': 1, 'adapter_autocast_dtype': True}


class CudaBackend:
    def __init__(self, model_dir, adapter_dir, data_root, device):
        self.model_dir, self.adapter_dir, self.data_root = model_dir, adapter_dir, data_root
        if re.fullmatch(r'cuda:\d+', device) is None:
            raise ValueError('device must name one CUDA device')
        self.device = device

    def prepare(self):
        try:
            installed = {name: metadata.version(name) for name in VERSIONS}
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f'CUDA evaluation dependency is missing: {error.name}') from error
        if installed != VERSIONS:
            raise RuntimeError(f'CUDA evaluation environment must match the pinned versions: {installed}')
        if os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', CUBLAS_CONFIG) != CUBLAS_CONFIG:
            raise RuntimeError(f'use CUBLAS_WORKSPACE_CONFIG={CUBLAS_CONFIG}')
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available() or int(self.device.split(':')[1]) >= torch.cuda.device_count():
            raise RuntimeError(f'{self.device} is unavailable')
        with torch.cuda.device(self.device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError('evaluation requires native CUDA BF16 support')
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_dir), dtype=torch.bfloat16, device_map=None, trust_remote_code=False,
            attn_implementation='sdpa', local_files_only=True, use_safetensors=True,
        )
        if model.dtype != torch.bfloat16:
            raise RuntimeError('base checkpoint must load as BF16')
        if self.adapter_dir is not None:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', message='.*missing adapter keys.*')
                    model = PeftModel.from_pretrained(
                        model, str(self.adapter_dir), is_trainable=False, local_files_only=True,
                        autocast_adapter_dtype=POLICY['adapter_autocast_dtype'],
                    )
            except UserWarning as error:
                raise RuntimeError(str(error)) from error
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir), trust_remote_code=False, local_files_only=True,
        )
        self.processor.image_processor.size = dict(POLICY['image_pixels'])
        self.generation = GenerationConfig(**copy.deepcopy(_GENERATION))
        self.torch = torch
        props = torch.cuda.get_device_properties(self.device)
        return {'device': self.device, 'name': props.name, 'capability': [props.major, props.minor],
                'cuda': torch.version.cuda, 'python': platform.python_version(),
                'pillow': metadata.version('Pillow'), 'cudnn': torch.backends.cudnn.version()}

    def generate(self, case):
        image = _safe_file(self.data_root, case['image'])
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': str(image)}, {'type': 'text', 'text': case['prompt']},
        ]}]
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
        ).to(self.device)
        if inputs.input_ids.ndim != 2 or inputs.input_ids.shape[0] != 1:
            raise ValueError('evaluation requires one case per generation')
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, generation_config=self.generation, max_new_tokens=MAX_TOKENS,
                use_model_defaults=POLICY['use_model_defaults'],
            )
        prompt_ids = inputs.input_ids.tolist()[0]
        rows = output.tolist()
        if len(rows) != 1 or rows[0][:len(prompt_ids)] != prompt_ids:
            raise ValueError('generation does not preserve the prompt token prefix')
        output_ids = rows[0][len(prompt_ids):]
        counts = token_receipt(prompt_ids, output_ids, max_new_tokens=MAX_TOKENS,
                               eos_token_ids=_GENERATION['eos_token_id'])
        text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return {**counts, 'text': text, 'output_token_ids': output_ids,
                'latency_s': time.perf_counter() - started}
