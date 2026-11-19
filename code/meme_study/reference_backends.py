"""Load pinned Qwen and InternVL weights on the first valid generation request.

Use Transformers 4.57.1 for Qwen and 4.48.3 for InternVL in separate environments.
"""
from __future__ import annotations

import copy
import importlib
from importlib import metadata
from pathlib import Path
import platform
import re
import time

from .backends import OFFICIAL_CHECKPOINTS, TransformersBackend, _GENERATION


INTERNVL_CHECKPOINT = 'OpenGVLab/InternVL3-9B'
INTERNVL_REVISION = '5f618513e35a9b85922341b8057feddfc8880e50'
REFERENCE_CHECKPOINTS = {**OFFICIAL_CHECKPOINTS, INTERNVL_CHECKPOINT: INTERNVL_REVISION}
_SPECIAL_IDS = {'<|im_end|>': 128131, '<img>': 128133, '</img>': 128134,
                '<IMG_CONTEXT>': 128135}
_DISTRIBUTIONS = ('torch', 'transformers', 'torchvision', 'accelerate', 'Pillow',
                  'timm', 'einops', 'sentencepiece', 'huggingface-hub', 'safetensors')
_JUDGE_GENERATION = {
    'do_sample': False, 'num_beams': 1, 'num_return_sequences': 1,
    'bos_token_id': 1, 'pad_token_id': 2, 'eos_token_id': 128131,
    'repetition_penalty': 1.0, 'use_cache': True,
    'return_dict_in_generate': False, 'output_scores': False,
}
_REMOTE_FILES = ('configuration_intern_vit.py', 'configuration_internlm2.py',
                 'configuration_internvl_chat.py', 'conversation.py',
                 'modeling_intern_vit.py', 'modeling_internlm2.py',
                 'modeling_internvl_chat.py', 'tokenization_internlm3.py')
_MEMORY_SCOPE = 'current CUDA process / torch allocator, not whole-device utilization'
_CUDA_SNAPSHOT_METHODS = ('is_available', 'device_count', 'get_device_properties',
                          'memory_allocated', 'memory_reserved',
                          'max_memory_allocated', 'max_memory_reserved')


def _versions():
    result = {}
    for name in _DISTRIBUTIONS:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _require_transformers(config, expected):
    if config['software_versions'].get('transformers') != expected:
        raise RuntimeError(f'this adapter requires a separate transformers=={expected} '
                           'environment; see requirements-reference-gpu.txt')


def _single_ids(tensor):
    if len(tensor.shape) != 2 or tensor.shape[0] != 1:
        raise ValueError('token tensors must contain exactly one sequence')
    rows = tensor.tolist()
    if len(rows) != 1 or len(rows[0]) != tensor.shape[1]:
        raise ValueError('token tensor shape and sequence disagree')
    return rows[0]


def token_receipt(input_ids, output_ids, *, max_new_tokens, eos_token_ids):
    """Count IDs obtained from actual request/result tensors, including final EOS."""
    if (not isinstance(input_ids, list) or not input_ids or not isinstance(output_ids, list)
            or any(type(token) is not int or token < 0 for token in input_ids + output_ids)
            or type(max_new_tokens) is not int or max_new_tokens <= 0):
        raise ValueError('receipt requires nonnegative integer token sequences and a positive cap')
    if len(output_ids) > max_new_tokens:
        raise ValueError('generation exceeds the protocol token cap')
    if any(token in eos_token_ids for token in output_ids[:-1]):
        raise ValueError('single-sequence generation contains tokens after EOS')
    reason = ('stop' if output_ids and output_ids[-1] in eos_token_ids
              else 'length' if len(output_ids) == max_new_tokens else 'unknown')
    return {'input_tokens': len(input_ids), 'output_tokens': len(output_ids),
            'finish_reason': reason}


class _CudaResourceTelemetry:
    """Read public CUDA allocator counters without loading or invoking a model.

    Counters cover this process on the selected device, including other models
    or allocations in this process. They do not measure utilization, CUDA
    context/non-PyTorch allocations, other processes, or GPU energy consumption.
    """
    def resource_snapshot(self):
        """Return JSON data; missing/unavailable readings never become zeros.

        A snapshot does not reset peaks. Its peaks start at the most recent
        process/device reset (or process start if there has been no reset).
        Calling this before prepare/generate is lazy and returns unavailable.
        """
        unavailable = {'available': False, 'memory_scope': _MEMORY_SCOPE}
        if self._model is None or self._torch is None:
            return {**unavailable, 'reason': 'backend_not_prepared'}
        cuda = getattr(self._torch, 'cuda', None)
        missing = [name for name in _CUDA_SNAPSHOT_METHODS
                   if not callable(getattr(cuda, name, None))]
        if missing:
            return {**unavailable, 'reason': 'cuda_snapshot_api_unavailable',
                    'missing_methods': missing}
        index = int(self.device.split(':')[1])
        try:
            if not cuda.is_available() or index >= cuda.device_count():
                return {**unavailable, 'reason': 'cuda_device_unavailable'}
            properties = cuda.get_device_properties(index)
            return {
                'available': True, 'memory_scope': _MEMORY_SCOPE,
                'device_index': index, 'device_name': str(properties.name),
                'total_memory_bytes': int(properties.total_memory),
                'torch_cuda_version': getattr(getattr(self._torch, 'version', None), 'cuda', None),
                'torch_memory_allocated_bytes': int(cuda.memory_allocated(index)),
                'torch_memory_reserved_bytes': int(cuda.memory_reserved(index)),
                'torch_peak_allocated_bytes': int(cuda.max_memory_allocated(index)),
                'torch_peak_reserved_bytes': int(cuda.max_memory_reserved(index)),
                'peak_scope': 'since most recent process/device peak reset, or process start',
            }
        except (RuntimeError, ValueError, AttributeError) as exc:
            return {**unavailable, 'reason': 'cuda_snapshot_failed',
                    'error_type': type(exc).__name__, 'error': str(exc)}

    def _begin_resource_usage(self):
        before = self.resource_snapshot()
        usage = {'available': False, 'memory_scope': _MEMORY_SCOPE,
                 'peak_stats_reset': False, 'before': before}
        if not before['available']:
            return {**usage, 'reason': before['reason']}
        reset = getattr(self._torch.cuda, 'reset_peak_memory_stats', None)
        if not callable(reset):
            return {**usage, 'reason': 'cuda_peak_reset_api_unavailable'}
        try:
            reset(int(self.device.split(':')[1]))
        except (RuntimeError, ValueError) as exc:
            return {**usage, 'reason': 'cuda_peak_reset_failed',
                    'error_type': type(exc).__name__, 'error': str(exc)}
        before = self.resource_snapshot()
        usage.update({
            'available': before['available'], 'peak_stats_reset': True, 'before': before,
            'peak_window': 'after model load, before preprocessing through generation and decoding; '
                           'includes loaded-model baseline',
        })
        if not before['available']:
            usage['reason'] = before['reason']
        return usage

    def _finish_resource_usage(self, usage):
        after = self.resource_snapshot()
        result = {**usage, 'after': after}
        if not after['available']:
            result.update({'available': False, 'reason': after['reason']})
        return result


class ReferenceQwenBackend(_CudaResourceTelemetry, TransformersBackend):
    """One image or OCR text, one direct call, exactly the 512-token protocol cap."""
    def __init__(self, checkpoint, revision, data_root, *, device='cuda:0',
                 max_pixels=1310720, min_pixels=262144):
        super().__init__(checkpoint, revision, data_root, device=device,
                         max_pixels=max_pixels, min_pixels=min_pixels)
        self._config.update({
            'backend': 'reference_qwen_transformers', 'protocol_id': 'reference_reuse_v1',
            'software_versions': _versions(), 'required_transformers': '4.57.1',
            'stage_token_caps': {'explanation': 512, 'pair': 512, 'permutation': 512},
            'frozen_weights': True,
            'token_accounting': 'actual input_ids and generated continuation IDs; includes EOS',
        })

    def _ensure_loaded(self):
        if self._model is not None:
            return
        _require_transformers(self._config, '4.57.1')
        super()._ensure_loaded()
        self._model.requires_grad_(False)

    def prepare(self):
        """Load/check the model before any budgeted dispatch is recorded."""
        self._ensure_loaded()

    def generate(self, messages, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens != 512:
            raise ValueError('reference tested-model calls require max_new_tokens=512')
        normalized = self._messages_with_local_images(messages)
        self._ensure_loaded()
        self._torch.cuda.synchronize(self.device)
        resource_usage = self._begin_resource_usage()
        start = time.perf_counter()
        inputs = self._processor.apply_chat_template(
            normalized, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt')
        prompt_ids = _single_ids(inputs['input_ids'])
        inputs = inputs.to(self.device)
        generation = self._transformers.GenerationConfig(**copy.deepcopy(_GENERATION))
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, generation_config=generation,
                                             max_new_tokens=max_new_tokens)
        self._torch.cuda.synchronize(self.device)
        sequence = _single_ids(generated)
        if sequence[:len(prompt_ids)] != prompt_ids:
            raise ValueError('generated Qwen sequence does not begin with the actual prompt IDs')
        output_ids = sequence[len(prompt_ids):]
        receipt = token_receipt(prompt_ids, output_ids, max_new_tokens=max_new_tokens,
                                eos_token_ids=_GENERATION['eos_token_id'])
        text = self._processor.batch_decode([output_ids], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
        latency_s = time.perf_counter() - start
        return {'text': text, **receipt, 'latency_s': latency_s,
                'resource_usage': self._finish_resource_usage(resource_usage)}


def internvl_image_tiles(image, *, max_tiles=4):
    """Official aspect-ratio grid policy, bounded at 4/6 tiles plus a thumbnail.

    RGB conversion precedes resizing. PIL's documented RGB default resize is
    BICUBIC; it is explicit here. Equal-area grid ties use (columns, rows) order
    so the profile is stable across Python versions.
    """
    from PIL import Image
    if type(max_tiles) is not int or max_tiles not in (4, 6):
        raise ValueError('bounded InternVL profiles permit max_tiles=4 or 6')
    image = image.convert('RGB')
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive')
    ratios = sorted({(columns, rows) for columns in range(1, max_tiles + 1)
                     for rows in range(1, max_tiles + 1) if columns * rows <= max_tiles},
                    key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))
    best, difference = (1, 1), float('inf')
    for columns, rows in ratios:
        candidate = abs(width / height - columns / rows)
        if candidate < difference or (candidate == difference and
                width * height > 0.5 * 448 * 448 * columns * rows):
            best, difference = (columns, rows), candidate
    columns, rows = best
    resized = image.resize((448 * columns, 448 * rows), Image.Resampling.BICUBIC)
    tiles = [resized.crop((x * 448, y * 448, (x + 1) * 448, (y + 1) * 448))
             for y in range(rows) for x in range(columns)]
    if len(tiles) > 1:
        tiles.append(image.resize((448, 448), Image.Resampling.BICUBIC))
    return tiles


def internvl_query(question, template, *, num_patches):
    """Reproduce single-turn remote chat setup using its actual Conversation."""
    if (not isinstance(question, str) or not question.strip()
            or any(token in question for token in (*_SPECIAL_IDS, '<image>', '<|im_start|>'))):
        raise ValueError('judge text must be nonempty and contain no reserved image/chat tokens')
    if type(num_patches) is not int or not 1 <= num_patches <= 7:
        raise ValueError('judge patch count is outside the bounded profile')
    conversation = template.copy()
    if conversation.messages:
        raise ValueError('judge template must not contain conversation history')
    conversation.append_message(conversation.roles[0], '<image>\n' + question)
    conversation.append_message(conversation.roles[1], None)
    query = conversation.get_prompt()
    if query.count('<image>') != 1:
        raise ValueError('judge query must contain exactly one original-image placeholder')
    return query.replace('<image>', '<img>' + '<IMG_CONTEXT>' * (256 * num_patches) + '</img>', 1)


def validate_internvl_contract(model, tokenizer):
    """Fail closed if pinned remote metadata/template/token IDs do not match."""
    if (model.template != 'internvl2_5' or model.num_image_token != 256
            or model.config.force_image_size != 448
            or model.conv_template.name != 'internvl2_5'
            or model.conv_template.sep != '<|im_end|>\n'
            or tuple(model.conv_template.roles) != ('<|im_start|>user\n', '<|im_start|>assistant\n')
            or model.conv_template.messages):
        raise ValueError('InternVL image-token or conversation metadata differs from the inspected revision')
    if tokenizer.bos_token_id != 1 or tokenizer.pad_token_id != 2:
        raise ValueError('InternVL BOS/PAD IDs differ from inspected tokenizer metadata')
    if any(tokenizer.convert_tokens_to_ids(token) != token_id
           for token, token_id in _SPECIAL_IDS.items()):
        raise ValueError('InternVL special token IDs differ from inspected tokenizer metadata')


class InternVLJudgeBackend(_CudaResourceTelemetry):
    """Original-image judge using the pinned remote model's tensor generate path."""
    _messages_with_local_images = TransformersBackend._messages_with_local_images
    run_config = TransformersBackend.run_config

    def prepare(self):
        """Load/check the model before any budgeted dispatch is recorded."""
        self._ensure_loaded()

    def __init__(self, checkpoint, revision, data_root, *, device='cuda:0', max_tiles=4):
        if checkpoint != INTERNVL_CHECKPOINT or revision != INTERNVL_REVISION:
            raise ValueError('use the official InternVL3-9B checkpoint and pinned revision')
        if not isinstance(device, str) or re.fullmatch(r'cuda:[0-9]+', device) is None:
            raise ValueError('device must name one CUDA device, for example cuda:0')
        if type(max_tiles) is not int or max_tiles not in (4, 6):
            raise ValueError('bounded InternVL profiles permit max_tiles=4 or 6')
        try:
            root = Path(data_root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError('data_root must be an existing directory') from exc
        if not root.is_dir():
            raise ValueError('data_root must be an existing directory')
        self.checkpoint, self.revision, self.data_root = checkpoint, revision, root
        self.device, self.max_tiles = device, max_tiles
        self._config = {
            'backend': 'reference_internvl_remote', 'protocol_id': 'reference_reuse_v1',
            'checkpoint': checkpoint, 'revision': revision, 'code_revision': revision,
            'data_root': str(root), 'device': device, 'dtype': 'bfloat16',
            'quantization': None, 'batch_size': 1, 'frozen_weights': True,
            'trust_remote_code': True, 'inspected_remote_files': list(_REMOTE_FILES),
            'attention_implementation': 'eager', 'use_flash_attn': False,
            'software_versions': _versions(), 'python_version': platform.python_version(),
            'required_transformers': '4.48.3', 'conversation_template': 'internvl2_5',
            'image_processing': {
                'image_size': 448, 'mode': 'RGB', 'resize': 'PIL BICUBIC',
                'dynamic_tiles': True, 'min_tiles': 1, 'max_tiles': max_tiles,
                'use_thumbnail': True, 'max_total_patches': max_tiles + 1,
                'grid_tie_break': 'area_then_columns_then_rows',
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'max_images_per_request': 1, 'tokens_per_patch': 256,
            },
            'generation': copy.deepcopy(_JUDGE_GENERATION), 'stage_token_caps': {'judge': 768},
            'max_input_and_output_tokens': 8192,
            'token_accounting': 'actual expanded input_ids; remote generate returns continuation only; includes EOS',
            'latency_scope': 'preprocessing_generation_decoding; excludes first model load',
        }
        self._model = self._tokenizer = self._torch = self._transformers = self._transform = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        _require_transformers(self._config, '4.48.3')
        try:
            torch = importlib.import_module('torch')
            transformers = importlib.import_module('transformers')
            transforms = importlib.import_module('torchvision.transforms')
        except ImportError as exc:
            raise RuntimeError('install the separate InternVL requirements-reference-gpu.txt profile') from exc
        for name, module in [('torch', torch), ('transformers', transformers)]:
            if module.__version__ != self._config['software_versions'][name]:
                raise RuntimeError(f'{name} differs from recorded package metadata')
        if not torch.cuda.is_available() or int(self.device.split(':')[1]) >= torch.cuda.device_count():
            raise RuntimeError(f'the requested GPU {self.device} is unavailable')
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.checkpoint, revision=self.revision, code_revision=self.revision,
            trust_remote_code=True, use_fast=False)
        model = transformers.AutoModel.from_pretrained(
            self.checkpoint, revision=self.revision, code_revision=self.revision,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            use_flash_attn=False, trust_remote_code=True, device_map=None)
        model = model.to(self.device).eval()
        model.requires_grad_(False)
        if model.dtype != torch.bfloat16:
            raise RuntimeError('loaded InternVL dtype differs from recorded BF16')
        validate_internvl_contract(model, tokenizer)
        if not callable(getattr(model.language_model, 'generate', None)):
            raise RuntimeError('InternVL language model lacks the required generation interface')
        model.img_context_token_id = _SPECIAL_IDS['<IMG_CONTEXT>']
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self._model, self._tokenizer, self._transform = model, tokenizer, transform
        self._torch, self._transformers = torch, transformers

    def _judge_input(self, messages):
        normalized = self._messages_with_local_images(messages)
        if len(normalized) != 1 or normalized[0]['role'] != 'user':
            raise ValueError('judge requires one user turn without chat history')
        blocks = normalized[0]['content']
        images = [block['image'] for block in blocks if block['type'] == 'image']
        question = '\n'.join(block['text'] for block in blocks if block['type'] == 'text')
        if len(images) != 1 or not question.strip():
            raise ValueError('judge requires the original image and nonempty checklist/candidate text')
        if any(token in question for token in (*_SPECIAL_IDS, '<image>', '<|im_start|>')):
            raise ValueError('judge text contains reserved image/chat tokens')
        return images[0], question

    def generate(self, messages, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens != 768:
            raise ValueError('reference judge calls require max_new_tokens=768')
        image_path, question = self._judge_input(messages)
        self._ensure_loaded()
        self._torch.cuda.synchronize(self.device)
        resource_usage = self._begin_resource_usage()
        start = time.perf_counter()
        from PIL import Image
        with Image.open(image_path) as original:
            tiles = internvl_image_tiles(original, max_tiles=self.max_tiles)
        pixel_values = self._torch.stack([self._transform(tile) for tile in tiles]).to(
            device=self.device, dtype=self._torch.bfloat16)
        template = self._model.conv_template.copy()
        template.system_message = self._model.system_message
        query = internvl_query(question, template, num_patches=len(tiles))
        inputs = self._tokenizer(query, return_tensors='pt')
        prompt_ids = _single_ids(inputs['input_ids'])
        if prompt_ids.count(_SPECIAL_IDS['<IMG_CONTEXT>']) != 256 * len(tiles):
            raise ValueError('actual tokenized visual placeholders do not match image patches')
        if len(prompt_ids) + max_new_tokens > self._config['max_input_and_output_tokens']:
            raise ValueError('judge request exceeds the conservative tokenizer context budget')
        generation = self._transformers.GenerationConfig(**copy.deepcopy(_JUDGE_GENERATION))
        with self._torch.inference_mode():
            generated = self._model.generate(
                pixel_values=pixel_values, input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                generation_config=generation, max_new_tokens=max_new_tokens)
        self._torch.cuda.synchronize(self.device)
        # Pinned remote generate forwards inputs_embeds, not input_ids, to its
        # language model. Transformers 4.48.3 therefore returns generated IDs only.
        output_ids = _single_ids(generated)
        receipt = token_receipt(prompt_ids, output_ids, max_new_tokens=max_new_tokens,
                                eos_token_ids=[_SPECIAL_IDS['<|im_end|>']])
        text = self._tokenizer.batch_decode([output_ids], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
        text = text.split(template.sep.strip())[0].strip()
        latency_s = time.perf_counter() - start
        return {'text': text, **receipt, 'latency_s': latency_s,
                'resource_usage': self._finish_resource_usage(resource_usage)}


def create_backend(checkpoint, data_root, *, device='cuda:0', **kwargs):
    """Construct only; exact public model ID selects a frozen revision/profile."""
    if checkpoint in OFFICIAL_CHECKPOINTS:
        return ReferenceQwenBackend(checkpoint, OFFICIAL_CHECKPOINTS[checkpoint], data_root,
                                    device=device, **kwargs)
    if checkpoint == INTERNVL_CHECKPOINT:
        return InternVLJudgeBackend(checkpoint, INTERNVL_REVISION, data_root,
                                    device=device, **kwargs)
    raise ValueError('unknown reference-protocol model; use an exact official checkpoint ID')
