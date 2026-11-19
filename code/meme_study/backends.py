"""Run pinned Qwen3-VL checkpoints on one BF16 device.

Callers verify material readiness and image hashes. The backend checks that
image paths remain inside the data root.
"""
import copy
import importlib
from importlib import metadata
from pathlib import Path
import platform
import re
import time


OFFICIAL_CHECKPOINTS = {
    'Qwen/Qwen3-VL-4B-Instruct': 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
    'Qwen/Qwen3-VL-8B-Instruct': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
}
_DISTRIBUTIONS = ('torch', 'transformers', 'torchvision', 'accelerate', 'Pillow')
# These token IDs were read from generation_config.json at both pinned revisions.
# Build a fresh GenerationConfig, not a copy of the checkpoint's sampling config.
_GENERATION = {
    'do_sample': False, 'num_beams': 1, 'num_return_sequences': 1,
    'bos_token_id': 151643, 'pad_token_id': 151643,
    'eos_token_id': [151645, 151643], 'repetition_penalty': 1.0,
    'use_cache': True, 'return_dict_in_generate': False, 'output_scores': False,
}


def _installed_versions():
    versions = {}
    for name in _DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class TransformersBackend:
    """One request at a time; weights are loaded on the first generate call.

    Construction and run_config inspect installed distribution metadata only.
    No weight/network access occurs until generate. Configuration is detached
    from callers and remains fixed for the runner's immutable run identity.
    """
    def __init__(self, checkpoint, revision, data_root, *, device='cuda:0',
                 max_pixels=1310720, min_pixels=262144):
        if checkpoint not in OFFICIAL_CHECKPOINTS or revision != OFFICIAL_CHECKPOINTS[checkpoint]:
            raise ValueError('use an official Qwen3-VL checkpoint and its pinned revision')
        if not isinstance(device, str) or re.fullmatch(r'cuda:[0-9]+', device) is None:
            raise ValueError('device must name one CUDA device, for example cuda:0')
        if (type(max_pixels) is not int or type(min_pixels) is not int
                or min_pixels <= 0 or max_pixels < min_pixels):
            raise ValueError('image pixel limits must be positive integers with min <= max')
        try:
            root = Path(data_root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError('data_root must be an existing directory') from exc
        if not root.is_dir():
            raise ValueError('data_root must be an existing directory')
        self.checkpoint, self.revision = checkpoint, revision
        self.data_root, self.device = root, device
        self.max_pixels, self.min_pixels = max_pixels, min_pixels
        self._config = {
            'backend': 'transformers', 'checkpoint': checkpoint, 'revision': revision,
            'data_root': str(root), 'device': device, 'dtype': 'bfloat16',
            'quantization': None, 'batch_size': 1, 'trust_remote_code': False,
            'attention_implementation': 'sdpa',
            'software_versions': _installed_versions(),
            'python_version': platform.python_version(),
            'image_processing': {'min_pixels': min_pixels, 'max_pixels': max_pixels,
                                 'max_images_per_request': 1},
            'generation': copy.deepcopy(_GENERATION),
            'stage_token_caps': {'intermediate': 80, 'final': 160},
            'latency_scope': 'preprocessing_generation_decoding; excludes first model load',
        }
        self._model = self._processor = self._torch = self._transformers = None

    @property
    def run_config(self):
        return copy.deepcopy(self._config)

    def _messages_with_local_images(self, messages):
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must describe one nonempty conversation')
        result, image_count = [], 0
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in {'user', 'assistant', 'system'}:
                raise ValueError('batched messages or unsupported message role')
            content = message.get('content')
            if not isinstance(content, list) or not content:
                raise ValueError('message content must be a nonempty list of blocks')
            blocks = []
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError('content blocks must be objects')
                if block.get('type') == 'text':
                    if not isinstance(block.get('text'), str):
                        raise ValueError('text block requires a string')
                    blocks.append({'type': 'text', 'text': block['text']})
                elif block.get('type') == 'image':
                    image_count += 1
                    if image_count > 1:
                        raise ValueError('this study permits one image per request')
                    if ('url' in block) == ('image' in block):
                        raise ValueError('image block requires exactly one local image or url path')
                    value = block.get('url', block.get('image'))
                    if (not isinstance(value, str) or not value or '\x00' in value
                            or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', value)
                            or Path(value).is_absolute()):
                        raise ValueError('image paths must be local and relative to data_root')
                    try:
                        path = (self.data_root / value).resolve(strict=True)
                        path.relative_to(self.data_root)
                    except (OSError, RuntimeError, ValueError) as exc:
                        raise ValueError('image path is missing or escapes data_root') from exc
                    if not path.is_file():
                        raise ValueError('image path must identify a regular local file')
                    blocks.append({'type': 'image', 'image': str(path)})
                else:
                    raise ValueError('only text and local image blocks are supported')
            result.append({'role': message['role'], 'content': blocks})
        return result

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            torch = importlib.import_module('torch')
            transformers = importlib.import_module('transformers')
        except ImportError as exc:
            raise RuntimeError('install and validate requirements-gpu.txt before GPU inference') from exc
        for name, module in [('torch', torch), ('transformers', transformers)]:
            if module.__version__ != self._config['software_versions'][name]:
                raise RuntimeError(f'{name} runtime version differs from recorded distribution metadata')
        if not torch.cuda.is_available() or int(self.device.split(':')[1]) >= torch.cuda.device_count():
            raise RuntimeError(f'the requested GPU {self.device} is unavailable')
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.bfloat16,
            device_map=None, trust_remote_code=False, attn_implementation='sdpa')
        model = model.to(self.device).eval()
        if model.dtype != torch.bfloat16:
            raise RuntimeError('loaded model dtype differs from the recorded BF16 setting')
        processor = transformers.AutoProcessor.from_pretrained(
            self.checkpoint, revision=self.revision, trust_remote_code=False)
        processor.image_processor.size = {
            'shortest_edge': self.min_pixels, 'longest_edge': self.max_pixels}
        self._torch, self._transformers = torch, transformers
        self._model, self._processor = model, processor

    def generate(self, messages, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens not in (80, 160):
            raise ValueError('study stages require max_new_tokens 80 or 160')
        normalized = self._messages_with_local_images(messages)
        self._ensure_loaded()
        self._torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        inputs = self._processor.apply_chat_template(
            normalized, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
            raise ValueError('processor produced a batch other than one')
        prompt_ids = input_ids.tolist()[0]
        inputs = inputs.to(self.device)
        generation = self._transformers.GenerationConfig(**copy.deepcopy(_GENERATION))
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs, generation_config=generation, max_new_tokens=max_new_tokens)
        self._torch.cuda.synchronize(self.device)
        output_rows = generated.tolist()
        if len(output_rows) != 1 or output_rows[0][:len(prompt_ids)] != prompt_ids:
            raise ValueError('model output batch or prompt prefix does not match the request')
        output_ids = output_rows[0][len(prompt_ids):]
        if len(output_ids) > max_new_tokens:
            raise ValueError('model returned more tokens than the stage cap')
        text = self._processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # EOS at the exact token cap is still a normal EOS stop. A shorter
        # sequence without EOS is unknown; never mislabel it as an EOS stop.
        finish_reason = ('stop' if output_ids and output_ids[-1] in _GENERATION['eos_token_id']
                         else 'length' if len(output_ids) == max_new_tokens else 'unknown')
        return {'text': text, 'input_tokens': len(prompt_ids), 'output_tokens': len(output_ids),
                'finish_reason': finish_reason, 'latency_s': time.perf_counter() - start}
