"""CPU contracts and image preprocessing only; no weights or CUDA inference."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from test_backends import RuntimeDouble, TensorDouble


QWEN = 'Qwen/Qwen3-VL-4B-Instruct'
QWEN_REV = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
JUDGE = 'OpenGVLab/InternVL3-9B'
JUDGE_REV = '5f618513e35a9b85922341b8057feddfc8880e50'
MEMORY_SCOPE = 'current CUDA process / torch allocator, not whole-device utilization'


class CudaResourceFixture:
    """CPU-only stub at the public CUDA boundary; these are not GPU measurements."""
    def __init__(self, index=0):
        self.index, self.resets = index, 0
        self.allocated, self.reserved = 16, 32
        self.peak_allocated, self.peak_reserved = 900, 1000

    def _device(self, device):
        if device not in (self.index, f'cuda:{self.index}'):
            raise AssertionError('telemetry queried a different CUDA device')

    def is_available(self):
        return True

    def device_count(self):
        return self.index + 1

    def synchronize(self, device):
        self._device(device)

    def get_device_properties(self, device):
        self._device(device)
        return SimpleNamespace(name='CPU test fixture; not an H100', total_memory=4096)

    def memory_allocated(self, device):
        self._device(device)
        return self.allocated

    def memory_reserved(self, device):
        self._device(device)
        return self.reserved

    def max_memory_allocated(self, device):
        self._device(device)
        return self.peak_allocated

    def max_memory_reserved(self, device):
        self._device(device)
        return self.peak_reserved

    def reset_peak_memory_stats(self, device):
        self._device(device)
        self.resets += 1
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    def record_usage(self, allocated, reserved):
        self.allocated, self.reserved = allocated, reserved
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)


class TemplateFixture:
    """The inspected remote Conversation boundary, without importing remote code."""
    name = 'internvl2_5'
    sep = '<|im_end|>\n'
    roles = ('<|im_start|>user\n', '<|im_start|>assistant\n')

    def __init__(self):
        self.messages = []
        self.system_message = 'system fixture'

    def copy(self):
        return copy.deepcopy(self)

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        result = '<|im_start|>system\n' + self.system_message + self.sep
        for role, message in self.messages:
            result += role + (message + self.sep if message else '')
        return result


class ReferenceBackendTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.reference_backends'),
                             'reference GPU adapters are not implemented yet')
        self.module = importlib.import_module('meme_study.reference_backends')
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        Image.new('RGB', (896, 448), 'white').save(self.root / 'meme.png')
        self.messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': 'meme.png'},
            {'type': 'text', 'text': '按给定清单评价候选回答。'}]}]

    def test_construction_is_lazy_and_configs_are_separate_immutable_run_identities(self):
        with patch('importlib.import_module', side_effect=AssertionError('eager runtime import')):
            qwen = self.module.create_backend(QWEN, self.root)
            judge = self.module.create_backend(JUDGE, self.root)
            config = judge.run_config
        self.assertEqual(qwen.run_config['stage_token_caps'],
                         {'explanation': 512, 'pair': 512, 'permutation': 512})
        self.assertEqual(config['checkpoint'], JUDGE)
        self.assertEqual(config['revision'], JUDGE_REV)
        self.assertEqual(config['code_revision'], JUDGE_REV)
        self.assertEqual(config['stage_token_caps'], {'judge': 768})
        self.assertEqual(config['image_processing']['max_total_patches'], 5)
        config['generation']['do_sample'] = True
        self.assertFalse(judge.run_config['generation']['do_sample'])
        self.assertIsNone(judge._model)

    def test_import_and_construction_do_not_import_torch_transformers_or_remote_code(self):
        code = """import sys, tempfile, json
from meme_study.reference_backends import create_backend
with tempfile.TemporaryDirectory() as root:
    for model in ['Qwen/Qwen3-VL-4B-Instruct', 'OpenGVLab/InternVL3-9B']:
        json.dumps(create_backend(model, root).run_config)
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert not any('transformers_modules' in name for name in sys.modules)
"""
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_resource_snapshot_before_prepare_is_unavailable_without_loading(self):
        for checkpoint in (QWEN, JUDGE):
            backend = self.module.create_backend(checkpoint, self.root)
            self.assertTrue(callable(getattr(backend, 'resource_snapshot', None)),
                            'reference backends must expose read-only resource snapshots')
            with patch('importlib.import_module', side_effect=AssertionError('runtime import')):
                snapshot = backend.resource_snapshot()
            self.assertFalse(snapshot['available'])
            self.assertEqual(snapshot['reason'], 'backend_not_prepared')
            self.assertEqual(snapshot['memory_scope'], MEMORY_SCOPE)
            self.assertNotIn('device_name', snapshot)
            json.dumps(snapshot, allow_nan=False)

    def test_resource_snapshot_reads_selected_device_without_model_calls_or_reset(self):
        for checkpoint in (QWEN, JUDGE):
            with self.subTest(checkpoint=checkpoint):
                backend = self.module.create_backend(checkpoint, self.root, device='cuda:1')
                cuda = CudaResourceFixture(index=1)
                backend._model = object()  # No generation interface exists on this sentinel.
                backend._torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='12.8-fixture'))
                self.assertTrue(callable(getattr(backend, 'resource_snapshot', None)))
                snapshot = backend.resource_snapshot()
                self.assertTrue(snapshot['available'])
                self.assertEqual(snapshot['device_index'], 1)
                self.assertEqual(snapshot['device_name'], 'CPU test fixture; not an H100')
                self.assertEqual(snapshot['total_memory_bytes'], 4096)
                self.assertEqual(snapshot['torch_cuda_version'], '12.8-fixture')
                self.assertEqual(snapshot['torch_memory_allocated_bytes'], 16)
                self.assertEqual(snapshot['torch_memory_reserved_bytes'], 32)
                self.assertEqual(snapshot['torch_peak_allocated_bytes'], 900)
                self.assertEqual(snapshot['torch_peak_reserved_bytes'], 1000)
                self.assertEqual(snapshot['memory_scope'], MEMORY_SCOPE)
                self.assertEqual(cuda.resets, 0)
                json.dumps(snapshot, allow_nan=False)

    def test_missing_cuda_sampling_methods_do_not_invent_resource_values(self):
        runtime = RuntimeDouble()  # Legacy CPU fixture deliberately has no memory sampling API.
        backend = self.module.create_backend(QWEN, self.root)
        backend._model, backend._processor = runtime.model, runtime.processor
        backend._torch, backend._transformers = runtime.torch, runtime.transformers
        result = backend.generate(self.messages, 512)
        self.assertIn('resource_usage', result)
        usage = result['resource_usage']
        self.assertFalse(usage['available'])
        self.assertEqual(usage['reason'], 'cuda_snapshot_api_unavailable')
        self.assertFalse(usage['peak_stats_reset'])
        self.assertNotIn('torch_memory_allocated_bytes', usage['after'])
        self.assertIn('memory_allocated', usage['after']['missing_methods'])

    def test_qwen_generation_resets_each_peak_after_loading_and_preserves_latency_scope(self):
        runtime = RuntimeDouble()
        cuda = CudaResourceFixture()
        runtime.torch.cuda = cuda
        runtime.torch.version = SimpleNamespace(cuda='12.8-fixture')
        runtime.model.requires_grad_ = lambda value: runtime.model
        runtime.transformers.__version__ = '4.57.1'
        backend = self.module.create_backend(QWEN, self.root)
        backend._config['software_versions']['torch'] = runtime.torch.__version__
        backend._config['software_versions']['transformers'] = '4.57.1'
        def preprocess(*args, **kwargs):
            # First preprocessing peak exceeds generation and must survive the reset window.
            cuda.record_usage(72 if cuda.resets == 1 else 20, 96 if cuda.resets == 1 else 36)
            cuda.record_usage(16, 32)
            return runtime.template(*args, **kwargs)
        runtime.processor.apply_chat_template = preprocess
        def generate(**kwargs):
            self.assertEqual(len(runtime.model_loads), 1)
            cuda.record_usage(48 if cuda.resets == 1 else 24, 80 if cuda.resets == 1 else 40)
            cuda.record_usage(16, 32)
            return runtime.generate(**kwargs)
        runtime.model.generate = generate
        with patch.dict('sys.modules', {'torch': runtime.torch, 'transformers': runtime.transformers}), \
                patch.object(self.module.time, 'perf_counter', side_effect=[10., 12., 20., 23.]):
            first = backend.generate(self.messages, 512)
            second = backend.generate(self.messages, 512)
        self.assertIn('resource_usage', first)
        for result, expected_peak, expected_reserved in ((first, 72, 96), (second, 24, 40)):
            usage = result['resource_usage']
            self.assertTrue(usage['available'])
            self.assertTrue(usage['peak_stats_reset'])
            self.assertIn('loaded-model baseline', usage['peak_window'])
            self.assertEqual(usage['before']['torch_peak_allocated_bytes'], 16)
            self.assertEqual(usage['before']['torch_peak_reserved_bytes'], 32)
            self.assertEqual(usage['after']['torch_peak_allocated_bytes'], expected_peak)
            self.assertEqual(usage['after']['torch_peak_reserved_bytes'], expected_reserved)
            self.assertEqual(usage['after']['torch_memory_allocated_bytes'], 16)
            json.dumps(result, allow_nan=False)
        self.assertEqual(cuda.resets, 2)
        self.assertEqual((first['latency_s'], second['latency_s']), (2., 3.))

    def test_cuda_snapshot_read_failure_is_explicit_and_contains_no_measurements(self):
        backend = self.module.create_backend(JUDGE, self.root)
        backend._model = object()
        cuda = CudaResourceFixture()
        backend._torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='12.8-fixture'))
        self.assertTrue(callable(getattr(backend, 'resource_snapshot', None)))
        with patch.object(cuda, 'memory_reserved', side_effect=RuntimeError('device lost')):
            snapshot = backend.resource_snapshot()
        self.assertFalse(snapshot['available'])
        self.assertEqual(snapshot['reason'], 'cuda_snapshot_failed')
        self.assertNotIn('torch_memory_allocated_bytes', snapshot)

    def test_unavailable_cuda_device_does_not_report_fixture_hardware(self):
        for availability, count in ((False, 1), (True, 0)):
            backend = self.module.create_backend(JUDGE, self.root)
            backend._model = object()
            cuda = CudaResourceFixture()
            backend._torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='12.8-fixture'))
            with patch.object(cuda, 'is_available', return_value=availability), \
                    patch.object(cuda, 'device_count', return_value=count):
                snapshot = backend.resource_snapshot()
            self.assertFalse(snapshot['available'])
            self.assertEqual(snapshot['reason'], 'cuda_device_unavailable')
            self.assertNotIn('device_name', snapshot)

    def test_missing_or_failed_peak_reset_cannot_be_mislabeled_as_per_call_peak(self):
        for reset in (None, unittest.mock.Mock(side_effect=RuntimeError('reset failed'))):
            runtime = RuntimeDouble()
            cuda = CudaResourceFixture()
            runtime.torch.cuda = cuda
            runtime.torch.version = SimpleNamespace(cuda='12.8-fixture')
            backend = self.module.create_backend(QWEN, self.root)
            backend._model, backend._processor = runtime.model, runtime.processor
            backend._torch, backend._transformers = runtime.torch, runtime.transformers
            with patch.object(cuda, 'reset_peak_memory_stats', reset):
                result = backend.generate(self.messages, 512)
            usage = result['resource_usage']
            self.assertFalse(usage['available'])
            self.assertFalse(usage['peak_stats_reset'])
            self.assertIn(usage['reason'], ('cuda_peak_reset_api_unavailable', 'cuda_peak_reset_failed'))
            self.assertNotIn('peak_window', usage)
            self.assertEqual(usage['after']['torch_peak_allocated_bytes'], 900)
            self.assertEqual(result['output_tokens'], 3)

    def test_caps_and_judge_input_errors_fail_before_loading(self):
        qwen = self.module.create_backend(QWEN, self.root)
        judge = self.module.create_backend(JUDGE, self.root)
        for backend, caps in [(qwen, [80, 160, 768, True, 512.0]),
                              (judge, [80, 512, True, 768.0])]:
            for cap in caps:
                with self.subTest(cap=cap, model=backend.checkpoint):
                    with self.assertRaises(ValueError): backend.generate(self.messages, cap)
        for messages in [
            [{'role': 'user', 'content': [{'type': 'text', 'text': 'no original image'}]}],
            self.messages + [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'history'}]}],
            [{'role': 'user', 'content': [{'type': 'image', 'image': '../outside.png'},
                                          {'type': 'text', 'text': 'x'}]}],
            [{'role': 'user', 'content': [{'type': 'image', 'image': 'meme.png'},
                                          {'type': 'text', 'text': 'injected <IMG_CONTEXT>'}]}],
        ]:
            with self.assertRaises(ValueError): judge.generate(messages, 768)
        self.assertIsNone(qwen._model)
        self.assertIsNone(judge._model)

    def test_unknown_model_revision_or_unbounded_tiling_is_rejected(self):
        for kwargs in [{'checkpoint': 'someone/InternVL3-9B'}, {'revision': 'main'},
                       {'max_tiles': 0}, {'max_tiles': 12}, {'max_tiles': True},
                       {'device': 'cpu'}]:
            args = dict(checkpoint=JUDGE, revision=JUDGE_REV, data_root=self.root)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError): self.module.InternVLJudgeBackend(**args)
        with self.assertRaises(ValueError): self.module.create_backend('9B', self.root)

    def test_qwen_receipt_counts_generated_ids_and_eos_at_512_without_retokenizing(self):
        runtime = RuntimeDouble()
        runtime.model.requires_grad_ = lambda value: runtime.model
        runtime.transformers.__version__ = '4.57.1'
        backend = self.module.create_backend(QWEN, self.root)
        backend._config['software_versions']['torch'] = runtime.torch.__version__
        backend._config['software_versions']['transformers'] = '4.57.1'
        runtime.output_rows = [[11, 12, 13] + [21] * 511 + [151645]]
        with patch.dict('sys.modules', {'torch': runtime.torch, 'transformers': runtime.transformers}):
            result = backend.generate(self.messages, 512)
        self.assertEqual((result['input_tokens'], result['output_tokens'], result['finish_reason']),
                         (3, 512, 'stop'))
        self.assertEqual(runtime.calls[0]['max_new_tokens'], 512)
        self.assertFalse(runtime.calls[0]['generation_config'].do_sample)
        self.assertEqual(runtime.model_loads[0][1]['revision'], QWEN_REV)

    def test_qwen_text_only_has_no_invented_visual_input(self):
        runtime = RuntimeDouble()
        backend = self.module.create_backend(QWEN, self.root)
        backend._model, backend._processor = runtime.model, runtime.processor
        backend._torch, backend._transformers = runtime.torch, runtime.transformers
        text_only = [{'role': 'user', 'content': [{'type': 'text', 'text': 'OCR文本'}]}]
        backend.generate(text_only, 512)
        self.assertEqual(runtime.chat_calls[0][0], text_only)

    def test_receipts_validate_real_token_sequences_not_decoded_text_lengths(self):
        receipt = self.module.token_receipt([3, 4, 5], [9, 128131], max_new_tokens=2,
                                           eos_token_ids=[128131])
        self.assertEqual(receipt, {'input_tokens': 3, 'output_tokens': 2, 'finish_reason': 'stop'})
        self.assertEqual(self.module.token_receipt([3], [9, 9], max_new_tokens=2,
                         eos_token_ids=[128131])['finish_reason'], 'length')
        self.assertEqual(self.module.token_receipt([3], [9], max_new_tokens=2,
                         eos_token_ids=[128131])['finish_reason'], 'unknown')
        for invalid in [[9, 9, 9], [128131, 9], [True], [-1]]:
            with self.assertRaises(ValueError):
                self.module.token_receipt([3], invalid, max_new_tokens=2, eos_token_ids=[128131])

    def test_tiling_matches_hand_calculated_wide_and_small_square_patch_counts(self):
        wide = Image.new('RGB', (1792, 448), 'red')
        patches = self.module.internvl_image_tiles(wide, max_tiles=4)
        self.assertEqual(len(patches), 5)
        self.assertTrue(all(p.size == (448, 448) and p.mode == 'RGB' for p in patches))
        self.assertEqual(len(self.module.internvl_image_tiles(
            Image.new('L', (100, 100)), max_tiles=4)), 1)
        self.assertEqual(len(self.module.internvl_image_tiles(
            Image.new('RGB', (2688, 448)), max_tiles=6)), 7)

    def test_tiling_preserves_left_to_right_crop_order_and_thumbnail(self):
        original = Image.new('RGB', (896, 448), 'red')
        original.paste('blue', (448, 0, 896, 448))
        tiles = self.module.internvl_image_tiles(original, max_tiles=4)
        self.assertEqual(len(tiles), 3)
        self.assertEqual(tiles[0].getpixel((224, 224)), (255, 0, 0))
        self.assertEqual(tiles[1].getpixel((224, 224)), (0, 0, 255))
        self.assertEqual(tiles[2].getpixel((10, 224)), (255, 0, 0))
        self.assertEqual(tiles[2].getpixel((438, 224)), (0, 0, 255))

    def test_prompt_uses_remote_template_and_exact_patch_count_without_mutating_it(self):
        template = TemplateFixture()
        prompt = self.module.internvl_query('候选文本', template, num_patches=3)
        self.assertTrue(prompt.startswith('<|im_start|>system\nsystem fixture<|im_end|>\n'))
        self.assertIn('<img>' + '<IMG_CONTEXT>' * 768 + '</img>\n候选文本', prompt)
        self.assertTrue(prompt.endswith('<|im_end|>\n<|im_start|>assistant\n'))
        self.assertEqual(template.messages, [])

    def test_metadata_guard_rejects_wrong_template_visual_count_and_special_token(self):
        model = SimpleNamespace(template='internvl2_5', num_image_token=256,
            config=SimpleNamespace(force_image_size=448), conv_template=TemplateFixture())
        ids = {'<|im_end|>': 128131, '<img>': 128133, '</img>': 128134, '<IMG_CONTEXT>': 128135}
        tokenizer = SimpleNamespace(convert_tokens_to_ids=ids.__getitem__, bos_token_id=1, pad_token_id=2)
        self.module.validate_internvl_contract(model, tokenizer)
        for obj, key, bad in [(model, 'num_image_token', 64), (model, 'template', 'Hermes-2'),
                              (tokenizer, 'pad_token_id', 99)]:
            prior = getattr(obj, key); setattr(obj, key, bad)
            with self.assertRaises(ValueError): self.module.validate_internvl_contract(model, tokenizer)
            setattr(obj, key, prior)
        ids['<IMG_CONTEXT>'] = 0
        with self.assertRaises(ValueError): self.module.validate_internvl_contract(model, tokenizer)

    def test_internvl_version_mismatch_fails_before_any_model_download(self):
        backend = self.module.create_backend(JUDGE, self.root)
        backend._config['software_versions']['transformers'] = '4.57.1'
        with self.assertRaisesRegex(RuntimeError, '4.48.3'):
            backend.generate(self.messages, 768)
        self.assertIsNone(backend._model)

    def test_judge_generate_uses_expanded_tensor_and_generated_only_output(self):
        """A strict boundary fixture catches prompt slicing or text retokenization."""
        runtime = RuntimeDouble()
        cuda = CudaResourceFixture()
        runtime.torch.cuda = cuda
        runtime.torch.version = SimpleNamespace(cuda='12.8-fixture')
        backend = self.module.create_backend(JUDGE, self.root)
        prompts, decoded, calls = [], [], []

        class MovableTensor(TensorDouble):
            def to(self, *args, **kwargs):
                return self

        class Pixels:
            def __init__(self, patches):
                self.shape = (len(patches), 3, 448, 448)

            def to(self, **kwargs):
                if kwargs != {'device': 'cuda:0', 'dtype': 'bf16-fixture'}:
                    raise AssertionError('pixels not transferred as recorded')
                return self

        def tokenize(prompt, *, return_tensors):
            if return_tensors != 'pt':
                raise AssertionError('must request tensor tokenization')
            prompts.append(prompt)
            ids = [1, 3] + [128135] * prompt.count('<IMG_CONTEXT>') + [4]
            return {'input_ids': MovableTensor([ids]),
                    'attention_mask': MovableTensor([[1] * len(ids)])}

        def generate(**kwargs):
            calls.append(kwargs)
            if kwargs['pixel_values'].shape != (3, 3, 448, 448):
                raise AssertionError('the wide image must include two tiles and its thumbnail')
            cuda.record_usage(48, 64)
            return TensorDouble([[21] * 767 + [128131]])

        def decode(rows, **kwargs):
            decoded.append(copy.deepcopy(rows))
            return ['  {"items":[]}  ']

        runtime.torch.stack = Pixels
        backend._model = SimpleNamespace(conv_template=TemplateFixture(), system_message='system fixture',
                                          generate=generate)
        backend._tokenizer = SimpleNamespace(__call__=tokenize, batch_decode=decode)
        # Special methods are resolved on types; give the external tokenizer boundary a callable type.
        class Tokenizer:
            __call__ = staticmethod(tokenize)
            batch_decode = staticmethod(decode)
        backend._tokenizer = Tokenizer()
        backend._torch, backend._transformers = runtime.torch, runtime.transformers
        backend._transform = lambda tile: tile
        result = backend.generate(self.messages, 768)
        self.assertEqual(result['text'], '{"items":[]}')
        self.assertEqual((result['input_tokens'], result['output_tokens'], result['finish_reason']),
                         (771, 768, 'stop'))
        self.assertEqual(len(decoded[0][0]), 768)
        self.assertEqual(calls[0]['input_ids'].shape, (1, 771))
        self.assertEqual(calls[0]['generation_config'].eos_token_id, 128131)
        self.assertNotIn('use_cache', calls[0])  # The remote wrapper supplies it exactly once.
        self.assertIn('resource_usage', result)
        self.assertEqual(result['resource_usage']['before']['torch_peak_allocated_bytes'], 16)
        self.assertEqual(result['resource_usage']['after']['torch_peak_allocated_bytes'], 48)
        self.assertEqual(result['resource_usage']['after']['torch_peak_reserved_bytes'], 64)
        self.assertEqual(cuda.resets, 1)


if __name__ == '__main__':
    unittest.main()
