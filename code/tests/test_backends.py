"""Backend checks with local runtime fixtures."""
import copy
from contextlib import nullcontext
import importlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODEL = 'Qwen/Qwen3-VL-4B-Instruct'
REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
VERSIONS = {'torch': '2.9.0+fixture', 'transformers': '4.57.1+fixture',
            'torchvision': '0.24.0+fixture', 'accelerate': '1.0.0+fixture',
            'Pillow': '11.0.0+fixture'}


class TensorDouble:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def tolist(self):
        return copy.deepcopy(self.rows)


class BatchDouble(dict):
    def to(self, device):
        self.moved_to = device
        return self


class RuntimeDouble:
    """Imitates only documented HF/PyTorch boundaries; supplies no learned model."""
    def __init__(self):
        self.model_loads, self.processor_loads, self.calls = [], [], []
        self.chat_calls, self.decode_calls, self.syncs = [], [], []
        self.prompt_rows = [[11, 12, 13]]
        self.output_rows = [[11, 12, 13, 21, 22, 151645]]
        self.model = SimpleNamespace(dtype='bf16-fixture',
                                     to=self.move, eval=self.eval, generate=self.generate)
        self.processor = SimpleNamespace(image_processor=SimpleNamespace(size={}),
                                         apply_chat_template=self.template,
                                         batch_decode=self.decode)
        self.torch = SimpleNamespace(__version__=VERSIONS['torch'], bfloat16='bf16-fixture',
                                     inference_mode=nullcontext,
                                     cuda=SimpleNamespace(is_available=lambda: True,
                                         device_count=lambda: 1, synchronize=self.syncs.append))
        self.transformers = SimpleNamespace(__version__=VERSIONS['transformers'],
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=self.load_model),
            AutoProcessor=SimpleNamespace(from_pretrained=self.load_processor),
            GenerationConfig=lambda **kw: SimpleNamespace(**kw))

    def load_model(self, checkpoint, **kwargs):
        self.model_loads.append((checkpoint, kwargs)); return self.model

    def load_processor(self, checkpoint, **kwargs):
        self.processor_loads.append((checkpoint, kwargs)); return self.processor

    def move(self, device):
        self.model_device = device; return self.model

    def eval(self):
        self.eval_called = True; return self.model

    def template(self, messages, **kwargs):
        self.chat_calls.append((copy.deepcopy(messages), kwargs))
        self.batch = BatchDouble(input_ids=TensorDouble(self.prompt_rows),
                                 attention_mask=TensorDouble([[1, 1, 1]]))
        return self.batch

    def generate(self, **kwargs):
        self.calls.append(kwargs); return TensorDouble(self.output_rows)

    def decode(self, rows, **kwargs):
        self.decode_calls.append((copy.deepcopy(rows), kwargs))
        return ['AB' if row == [21, 22, 151645] else 'fixture output' for row in rows]


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.backends'),
                             'GPU backend implementation is not present yet')
        self.module = importlib.import_module('meme_study.backends')
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        (self.root / 'images' / 'candidate.jpg').write_bytes(b'test-only image placeholder')
        self.runtime = RuntimeDouble()
        self.patches = [patch.dict('sys.modules', {'torch': self.runtime.torch,
                       'transformers': self.runtime.transformers}),
                        patch('importlib.metadata.version', side_effect=VERSIONS.__getitem__)]
        for active in self.patches: active.start(); self.addCleanup(active.stop)
        self.backend = self.module.TransformersBackend(MODEL, REVISION, self.root)
        self.messages = [{'role': 'user', 'content': [
            {'type': 'image', 'url': 'images/candidate.jpg'},
            {'type': 'text', 'text': '仅输出给定材料的答案。'}]}]

    def test_construction_and_run_config_do_not_load_weights(self):
        config = self.backend.run_config
        self.assertEqual(self.runtime.model_loads, [])
        self.assertEqual(config['software_versions'], VERSIONS)
        self.assertEqual(config['revision'], REVISION)
        self.assertEqual(config['dtype'], 'bfloat16')
        self.assertEqual(config['generation']['do_sample'], False)
        config['generation']['do_sample'] = True
        self.assertFalse(self.backend.run_config['generation']['do_sample'])

    def test_normalizes_local_image_and_preserves_text_without_mutating_messages(self):
        original = copy.deepcopy(self.messages)
        self.backend.generate(self.messages, max_new_tokens=80)
        seen, kwargs = self.runtime.chat_calls[0]
        self.assertEqual(self.messages, original)
        self.assertEqual(seen[0]['content'][0], {'type': 'image',
            'image': str((self.root / 'images/candidate.jpg').resolve())})
        self.assertEqual(seen[0]['content'][1], original[0]['content'][1])
        self.assertEqual(kwargs, {'tokenize': True, 'add_generation_prompt': True,
                                  'return_dict': True, 'return_tensors': 'pt'})

    def test_passes_both_stage_caps_and_loads_pinned_single_device_without_remote_code(self):
        for cap in (80, 160): self.backend.generate(self.messages, max_new_tokens=cap)
        self.assertEqual([x['max_new_tokens'] for x in self.runtime.calls], [80, 160])
        self.assertEqual(len(self.runtime.model_loads), 1)
        model, kwargs = self.runtime.model_loads[0]
        self.assertEqual(model, MODEL)
        self.assertEqual(kwargs['revision'], REVISION)
        self.assertIs(kwargs['trust_remote_code'], False)
        self.assertEqual(kwargs['dtype'], 'bf16-fixture')
        self.assertIsNone(kwargs['device_map'])
        self.assertNotIn('quantization_config', kwargs)
        self.assertEqual(self.runtime.processor_loads[0][1]['revision'], REVISION)
        self.assertIs(self.runtime.processor_loads[0][1]['trust_remote_code'], False)
        self.assertEqual(self.runtime.model_device, 'cuda:0')
        self.assertEqual(self.runtime.batch.moved_to, 'cuda:0')
        self.assertTrue(self.runtime.eval_called)
        self.assertEqual(self.runtime.processor.image_processor.size,
                         {'shortest_edge': 262144, 'longest_edge': 1310720})
        generation = self.runtime.calls[0]['generation_config']
        self.assertFalse(generation.do_sample)
        self.assertEqual((generation.num_beams, generation.num_return_sequences), (1, 1))

    def test_strips_prompt_tokens_and_counts_returned_eos_token(self):
        result = self.backend.generate(self.messages, max_new_tokens=160)
        self.assertEqual(result['text'], 'AB')
        self.assertEqual((result['input_tokens'], result['output_tokens']), (3, 3))
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertGreaterEqual(result['latency_s'], 0)
        self.assertEqual(self.runtime.decode_calls[0][0], [[21, 22, 151645]])
        self.assertEqual(self.runtime.decode_calls[0][1],
                         {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False})
        self.assertEqual(self.runtime.syncs, ['cuda:0', 'cuda:0'])

    def test_length_without_eos_and_eos_exactly_at_cap_have_different_finish_reasons(self):
        for tail, expected in [([21] * 80, 'length'),
                               ([21] * 79 + [151643], 'stop'), ([21], 'unknown')]:
            with self.subTest(expected=expected):
                self.runtime.output_rows = [[11, 12, 13] + tail]
                result = self.backend.generate(self.messages, max_new_tokens=80)
                self.assertEqual(result['finish_reason'], expected)
                self.assertEqual(result['output_tokens'], len(tail))

    def test_remote_absolute_traversal_missing_and_symlink_images_fail_before_loading(self):
        outside = self.root.parent / (self.root.name + '-outside.jpg')
        outside.write_bytes(b'outside'); self.addCleanup(outside.unlink)
        (self.root / 'images' / 'link.jpg').symlink_to(outside)
        for value in ['https://example.com/image.jpg', 'file:///tmp/image.jpg',
                      str(self.root / 'images/candidate.jpg'), '../' + outside.name,
                      'images/link.jpg', 'images/missing.jpg']:
            with self.subTest(path=value):
                self.messages[0]['content'][0]['url'] = value
                with self.assertRaises(ValueError):
                    self.backend.generate(self.messages, max_new_tokens=160)
        self.assertEqual(self.runtime.model_loads, [])

    def test_text_only_request_is_supported_without_adding_an_image(self):
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'caption'}]}]
        self.backend.generate(messages, max_new_tokens=160)
        self.assertEqual(self.runtime.chat_calls[0][0], messages)

    def test_batched_or_multi_image_messages_and_invalid_caps_are_rejected(self):
        multi_image = copy.deepcopy(self.messages)
        multi_image[0]['content'].append({'type': 'image', 'url': 'images/candidate.jpg'})
        for messages in [[self.messages, self.messages], multi_image]:
            with self.assertRaises(ValueError): self.backend.generate(messages, max_new_tokens=80)
        for cap in [0, 1, 159, True, 80.0]:
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError): self.backend.generate(self.messages, max_new_tokens=cap)
        self.assertEqual(self.runtime.model_loads, [])

    def test_processor_or_model_batch_expansion_cannot_silently_score_first_item(self):
        self.runtime.prompt_rows = [[11, 12, 13], [11, 12, 13]]
        with self.assertRaises(ValueError): self.backend.generate(self.messages, max_new_tokens=80)
        self.assertEqual(self.runtime.calls, [])
        self.runtime.prompt_rows = [[11, 12, 13]]
        self.runtime.output_rows = [[11, 12, 13, 21], [11, 12, 13, 22]]
        with self.assertRaises(ValueError): self.backend.generate(self.messages, max_new_tokens=80)

    def test_wrong_prompt_prefix_and_exceeded_token_cap_do_not_produce_receipts(self):
        for output in [[99, 12, 13, 21], [11, 12, 13] + [21] * 81]:
            self.runtime.output_rows = [output]
            with self.assertRaises(ValueError): self.backend.generate(self.messages, max_new_tokens=80)

    def test_unknown_checkpoint_revision_device_and_pixel_bounds_fail_closed(self):
        for changes in [{'checkpoint': 'someone/custom-model'}, {'revision': 'main'},
                        {'device': 'cpu'}, {'device': 'cuda'}, {'max_pixels': 0},
                        {'min_pixels': 1310721}, {'max_pixels': True}]:
            args = dict(checkpoint=MODEL, revision=REVISION, data_root=self.root)
            args.update(changes)
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError): self.module.TransformersBackend(**args)

    def test_cuda_unavailable_or_unlogged_runtime_version_fails_without_model_loading(self):
        self.runtime.torch.cuda.is_available = lambda: False
        with self.assertRaises(RuntimeError): self.backend.generate(self.messages, max_new_tokens=80)
        self.runtime.torch.cuda.is_available = lambda: True
        self.runtime.torch.__version__ = 'unrecorded-runtime'
        with self.assertRaises(RuntimeError): self.backend.generate(self.messages, max_new_tokens=80)
        self.assertEqual(self.runtime.model_loads, [])


class LauncherTests(unittest.TestCase):
    def test_reference_job_forwards_model_task_split_and_paths(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'reference_job.sh'
        with tempfile.TemporaryDirectory() as directory:
            double = Path(directory) / 'python-fixture'
            double.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            double.chmod(0o700)
            launched = subprocess.run(
                ['/bin/bash', str(script), str(double), 'study prepared', 'data root',
                 'shared runs', MODEL, 'explanation', 'dev'],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(launched.stdout.splitlines(), [
                '-m', 'meme_study.reference_cli', 'run', '--prepared', 'study prepared',
                '--data-root', 'data root', '--runs', 'shared runs', '--model', MODEL,
                '--task', 'explanation', '--split', 'dev',
            ])


if __name__ == '__main__':
    unittest.main()
