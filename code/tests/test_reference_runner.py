"""Synthetic CPU receipts only: no weights, network, GPU or research outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from meme_study.data import content_hash


def job(name='one', model='model4', kind='explanation', **changes):
    row = {'protocol_id': 'reference_reuse_v1', 'job_id': name, 'model': model,
           'task_kind': kind, 'family_id': 'synthetic-family', 'split': 'dev',
           'template_group': 'synthetic-family', 'condition': 'image_only',
           'max_new_tokens': 768 if kind == 'judge' else 512, 'readiness': 'ready',
           'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': '合成测试输入'}]}]}
    row.update(changes)
    row['input_sha256'] = content_hash(row)
    return row


class ReceiptBackend:
    def __init__(self, model='model4', *, effect=None):
        self.run_config = {'checkpoint': model, 'revision': 'synthetic-revision', 'device': 'cpu-test'}
        self.calls = []
        self.effect = effect

    def generate(self, messages, max_new_tokens):
        self.calls.append((copy.deepcopy(messages), max_new_tokens))
        if self.effect is not None:
            return self.effect(messages, max_new_tokens)
        return {'text': '合成输出，并非真实模型结果', 'input_tokens': 11, 'output_tokens': 12,
                'finish_reason': 'stop', 'latency_s': .25}


class ReferenceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.reference_runner'),
                             'reference runner is not implemented')
        from meme_study.reference_runner import run_reference_jobs
        self.run = run_reference_jobs
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'study'

    def execute(self, jobs, backend, **kwargs):
        args = {'batch_id': 'batch4', 'study_id': 'synthetic-study', 'attempt_cap': 913}
        args.update(kwargs)
        return self.run(jobs, backend, self.root, **args)

    def ledger(self):
        return [json.loads(line) for line in (self.root / 'reference-ledger.jsonl').read_text().splitlines()]

    def test_completed_batch_resumes_without_any_additional_generation(self):
        backend = ReceiptBackend()
        jobs = [job('one'), job('two')]
        first = self.execute(jobs, backend)
        before = (self.root / 'reference-ledger.jsonl').read_bytes()
        second = self.execute(jobs, backend)
        self.assertEqual(first, second)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(before, (self.root / 'reference-ledger.jsonl').read_bytes())
        self.assertEqual([r['status'] for r in first], ['ok', 'ok'])
        self.assertEqual([r['attempt'] for r in first], [1, 2])
        self.assertEqual(first[0]['text'], '合成输出，并非真实模型结果')
        self.assertEqual(first[0]['input_sha256'], jobs[0]['input_sha256'])

    def test_two_model_batches_share_one_global_attempt_limit(self):
        a, b = ReceiptBackend(), ReceiptBackend('model8')
        self.execute([job('a'), job('b')], a, attempt_cap=3)
        result = self.execute([job('c', 'model8'), job('d', 'model8')], b,
                              batch_id='batch8', attempt_cap=3)
        self.assertEqual((len(a.calls), len(b.calls)), (2, 1))
        self.assertEqual(result[0]['attempt'], 3)
        self.assertEqual(result[1]['status'], 'failed')
        self.assertEqual(result[1]['error_type'], 'BudgetExhausted')
        self.assertIsNone(result[1]['attempt'])
        self.assertEqual(result[1]['input_sha256'], job('d', 'model8')['input_sha256'])
        self.assertEqual(sum(e['event'] == 'dispatch' for e in self.ledger()), 3)
        self.assertEqual(result, self.execute([job('c', 'model8'), job('d', 'model8')], b,
                         batch_id='batch8', attempt_cap=3))
        self.assertEqual(len(b.calls), 1)

    def test_same_job_cannot_be_scored_again_under_a_different_batch(self):
        backend = ReceiptBackend()
        self.execute([job()], backend)
        with self.assertRaises(ValueError):
            self.execute([job()], backend, batch_id='other')
        self.assertEqual(len(backend.calls), 1)

    def test_input_or_job_order_change_refuses_batch_resume(self):
        backend = ReceiptBackend()
        self.execute([job('one'), job('two')], backend)
        changed = job('one', condition='hinted')
        for jobs in ([changed, job('two')], [job('two'), job('one')], [job('one')]):
            with self.subTest(jobs=jobs):
                with self.assertRaises(ValueError):
                    self.execute(jobs, backend)
        self.assertEqual(len(backend.calls), 2)

    def test_backend_config_change_refuses_resume(self):
        backend = ReceiptBackend()
        self.execute([job()], backend)
        backend.run_config['revision'] = 'changed'
        with self.assertRaises(ValueError):
            self.execute([job()], backend)
        self.assertEqual(len(backend.calls), 1)

    def test_study_id_and_fixed_cap_changes_refuse_reuse(self):
        backend = ReceiptBackend()
        self.execute([job()], backend, attempt_cap=3)
        for change in ({'study_id': 'different', 'attempt_cap': 3}, {'attempt_cap': 4}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.execute([job()], backend, **change)
        self.assertEqual(len(backend.calls), 1)

    def test_pending_material_or_wrong_protocol_is_rejected_before_generation(self):
        backend = ReceiptBackend()
        for bad in (job(kind='pair', readiness='pending_material_check'),
                    job(protocol_id='legacy_drv_v1')):
            with self.subTest(job=bad):
                with self.assertRaises(ValueError):
                    self.execute([bad], backend)
        self.assertEqual(backend.calls, [])
        self.assertFalse(self.root.exists())

    def test_bad_input_hash_or_task_token_cap_cannot_consume_an_attempt(self):
        backend = ReceiptBackend()
        bad_hash = job(); bad_hash['messages'][0]['content'][0]['text'] = 'changed without hash'
        for bad in (bad_hash, job(max_new_tokens=160), job(kind='judge', max_new_tokens=512),
                    job(kind='unexpected'), job(max_new_tokens=True)):
            with self.subTest(job=bad):
                with self.assertRaises(ValueError):
                    self.execute([bad], backend)
        self.assertEqual(backend.calls, [])

    def test_empty_duplicate_mixed_model_or_checkpoint_mismatch_is_rejected(self):
        backend = ReceiptBackend()
        for jobs in ([], [job(), job()], [job('a'), job('b', 'model8')], [job(model='other')]):
            with self.subTest(jobs=jobs):
                with self.assertRaises(ValueError):
                    self.execute(jobs, backend)
        self.assertEqual(backend.calls, [])

    def test_attempt_cap_must_be_a_positive_integer_at_most_913(self):
        backend = ReceiptBackend()
        for cap in (0, -1, 914, True, 1.5):
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    self.execute([job()], backend, attempt_cap=cap)
        self.assertEqual(backend.calls, [])

    def test_backend_failure_consumes_one_attempt_and_never_retries(self):
        def fail(messages, cap):
            raise RuntimeError('synthetic backend failure')
        backend = ReceiptBackend(effect=fail)
        first = self.execute([job()], backend, attempt_cap=1)
        self.assertEqual(first[0]['status'], 'failed')
        self.assertEqual(first[0]['job_id'], 'one')
        self.assertEqual(first[0]['input_sha256'], job()['input_sha256'])
        self.assertEqual(first[0]['error_type'], 'RuntimeError')
        self.assertIn('synthetic backend failure', first[0]['error'])
        self.assertEqual(first, self.execute([job()], backend, attempt_cap=1))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(sum(e['event'] == 'dispatch' for e in self.ledger()), 1)

    def test_interruption_leaves_durable_dispatch_and_is_never_reissued(self):
        def interrupt(messages, cap):
            self.assertEqual(sum(e['event'] == 'dispatch' for e in self.ledger()), 1)
            raise KeyboardInterrupt('synthetic interruption')
        backend = ReceiptBackend(effect=interrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.execute([job()], backend, attempt_cap=2)
        self.assertEqual(self.ledger()[-1]['event'], 'dispatch')
        resumed_backend = ReceiptBackend()
        resumed = self.execute([job()], resumed_backend, attempt_cap=2)
        self.assertEqual(resumed[0]['status'], 'interrupted')
        self.assertEqual(resumed[0]['job_id'], 'one')
        self.assertEqual(resumed[0]['input_sha256'], job()['input_sha256'])
        self.assertEqual(resumed[0]['attempt'], 1)
        self.assertEqual(resumed_backend.calls, [])
        self.assertEqual(resumed, self.execute([job()], resumed_backend, attempt_cap=2))
        self.assertEqual(sum(e['event'] == 'dispatch' for e in self.ledger()), 1)

    def test_bad_receipts_are_failed_with_raw_evidence_and_not_retried(self):
        valid = ReceiptBackend().generate([], 512)
        bad = [None, 'not a receipt', dict(valid, text=2), dict(valid, output_tokens=513),
               dict(valid, input_tokens=True), dict(valid, latency_s=-1),
               dict(valid, latency_s=float('nan')), dict(valid, finish_reason='')]
        for index, receipt in enumerate(bad):
            with self.subTest(receipt=receipt):
                backend = ReceiptBackend(effect=lambda messages, cap, value=receipt: value)
                jobs = [job(f'bad-{index}')]
                result = self.execute(jobs, backend, batch_id=f'bad-{index}')
                self.assertEqual(result[0]['status'], 'failed')
                self.assertIn('raw_receipt', result[0])
                self.assertEqual(result, self.execute(jobs, backend, batch_id=f'bad-{index}'))
                self.assertEqual(len(backend.calls), 1)

    def test_generation_content_does_not_trigger_answer_repair(self):
        valid = ReceiptBackend().generate([], 512)
        backend = ReceiptBackend(effect=lambda messages, cap: dict(valid, text='invalid JSON answer'))
        result = self.execute([job(kind='pair')], backend)
        self.assertEqual(result[0]['status'], 'ok')
        self.assertEqual(result[0]['text'], 'invalid JSON answer')
        self.execute([job(kind='pair')], backend)
        self.assertEqual(len(backend.calls), 1)

    def test_truncated_generated_text_is_retained_for_downstream_scoring(self):
        valid = ReceiptBackend().generate([], 512)
        backend = ReceiptBackend(effect=lambda messages, cap: dict(valid, output_tokens=cap,
                                                                   finish_reason='length', text='partial'))
        result = self.execute([job()], backend)
        self.assertEqual(result[0]['status'], 'ok')
        self.assertTrue(result[0]['truncated'])
        self.assertEqual(result[0]['text'], 'partial')

    def test_task_caps_and_messages_are_forwarded_to_backend_without_mutating_jobs(self):
        for kind, cap in (('explanation', 512), ('pair', 512), ('permutation', 512), ('judge', 768)):
            with self.subTest(kind=kind):
                jobs = [job(kind, kind=kind)]
                original = copy.deepcopy(jobs)
                backend = ReceiptBackend()
                self.execute(jobs, backend, batch_id=kind)
                self.assertEqual(backend.calls, [(jobs[0]['messages'], cap)])
                self.assertEqual(jobs, original)

    def test_backend_message_mutation_cannot_change_bound_jobs_or_resumed_output(self):
        valid = ReceiptBackend().generate([], 512)
        def mutate(messages, cap):
            messages[0]['content'][0]['text'] = 'backend mutated a detached copy'
            return valid
        backend = ReceiptBackend(effect=mutate)
        jobs = [job()]
        original = copy.deepcopy(jobs)
        first = self.execute(jobs, backend)
        self.assertEqual(jobs, original)
        self.assertEqual(first, self.execute(jobs, backend))
        self.assertEqual(len(backend.calls), 1)

    def test_global_lock_refuses_overlapping_batch_execution(self):
        other = ReceiptBackend('model8')
        valid = ReceiptBackend().generate([], 512)
        def overlap(messages, cap):
            with self.assertRaises(RuntimeError):
                self.execute([job('other', 'model8')], other, batch_id='other')
            return valid
        result = self.execute([job()], ReceiptBackend(effect=overlap))
        self.assertEqual(result[0]['status'], 'ok')
        self.assertEqual(other.calls, [])

    def test_torn_ledger_refuses_new_calls(self):
        backend = ReceiptBackend()
        self.execute([job()], backend)
        with (self.root / 'reference-ledger.jsonl').open('ab') as stream:
            stream.write(b'{"event":')
        with self.assertRaises(ValueError):
            self.execute([job('new')], backend, batch_id='new')
        self.assertEqual(len(backend.calls), 1)


if __name__ == '__main__':
    unittest.main()
