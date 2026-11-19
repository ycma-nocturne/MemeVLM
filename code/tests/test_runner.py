"""Runner behavior tests. RecordingBackend is a test fixture, never a model."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_workflows import case_fixture, text_parts, image_parts


VALID = '{"option_id":"A","fact":true,"explanation":"synthetic fixture"}'


def response(text=VALID, finish_reason='stop'):
    return {'text': text, 'input_tokens': 20, 'output_tokens': 9,
            'finish_reason': finish_reason, 'latency_s': 0.01}


def job_fixture(job_id='j1', workflow='D', background='relevant'):
    case = case_fixture(background=background)
    case['case_id'] = job_id + '-case'
    return {'job_id': job_id, 'model': 'fixture-model', 'workflow': workflow,
            'case': case, 'input_sha256': job_id + '-input-digest',
            'planned_calls': 1 if workflow == 'D' else 2}


class RecordingBackend:
    run_config = {'checkpoint': 'fixture-model', 'revision': 'fixture-revision',
                  'device': 'cpu-test-fixture', 'do_sample': False}

    def __init__(self, responses=None, on_call=None):
        self.responses = iter(responses) if responses is not None else None
        self.calls = []
        self.on_call = on_call

    def generate(self, messages, max_new_tokens):
        self.calls.append((copy.deepcopy(messages), max_new_tokens))
        if self.on_call:
            self.on_call(len(self.calls))
        result = next(self.responses) if self.responses is not None else response()
        if isinstance(result, BaseException):
            raise result
        return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.runner'),
                             'resumable runner is not implemented')
        from meme_study import runner
        self.runner = runner
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def run_jobs(self, jobs, backend=None, cap=20):
        return self.runner.run_jobs(jobs, backend or RecordingBackend(), self.out, attempt_cap=cap)

    def test_call_caps_stage_names_and_raw_previous_text(self):
        backend = RecordingBackend([response(), response('draft'), response(),
                                    response('observation', 'length'), response()])
        summary = self.run_jobs([job_fixture('d'), job_fixture('r', 'R'), job_fixture('v', 'V')], backend)
        self.assertEqual([cap for _, cap in backend.calls], [160, 80, 160, 80, 160])
        self.assertIn('draft', text_parts(backend.calls[2][0]))
        self.assertIn('observation', text_parts(backend.calls[4][0]))
        self.assertTrue(all(len(image_parts(messages)) == 1 for messages, _ in backend.calls))
        self.assertEqual((summary['attempts'], summary['completed']), (5, 3))
        events = read_jsonl(self.out / 'events.jsonl')
        self.assertEqual([e['stage'] for e in events if e['event'] == 'dispatch'],
                         ['final', 'draft', 'final', 'observation', 'final'])

    def test_dispatch_intent_is_durable_before_backend_call(self):
        def check_intent(call_number):
            events = read_jsonl(self.out / 'events.jsonl')
            self.assertEqual(events[-1]['event'], 'dispatch')
            self.assertEqual(events[-1]['attempt'], call_number)
            self.assertEqual(events[-1]['max_new_tokens'], 160)
            self.assertIn('messages_sha256', events[-1])
            self.assertIn('run_config', events[-1])
        self.run_jobs([job_fixture()], RecordingBackend(on_call=check_intent))
        records = read_jsonl(self.out / 'records.jsonl')
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['text'], VALID)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['input_tokens'], 20)
        self.assertEqual(record['output_tokens'], 9)
        self.assertEqual(record['latency_s'], 0.01)
        self.assertEqual(record['model'], 'fixture-model')

    def test_completed_resume_has_zero_calls_and_does_not_rewrite_logs(self):
        jobs = [job_fixture('r', 'R')]
        self.run_jobs(jobs)
        before = {name: (self.out / name).read_bytes() for name in ['events.jsonl', 'records.jsonl', 'run.json']}
        backend = RecordingBackend()
        summary = self.run_jobs(jobs, backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual((summary['attempts'], summary['completed']), (2, 1))
        self.assertEqual(before, {name: (self.out / name).read_bytes() for name in before})

    def test_completed_first_stage_resumes_without_repeating_it(self):
        jobs = [job_fixture('v', 'V')]
        real_build = self.runner.build_messages
        def interrupt_before_final(case, workflow, stage, previous_text=None):
            if stage == 'final': raise KeyboardInterrupt('controlled crash between stages')
            return real_build(case, workflow, stage, previous_text)
        first = RecordingBackend([response('saved observation')])
        with patch.object(self.runner, 'build_messages', side_effect=interrupt_before_final):
            with self.assertRaises(KeyboardInterrupt): self.run_jobs(jobs, first)
        self.assertEqual(len(first.calls), 1)
        resumed = RecordingBackend()
        summary = self.run_jobs(jobs, resumed)
        self.assertEqual(len(resumed.calls), 1)
        self.assertIn('saved observation', text_parts(resumed.calls[0][0]))
        self.assertEqual(summary['attempts'], 2)
        self.assertEqual(len(read_jsonl(self.out / 'records.jsonl')), 1)

    def test_unfinished_dispatch_requires_reconciliation_and_never_repeats(self):
        jobs = [job_fixture('broken'), job_fixture('other')]
        with self.assertRaises(KeyboardInterrupt):
            self.run_jobs(jobs, RecordingBackend([KeyboardInterrupt('crash after dispatch')]))
        backend = RecordingBackend()
        summary = self.run_jobs(jobs, backend)
        self.assertEqual(summary['reconciliation_required'], 1)
        self.assertEqual(summary['attempts'], 2)
        self.assertEqual(len(backend.calls), 1)
        pending = next(r for r in summary['records'] if r['job_id'] == 'broken')
        self.assertEqual(pending['status'], 'reconciliation_required')
        self.assertIn('manual', pending['error'].lower())
        again = RecordingBackend(); self.run_jobs(jobs, again)
        self.assertEqual(again.calls, [])

    def test_budget_is_checked_before_every_call_including_second_stage(self):
        jobs = [job_fixture('r', 'R'), job_fixture('d')]
        backend = RecordingBackend()
        summary = self.run_jobs(jobs, backend, cap=1)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(summary['attempts'], 1)
        self.assertEqual(summary['budget_exhausted'], 2)
        resumed = RecordingBackend(); self.run_jobs(jobs, resumed, cap=1)
        self.assertEqual(resumed.calls, [])

    def test_zero_budget_makes_no_dispatch(self):
        backend = RecordingBackend()
        summary = self.run_jobs([job_fixture()], backend, cap=0)
        self.assertEqual(summary['attempts'], 0)
        self.assertEqual(backend.calls, [])
        self.assertEqual(summary['budget_exhausted'], 1)

    def test_infrastructure_failure_is_counted_terminal_and_other_jobs_continue(self):
        jobs = [job_fixture('r', 'R'), job_fixture('d')]
        backend = RecordingBackend([RuntimeError('fixture network failure'), response()])
        summary = self.run_jobs(jobs, backend, cap=2)
        self.assertEqual((summary['attempts'], summary['error'], summary['completed']), (2, 1, 1))
        self.assertEqual([cap for _, cap in backend.calls], [80, 160])
        self.assertEqual(len(read_jsonl(self.out / 'records.jsonl')), 2)
        resumed = RecordingBackend(); self.run_jobs(jobs, resumed, cap=2)
        self.assertEqual(resumed.calls, [])

    def test_bad_json_wrong_answer_and_final_length_are_retained_without_retry(self):
        jobs = [job_fixture('bad'), job_fixture('answer'), job_fixture('short')]
        texts = ['not JSON', '{"option_id":"D","fact":false,"explanation":"wrong fixture"}', '{']
        backend = RecordingBackend([response(texts[0]), response(texts[1]), response(texts[2], 'length')])
        summary = self.run_jobs(jobs, backend)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual([r['text'] for r in summary['records']], texts)
        self.assertEqual([r['status'] for r in summary['records']], ['completed', 'completed', 'truncated'])
        self.assertTrue(summary['records'][-1]['truncated'])

    def test_no_first_stage_cache_across_jobs_or_backgrounds(self):
        jobs = [job_fixture('none', 'V', 'none'), job_fixture('relevant', 'V', 'relevant')]
        backend = RecordingBackend([response('first'), response(), response('second'), response()])
        summary = self.run_jobs(jobs, backend)
        self.assertEqual(summary['attempts'], 4)
        self.assertEqual([cap for _, cap in backend.calls], [80, 160, 80, 160])
        self.assertIn('first', text_parts(backend.calls[1][0]))
        self.assertIn('second', text_parts(backend.calls[3][0]))

    def test_resume_rejects_changed_jobs_hash_config_or_budget_before_call(self):
        jobs = [job_fixture()]; self.run_jobs(jobs)
        variations = []
        for key, value in [('input_sha256', 'changed'), ('model', 'changed'), ('planned_calls', 2)]:
            changed = copy.deepcopy(jobs); changed[0][key] = value; variations.append(changed)
        changed = copy.deepcopy(jobs); changed[0]['case']['note'] = 'changed'; variations.append(changed)
        variations.append(jobs + [job_fixture('new')])
        for changed in variations:
            backend = RecordingBackend()
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError): self.run_jobs(changed, backend)
                self.assertEqual(backend.calls, [])
        with self.assertRaises(ValueError): self.run_jobs(jobs, cap=21)
        backend = RecordingBackend(); backend.run_config = {'revision': 'changed'}
        with self.assertRaises(ValueError): self.run_jobs(jobs, backend)
        self.assertEqual(backend.calls, [])

    def test_duplicate_jobs_bad_cap_and_non_direct_controls_fail_before_call(self):
        for jobs, cap in [([job_fixture(), job_fixture()], 20), ([job_fixture()], -1),
                          ([job_fixture()], True), ([job_fixture()], 1.5)]:
            backend = RecordingBackend()
            with self.assertRaises(ValueError): self.run_jobs(jobs, backend, cap)
            self.assertEqual(backend.calls, [])
        bad = job_fixture(workflow='V'); bad['case']['control'] = 'text'
        with self.assertRaises(ValueError): self.run_jobs([bad])

    def test_process_lock_refuses_concurrent_budget_owner(self):
        self.out.mkdir(exist_ok=True)
        with (self.out / '.run.lock').open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            backend = RecordingBackend()
            with self.assertRaises(RuntimeError): self.run_jobs([job_fixture()], backend)
            self.assertEqual(backend.calls, [])

    def test_corrupt_or_torn_log_refuses_dispatch(self):
        jobs = [job_fixture()]; self.run_jobs(jobs)
        with (self.out / 'events.jsonl').open('ab') as handle: handle.write(b'{"event":')
        backend = RecordingBackend()
        with self.assertRaises(ValueError): self.run_jobs(jobs, backend)
        self.assertEqual(backend.calls, [])

    def test_completed_final_event_reconstructs_missing_final_record_without_call(self):
        jobs = [job_fixture()]; self.run_jobs(jobs)
        (self.out / 'records.jsonl').unlink()
        backend = RecordingBackend(); summary = self.run_jobs(jobs, backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(len(read_jsonl(self.out / 'records.jsonl')), 1)

    def test_invalid_backend_receipt_counts_as_error_and_never_retries(self):
        jobs = [job_fixture()]
        summary = self.run_jobs(jobs, RecordingBackend([{'text': VALID}]))
        self.assertEqual((summary['attempts'], summary['error']), (1, 1))
        backend = RecordingBackend(); self.run_jobs(jobs, backend)
        self.assertEqual(backend.calls, [])

    def test_invalid_receipt_preserves_returned_raw_text_for_audit(self):
        summary = self.run_jobs([job_fixture()], RecordingBackend([{'text': 'returned but unmetered'}]))
        self.assertEqual(summary['records'][0].get('text'), 'returned but unmetered')
        errors = [event for event in read_jsonl(self.out / 'events.jsonl') if event['event'] == 'error']
        self.assertEqual(errors[0]['raw_receipt'], {'text': 'returned but unmetered'})

    def test_final_records_without_matching_events_refuse_before_any_dispatch(self):
        jobs = [job_fixture()]; self.run_jobs(jobs)
        (self.out / 'events.jsonl').unlink()
        backend = RecordingBackend()
        with self.assertRaises(ValueError): self.run_jobs(jobs, backend)
        self.assertEqual(backend.calls, [])

    def test_second_stage_without_first_stage_history_refuses_before_dispatch(self):
        jobs = [job_fixture('v', 'V')]; self.run_jobs(jobs)
        events = read_jsonl(self.out / 'events.jsonl')[2:]
        for event in events: event['attempt'] = 1
        (self.out / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
        backend = RecordingBackend()
        with self.assertRaises(ValueError): self.run_jobs(jobs, backend)
        self.assertEqual(backend.calls, [])

    def test_second_stage_failure_consumes_budget_without_repeating_first_stage(self):
        jobs = [job_fixture('r', 'R'), job_fixture('d')]
        summary = self.run_jobs(jobs, RecordingBackend([response(''), OSError('stage two failed')]), cap=2)
        self.assertEqual((summary['attempts'], summary['error'], summary['budget_exhausted']), (2, 1, 1))
        backend = RecordingBackend(); self.run_jobs(jobs, backend, cap=2)
        self.assertEqual(backend.calls, [])

    def test_external_guard_blocks_before_local_attempt_and_backend_dispatch(self):
        import inspect
        self.assertIn('before_dispatch', inspect.signature(self.runner.run_jobs).parameters)
        backend = RecordingBackend()
        seen = []
        def guard(context):
            seen.append(context)
            return False
        summary = self.runner.run_jobs([job_fixture()], backend, self.out,
                                       attempt_cap=4180, before_dispatch=guard)
        self.assertEqual(backend.calls, [])
        self.assertEqual(summary['attempts'], 0)
        self.assertEqual(summary['budget_exhausted'], 1)
        self.assertEqual(read_jsonl(self.out / 'events.jsonl'), [])
        self.assertEqual((seen[0]['job_id'], seen[0]['stage'], seen[0]['attempt']), ('j1', 'final', 1))

    def test_copied_results_from_different_backend_revision_refuse_resume(self):
        jobs = [job_fixture()]
        self.run_jobs(jobs)
        other = self.out / 'other-revision'
        backend_b = RecordingBackend()
        backend_b.run_config = dict(backend_b.run_config, revision='different-revision')
        self.runner.run_jobs(jobs, backend_b, other, attempt_cap=20)
        for name in ('events.jsonl', 'records.jsonl'):
            (other / name).write_bytes((self.out / name).read_bytes())
        resumed = RecordingBackend(); resumed.run_config = backend_b.run_config
        with self.assertRaises(ValueError):
            self.runner.run_jobs(jobs, resumed, other, attempt_cap=20)
        self.assertEqual(resumed.calls, [])

    def test_event_caps_and_reconstructed_stage_message_hashes_are_bound(self):
        jobs = [job_fixture('r', 'R')]; self.run_jobs(jobs)
        original = (self.out / 'events.jsonl').read_bytes()
        for field, changed in [('max_new_tokens', 999), ('messages_sha256', '0' * 64),
                               ('checkpoint', 'wrong-checkpoint'), ('model', 'wrong-model'),
                               ('runner_sha256', '0' * 64), ('prompts_sha256', '0' * 64)]:
            events = [json.loads(line) for line in original.splitlines()]
            for event in events[:2]: event[field] = changed
            (self.out / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
            backend = RecordingBackend()
            with self.subTest(field=field):
                with self.assertRaises(ValueError): self.run_jobs(jobs, backend)
                self.assertEqual(backend.calls, [])
        (self.out / 'events.jsonl').write_bytes(original)


if __name__ == '__main__':
    unittest.main()
