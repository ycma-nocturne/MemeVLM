"""Synthetic receipt/accounting checks only; no model, network or GPU access."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from meme_study.data import content_hash
from meme_study.reference_runner import run_reference_jobs
from test_reference_runner import ReceiptBackend, job


class ReferenceCostTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.reference_costs'),
                             'reference cost exporter is not implemented')
        from meme_study.reference_costs import export_reference_costs
        self.export = export_reference_costs
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.profiles = self.root / 'profiles'
        self.out = self.root / 'costs'

    def execute(self, jobs, backend=None, batch='batch', cap=913):
        return run_reference_jobs(jobs, backend or ReceiptBackend(), self.runs,
                                  batch_id=batch, study_id='reference-reuse-v1', attempt_cap=cap)

    def export_empty_profiles(self):
        return self.export(runs=self.runs, profiles=self.profiles, output_dir=self.out)

    def rows(self, name='per-call.csv'):
        with (self.out / name).open() as stream:
            return list(csv.DictReader(stream))

    def rewrite_ledger(self, change):
        path = self.runs / 'reference-ledger.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        change(events)
        previous = None
        for event in events:
            event['previous_sha256'] = previous
            event.pop('event_sha256', None)
            event['event_sha256'] = content_hash(event)
            previous = event['event_sha256']
        path.write_text(''.join(json.dumps(event) + '\n' for event in events))

    def test_all_unique_attempts_across_batches_count_once_even_when_reused(self):
        original = job('release-one')
        self.execute([original])
        self.execute([original])
        self.execute([job('release-two', condition='hinted')], batch='second-release')
        summary = self.export_empty_profiles()
        self.assertEqual(summary['attempts_total'], 2)
        self.assertEqual(summary['receipts']['latency_s'],
                         {'known_count': 2, 'missing_count': 0, 'known_sum': .5})
        self.assertEqual(summary['receipts']['input_tokens']['known_sum'], 22)
        self.assertEqual(len(self.rows()), 2)
        self.assertFalse(summary['profiles_complete'])
        self.assertIsNone(summary['actual_allocated_gpu_hours'])
        self.assertEqual(summary['hypothetical_cost_proxy']['hkd_per_card_hour'], 12)
        self.assertIsNone(summary['hypothetical_cost_proxy']['cost_hkd'])

    def test_failed_or_unresolved_attempts_leave_missing_receipts_null(self):
        def fail(*args):
            raise RuntimeError('synthetic setup-free generation error')
        self.execute([job('failed')], ReceiptBackend(effect=fail))
        self.execute([job('pending')], batch='second')
        self.rewrite_ledger(lambda events: events.pop())
        summary = self.export_empty_profiles()
        self.assertEqual(summary['attempts_total'], 2)
        self.assertEqual(summary['receipts']['latency_s'],
                         {'known_count': 0, 'missing_count': 2, 'known_sum': None})
        rows = self.rows()
        self.assertEqual([row['status'] for row in rows], ['failed', 'unresolved'])
        self.assertTrue(all(row['latency_s'] == '' and row['input_tokens'] == '' for row in rows))

    def test_budget_blocked_job_is_visible_but_never_counted_as_attempt(self):
        self.execute([job('called'), job('blocked')], cap=1)
        summary = self.export_empty_profiles()
        self.assertEqual(summary['attempts_total'], 1)
        self.assertEqual(summary['blocked_without_attempt'], 1)
        self.assertEqual(summary['receipts']['output_tokens']['missing_count'], 0)
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(self.rows()[1]['attempt'], '')

    def test_judge_row_retains_candidate_model_attribution(self):
        parent = job('candidate', model='model4')
        self.execute([parent])
        child = job('judge', model='judge8', kind='judge', parent_job_id='candidate',
                    parent_input_sha256=parent['input_sha256'])
        self.execute([child], ReceiptBackend('judge8'), batch='judging')
        self.export_empty_profiles()
        row = self.rows()[1]
        self.assertEqual(row['model'], 'judge8')
        self.assertEqual(row['candidate_model'], 'model4')
        self.assertEqual(row['parent_job_id'], 'candidate')

    def test_terminal_raw_token_disagreement_is_rejected_before_publication(self):
        self.execute([job()])
        self.rewrite_ledger(lambda events: events[-1]['payload']['raw_receipt'].update(input_tokens=999))
        with self.assertRaisesRegex(ValueError, 'raw.*input_tokens|input_tokens.*raw'):
            self.export_empty_profiles()
        self.assertFalse(self.out.exists())

    def ledger_bytes(self):
        path = self.runs / 'reference-ledger.jsonl'
        return path.read_bytes() if path.exists() else b''

    def profile(self, *, before=b'', after=None, selected=('one',), identifier='a' * 32,
                wall=3600, status='completed', incomplete=False, setup=60):
        after = self.ledger_bytes() if after is None else after
        def attempts(raw):
            return sum(json.loads(line)['event'] == 'dispatch' for line in raw.splitlines())
        directory = self.profiles / identifier
        directory.mkdir(parents=True)
        started = {'schema_version': 1, 'protocol_id': 'reference_reuse_v1',
                   'invocation_id': identifier, 'started_at': '2026-10-27T01:00:00+00:00',
                   'runs_path': '/old/machine/runs', 'prepared_sha256': 'a' * 64,
                   'model': 'model4', 'task': 'explanation', 'split': 'dev',
                   'selected_job_ids': list(selected), 'attempts_before': attempts(before),
                   'ledger_before_sha256': hashlib.sha256(before).hexdigest() if before else None,
                   'ledger_bytes_before': len(before), 'timing_scope': 'backend setup and execution'}
        raw = (json.dumps(started, sort_keys=True, indent=2) + '\n').encode()
        (directory / 'started.json').write_bytes(raw)
        if not incomplete:
            finished = {'schema_version': 1, 'protocol_id': 'reference_reuse_v1',
                        'invocation_id': identifier, 'started_sha256': hashlib.sha256(raw).hexdigest(),
                        'status': status, 'finished_at': '2026-10-27T02:00:00+00:00',
                        'wall_seconds': wall, 'setup_seconds': setup,
                        'setup_attempted': True, 'setup_completed': status == 'completed',
                        'resource_after_setup': {'available': False, 'reason': 'fixture CPU'},
                        'attempts_after': attempts(after), 'ledger_bytes_after': len(after),
                        'ledger_after_sha256': hashlib.sha256(after).hexdigest() if after else None,
                        'output_summary': None,
                        'error': {'type': 'RuntimeError', 'message': 'fixture setup failed'}
                            if status == 'error' else None,
                        'profile_code_sha256': 'b' * 64}
            (directory / 'finished.json').write_text(json.dumps(finished) + '\n')
        return directory

    def test_complete_profiles_include_setup_time_but_never_claim_allocated_gpu_hours(self):
        self.execute([job()])
        self.profile()
        summary = self.export_empty_profiles()
        self.assertTrue(summary['profiles_complete'])
        self.assertEqual(summary['serial_process_wall_seconds'], 3600)
        self.assertEqual(summary['hypothetical_cost_proxy']['cost_hkd'], 12)
        self.assertIsNone(summary['actual_allocated_gpu_hours'])
        self.assertEqual(summary['profile_coverage']['attempts_covered'], 1)

    def test_setup_failure_can_have_nonzero_process_cost_with_no_attempts(self):
        self.profile(selected=(), status='error', wall=1800, setup=1800)
        summary = self.export_empty_profiles()
        self.assertEqual(summary['attempts_total'], 0)
        self.assertEqual(summary['serial_process_wall_seconds'], 1800)
        self.assertEqual(summary['hypothetical_cost_proxy']['cost_hkd'], 6)
        self.assertEqual(self.rows('per-invocation.csv')[0]['status'], 'error')

    def test_incomplete_profile_makes_total_wall_and_proxy_unknown(self):
        self.execute([job()])
        self.profile(incomplete=True)
        summary = self.export_empty_profiles()
        self.assertFalse(summary['profiles_complete'])
        self.assertIsNone(summary['serial_process_wall_seconds'])
        self.assertIsNone(summary['hypothetical_cost_proxy']['cost_hkd'])
        self.assertEqual(summary['profile_coverage']['incomplete_invocations'], 1)

    def test_profile_prefix_or_selected_job_mismatch_is_rejected(self):
        self.execute([job()])
        directory = self.profile(selected=('another',))
        with self.assertRaisesRegex(ValueError, 'selected|membership'):
            self.export_empty_profiles()
        self.assertFalse(self.out.exists())
        path = directory / 'finished.json'
        finished = json.loads(path.read_text())
        finished['ledger_after_sha256'] = '0' * 64
        path.write_text(json.dumps(finished))
        with self.assertRaisesRegex(ValueError, 'prefix|ledger'):
            self.export_empty_profiles()
        self.assertFalse(self.out.exists())

    def test_overlapping_attempt_profiles_cannot_double_count_walltime(self):
        self.execute([job()])
        self.profile()
        self.profile(identifier='b' * 32)
        with self.assertRaisesRegex(ValueError, 'overlap|duplicate'):
            self.export_empty_profiles()
        self.assertFalse(self.out.exists())

    def test_available_allocator_snapshots_are_maxima_not_gpu_utilization(self):
        def receipt(*args):
            snapshot = {'available': True, 'device_index': 0, 'device_name': 'synthetic GPU',
                        'total_memory_bytes': 1000, 'torch_cuda_version': 'fixture',
                        'memory_scope': 'current CUDA process / torch allocator, not whole-device utilization',
                        'peak_scope': 'since last reset', 'torch_memory_allocated_bytes': 100,
                        'torch_memory_reserved_bytes': 200, 'torch_peak_allocated_bytes': 120,
                        'torch_peak_reserved_bytes': 220}
            after = {**snapshot, 'torch_peak_allocated_bytes': 300, 'torch_peak_reserved_bytes': 400}
            return {'text': 'fixture', 'input_tokens': 11, 'output_tokens': 12,
                    'finish_reason': 'stop', 'latency_s': .25,
                    'resource_usage': {'available': True, 'peak_stats_reset': True,
                                       'before': snapshot, 'after': after}}
        self.execute([job()], ReceiptBackend(effect=receipt))
        summary = self.export_empty_profiles()
        self.assertEqual(summary['memory']['max_allocated_bytes'], 300)
        self.assertEqual(summary['memory']['max_reserved_bytes'], 400)
        self.assertEqual(summary['memory']['attempts_with_snapshots'], 1)
        self.assertEqual(self.rows()[0]['resource_snapshot_count'], '2')
        self.assertIn('not whole-device utilization', summary['memory']['scope'])

    def test_available_per_call_resource_cannot_omit_its_memory_snapshots(self):
        self.execute([job()])
        self.rewrite_ledger(lambda events: events[-1]['payload']['raw_receipt'].update(
            resource_usage={'available': True, 'peak_stats_reset': True}))
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            self.export_empty_profiles()
        self.assertFalse(self.out.exists())

    def test_failed_receipt_without_finish_reason_has_unknown_truncation(self):
        def fail(*args):
            raise RuntimeError('fixture failed')
        self.execute([job()], ReceiptBackend(effect=fail))
        self.export_empty_profiles()
        self.assertEqual(self.rows()[0]['truncated'], '')

    def test_finished_profile_with_unknown_ledger_end_retains_wall_but_excludes_total(self):
        self.execute([job()])
        directory = self.profile()
        path = directory / 'finished.json'
        finished = json.loads(path.read_text())
        finished.update(attempts_after=None, ledger_after_sha256=None, ledger_bytes_after=None)
        path.write_text(json.dumps(finished))
        summary = self.export_empty_profiles()
        self.assertFalse(summary['profiles_complete'])
        self.assertIsNone(summary['serial_process_wall_seconds'])
        self.assertEqual(self.rows('per-invocation.csv')[0]['reported_wall_seconds'], '3600')
        self.assertEqual(self.rows('per-invocation.csv')[0]['wall_seconds'], '')

    def test_real_profile_writer_binds_a_fixture_call_and_cached_invocation(self):
        from meme_study.reference_execution import read_reference_run
        from meme_study.reference_profiling import ResourceInvocation
        self.profiles = self.runs.parent / (self.runs.name + '.profiles')
        metadata = {'prepared_sha256': 'a' * 64, 'model': 'model4', 'task': 'explanation',
                    'split': 'dev', 'selected_job_ids': ['one']}
        for _ in range(2):
            with ResourceInvocation(runs=self.runs, metadata=metadata,
                                    snapshot=read_reference_run(self.runs),
                                    read_snapshot=read_reference_run) as profile:
                with profile.setup(True):
                    pass
                self.execute([job()])
        summary = self.export_empty_profiles()
        self.assertEqual(summary['attempts_total'], 1)
        self.assertEqual(summary['profile_coverage']['invocations'], 2)
        self.assertTrue(summary['profiles_complete'])
        self.assertEqual(sorted(row['new_attempts'] for row in self.rows('per-invocation.csv')), ['0', '1'])

    def test_bound_job_metadata_cannot_supply_failed_token_measurements(self):
        def fail(*args):
            raise RuntimeError('fixture failed')
        self.execute([job(input_tokens=999, output_tokens=999, latency_s=999)], ReceiptBackend(effect=fail))
        summary = self.export_empty_profiles()
        self.assertIsNone(summary['receipts']['input_tokens']['known_sum'])
        self.assertIsNone(summary['receipts']['latency_s']['known_sum'])

    def test_plain_keyboard_interrupt_has_valid_empty_error_message(self):
        from meme_study.reference_execution import read_reference_run
        from meme_study.reference_profiling import ResourceInvocation
        self.profiles=self.runs.parent/(self.runs.name+'.profiles')
        metadata={'prepared_sha256':'a'*64,'model':'model4','task':'explanation',
                  'split':'dev','selected_job_ids':['one']}
        with self.assertRaises(KeyboardInterrupt):
            with ResourceInvocation(runs=self.runs,metadata=metadata,
                                    snapshot=read_reference_run(self.runs),
                                    read_snapshot=read_reference_run) as profile:
                with profile.setup(True):raise KeyboardInterrupt()
        summary=self.export_empty_profiles()
        self.assertEqual(summary['attempts_total'],0)
        self.assertTrue(summary['profiles_complete'])
        self.assertIsNotNone(summary['serial_process_wall_seconds'])
        self.assertEqual(self.rows('per-invocation.csv')[0]['status'],'error')

    def test_existing_destination_is_never_overwritten_and_source_hash_is_bound(self):
        self.execute([job()])
        self.export_empty_profiles()
        provenance = json.loads((self.out / 'provenance.json').read_text())
        import hashlib
        expected = hashlib.sha256((self.runs / 'reference-ledger.jsonl').read_bytes()).hexdigest()
        self.assertEqual(provenance['ledger_sha256'], expected)
        self.assertEqual(set(provenance['artifact_sha256']),
                         {'summary.json', 'per-call.csv', 'per-invocation.csv'})
        before = (self.out / 'summary.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.export_empty_profiles()
        self.assertEqual((self.out / 'summary.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
