"""CLI boundary checks on real prepared fixtures; no weights or network."""
import contextlib
import io
import json
from pathlib import Path
import unittest
import test_reference_cli as fixtures
from meme_study.reference_cli import main
from meme_study.reference_execution import load_scores


class ReferenceCommandsTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ReferenceCliTests('runTest');self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.fixture.prepare()
    def cli(self,*args):
        out,err=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:code=main([str(a) for a in args])
            except SystemExit as e:code=e.code
        return code,out.getvalue(),err.getvalue()
    def test_inspect_validates_prepared_images_without_inference(self):
        code,out,err=self.cli('inspect','--prepared',self.fixture.out,'--data-root',self.fixture.data)
        self.assertEqual(code,0,err)
        result=json.loads(out)
        self.assertEqual(result['explanation_calls'],12)
        self.assertEqual(result['actual_model_calls'],0)
        self.assertEqual(result['images_checked'],8)
    def test_run_refuses_pending_pair_before_gpu_import(self):
        runs=self.fixture.root/'runs'
        code,out,err=self.cli('run','--prepared',self.fixture.out,'--data-root',self.fixture.data,
                             '--runs',runs,'--model','fixture-small','--task','pair')
        self.assertEqual(code,2)
        self.assertIn('recorded current-content material review',err)
        self.assertFalse(runs.exists())
    def test_partial_score_to_analysis_retains_all_planned_denominators(self):
        scores=self.fixture.root/'scores';analysis=self.fixture.root/'analysis'
        code,out,err=self.cli('score','--prepared',self.fixture.out,'--runs',self.fixture.root/'empty',
                             '--output',scores,'--allow-partial')
        self.assertEqual(code,0,err)
        self.assertFalse(json.loads(out)['complete'])
        self.assertEqual(len(load_scores(scores)),26)
        code,out,err=self.cli('analyze','--scores',scores,'--output',analysis,'--draws',100)
        self.assertEqual(code,0,err)
        report=json.loads((analysis/'analysis.json').read_text())
        self.assertEqual(set(report['splits']),{'dev','test'})
        model=report['splits']['test']['explanation']['models']['fixture-small']
        self.assertIsNone(model['conditions']['image_only']['dimensions']['V']['estimate'])
    def test_changed_scored_rows_cannot_enter_analysis(self):
        scores=self.fixture.root/'scores';analysis=self.fixture.root/'analysis'
        self.cli('score','--prepared',self.fixture.out,'--runs',self.fixture.root/'empty',
                 '--output',scores,'--allow-partial')
        path=scores/'scored.jsonl';path.write_text(path.read_text()+'\n')
        code,out,err=self.cli('analyze','--scores',scores,'--output',analysis)
        self.assertEqual(code,2)
        self.assertIn('changed',err)
        self.assertFalse(analysis.exists())
    def test_audit_cannot_fill_missing_quotas_with_planned_outputs(self):
        scores=self.fixture.root/'scores';audit=self.fixture.root/'audit'
        self.cli('score','--prepared',self.fixture.out,'--runs',self.fixture.root/'empty',
                 '--output',scores,'--allow-partial')
        code,out,err=self.cli('audit','--scores',scores,'--output',audit,
                             '--model-ids','fixture-small','fixture-large')
        self.assertEqual(code,2)
        self.assertFalse(audit.exists())

    def execute_fixture(self,model,task='explanation'):
        from meme_study.reference_execution import execute_reference
        from test_reference_execution import Backend
        return execute_reference(prepared=self.fixture.out,data_root=self.fixture.data,
            runs=self.fixture.root/'runs',model=model,task=task,split='all',
            backend_factory=lambda checkpoint,*args,**kwargs:Backend(checkpoint,
                judge=checkpoint==self.fixture.config['judge_model']))

    def test_cost_command_reads_actual_fixture_profiles_without_inventing_billing(self):
        self.execute_fixture(self.fixture.config['model_ids'][0])
        costs=self.fixture.root/'costs'
        code,out,err=self.cli('costs','--runs',self.fixture.root/'runs','--output',costs)
        self.assertEqual(code,0,err)
        summary=json.loads(out)
        self.assertEqual(summary['attempts_total'],6)
        self.assertIsNone(summary['actual_allocated_gpu_hours'])
        self.assertFalse(summary['hypothetical_cost_proxy']['is_actual_bill'])
        self.assertTrue(summary['profiles_complete'])

    def test_cost_default_profiles_follow_resolved_run_path(self):
        self.execute_fixture(self.fixture.config['model_ids'][0])
        runs=self.fixture.root/'runs'
        alias=self.fixture.root/'runs-alias'
        alias.symlink_to(runs,target_is_directory=True)
        for name,path in [('alias',alias),('current',Path('.'))]:
            with self.subTest(path=name),contextlib.chdir(runs):
                code,out,err=self.cli('costs','--runs',path,
                    '--output',self.fixture.root/('costs-'+name))
                self.assertEqual(code,0,err)
                summary=json.loads(out)
                self.assertEqual(summary['attempts_total'],6)
                self.assertTrue(summary['profiles_complete'])
                self.assertIsNotNone(summary['serial_process_wall_seconds'])

    def test_result_command_preserves_synthetic_partial_evidence_label(self):
        for model in self.fixture.config['model_ids']:self.execute_fixture(model)
        self.execute_fixture(self.fixture.config['judge_model'],'judge')
        scores=self.fixture.root/'scores'
        self.cli('score','--prepared',self.fixture.out,'--runs',self.fixture.root/'runs',
                 '--output',scores,'--allow-partial')
        output=self.fixture.root/'tables'
        code,out,err=self.cli('report-results','--scores',scores,'--output',output,
                             '--allow-partial','--synthetic-preview','--draws',100)
        self.assertEqual(code,0,err)
        manifest=json.loads(out)
        self.assertTrue(manifest['synthetic_preview'])
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['model_calls'],0)

if __name__=='__main__':unittest.main()
