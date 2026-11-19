"""Exercise the prepared-input -> runtime ledger -> scores boundary with local fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import test_reference_cli as fixtures
from meme_study.reference_execution import (load_prepared, execute_reference, score_reference,
                                            read_reference_run)


class Backend:
    def __init__(self,checkpoint,*,judge=False):
        self.run_config={'checkpoint':checkpoint,'fixture_backend':True}
        self.calls=[];self.judge=judge
    def generate(self,messages,max_new_tokens):
        self.calls.append((copy.deepcopy(messages),max_new_tokens))
        text='有一个角色，但称呼与外观不符。'
        if self.judge:
            payload=json.loads(messages[0]['content'][-1]['text'].rsplit('\n',1)[1])
            text=json.dumps({'items':[{'item_id':r['item_id'],'verdict':'not_covered','evidence':[]}
                                     for r in payload['检查项']]})
        return {'text':text,'input_tokens':30,'output_tokens':20,'finish_reason':'stop','latency_s':0.01}


class ReferenceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ReferenceCliTests('runTest');self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.fixture.prepare()
        self.prepared=self.fixture.out;self.data=self.fixture.data
        self.runs=self.fixture.root/'runs';self.models=self.fixture.config['model_ids']
        self.backends=[]
    def factory(self,checkpoint,data_root,**kwargs):
        b=Backend(checkpoint,judge=checkpoint==self.fixture.config['judge_model']);self.backends.append(b);return b
    def run_model(self,model,task='explanation',split='all'):
        return execute_reference(prepared=self.prepared,data_root=self.data,runs=self.runs,
                                 model=model,task=task,split=split,backend_factory=self.factory)
    def test_explanations_and_judge_share_budget_and_resume_without_calls(self):
        for model in self.models:self.run_model(model)
        self.run_model(self.fixture.config['judge_model'],'judge')
        snap=read_reference_run(self.runs)
        self.assertEqual(snap['attempts'],24)
        self.assertEqual(len(snap['outputs']),24)
        self.assertEqual([len(b.calls) for b in self.backends],[6,6,12])
        self.run_model(self.models[0]);self.assertEqual(len(self.backends[-1].calls),0)
        result=score_reference(prepared=self.prepared,runs=self.runs,output=self.fixture.root/'scores',allow_partial=True)
        self.assertEqual(result['explanations_with_valid_judge'],12)
        rows=[json.loads(x) for x in (self.fixture.root/'scores/scored.jsonl').read_text().splitlines()]
        a=[r for r in rows if r['task_kind']=='explanation']
        self.assertEqual(len(a),12)
        self.assertTrue(all(r['score']['dimensions']['V']['coverage']==0 for r in a))
        self.assertTrue(all(r['candidate_text']=='有一个角色，但称呼与外观不符。' for r in a))
        self.assertEqual(sum(r['run_status']=='pending' for r in rows),14)
    def test_pending_pairs_cannot_execute(self):
        with self.assertRaises(ValueError):self.run_model(self.models[0],'pair')
        self.assertEqual(self.backends,[])
        self.assertFalse(self.runs.exists())
    def test_image_change_is_rejected_before_backend_creation(self):
        path=self.data/self.fixture.selection[0]['image_path'];path.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.run_model(self.models[0])
        self.assertEqual(self.backends,[])
    def test_runtime_rechecks_each_image_after_preflight(self):
        original_factory=self.factory
        def factory(checkpoint,data_root,**kwargs):
            b=original_factory(checkpoint,data_root,**kwargs)
            path=self.data/self.fixture.selection[0]['image_path'];path.write_bytes(b'changed after check')
            return b
        result=execute_reference(prepared=self.prepared,data_root=self.data,runs=self.runs,
                 model=self.models[0],task='explanation',split='dev',backend_factory=factory)
        self.assertEqual(result['failed'],2)  # two image conditions; OCR-only is still executable
        self.assertEqual(len(self.backends[0].calls),1)
    def test_changed_prepared_artifact_or_missing_parent_cannot_be_silently_used(self):
        path=self.prepared/'explanation_jobs.jsonl';path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):load_prepared(self.prepared)
        self.assertFalse(self.runs.exists())
    def test_judge_requires_completed_selected_parent_batch(self):
        with self.assertRaises(ValueError):self.run_model(self.fixture.config['judge_model'],'judge')
        self.assertEqual(self.backends,[])
    def test_partial_scores_require_explicit_option_and_never_turn_missing_into_zero(self):
        self.run_model(self.models[0],split='dev')
        with self.assertRaises(ValueError):score_reference(prepared=self.prepared,runs=self.runs,output=self.fixture.root/'reject')
        self.assertFalse((self.fixture.root/'reject').exists())
        result=score_reference(prepared=self.prepared,runs=self.runs,output=self.fixture.root/'partial',allow_partial=True)
        self.assertFalse(result['complete'])
        rows=[json.loads(x) for x in (self.fixture.root/'partial/scored.jsonl').read_text().splitlines()]
        missing=[r for r in rows if r['run_status']=='missing']
        self.assertTrue(missing);self.assertTrue(all(r['score'] is None for r in missing))
    def test_notes_only_release_reuses_identical_inputs_without_extra_attempts(self):
        self.run_model(self.models[0])
        self.fixture.hints['public-7']['preparation_note']='new preparation note'
        self.fixture.save_inputs();second=self.fixture.root/'prepared2';self.fixture.prepare(output=second)
        execute_reference(prepared=second,data_root=self.data,runs=self.runs,model=self.models[0],
                          task='explanation',split='all',backend_factory=self.factory)
        snap=read_reference_run(self.runs)
        self.assertEqual(snap['attempts'],6)
        self.assertEqual(len({r['job_id'] for r in snap['outputs']}),6)

    def test_source_change_only_dispatches_new_inputs_with_shared_budget(self):
        from meme_study.data import content_hash
        self.run_model(self.models[0])
        records=[json.loads(s) for s in self.fixture.metadata.read_text().splitlines()]
        records[0]['knowledge']['facts'].append('第二条可独立核对的事实。')
        self.fixture.metadata.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        self.fixture.config['metadata']['sha256']=hashlib.sha256(self.fixture.metadata.read_bytes()).hexdigest()
        hint=self.fixture.hints['public-7']
        hint.update(source_record_sha256=content_hash(records[0]),source_fact_indices=[1],text='第二条可独立核对的事实。')
        self.fixture.save_inputs();second=self.fixture.root/'prepared2';self.fixture.prepare(output=second)
        result=execute_reference(prepared=second,data_root=self.data,runs=self.runs,model=self.models[0],
                                 backend_factory=self.factory)
        self.assertEqual(result['new_attempts'],3)
        self.assertEqual(read_reference_run(self.runs)['attempts'],9)

    def test_environment_setup_failure_does_not_consume_any_attempts(self):
        class CannotLoad(Backend):
            def prepare(self):
                raise RuntimeError('fixture CUDA unavailable')
        with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):
            execute_reference(prepared=self.prepared,data_root=self.data,runs=self.runs,model=self.models[0],
                              backend_factory=lambda checkpoint,*args,**kwargs:CannotLoad(checkpoint))
        self.assertFalse(self.runs.exists())

    def test_two_preparations_cannot_race_during_backend_setup(self):
        self.fixture.hints['public-7']['preparation_note']='alternate immutable preparation'
        self.fixture.save_inputs();second=self.fixture.root/'prepared2';self.fixture.prepare(output=second)
        def overlapping_factory(checkpoint,data_root,**kwargs):
            with self.assertRaisesRegex(RuntimeError,'another process'):
                self.run_model(self.models[0])
            return self.factory(checkpoint,data_root,**kwargs)
        execute_reference(prepared=second,data_root=self.data,runs=self.runs,model=self.models[0],
                          backend_factory=overlapping_factory)
        self.assertEqual(read_reference_run(self.runs)['attempts'],6)

    def test_test_judge_calibration_preserves_completed_dev_profile(self):
        for model in self.models:self.run_model(model)
        judge=self.fixture.config['judge_model'];self.run_model(judge,'judge','dev')
        self.fixture.config['judge_notes_by_split']={'dev':'','test':'逐项保留不确定判断，不从参考中替候选补写内容。'}
        self.fixture.save_inputs();second=self.fixture.root/'calibrated';self.fixture.prepare(output=second)
        dev=execute_reference(prepared=second,data_root=self.data,runs=self.runs,model=judge,
                              task='judge',split='dev',backend_factory=self.factory)
        self.assertEqual(dev['new_attempts'],0)
        test=execute_reference(prepared=second,data_root=self.data,runs=self.runs,model=judge,
                               task='judge',split='test',backend_factory=self.factory)
        self.assertEqual(test['new_attempts'],6)
        snapshot=read_reference_run(self.runs)
        judged=[j for j in snapshot['jobs'].values() if j['task_kind']=='judge']
        self.assertEqual(len({j['judge_profile_sha256'] for j in judged}),2)
        for job in judged:
            note=self.fixture.config['judge_notes_by_split']['test']
            self.assertEqual(note in job['messages'][0]['content'][-1]['text'],job['split']=='test')
        self.assertEqual(snapshot['attempts'],24)
        summary=score_reference(prepared=second,runs=self.runs,output=self.fixture.root/'scores',allow_partial=True)
        self.assertEqual(summary['explanations_with_valid_judge'],12)

    def profiles(self):
        directory=self.runs.parent/(self.runs.name+'.profiles')
        return [(json.loads((p/'started.json').read_text()),json.loads((p/'finished.json').read_text()))
                for p in sorted(directory.iterdir())]

    def test_invocation_profile_binds_ledger_prefixes_and_cache_reuse(self):
        self.run_model(self.models[0],split='dev')
        self.run_model(self.models[0],split='dev')
        profiles=self.profiles();self.assertEqual(len(profiles),2)
        self.assertEqual(sorted(end['attempts_after']-start['attempts_before'] for start,end in profiles),[0,3])
        ledger=(self.runs/'reference-ledger.jsonl').read_bytes()
        for start,end in profiles:
            self.assertEqual(end['status'],'completed')
            self.assertGreaterEqual(end['wall_seconds'],end['setup_seconds'])
            self.assertEqual(hashlib.sha256(ledger[:end['ledger_bytes_after']]).hexdigest(),end['ledger_after_sha256'])
            self.assertFalse(end['resource_after_setup']['available'])

    def test_setup_failure_keeps_resource_profile_with_zero_dispatches(self):
        class CannotLoad(Backend):
            def prepare(self):raise RuntimeError('fixture CUDA unavailable')
        with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):
            execute_reference(prepared=self.prepared,data_root=self.data,runs=self.runs,model=self.models[0],
                              backend_factory=lambda checkpoint,*args,**kwargs:CannotLoad(checkpoint))
        start,end=self.profiles()[0]
        self.assertEqual((start['attempts_before'],end['attempts_after']),(0,0))
        self.assertEqual(end['status'],'error')
        self.assertTrue(end['setup_attempted']);self.assertFalse(end['setup_completed'])
        self.assertGreaterEqual(end['setup_seconds'],0)
        self.assertEqual(end['error']['type'],'RuntimeError')
        self.assertFalse(self.runs.exists())

    def test_interrupted_generation_keeps_profile_and_pending_dispatch(self):
        class Interrupted(Backend):
            def generate(self,messages,max_new_tokens):raise KeyboardInterrupt('fixture interruption')
        with self.assertRaises(KeyboardInterrupt):
            execute_reference(prepared=self.prepared,data_root=self.data,runs=self.runs,model=self.models[0],
                              split='dev',backend_factory=lambda checkpoint,*args,**kwargs:Interrupted(checkpoint))
        start,end=self.profiles()[0]
        self.assertEqual(end['status'],'error');self.assertEqual(end['attempts_after'],1)
        self.assertEqual(end['error']['type'],'KeyboardInterrupt')
        snapshot=read_reference_run(self.runs)
        self.assertEqual(snapshot['outputs'],[]);self.assertEqual(len(snapshot['dispatches']),1)

    def test_dev_prefix_cap_resume_and_full_continuation(self):
        args=dict(prepared=self.prepared,data_root=self.data,runs=self.runs,
                  model=self.models[0],split='dev',backend_factory=self.factory)
        first=execute_reference(**args,max_jobs=2)
        self.assertEqual(first['new_attempts'],2)
        self.assertEqual(first['selected_jobs'],2)
        again=execute_reference(**args,max_jobs=2)
        self.assertEqual(again['new_attempts'],0)
        full=execute_reference(**args)
        self.assertEqual(full['new_attempts'],1)
        snap=read_reference_run(self.runs)
        self.assertEqual(snap['attempts'],3)
        self.assertTrue(all(j['split']=='dev' for j in snap['jobs'].values()))

    def test_dev_prefix_rejects_invalid_caps_and_test_selection(self):
        for limit,split in [(0,'dev'),(-1,'dev'),(True,'dev'),(1.2,'dev'),(1,'test'),(1,'all')]:
            with self.subTest(limit=limit,split=split):
                with self.assertRaisesRegex(ValueError,'max_jobs'):
                    execute_reference(prepared=self.prepared,data_root=self.data,runs=self.runs,
                        model=self.models[0],split=split,backend_factory=self.factory,max_jobs=limit)
        self.assertEqual(self.backends,[])
        self.assertFalse(self.runs.exists())

    def test_dev_prefix_does_not_weaken_judge_dependencies(self):
        self.run_model(self.models[0],split='dev')
        with self.assertRaisesRegex(ValueError,'parent'):
            execute_reference(prepared=self.prepared,data_root=self.data,runs=self.runs,
                model=self.fixture.config['judge_model'],task='judge',split='dev',
                backend_factory=self.factory,max_jobs=1)
        self.run_model(self.models[1],split='dev')
        result=execute_reference(prepared=self.prepared,data_root=self.data,runs=self.runs,
            model=self.fixture.config['judge_model'],task='judge',split='dev',
            backend_factory=self.factory,max_jobs=1)
        self.assertEqual(result['new_attempts'],1)
        self.assertEqual(result['selected_jobs'],1)

    def test_dev_prefix_does_not_retry_failed_answers(self):
        def factory(*a,**kw):
            b=self.factory(*a,**kw)
            def fail(*a,**kw):raise RuntimeError('fixture failure')
            b.generate=fail
            return b
        args=dict(prepared=self.prepared,data_root=self.data,runs=self.runs,
                  model=self.models[0],split='dev',backend_factory=factory,max_jobs=1)
        self.assertEqual(execute_reference(**args)['new_attempts'],1)
        self.assertEqual(execute_reference(**args)['new_attempts'],0)
        self.assertEqual(read_reference_run(self.runs)['attempts'],1)

    def test_cli_dev_prefix_forwarding(self):
        from unittest.mock import patch
        from meme_study.reference_cli import main
        with patch('meme_study.reference_execution.execute_reference',return_value={}) as run:
            code=main(['run','--prepared',str(self.prepared),'--data-root',str(self.data),
                '--runs',str(self.runs),'--model',self.models[0],'--split','dev','--max-jobs','3'])
        self.assertEqual(code,0)
        self.assertEqual(run.call_args.kwargs['max_jobs'],3)

if __name__=='__main__':unittest.main()
