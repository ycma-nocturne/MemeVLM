"""Behavioral checks for the reference-reuse matrix and evidence isolation."""
import copy
import json
import unittest
from meme_study.reference_protocol import (compile_explanation_jobs, compile_pair_jobs,
                                         compile_judge_jobs, summarize_plan)


def records():
    return [{'family_id':f'f{i:02}', 'source_id':i+1,
             'split':'dev' if i<20 else 'test','template_group':f't{i}',
             'source':{'record_sha256':'a'*64},
             'image':{'path':f'images/{i}.png','sha256':'b'*64},
             'ocr_text':None if i==0 else '只属于文字条件的字幕',
             'knowledge_facts':['独立事实甲。','这张图的笑点来自偷换概念。'],
             'reference':{'explanation':'PRIVATE_GOLD_EXPLANATION',
                          'checklist':[{'item_id':'V001','dimension':'V','text':'PRIVATE_CHECKLIST'}]}}
            for i in range(60)]


def hints(rows):
    return {r['family_id']:{'source_record_sha256':'a'*64,
                            'source_fact_indices':[0], 'text':'独立事实甲。'} for r in rows}


def pairs(rows):
    result=[]
    for row in rows[:5]+rows[20:30]:
        result.append({'family_id':row['family_id'],'split':row['split'],
                       'template_group':row['template_group'],
                       'cue_question':'是否有尖刺？', 'relation_question':'哪种关系有图像支持？',
                       'options':[{'label':l,'semantic_id':s,'text':t} for l,s,t in
                                  [('A','barrier','没有上锁却难接近'),('B','smooth','把手容易使用'),
                                   ('C','other','另一个解释'),('D','insufficient','证据不足')]],
                       'variants':{v:{'image_path':f'images/{row["family_id"]}-{v}.png',
                                      'image_sha256':'c'*64,
                                      'cue_answers':['no' if v=='edited' else 'yes'],
                                      'relation_answers':['smooth' if v=='edited' else 'barrier']}
                                   for v in ['original','edited','sham']},
                       'material_check':None})
    return result


class ReferenceProtocolTests(unittest.TestCase):
    def test_three_conditions_do_not_leak_other_inputs_or_references(self):
        rows=records(); jobs=compile_explanation_jobs(rows,hints(rows),['m4','m8'])
        self.assertEqual(len(jobs),360)
        self.assertEqual(len({j['job_id'] for j in jobs}),360)
        for j in jobs:
            msg=json.dumps(j['messages'],ensure_ascii=False)
            self.assertNotIn('PRIVATE_',msg)
            self.assertNotIn('笑点来自偷换概念',msg)
            if j['condition']=='ocr_only':
                self.assertNotIn('"type": "image"',msg)
                self.assertNotIn('独立事实甲',msg)
            else:
                self.assertIn('"type": "image"',msg)
                self.assertNotIn('只属于文字条件',msg)
                self.assertEqual('独立事实甲' in msg,j['condition']=='hinted')
            self.assertEqual(j['max_new_tokens'],512)
        empty=[j for j in jobs if j['family_id']=='f00' and j['condition']=='ocr_only']
        self.assertTrue(all(j['ocr_available'] is False for j in empty))

    def test_stale_or_non_source_hint_rejected(self):
        rows=records(); h=hints(rows)
        h['f00']['text']='PRIVATE_GOLD_EXPLANATION'
        with self.assertRaises(ValueError):compile_explanation_jobs(rows,h,['m4'])
        h=hints(rows);h['f00']['source_record_sha256']='d'*64
        with self.assertRaises(ValueError):compile_explanation_jobs(rows,h,['m4'])

    def test_model_and_family_duplicates_cannot_double_count(self):
        rows=records()
        with self.assertRaises(ValueError):compile_explanation_jobs(rows,hints(rows),['m4','m4'])
        with self.assertRaises(ValueError):compile_explanation_jobs(rows+[rows[0]],hints(rows),['m4'])

    def test_preview_pairs_are_explicitly_unready_and_do_not_inherit_checklist(self):
        ps=pairs(records())
        with self.assertRaises(ValueError):compile_pair_jobs(ps,['m4','m8'])
        jobs,gold=compile_pair_jobs(ps,['m4','m8'],require_material_checks=False)
        self.assertEqual(len(jobs),110)
        self.assertEqual(sum(j['task_kind']=='pair' for j in jobs),90)
        self.assertEqual(sum(j['task_kind']=='permutation' for j in jobs),20)
        self.assertTrue(all(j['readiness']=='pending_material_check' for j in jobs))
        self.assertTrue(all('gold' not in j and 'reference' not in j for j in jobs))
        self.assertEqual(len(gold),110)
        e=next(j for j in jobs if j['variant']=='edited')
        self.assertEqual(gold[e['job_id']]['cue_answers'],['no'])
        self.assertEqual(gold[e['job_id']]['relation_answers'],['smooth'])
        original=next(j for j in jobs if j['task_kind']=='pair' and j['family_id']=='f20' and j['variant']=='original' and j['model']=='m4')
        repeat=next(j for j in jobs if j['task_kind']=='permutation' and j['family_id']=='f20' and j['model']=='m4')
        self.assertNotEqual(original['options'],repeat['options'])
        self.assertEqual(gold[original['job_id']],gold[repeat['job_id']])

    def test_invalid_sham_target_or_answer_set_rejected(self):
        ps=pairs(records());ps[0]['variants']['sham']['relation_answers']=['smooth']
        with self.assertRaises(ValueError):compile_pair_jobs(ps,['m4'],require_material_checks=False)
        ps=pairs(records());ps[0]['variants']['edited']['cue_answers']=['maybe']
        with self.assertRaises(ValueError):compile_pair_jobs(ps,['m4'],require_material_checks=False)

    def test_pair_images_must_stay_in_data_root_with_real_hash_format(self):
        for field,value in [('image_path','../outside.png'),('image_path','/tmp/a.png'),
                            ('image_sha256',''),('image_sha256','not-a-hash')]:
            ps=pairs(records());ps[0]['variants']['edited'][field]=value
            with self.assertRaises(ValueError):
                compile_pair_jobs(ps,['m4'],require_material_checks=False)

    def test_judge_jobs_bind_actual_output_and_hide_model_identity(self):
        rows=records();js=compile_explanation_jobs(rows,hints(rows),['m4','m8'])
        outputs=[{'job_id':js[0]['job_id'],'input_sha256':js[0]['input_sha256'],'status':'ok','text':'模型回答：有一个角色。',
                  'finish_reason':'stop'}]
        judge=compile_judge_jobs(js,outputs,rows,'judge9')
        self.assertEqual(len(judge),1)
        self.assertEqual(judge[0]['max_new_tokens'],768)
        text=json.dumps(judge[0]['messages'],ensure_ascii=False)
        self.assertIn('PRIVATE_CHECKLIST',text)
        self.assertIn('模型回答：有一个角色。',text)
        self.assertNotIn('m4',text)
        self.assertNotIn('m8',text)
        changed=copy.deepcopy(outputs);changed[0]['text']='另一条输出'
        other=compile_judge_jobs(js,changed,rows,'judge9')[0]
        self.assertNotEqual(judge[0]['input_sha256'],other['input_sha256'])
        self.assertNotEqual(judge[0]['source_output_sha256'],other['source_output_sha256'])
        self.assertEqual(judge[0]['parent_job_id'],js[0]['job_id'])
        with self.assertRaises(ValueError):compile_judge_jobs(js,outputs+outputs,rows,'judge9')

    def test_judge_rejects_stale_or_unbound_parent_response(self):
        rows=records();js=compile_explanation_jobs(rows,hints(rows),['m4'])
        output={'job_id':js[0]['job_id'],'status':'ok','text':'old output'}
        with self.assertRaises(ValueError):compile_judge_jobs(js,[output],rows,'judge9')
        output['input_sha256']='e'*64
        with self.assertRaises(ValueError):compile_judge_jobs(js,[output],rows,'judge9')
        output['input_sha256']=js[0]['input_sha256']
        altered=copy.deepcopy(rows);altered[0]['source']['record_sha256']='f'*64
        with self.assertRaises(ValueError):compile_judge_jobs(js,[output],altered,'judge9')

    def test_summary_counts_pending_judge_slots_without_fake_outputs(self):
        rows=records();a=compile_explanation_jobs(rows,hints(rows),['m4','m8'])
        b,_=compile_pair_jobs(pairs(rows),['m4','m8'],require_material_checks=False)
        summary=summarize_plan(a,b)
        self.assertEqual(summary['planned_calls'],830)
        self.assertEqual(summary['attempt_cap'],913)
        self.assertEqual(summary['judge_slots'],360)
        self.assertEqual(summary['ready_tested_calls'],360)
        self.assertEqual(summary['pending_material_calls'],110)
        self.assertEqual(summary['actual_model_calls'],0)

if __name__=='__main__':unittest.main()
