import importlib.util
import unittest
from helpers import family, full_design
class DataContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.data'), 'case compiler is not implemented')
        from meme_study import data
        self.data=data
    def test_paired_cases_keep_notes_and_options_but_hide_gold(self):
        cases,gold=self.data.compile_cases([family()],permutation_family_ids=[])
        self.assertEqual(len(cases),8); self.assertEqual(len(gold),8)
        self.assertEqual([c['note'] for c in cases if c['background']=='relevant'],['测试相关背景']*3)
        self.assertTrue(all('gold_option' not in c and 'gold_fact' not in c and 'reviews' not in c for c in cases))
        self.assertEqual({tuple(o['semantic_id'] for o in c['options']) for c in cases},{('o1','o2','o3','none')})
    def test_full_design_counts_and_controls_only_use_direct(self):
        cases,gold=self.data.compile_cases(full_design(),permutation_family_ids=[f'f{i:02d}' for i in range(20,40)])
        jobs=self.data.build_jobs(cases,['qwen4b','qwen8b'],['D','R','V'])
        summary=self.data.design_summary(cases,jobs)
        self.assertEqual(summary['core_cases'],360); self.assertEqual(summary['core_dev_cases'],120)
        self.assertEqual(summary['core_test_cases'],240); self.assertEqual(summary['final_responses'],2360)
        self.assertEqual(summary['planned_calls'],3800); self.assertEqual(summary['intermediate_responses'],1440)
        self.assertTrue(all(j['workflow']=='D' for j in jobs if j['case']['control']!='core'))
        self.assertEqual(len({j['job_id'] for j in jobs}),2360)
    def test_permutation_changes_labels_but_preserves_semantic_gold(self):
        cases,gold=self.data.compile_cases([family('test','test',False)],permutation_family_ids=['test'])
        original=next(c for c in cases if c['control']=='core' and c['background']=='none')
        perm=next(c for c in cases if c['control']=='permutation')
        self.assertEqual(original['options'][0]['semantic_id'],'o1'); self.assertEqual(perm['options'][0]['semantic_id'],'o2')
        self.assertEqual(gold[original['case_id']]['option'],gold[perm['case_id']]['option'])
        text=next(c for c in cases if c['control']=='text')
        self.assertIsNone(text['image_path']); self.assertIsNone(text['fact_question'])
    def test_rejects_template_leakage_and_changed_sham(self):
        a,b=family(),family('f01','test'); b['template_group']=a['template_group']
        self.assertTrue(any('crosses splits' in e for e in self.data.validate_families([a,b])))
        a['variants']['sham']['gold_option']='o2'
        self.assertTrue(any('sham' in e for e in self.data.validate_families([a])))
    def test_rejects_non_test_permutation_and_path_escape(self):
        with self.assertRaises(ValueError): self.data.compile_cases([family()],permutation_family_ids=['f00'])
        a=family(); a['variants']['original']['image_path']='../secret.png'
        self.assertTrue(any('path' in e for e in self.data.validate_families([a])))
    def test_two_current_human_reviews_required_for_release(self):
        a=family(); self.assertTrue(self.data.validate_families([a],require_reviews=True))
        h=self.data.family_content_hash(a)
        a['reviews']=[{'reviewer':r,'role':'human','decision':'accept','reviewed_content_sha256':h,'reviewed_at':'2026-10-24T09:00:00+08:00'} for r in ['fixture-person-a','fixture-person-b']]
        self.assertEqual(self.data.validate_families([a],require_reviews=True),[])
        a['caption_zh']='changed after review'
        self.assertTrue(any('review' in e for e in self.data.validate_families([a],require_reviews=True)))
    def test_malformed_records_return_errors_instead_of_crashing(self):
        for bad in (None, [], {'family_id': []}, {'family_id':'x','split':[],'source':None,'option_order':None}):
            with self.subTest(bad=bad):
                self.assertTrue(self.data.validate_families([bad],require_reviews=True))
    def test_non_mapping_variants_and_reviews_cannot_pass(self):
        a=family();a['variants']['original']=[];a['reviews']=None
        self.assertTrue(self.data.validate_families([a],require_reviews=True))
    def test_current_reviewers_are_normalized_and_conflicts_rejected(self):
        a=family();h=self.data.family_content_hash(a)
        def review(name,decision='accept',when='2026-10-24T09:00:00+08:00'):
            return dict(reviewer=name,role='human',decision=decision,reviewed_content_sha256=h,reviewed_at=when)
        for reviews in ([review('Alice'),review('alice')],
                        [review('Alice'),review('Bob'),review('Bob','reject')],
                        [review('Alice'),review('Bob',when='')],
                        [review('Alice'),review('Bob',when='2026-10-24T09:00:00')]):
            a['reviews']=reviews
            with self.subTest(reviews=reviews):
                self.assertTrue(self.data.validate_families([a],require_reviews=True))
        a['reviews']=[review('Alice'),review('Bob')]
        self.assertEqual(self.data.validate_families([a],require_reviews=True),[])
    def test_identical_option_wording_is_rejected(self):
        a=family();a['options']['o2']=a['options']['o1']
        self.assertTrue(any('option' in e for e in self.data.validate_families([a])))
if __name__=='__main__': unittest.main()
