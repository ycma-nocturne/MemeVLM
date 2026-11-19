"""Synthetic scoring contracts; these fixtures are not model or human results."""
import copy
import importlib.util
import json
import unittest


class ReferenceScoringTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.reference_scoring'),
                             'reference scoring is not implemented')
        from meme_study import reference_scoring
        self.scoring = reference_scoring
        self.candidate = '人物拿着奖杯。表演令人信服，但这里没有解释反讽。'
        self.checklist = [
            {'item_id': 'V001', 'dimension': 'V', 'text': '可见奖杯'},
            {'item_id': 'V002', 'dimension': 'V', 'text': '可见动作'},
            {'item_id': 'I001', 'dimension': 'I', 'text': '角色身份'},
            {'item_id': 'R001', 'dimension': 'R', 'text': '解释关系'},
        ]
        self.items = [
            {'item_id': 'V001', 'verdict': 'covered', 'evidence': ['拿着奖杯']},
            {'item_id': 'V002', 'verdict': 'uncertain', 'evidence': ['人物']},
            {'item_id': 'I001', 'verdict': 'not_covered', 'evidence': []},
            {'item_id': 'R001', 'verdict': 'covered', 'evidence': ['表演令人信服']},
        ]
        self.options = [
            {'label': 'A', 'semantic_id': 'barrier', 'text': '形成阻碍'},
            {'label': 'B', 'semantic_id': 'invitation', 'text': '发出邀请'},
            {'label': 'C', 'semantic_id': 'none', 'text': '证据不足'},
            {'label': 'D', 'semantic_id': 'display', 'text': '用于展示'},
        ]
        self.gold = {'cue_answers': ['yes'], 'relation_answers': ['barrier']}
        self.pair = {'cue': 'yes', 'relation': 'A', 'explanation': '尖刺阻碍使用。'}

    def judge(self, items=None, **kwargs):
        text = json.dumps({'items': self.items if items is None else items}, ensure_ascii=False)
        return self.scoring.score_checklist(self.candidate, self.checklist, text, **kwargs)

    def assert_unavailable(self, result):
        self.assertEqual(result['status'], 'unavailable')
        self.assertTrue(result['failure_reason'])
        for dimension in 'VIKR':
            self.assertIsNone(result['dimensions'][dimension]['coverage'])
            self.assertIsNone(result['dimensions'][dimension]['uncertainty_rate'])

    def assert_pair_error(self, result):
        self.assertIn(result['status'], ('invalid', 'truncated'))
        self.assertEqual((result['cue_correct'], result['relation_correct'], result['joint_correct']),
                         (0, 0, 0))
        self.assertTrue(result['reason'])

    def test_mixed_coverage_counts_uncertain_in_applicable_denominator(self):
        result = self.judge()
        self.assertEqual(result['status'], 'ok')
        self.assertIsNone(result['failure_reason'])
        self.assertEqual(result['dimensions']['V'],
                         {'applicable': 2, 'covered': 1, 'uncertain': 1,
                          'coverage': .5, 'uncertainty_rate': .5})
        self.assertEqual(result['dimensions']['I'],
                         {'applicable': 1, 'covered': 0, 'uncertain': 0,
                          'coverage': 0., 'uncertainty_rate': 0.})
        self.assertEqual(result['dimensions']['R']['coverage'], 1.)

    def test_absent_dimensions_are_na_not_zero(self):
        self.assertEqual(self.judge()['dimensions']['K'],
                         {'applicable': 0, 'covered': 0, 'uncertain': 0,
                          'coverage': None, 'uncertainty_rate': None})
        result = self.scoring.score_checklist('', [], '{"items":[]}')
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(all(d['coverage'] is None for d in result['dimensions'].values()))

    def test_all_uncertain_items_are_reported_without_coverage_credit(self):
        items = [dict(item, verdict='uncertain', evidence=[]) for item in self.items]
        result = self.judge(items)
        self.assertEqual(result['dimensions']['V']['coverage'], 0.)
        self.assertEqual(result['dimensions']['V']['uncertainty_rate'], 1.)

    def test_judge_item_order_does_not_change_scores_or_evidence_identity(self):
        result = self.judge(list(reversed(self.items)))
        self.assertEqual(result['dimensions'], self.judge()['dimensions'])
        self.assertEqual([item['item_id'] for item in result['items']],
                         ['V001', 'V002', 'I001', 'R001'])
        self.assertEqual(result['items'][0]['evidence'], ['拿着奖杯'])

    def test_fabricated_quote_invalidates_entire_judge_record(self):
        for verdict in ('covered', 'not_covered', 'uncertain'):
            items = copy.deepcopy(self.items)
            items[0].update(verdict=verdict, evidence=['根本不存在的引文'])
            with self.subTest(verdict=verdict):
                self.assert_unavailable(self.judge(items))

    def test_covered_requires_at_least_one_nonempty_verbatim_span(self):
        for evidence in ([], [''], [' '], '拿着奖杯', [1], [None], ['拿着 奖杯']):
            items = copy.deepcopy(self.items)
            items[0]['evidence'] = evidence
            with self.subTest(evidence=evidence):
                self.assert_unavailable(self.judge(items))

    def test_missing_extra_or_repeated_item_ids_make_scores_unavailable(self):
        extra = {'item_id': 'K999', 'verdict': 'not_covered', 'evidence': []}
        for items in (self.items[:-1], self.items + [extra], self.items + [self.items[0]],
                      [self.items[0], self.items[0], self.items[2], self.items[3]]):
            with self.subTest(items=items):
                self.assert_unavailable(self.judge(items))

    def test_duplicate_json_keys_are_rejected_at_both_levels(self):
        valid = json.dumps({'items': self.items}, ensure_ascii=False)
        texts = [valid[:-1] + ',"items":[]}',
                 valid.replace('"verdict": "covered"',
                               '"verdict":"not_covered","verdict":"covered"', 1)]
        for text in texts:
            with self.subTest(text=text):
                self.assert_unavailable(self.scoring.score_checklist(
                    self.candidate, self.checklist, text))

    def test_malformed_json_or_top_level_schema_is_unavailable(self):
        valid = json.dumps({'items': self.items})
        for text in ('not JSON', '', '[]', 'null', '{"items":null}',
                     '{"items":{}}', '{}', valid[:-1] + ',"score":1}',
                     '```json\n' + valid + '\n```', valid + ' trailing',
                     '{"items":NaN}', '{"items":Infinity}', None, 1):
            with self.subTest(text=text):
                self.assert_unavailable(self.scoring.score_checklist(
                    self.candidate, self.checklist, text))

    def test_wrong_judge_item_fields_types_and_verdict_are_unavailable(self):
        bad = [None, {'item_id': 'V001', 'verdict': 'covered'},
               dict(self.items[0], item_id=1), dict(self.items[0], verdict=True),
               dict(self.items[0], verdict='partial'), dict(self.items[0], score=1)]
        for item in bad:
            with self.subTest(item=item):
                self.assert_unavailable(self.judge([item] + self.items[1:]))

    def test_truncated_judge_is_unavailable_even_if_json_is_valid(self):
        self.assert_unavailable(self.judge(truncated=True))

    def test_bad_checklist_and_candidate_contract_raise_instead_of_scoring(self):
        bad = [None, {}, [None], [self.checklist[0], self.checklist[0]],
               [dict(self.checklist[0], dimension='X')],
               [dict(self.checklist[0], text=' ')],
               [dict(self.checklist[0], item_id=[])],
               [dict(self.checklist[0], dimension=[])]]
        for checklist in bad:
            with self.subTest(checklist=checklist):
                with self.assertRaises(ValueError):
                    self.scoring.score_checklist(self.candidate, checklist, '{}')
        with self.assertRaises(ValueError):
            self.scoring.score_checklist(None, self.checklist, '{}')
        with self.assertRaises(ValueError):
            self.judge(truncated='false')

    def test_source_hashes_bind_inputs_without_mutating_checklists(self):
        original = copy.deepcopy(self.checklist)
        result = self.judge()
        self.assertEqual(self.checklist, original)
        hashes = result['source_sha256']
        self.assertEqual(set(hashes), {'candidate', 'checklist', 'judge'})
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        changed = self.scoring.score_checklist(self.candidate + '新增文字', self.checklist,
                    json.dumps({'items': self.items}, ensure_ascii=False))
        self.assertNotEqual(hashes['candidate'], changed['source_sha256']['candidate'])
        self.assertEqual(hashes['checklist'], changed['source_sha256']['checklist'])

    def test_pair_correct_cue_and_relation_receive_joint_credit(self):
        result = self.scoring.score_pair(json.dumps(self.pair), self.options, self.gold)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual((result['cue_correct'], result['relation_correct'], result['joint_correct']),
                         (1, 1, 1))
        self.assertEqual(result['cue'], 'yes')
        self.assertEqual(result['semantic_relation'], 'barrier')
        self.assertEqual(result['explanation'], '尖刺阻碍使用。')
        self.assertIsNone(result['reason'])

    def test_valid_but_wrong_pair_components_are_scored_separately(self):
        for cue, relation, expected in [('yes', 'B', (1, 0, 0)),
                                         ('no', 'A', (0, 1, 0)),
                                         ('no', 'B', (0, 0, 0))]:
            raw = json.dumps(dict(self.pair, cue=cue, relation=relation))
            with self.subTest(cue=cue, relation=relation):
                result = self.scoring.score_pair(raw, self.options, self.gold)
                self.assertEqual(result['status'], 'ok')
                self.assertEqual((result['cue_correct'], result['relation_correct'],
                                  result['joint_correct']), expected)

    def test_multiple_acceptable_answers_and_insufficient_are_supported(self):
        gold = {'cue_answers': ['no', 'insufficient'], 'relation_answers': ['none', 'display']}
        for cue, relation in [('no', 'D'), ('insufficient', 'C')]:
            result = self.scoring.score_pair(
                json.dumps(dict(self.pair, cue=cue, relation=relation)), self.options, gold)
            self.assertEqual(result['joint_correct'], 1)

    def test_permutation_scores_by_semantics_and_not_display_label(self):
        options = copy.deepcopy(self.options)
        options[0]['label'], options[3]['label'] = 'D', 'A'
        moved = self.scoring.score_pair(json.dumps(dict(self.pair, relation='D')), options, self.gold)
        unchanged = self.scoring.score_pair(json.dumps(self.pair), options, self.gold)
        self.assertEqual(moved['semantic_relation'], 'barrier')
        self.assertEqual(moved['joint_correct'], 1)
        self.assertEqual(unchanged['semantic_relation'], 'display')
        self.assertEqual(unchanged['relation_correct'], 0)

    def test_bad_pair_schema_never_receives_partial_credit(self):
        values = [dict(self.pair, cue=True), dict(self.pair, cue='maybe'),
                  dict(self.pair, relation='E'), dict(self.pair, relation=1),
                  dict(self.pair, explanation=' '), dict(self.pair, explanation=[]),
                  dict(self.pair, score=1), {'cue': 'yes', 'relation': 'A'}, [], None]
        for value in values:
            with self.subTest(value=value):
                self.assert_pair_error(self.scoring.score_pair(json.dumps(value), self.options, self.gold))

    def test_malformed_or_duplicate_key_pair_json_is_wrong_not_na(self):
        valid = json.dumps(self.pair)
        texts = ['{', '', valid + ' trailing', '```json\n' + valid + '\n```',
                 valid[:-1] + ',"cue":"no"}',
                 '{"cue":NaN,"relation":"A","explanation":"x"}', None, 3]
        for text in texts:
            with self.subTest(text=text):
                self.assert_pair_error(self.scoring.score_pair(text, self.options, self.gold))

    def test_truncated_pair_is_wrong_even_if_complete_json(self):
        result = self.scoring.score_pair(json.dumps(self.pair), self.options, self.gold, truncated=True)
        self.assertEqual(result['status'], 'truncated')
        self.assert_pair_error(result)

    def test_bad_gold_is_a_configuration_error_even_for_truncated_output(self):
        bad = [None, {}, {'cue_answers': [], 'relation_answers': ['barrier']},
               dict(self.gold, cue_answers=['true']), dict(self.gold, cue_answers=['yes', 'yes']),
               dict(self.gold, cue_answers='yes'), dict(self.gold, relation_answers=[]),
               dict(self.gold, relation_answers=['missing']),
               dict(self.gold, relation_answers=['barrier', 'barrier']),
               dict(self.gold, relation_answers=[None]), dict(self.gold, extra=1)]
        for gold in bad:
            with self.subTest(gold=gold):
                with self.assertRaises(ValueError):
                    self.scoring.score_pair('{}', self.options, gold, truncated=True)

    def test_bad_options_are_configuration_errors_not_candidate_errors(self):
        bad = [None, self.options[:-1], self.options + [self.options[0]],
               [self.options[0], self.options[0], *self.options[2:]],
               [dict(self.options[0], semantic_id='invitation'), *self.options[1:]],
               [dict(self.options[0], label='E'), *self.options[1:]],
               [dict(self.options[0], text=''), *self.options[1:]],
               [dict(self.options[0], semantic_id=[]), *self.options[1:]]]
        for options in bad:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.scoring.score_pair('{}', options, self.gold)
        with self.assertRaises(ValueError):
            self.scoring.score_pair('{}', self.options, self.gold, truncated=1)

    def test_pair_input_objects_are_not_mutated_and_gold_is_hashed(self):
        options, gold = copy.deepcopy(self.options), copy.deepcopy(self.gold)
        result = self.scoring.score_pair(json.dumps(self.pair), options, gold)
        self.assertEqual(options, self.options)
        self.assertEqual(gold, self.gold)
        self.assertEqual(set(result['source_sha256']), {'response', 'options', 'gold'})
        changed = self.scoring.score_pair(json.dumps(self.pair), self.options,
                    {'cue_answers': ['yes', 'no'], 'relation_answers': ['barrier']})
        self.assertNotEqual(result['source_sha256']['gold'], changed['source_sha256']['gold'])


if __name__ == '__main__':
    unittest.main()
