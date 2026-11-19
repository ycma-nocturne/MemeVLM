"""CPU tests; all content below is a synthetic protocol fixture."""
import copy
import importlib.util
import json
import unittest


def case_fixture(control='core', background='relevant'):
    return {
        'case_id': 'fixture-case', 'family_id': 'fixture-family',
        'template_group': 'fixture-template', 'split': 'dev',
        'variant': 'original', 'background': background, 'control': control,
        'image_path': None if control == 'text' else 'images/fixture.png',
        'image_sha256': None if control == 'text' else 'a' * 64,
        'caption_zh': 'CAPTION_SENTINEL', 'note': 'NOTE_SENTINEL',
        'options': [{'label': label, 'semantic_id': sid, 'text': 'OPTION_' + label}
                    for label, sid in zip('ABCD', ['o1', 'o2', 'o3', 'none'])],
        'fact_question': None if control == 'text' else 'FACT_SENTINEL',
    }


def text_parts(messages):
    return [item['text'] for message in messages for item in message['content']
            if item['type'] == 'text']


def image_parts(messages):
    return [item for message in messages for item in message['content']
            if item['type'] == 'image']


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.prompts'),
                             'workflow prompts are not implemented')
        from meme_study.prompts import build_messages
        self.build = build_messages

    def test_visual_observation_contains_only_image_and_neutral_instruction(self):
        case = case_fixture()
        messages = self.build(case, 'V', 'observation')
        serialized = json.dumps(messages, ensure_ascii=False)
        for forbidden in ['CAPTION_SENTINEL', 'NOTE_SENTINEL', 'FACT_SENTINEL',
                          'OPTION_', 'relevant', 'fixture-case', 'o1']:
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(len(image_parts(messages)), 1)
        self.assertEqual(len(text_parts(messages)), 1)
        altered = copy.deepcopy(case)
        altered.update(caption_zh='OTHER', note='OTHER', fact_question='OTHER', options=[])
        self.assertEqual(messages, self.build(altered, 'V', 'observation'))

    def test_draft_and_finals_include_full_input_but_not_metadata_or_gold(self):
        case = case_fixture()
        case.update(gold_option='SECRET_GOLD', gold_fact='SECRET_GOLD',
                    reference_explanation='SECRET_GOLD')
        for workflow, stage, previous in [('D', 'final', None), ('R', 'draft', None),
                                          ('R', 'final', 'prior'), ('V', 'final', 'prior')]:
            with self.subTest(workflow=workflow, stage=stage):
                messages = self.build(case, workflow, stage, previous)
                self.assertEqual([m['role'] for m in messages], ['user'])
                text = '\n'.join(text_parts(messages))
                for wanted in ['CAPTION_SENTINEL', 'NOTE_SENTINEL', 'FACT_SENTINEL',
                               'OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D']:
                    self.assertIn(wanted, text)
                for forbidden in ['SECRET_GOLD', 'fixture-case', 'semantic_id', 'gold_option']:
                    self.assertNotIn(forbidden, text)
                self.assertEqual(image_parts(messages), [{'type': 'image', 'url': 'images/fixture.png'}])
                if stage == 'final':
                    for key in ['option_id', 'fact', 'explanation', 'A', 'D', 'JSON']:
                        self.assertIn(key, text)

    def test_final_preserves_previous_text_byte_for_byte_as_text_block(self):
        previous = ' \n原样草稿: {"odd": true}\n```\n '
        for workflow in ['R', 'V']:
            self.assertIn(previous, text_parts(self.build(case_fixture(), workflow, 'final', previous)))

    def test_text_and_permutation_controls_are_direct_only(self):
        for control in ['text', 'permutation']:
            case = case_fixture(control)
            self.build(case, 'D', 'final')
            for workflow, stage in [('R', 'draft'), ('V', 'observation'), ('R', 'final'), ('V', 'final')]:
                with self.assertRaises(ValueError):
                    self.build(case, workflow, stage, 'prior' if stage == 'final' else None)
        messages = self.build(case_fixture('text'), 'D', 'final')
        self.assertEqual(image_parts(messages), [])
        self.assertIn('null', '\n'.join(text_parts(messages)))
        self.assertNotIn('FACT_SENTINEL', '\n'.join(text_parts(messages)))

    def test_illegal_stages_and_missing_previous_or_image_are_rejected(self):
        for workflow, stage, previous in [('X', 'final', None), ('D', 'draft', None),
                                          ('V', 'draft', None), ('R', 'observation', None),
                                          ('R', 'final', None), ('V', 'final', None),
                                          ('D', 'final', 'unexpected')]:
            with self.subTest(workflow=workflow, stage=stage, previous=previous):
                with self.assertRaises(ValueError):
                    self.build(case_fixture(), workflow, stage, previous)
        case = case_fixture(); case['image_path'] = None
        with self.assertRaises(ValueError): self.build(case, 'V', 'observation')


if __name__ == '__main__':
    unittest.main()
