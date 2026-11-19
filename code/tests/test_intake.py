import importlib.util
import unittest


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.intake'), 'intake module missing')
        from meme_study import intake
        self.m = intake

    def record(self, item=7, lang='zh'):
        return {'id':item,'file_name':f'images/image_{item:04d}.png',
                'image_path':f'images/image_{item:04d}.png','meta':{'language':lang,'domain':'ACG'},
                'visual':{'ocr':'甲','description':'face'},'checked_gt':'Existing interpretation'}

    def test_filter_preserve_reference_and_do_not_invent_reviews_or_gold(self):
        rows=self.m.candidate_index([self.record(),self.record(8,'en')],revision='a'*40)
        self.assertEqual(len(rows),1)
        row=rows[0]
        self.assertEqual(row['candidate_id'],'memebench-0007')
        self.assertIn('/'+'a'*40+'/',row['image_url'])
        self.assertEqual(row['existing_annotation']['checked_gt'],'Existing interpretation')
        self.assertEqual(row['status'],'unreviewed_candidate')
        self.assertEqual(row['human_reviews'],[])
        self.assertNotIn('gold_option',row)
        self.assertIsNone(row['original_post_url'])

    def test_duplicate_source_and_nonlocal_image_reference_fail(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.m.candidate_index([self.record(),self.record()],revision='a'*40)
        for path in ('../outside.png','/tmp/image.png','https://x/y.png','images/../../a.png'):
            r=self.record();r['image_path']=r['file_name']=path
            with self.subTest(path=path),self.assertRaises(ValueError):
                self.m.candidate_index([r],revision='a'*40)

    def test_actual_image_signature_and_integrity_are_distinct(self):
        jpeg=b'\xff\xd8\xff\xe0hello\xff\xd9'
        result=self.m.image_receipt(jpeg,source_name='wrong.png')
        self.assertEqual(result['detected_format'],'jpeg')
        self.assertTrue(result['extension_mismatch'])
        self.assertFalse(result['decoded'])
        self.assertEqual(result['bytes'],len(jpeg))
        with self.assertRaises(ValueError):
            self.m.image_receipt(b'<html>error</html>',source_name='image.png')
        with self.assertRaises(ValueError):
            self.m.image_receipt(jpeg,source_name='image.png',expected_sha256='0'*64)

if __name__=='__main__': unittest.main()
