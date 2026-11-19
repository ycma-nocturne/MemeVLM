"""Generated image fixtures only; pixel decoding is real, no research images."""
import hashlib
import importlib.util
import io
import struct
import unittest
from unittest.mock import patch
import zlib

from PIL import Image, ImageFile


def encoded_image(format='PNG', size=(3, 2)):
    image = Image.new('RGB', size, (25, 50, 75))
    output = io.BytesIO()
    image.save(output, format=format)
    return output.getvalue()


def png_with_stream(width, height, stream):
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', stream) + chunk(b'IEND', b''))


class ImageDecodeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.image_io'), 'strict image decoding is not implemented')
        from meme_study.image_io import decode_image_receipt
        self.decode = decode_image_receipt

    def test_loads_pixels_and_records_dimensions_encoding_hash_without_rewriting(self):
        raw = encoded_image()
        receipt = self.decode(raw, source_name='fixture.png', expected_sha256=hashlib.sha256(raw).hexdigest())
        self.assertTrue(receipt['decoded'])
        self.assertEqual((receipt['width'], receipt['height']), (3, 2))
        self.assertEqual(receipt['detected_format'], 'png')
        self.assertEqual(receipt['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(receipt['bytes'], len(raw))

    def test_actual_encoding_wins_over_mislabeled_filename(self):
        receipt = self.decode(encoded_image('JPEG'), source_name='mislabeled.png')
        self.assertEqual(receipt['detected_format'], 'jpeg')
        self.assertTrue(receipt['extension_mismatch'])

    def test_signature_and_valid_chunks_cannot_substitute_for_decodable_pixels(self):
        bad_images = [b'\x89PNG\r\n\x1a\nnot an image', png_with_stream(2, 2, b'invalid zlib payload'),
                      encoded_image()[:-12], b'\xff\xd8\xffbad pixels\xff\xd9']
        for raw in bad_images:
            with self.subTest(raw=raw[:12]):
                with self.assertRaisesRegex(ValueError, 'decode|image|truncated'):
                    self.decode(raw, source_name='bad.png')

    def test_rejects_animated_content_instead_of_silently_reading_first_frame(self):
        output = io.BytesIO()
        frames = [Image.new('RGB', (2, 2), color) for color in ['red', 'blue']]
        frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], duration=10)
        with self.assertRaisesRegex(ValueError, 'animated|frame'):
            self.decode(output.getvalue(), source_name='animated.gif')

    def test_pixel_limit_is_checked_before_loading_large_declared_canvas(self):
        raw = png_with_stream(10000, 10000, zlib.compress(b'not enough pixels'))
        with self.assertRaisesRegex(ValueError, 'pixel|large'):
            self.decode(raw, source_name='oversized.png')
        with self.assertRaisesRegex(ValueError, 'pixel|large'):
            self.decode(encoded_image(), source_name='small.png', max_pixels=5)

    def test_hash_mismatch_and_invalid_limits_fail_closed(self):
        raw = encoded_image()
        with self.assertRaisesRegex(ValueError, 'checksum|hash'):
            self.decode(raw, source_name='small.png', expected_sha256='0' * 64)
        for limit in [0, -1, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.decode(raw, source_name='small.png', max_pixels=limit)

    def test_global_permissive_truncated_decoder_setting_is_not_silently_used(self):
        with patch.object(ImageFile, 'LOAD_TRUNCATED_IMAGES', True):
            with self.assertRaisesRegex(RuntimeError, 'truncated|strict'):
                self.decode(encoded_image(), source_name='small.png')


if __name__ == '__main__':
    unittest.main()
