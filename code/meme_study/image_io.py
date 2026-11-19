"""Decode and verify source images in place."""
import io
import warnings

from PIL import Image, ImageFile

from .intake import image_receipt


MAX_DECODE_PIXELS = 40_000_000


def decode_image_receipt(data, *, source_name, expected_sha256=None,
                         max_pixels=MAX_DECODE_PIXELS):
    """Verify the encoded file, then reopen and load every static image pixel.

    Only intake-supported JPEG/PNG/GIF/WebP encodings are accepted. Actual
    encoding and untransformed pixel dimensions are recorded, regardless of the
    filename. This is a readability check, not a semantic or rights judgment.
    """
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError('max_pixels must be a positive integer')
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError('strict decoding requires LOAD_TRUNCATED_IMAGES=False')
    receipt = image_receipt(data, source_name=source_name, expected_sha256=expected_sha256)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as header:
                width, height = header.size
                if width < 1 or height < 1 or width * height > max_pixels:
                    raise ValueError(f'image pixel count exceeds the {max_pixels} pixel limit')
                if getattr(header, 'is_animated', False) or getattr(header, 'n_frames', 1) != 1:
                    raise ValueError('animated or multiframe images are not supported')
                encoding = (header.format or '').lower()
                if encoding != receipt['detected_format']:
                    raise ValueError('image encoding disagrees with its signature')
                header.verify()
            # verify() checks structure without decoding pixels. Reopening and
            # load() are both necessary to catch valid chunks with corrupt data.
            with Image.open(io.BytesIO(data)) as decoded:
                decoded.load()
                if decoded.size != (width, height) or (decoded.format or '').lower() != encoding:
                    raise ValueError('image metadata changed during decoding')
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError('image pixel canvas is too large for strict decoding') from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f'image decode failed: {exc}') from exc
    return dict(receipt, decoded=True, width=width, height=height)
