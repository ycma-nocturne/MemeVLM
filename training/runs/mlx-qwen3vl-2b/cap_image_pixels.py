"""Cap pixel count on a derived copy of the MLX data, keeping image proportions."""
import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image

MAX_PIXELS = 262_144
root = Path(sys.argv[1])
receipt = []
for path in sorted((root / 'images').glob('*.png')):
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        width, height = image.size
        dimensions = image.size
        if width * height > MAX_PIXELS:
            ratio = math.sqrt(MAX_PIXELS / (width * height))
            dimensions = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            resized = image.convert('RGB').resize(dimensions, Image.Resampling.LANCZOS)
    if dimensions != (width, height):
        temp = path.with_suffix('.resized.png')
        resized.save(temp, optimize=True)
        temp.replace(path)
    receipt.append({'image': path.relative_to(root).as_posix(), 'input_size': [width, height],
                    'output_size': list(dimensions), 'input_sha256': original,
                    'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
assert len(receipt) == 131
(root / 'images-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(f"Capped {sum(r['input_size'] != r['output_size'] for r in receipt)} of {len(receipt)} images")
