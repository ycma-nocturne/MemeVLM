"""Render the project's Chinese contrast memes and their authored image QA rows."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SOURCE = Path(__file__).resolve().parent
PROMPT = '用两三句话说明这张梗图的笑点，结合两格里的文字和图标。'
THEMES = (
    ('#181c26', '#f8dcd7', '#e0eafe', '#23232d', 'horizontal'),
    ('#e0e2da', '#143c44', '#402f51', '#f7f4ef', 'vertical'),
    ('#11222a', '#b9e6df', '#ffdb90', '#202628', 'horizontal'),
    ('#ead6c4', '#582e41', '#2a405b', '#fff5ea', 'vertical'),
)


def draw_panel(canvas, box, tag, caption, icon, ink, paper, fonts):
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    center = (left + right) // 2
    draw.rounded_rectangle(box, radius=22, fill=paper)
    draw.text((left + 22, top + 16), tag, font=fonts['tag'], fill=ink)
    sprite = icon.resize((114, 114), Image.Resampling.LANCZOS)
    canvas.paste(sprite, (center - 57, top + 69), sprite)
    if draw.textlength(caption, font=fonts['caption']) > right - left - 24:
        raise ValueError(f'caption does not fit: {caption}')
    draw.text((center, bottom - 52), caption, font=fonts['caption'], fill=ink, anchor='mm')


def render(row, theme, fonts):
    background, left_color, right_color, ink, layout = theme
    if layout == 'horizontal':
        size = (720, 322)
        boxes = ((18, 18, 351, 304), (369, 18, 702, 304))
    else:
        size = (500, 610)
        boxes = ((18, 18, 482, 297), (18, 313, 482, 592))
    canvas = Image.new('RGB', size, background)
    for side, box, tag, color in zip(('left', 'right'), boxes, ('想象', '现实'), (left_color, right_color)):
        with Image.open(SOURCE / 'twemoji' / f"{row[side + '_icon']}.png") as sprite:
            draw_panel(canvas, box, tag, row[side + '_text'], sprite.convert('RGBA'), ink, color, fonts)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', required=True, type=Path, help='NotoSansCJKsc-Bold.otf')
    parser.add_argument('--output', required=True, type=Path, help='New directory for the pilot images and JSONL')
    args = parser.parse_args()
    fonts = {name: ImageFont.truetype(str(args.font), size) for name, size in [('tag', 23), ('caption', 25)]}
    with (SOURCE / 'scenes.tsv').open(encoding='utf-8', newline='') as stream:
        scenes = list(csv.DictReader(stream, delimiter='\t'))
    args.output.mkdir()
    (args.output / 'images').mkdir()
    rows = {'train': [], 'validation': []}
    for scene in scenes:
        group = 'pilot-' + scene['id']
        for index, theme in enumerate(THEMES, 1):
            item = f'{group}-{index:02d}'
            relative = f'images/{item}.png'
            output = args.output / relative
            render(scene, theme, fonts).save(output, optimize=True)
            rows[scene['split']].append({
                'id': item, 'group_id': group, 'image': relative, 'prompt': PROMPT,
                'answer': scene['answer'], 'source': 'memevlm-authored-synthetic',
                'image_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            })
    for split, examples in rows.items():
        with (args.output / f'{split}.jsonl').open('w', encoding='utf-8') as out:
            for row in examples:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print({split: len(items) for split, items in rows.items()})


if __name__ == '__main__':
    main()
