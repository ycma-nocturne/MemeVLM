"""Build or verify the project-authored English meme family dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, features, __version__ as pillow_version

from pictograms import draw_icon


SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parent
SIZE = (640, 560)
SPLIT_COUNTS = {"train": 21, "validation": 5, "test": 4}
FORMATS = ("comic", "status", "chat", "poster")
PALETTES = (
    ("#201d37", "#6254a1", "#eee8fb", "#faf8ff"),
    ("#152d27", "#20755b", "#dcece1", "#f5fbf7"),
    ("#462120", "#a14244", "#f7dfdc", "#fffcf7"),
    ("#112b43", "#286695", "#dcecf6", "#f2f9fd"),
    ("#443121", "#8a5738", "#f3e4ce", "#fffaf1"),
)
VARIANTS = ("original", "edited", "sham")
SHAM_REGION = (546, 43, 590, 88)
OUTCOME_REGIONS = {
    "comic": (331, 164, 588, 493),
    "status": (77, 270, 565, 477),
    "chat": (177, 295, 574, 521),
    "poster": (83, 247, 558, 486),
}


@dataclass(frozen=True)
class Scene:
    group: str
    mechanism: str
    title: str
    caption: str
    original: str
    edited: str
    original_relation: str
    edited_relation: str

    @property
    def format(self) -> str:
        return FORMATS[index("format", self.group, len(FORMATS))]

    @property
    def palette(self) -> tuple[str, str, str, str]:
        return PALETTES[index("palette", self.group, len(PALETTES))]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def index(namespace: str, name: str, count: int) -> int:
    return int(digest(f"english-v1/{namespace}/{name}".encode())[:12], 16) % count


def read_scenes() -> list[Scene]:
    rows = []
    mechanism = ""
    for number, raw in enumerate((SOURCE / "scenes.txt").read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            mechanism = line[1:-1]
            continue
        columns = [value.strip() for value in line.split("|")]
        if len(columns) != 6 or not mechanism or not all(columns):
            raise ValueError(f"invalid scene at line {number}")
        for outcome in columns[2:4]:
            if ":" not in outcome or not outcome.split(":", 1)[1]:
                raise ValueError(f"expected icon:label at line {number}")
        if columns[2].split(":", 1)[1].casefold() == columns[3].split(":", 1)[1].casefold():
            raise ValueError(f"outcome label must change at line {number}")
        rows.append(Scene(f"en-{len(rows)+1:03}", mechanism, *columns))
    for attr in ("title", "caption"):
        if len({getattr(scene, attr).casefold() for scene in rows}) != len(rows):
            raise ValueError(f"duplicate scene {attr}")
    return rows


def make_splits(scenes: list[Scene]) -> dict[str, str]:
    by_mechanism = defaultdict(list)
    for scene in scenes:
        by_mechanism[scene.mechanism].append(scene.group)
    assignments = {}
    for groups in by_mechanism.values():
        if len(groups) != sum(SPLIT_COUNTS.values()):
            raise ValueError("each mechanism needs the same split quotas")
        ranks = sorted(groups, key=lambda group: digest(f"english-v1/split/{group}".encode()))
        for split, count in SPLIT_COUNTS.items():
            assignments.update({group: split for group in ranks[:count]})
            ranks = ranks[count:]
    return assignments


@lru_cache(maxsize=None)
def font(size: int, bold: bool):
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(str(SOURCE / "fonts" / f"DejaVuSans{suffix}.ttf"), size)


def wrap(text: str, logical_font, width: int) -> list[str]:
    lines = []
    pending = ""
    for word in text.split():
        candidate = f"{pending} {word}" if pending else word
        if pending and logical_font.getlength(candidate) > width:
            lines.append(pending)
            pending = word
        else:
            pending = candidate
    return lines + [pending]


def put(draw, text, box, color, initial_size=29, bold=False, center=False):
    x1, y1, x2, y2 = box
    for size in range(initial_size, 14, -1):
        logical_font = font(size, bold)
        lines = wrap(text, logical_font, x2 - x1)
        stride = size + 7
        height = stride * len(lines)
        if height <= y2 - y1 and all(logical_font.getlength(line) <= x2 - x1 for line in lines):
            break
    else:
        raise ValueError(f"text will not fit: {text}")
    baseline = y1 + (y2 - y1 - height) // 2 + size // 2
    for line in lines:
        draw.text(((x1+x2)//2 if center else x1, baseline), line,
                  font=logical_font, fill=color, anchor="mm" if center else "lm")
        baseline += stride


def surface(draw, box, fill, border=None, radius=19, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=border, width=width)


def scene_frame(draw, scene: Scene, sham: bool) -> None:
    ink, accent, tint, white = scene.palette
    kind = scene.format
    ring = bool(index("corner-mark", scene.group, 2)) != sham
    if kind == "comic":
        surface(draw, (25, 28, 615, 532), white, ink, 20)
        put(draw, scene.title, (45, 42, 523, 113), ink, 37, True)
        surface(draw, (43, 132, 312, 510), tint, ink, 13)
        surface(draw, (328, 132, 592, 510), white, ink, 13)
        surface(draw, (59, 168, 297, 339), white, ink, 21)
        put(draw, scene.caption, (76, 175, 280, 335), ink, 29, True)
        draw.polygon(((142, 336), (153, 360), (172, 336)), fill=white)
        draw_icon(draw, "person", 116, 367, ink)
        draw.line((217, 430, 283, 430), fill=accent, width=6)
        draw.line((267, 414, 283, 430, 267, 446), fill=accent, width=6)
        draw_mark(draw, accent, white, ring)
        return
    if kind == "status":
        surface(draw, (25, 27, 614, 533), ink, radius=23)
        surface(draw, (43, 113, 595, 512), white, radius=14)
        put(draw, scene.title, (54, 41, 514, 103), white, 35, True)
        put(draw, scene.caption, (68, 132, 572, 242), ink, 29)
        surface(draw, (61, 253, 579, 490), tint, radius=14)
        draw.line((68, 245, 572, 245), fill=accent, width=3)
        draw_mark(draw, tint, ink, ring)
        return
    if kind == "chat":
        surface(draw, (32, 24, 609, 540), white, ink, 25)
        draw.rounded_rectangle((33, 25, 608, 106), radius=21, fill=tint)
        put(draw, scene.title, (65, 40, 514, 101), ink, 35, True)
        surface(draw, (56, 125, 467, 255), tint, radius=23)
        put(draw, scene.caption, (81, 133, 442, 249), ink, 26)
        draw.polygon(((63, 223), (44, 253), (82, 253)), fill=tint)
        surface(draw, (165, 279, 584, 526), accent, radius=19)
        draw_mark(draw, accent, tint, ring)
        return
    surface(draw, (35, 32, 606, 526), white, ink, 9, 5)
    surface(draw, (44, 42, 596, 514), white, accent, 6)
    put(draw, scene.title, (74, 58, 518, 123), ink, 36, True)
    put(draw, scene.caption, (69, 129, 573, 228), ink, 30, True, True)
    draw.line((75, 239, 563, 239), fill=accent, width=4)
    draw.line((76, 494, 562, 494), fill=accent, width=4)
    draw_mark(draw, accent, white, ring)


def draw_mark(draw, accent, ground, ring):
    """Decorative mark in an isolated location shared by all four formats."""
    x1, y1, x2, y2 = SHAM_REGION
    if ring:
        draw.ellipse((x1+8, y1+8, x2-8, y2-8), outline=accent, width=5, fill=ground)
    else:
        draw.polygon((((x1+x2)//2, y1+6), (x2-6, (y1+y2)//2),
                      ((x1+x2)//2, y2-6), (x1+6, (y1+y2)//2)), fill=accent)


def render(scene: Scene, variant: str):
    ink, accent, tint, white = scene.palette
    image = Image.new("RGB", SIZE, tint)
    draw = ImageDraw.Draw(image)
    scene_frame(draw, scene, sham=(variant == "sham"))
    x1, y1, x2, y2 = OUTCOME_REGIONS[scene.format]
    icon, text = (scene.edited if variant == "edited" else scene.original).split(":", 1)
    height = y2 - y1
    color = white if scene.format == "chat" else accent
    text_color = white if scene.format == "chat" else ink
    draw_icon(draw, icon, (x1+x2)//2 - 50, y1 + (32 if height > 240 else 9), color)
    put(draw, text, (x1+13, y1+118, x2-13, y2-15), text_color,
        initial_size=34 if scene.format != "comic" else 31, bold=True, center=True)
    return image


def validate_edit(original, variant, region, description):
    diff = ImageChops.difference(original, variant).getbbox()
    x1, y1, x2, y2 = region
    if diff is None or diff[0] < x1 or diff[1] < y1 or diff[2] > x2 or diff[3] > y2:
        raise ValueError(f"{description}: expected a nonempty change inside {region}, got {diff}")


def json_bytes(rows) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()


def write_or_check(relative, content, check):
    path = ROOT / relative
    if check:
        if path.read_bytes() != content:
            raise ValueError(f"regeneration differs: {relative}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def family_records(scene, split, images):
    outcomes = {
        "original": scene.original.split(":", 1)[1],
        "edited": scene.edited.split(":", 1)[1],
        "sham": scene.original.split(":", 1)[1],
    }
    true_for = ("original", "edited")[index("cue-polarity", scene.group, 2)]
    cue = f'Does the image display the wording "{outcomes[true_for]}"?'
    options = {
        "original_relation": scene.original_relation,
        "edited_relation": scene.edited_relation,
        "caption_repeated": "The image repeats the instruction without showing what happened.",
        "unrelated": "The image changes the subject instead of responding to the setup.",
    }
    order = sorted(options, key=lambda name: digest(f"english-v1/sft/{scene.group}/{name}".encode()))
    labels = dict(zip(order, "ABCD"))
    choices = "\n".join(f"{labels[name]}. {options[name]}" for name in order)
    prompt = (
        f"Inspect the image. Answer the cue question: {cue}\n"
        "Which option best explains the relation between the setup and the punchline?\n"
        f"{choices}\n"
        'Return one JSON object with exactly three keys. Set "cue" to "yes" or "no", '
        '"relation" to "A", "B", "C", or "D", and "explanation" to one short sentence.'
    )
    rows = []
    variants = {}
    for variant in VARIANTS:
        relation = "edited_relation" if variant == "edited" else "original_relation"
        visible = outcomes[variant] == outcomes[true_for]
        evidence = outcomes[variant].rstrip(".")
        item = dict(images[variant], cue="yes" if visible else "no", relation=relation,
                    explanation=f'The visible outcome is "{evidence}".')
        variants[variant] = item
        answer = json.dumps({"cue": item["cue"], "relation": labels[relation],
                             "explanation": item["explanation"]}, ensure_ascii=False, separators=(",", ":"))
        rows.append({
            "id": f"{scene.group}-paired-{variant}",
            "group_id": scene.group,
            **images[variant],
            "prompt": prompt,
            "answer": answer,
            "source": "project-authored-english-v1",
            "task": "paired",
            "variant": variant,
            "language": "en",
        })
    rows.append({
        "id": f"{scene.group}-explanation",
        "group_id": scene.group,
        **images["original"],
        "prompt": "Explain the joke in one or two English sentences. Identify how the visible outcome relates to the setup.",
        "answer": f'The result shows "{outcomes["original"]}". {scene.original_relation}',
        "source": "project-authored-english-v1",
        "task": "explanation",
        "variant": "original",
        "language": "en",
    })
    pair = {
        "group_id": scene.group,
        "split": split,
        "mechanism": scene.mechanism,
        "format": scene.format,
        "cue_question": cue,
        "options": options,
        "variants": variants,
    }
    return rows, pair


def build(check):
    scenes = read_scenes()
    splits = make_splits(scenes)
    pairs = []
    by_split = defaultdict(list)
    png_hashes, rgb_hashes, paths = set(), set(), set()
    for scene in scenes:
        images = {variant: render(scene, variant) for variant in VARIANTS}
        validate_edit(images["original"], images["edited"], OUTCOME_REGIONS[scene.format], scene.group)
        validate_edit(images["original"], images["sham"], SHAM_REGION, scene.group)
        refs = {}
        for variant, image in images.items():
            out = io.BytesIO()
            image.save(out, format="PNG", compress_level=7)
            raw = out.getvalue()
            png, rgb = digest(raw), digest(image.tobytes())
            if png in png_hashes or rgb in rgb_hashes:
                raise ValueError(f"duplicate output: {scene.group}/{variant}")
            png_hashes.add(png)
            rgb_hashes.add(rgb)
            path = f"images/{scene.group}/{variant}.png"
            paths.add(path)
            write_or_check(path, raw, check)
            refs[variant] = {"image": path, "image_sha256": png}
        split = splits[scene.group]
        rows, pair = family_records(scene, split, refs)
        by_split[split].extend(rows)
        pairs.append(pair)
    for split, rows in by_split.items():
        write_or_check(f"{split}.jsonl", json_bytes(rows), check)
    write_or_check("pairs.jsonl", json_bytes(pairs), check)
    actual_paths = {path.relative_to(ROOT).as_posix() for path in (ROOT / "images").rglob("*.png")}
    if paths != actual_paths:
        raise ValueError("dataset image tree contains unexpected PNG paths")
    print(json.dumps({
        "families": len(scenes),
        "images": len(paths),
        "rows": {split: len(rows) for split, rows in by_split.items()},
        "mechanisms": dict(Counter(scene.mechanism for scene in scenes)),
        "formats": dict(Counter(scene.format for scene in scenes)),
        "pillow": pillow_version,
        "freetype": features.version("freetype2"),
        "zlib": features.version("zlib"),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare regenerated PNGs and manifests byte for byte")
    build(parser.parse_args().check)
