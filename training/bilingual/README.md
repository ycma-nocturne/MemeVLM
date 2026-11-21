# English synthetic meme corpus

The private course handoff contains 240 English scenarios written with generative AI for this project. Each family has its own setup and readings. The eight mechanisms are literal interpretation, automation errors, understatement, role reversal, tradeoffs, misplaced priorities, timing, and context mismatch. Four layouts show the scenarios as comics, status screens, chats, or posters. The handoff includes the renderer's DejaVu fonts and their license.

| Split | Families | PNGs | Explanation rows | Paired rows |
|---|---:|---:|---:|---:|
| Train | 168 | 504 | 168 | 504 |
| Validation | 40 | 120 | 40 | 120 |
| Test | 32 | 96 | 32 | 96 |

Each family has one original, one target edit that changes the outcome and its interpretation, and one sham edit that changes a corner mark. The renderer checks that each edit stays inside its designated region. Every mechanism contributes 21 train, 5 validation, and 4 test families. All variants of a family share one split. Captions and images were created for this corpus. The English corpus uses zero benchmark and Chinese-pilot assets.

The split files are line-delimited JSON. Each row has a dataset-root-relative image path, image hash, prompt and answer. The paired answer is a JSON string with `cue`, `relation`, and `explanation`. The family references used for paired evaluation are in `pairs.jsonl`. Use the train and validation files for fine-tuning. Reserve the test file for held-out scoring.

The task relies on visual text and simple pictograms. The held-out set measures transfer to new scenarios under the same renderer and rubric. The original project benchmark measures a separate scope.

## Reproduce

Use `source/scenes.txt` and the fonts from the private handoff. From this directory, run:

```sh
python -m pip install Pillow==13.0.0
python source/render.py
python source/render.py --check
```

The released PNGs were rendered with Pillow 13.0.0, FreeType 2.14.3, and zlib-ng 1.3.1. The check regenerates the manifests and pixels and compares the PNG files byte for byte. Font rasterization and PNG compression can vary with native library builds. Use the released PNGs and hashes when reproducing evaluation.
