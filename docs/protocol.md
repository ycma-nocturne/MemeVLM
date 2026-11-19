# Study protocol

MemeVLM evaluates how visual evidence and factual background shape Chinese meme explanations. The study uses 60 MemeBench families (20 development, 40 test). Each group of shared templates or near-duplicates stays within a single split.

## A. Explanation coverage

Both Qwen3-VL models explain each meme under three conditions:

1. Original image only.
2. Original image with a hint drawn from selected MemeBench knowledge facts. Hints exclude facts that directly state the target relation.
3. Source OCR text only. Seven families lack a usable transcript (two development, five test). Analysis reports coverage by transcript availability. This condition measures what the same vision-language models can recover from text.

The design yields 360 explanations and up to 360 dependent judgments. InternVL3-9B receives the original image, the source reference and checklist, and a candidate response with its model identity hidden. It scores visual clues (V), identity links (I), knowledge units (K), and reasoning mechanisms (R). Coverage is the fraction of checklist items judged covered in each dimension. The scorer verifies quoted spans against the response and records invalid judgments as unavailable.

Two reviewers independently rate the same 10 blinded development responses for judge calibration and 20 blinded test responses for assessment. They check coverage and grounding, and record their disagreements. Freeze the test judge prompt and runtime settings after development calibration.

## B. Controlled visual edits

Fifteen families (five development, ten test) have an original, a target-changing edit, and a sham edit that preserves the target relation. AI-assisted material checks document the variants and their answer keys. Both Qwen models answer the same cue and relation questions for each variant, with 90 base calls and 20 option-order repeats on test originals.

Exact scoring uses variant-specific answer keys. A triple succeeds when both answers are correct on all three variants and the sham preserves the original relation. Malformed or truncated answers count as incorrect. Missing runs and infrastructure failures are recorded as unavailable.

## Analysis

The plan allocates 830 calls across both tasks with a shared cap of 913 attempts. Report development and test separately, giving each family equal weight. Compare matched families, report planned and available denominators, and use family bootstrap intervals for paired estimates (10,000 draws, seed 6000). Treat the ten test triples as exploratory evidence.

## Sources and contributions

- MemeBench provides the source images, OCR, knowledge facts, reference explanations, and checklists. The study pins dataset revision `d421198c5fd8108b17d336a5d1e8a7c229bd2b1f`. [MemeBench paper](https://arxiv.org/abs/2607.27798v1) · [Dataset card and license](https://huggingface.co/datasets/anonymous-neurips-2026/memebench)
- Qwen and OpenGVLab provide the pretrained models and base inference interfaces. [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) · [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) · [InternVL3-9B](https://huggingface.co/OpenGVLab/InternVL3-9B/commit/5f618513e35a9b85922341b8057feddfc8880e50)
- MemeVLM adds the grouped sample, hint selection, controlled edits, task prompts, scoring, and reproducible workflow.
