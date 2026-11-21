# Local adapter evaluation

## Training

The 2B MLX 4-bit checkpoint and the resized training copy are pinned to the first local run. Continue with the settings in `training/selection-plan.json`. Select the adapter with the lowest token-weighted loss on the same 33 validation answers. An epoch boundary starts fresh AdamW state. Use the validation loss and artifact checks for selection.

## Evaluation

Compare the pinned 2B base and the selected adapter with identical image processing, prompts, decoding settings, and output-length rules. Use the existing v4 family split and material files. The development split verifies generation, result parsing, and judge behavior. Record the selected adapter digest and freeze configurations before generating held-out outputs.

- Task A uses the three original conditions on the selected development and held-out family splits. Record V/I/K/R coverage with availability denominators. The local judge uses the pinned MLX 4-bit InternVL snapshot and its native processor. Save the raw judgments and export blinded ratings for human review.
- Task B reuses the original and control variants, option permutations, and exact answer-key scorer. Report strict triple success by family, per-variant accuracy, output validity, and available paired denominators. Report the ten held-out triples as exploratory.

This extension evaluates two states of one quantized 2B base. The 4B/8B BF16 study has its own runtime, ledger, and protocol. Report grouped estimates and retain raw outputs, hashes, and stage settings alongside the results.

## Supplementary Task B decoding

All 30 unconstrained development calls produced responses. Zero conformed to the answer schema. Keep unconstrained format adherence and exact integrated scores. For a separate semantic-choice comparison, constrain both arms to the same JSON keys, the original cue and option enums, and a nonempty explanation. Use the same prompts and scorer. Freeze this decoding policy separately before its held-out calls and report its metrics alongside the unconstrained run.

## Supplementary Task A analysis

An independent development probe produced two complete judgments in single Markdown JSON fences. Removing the outer fence made one judgment scorable. The other contained evidence that did not match the candidate text. Preserve the strict scores on all raw outputs and add a separate score after stripping only one enclosing JSON fence. Keep the original checklist completeness and exact-quotation rules in both analyses.

## Task A judge runtime check

Generate all 120 development explanations. Check the local judge on the two recorded development inputs and on a fixed 12-judgment prefix of the seed-6000 development queue. For batched inference, encode each image and prompt with native single-request preprocessing and batch the greedy language-model decode. Freeze the selected runtime, batch size, and scoring rules before held-out generation. Use the full 240-explanation held-out matrix and retain one judgment per returned explanation. Report strict and fence-normalized availability separately. The blinded development and held-out forms support independent group review.
