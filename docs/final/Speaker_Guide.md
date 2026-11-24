# MemeVLM

| Speaker | Time | Slides |
|---|---|---|
| Speaker 1 | 0:00–2:28 | 1–3, including the handoff |
| Speaker 2 | 2:28–5:23 | 4–7, including the handoff |
| Speaker 3 | 5:23–8:45 | 8–11 |

The main talk takes 8:45 including two four-second handoffs. Use the appendix for the separate 3:00 Q&A.

## 1 · Speaker 1 · 0:00–0:25

Point to the tall and short blocks in the scale-reversal schematic.

Imagine four kilobytes as a towering figure and eight gigabytes as a much smaller one. Reading the numbers is easy. Explaining why the size relationship is reversed takes more. We tested whether a vision-language model could follow the evidence behind a joke as that evidence changed.

## 2 · Speaker 1 · 0:25–1:15

Pause after each question, then point to the caption and expression in the schematics.

These schematics illustrate two benchmark cases. On the left, why would eight gigabytes fear four kilobytes? The joke contrasts the computing resources associated with a Moon mission and the appetite of a modern web browser. On the right, a teacher announces a fun project, while the cat looks thoroughly unimpressed. Naming a browser or recognizing a cat misses the relationship that makes the joke. We built a controlled task to test that difference.

## 3 · Speaker 1 · 1:15–2:24

Compare the first and middle panels, then the corner marks in the outer panels. Handoff at 2:24.

The instruction here is identical in all three versions: put the files in the cloud. On the left, the intern takes the phrase literally and puts paper files in the sky. The target edit replaces that outcome with remote storage. The sham changes only a corner mark, so the literal joke survives. Each version has a reference cue and relation. We created 240 such families across eight mechanisms and assigned each whole family to one split. That leaves 32 families for the held-out test. Speaker 2 will explain how we scored them.

## 4 · Speaker 2 · 2:28–3:10

Trace the original, target edit, and sham with their reference readings, then point to the MemeQA branch.

The controlled test requires six correct decisions per family: one cue judgment and one relation choice on each of its three variants. On natural memes we use the existing MemeQA questions and choices, and pool exact accuracy across eligible questions. The same base and six adapters take both tests. Our paired bootstrap resamples individual synthetic families or whole natural images, retaining every question on each sampled image.

## 5 · Speaker 2 · 3:10–3:47

Follow the watch image and question to the recorded model choice, then point to the amber training path.

The frozen vision encoder reads the image. Its visual tokens join the prompt in the language decoder. We add rank-eight LoRA adapters to the attention and feed-forward projections in each of 36 language blocks. Only the adapters update during training. At inference we use fixed prompts and greedy decoding across all seven model arms.

## 6 · Speaker 2 · 3:47–4:27

Use the two training thumbnails to distinguish a supervision row from an image, then follow the shared budget.

Both thumbnails are actual training images. Airplane Mode supplies two supervision rows. Morning Run appears eight times per replay epoch. Both recipes process seven hundred seventy rows per epoch, with three epochs and three paired seeds. Each run selects its checkpoint using the same validation split. The language mix and the diversity of examples vary together.

## 7 · Speaker 2 · 4:27–5:19

Read the left and right axes separately. Point to the first measured checkpoint. Handoff at 5:19.

All six runs completed the full training schedule. On the left, training loss keeps falling on each recipe's own examples. On the right, shared-set validation loss is lowest at the first measured checkpoint, step ninety-seven, for every run. It rises at the two later evaluations. We therefore use each run's step-ninety-seven adapter for the held-out tests. This is the best checkpoint among the three we measured. Speaker 3 will show what those adapters achieve.

## 8 · Speaker 3 · 5:23–6:10

Name the strict family metric, then compare mixed-corpus LoRA with the matched replay.

On 32 held-out families, mixed-corpus LoRA passes 92.7 percent on average. replay passes 57.3 percent and the frozen base passes 37.5 percent. The mixed-corpus advantage over replay is 35.42 percentage points, with a paired family interval from 21.88 to 50 points. Every arm recognized the specified cue on all 96 variants. It was their choice of relation that separated the models.

## 9 · Speaker 3 · 6:10–7:00

State the denominator, compare the two bars for one seed, and point to both intervals.

On 816 questions across 134 natural memes, every matched seed fixes some replay errors and makes some new ones. At seed 42, 46 answers improve and 44 become wrong. Across seeds, mixed-corpus LoRA averages 54.82 percent, a gain of 0.90 points over replay. Its gain over the frozen base is 1.27 points. Both paired image-level intervals span zero, so the direction and size of transfer remain uncertain.

## 10 · Speaker 3 · 7:00–7:58

Point first to the words on the edited watch, then to each interpretation.

Here is an actual model disagreement. On the original watch, the displayed activity is three hours of clapping. The target edit changes the wording to walking and replaces the pictogram. Both models correctly answer that the edited watch no longer says clapping. The frozen base still chooses the relation that the watch counted clapping as exercise. The mixed-corpus adapter, seed 42, chooses the walking interpretation. Both were correct on the original and the sham. This is why we score the cue and the relation separately.

## 11 · Speaker 3 · 7:58–8:45

Compare the two estimates using the same replay control. Finish at 8:45.

Against matched replay, mixed-corpus training improved authored edits. The MemeQA transfer estimate remained uncertain. In the frozen-model controls, both frozen models detected every cue and passed five of ten triples. Across both edit studies, choosing the meaning remained the harder step. Two teammates later rated all 240 held-out explanations. Human ratings support the image–OCR result. Hint ordering varies. Thank you.

## Questions · 3:00

**Which pieces did you build yourselves?** · Speaker 1

We authored the 240 controlled scenes and their edits, constructed matched mixed-corpus and replay recipes, and implemented exact cue and relation scoring for each family. We evaluated independently authored memes using the existing MemeQA questions and choices.

**Why include a sham edit?** · Speaker 1

A model can change its answer whenever it sees an edit. The sham checks whether it keeps the same interpretation when the relevant cue stays the same. The complete-family score requires correct cue and relation answers on the original, the target edit, and the sham.

**Were 291 steps enough, and why use step 97?** · Speaker 2

Every run completed 291 updates. We measured validation loss at steps 97, 194, and 291. Step 97 had the lowest observed loss in all six runs. The shared validation mix had 160 English and 33 Chinese rows, so this selection is tied to that mix.

**What differs between the two training recipes?** · Speaker 2

The two recipes matched exposures and optimizer steps. They differed in language, unique examples, task diversity and question formats. The measured contrast therefore captures the combined effect of the full recipe.

**Why does MemeQA show a smaller gain?** · Speaker 3

Our authored families share a renderer and task rubric with the authored training set. MemeQA uses independently authored memes and its native multiple-choice questions. The current interval includes zero, so this test leaves the size and direction of any transfer gain uncertain.

**What did the earlier 2B edit pilot show?** · Speaker 3

The frozen 2B model and its LoRA variant each got one of ten target edits and zero complete triples in the structured pilot. That motivated checking the relation as well as the cue. The 4B controlled study uses a different set of examples and estimates the effect of two matched training recipes.

**How do the frozen and replay scores compare with human ratings?** · Speaker 3

R1 and R2 each scored 360 explanations post hoc, including 240 held out. They agreed on 2,929/2,952 held-out checklist items. On the held-out set, frozen scoring produced 91/240 usable replies and replay produced 188/240. Their scores were identical on all 91 shared explanations. At the item level, replay agreed with each rater on 1,441/2,297 and 1,440/2,297 items respectively. Frozen scoring agreed on 560/1,110 with each rater. The procedures cover different sets of items. The human reviewers’ blinded display masked spans in 80 of the 240 held-out explanations. The automated judge saw the full answers. Archived AI-generated development ratings informed the frozen scoring format.

**What do the human-rated image and hint comparisons show?** · Speaker 3

On the 35 families with transcripts per model, both human raters found higher image-based coverage on every dimension. All paired 95% intervals were positive. Estimated gains in knowledge and reasoning are positive on the full human-rated set. Every paired interval for reasoning gain minus knowledge gain spans zero. Excluding the 13 families flagged by either reviewer for directly stating the relation attenuates the gains. The knowledge-over-reasoning pattern under citation replay describes that scorer and its available pairs.
