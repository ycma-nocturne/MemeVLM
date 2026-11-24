const count = pair => pair.map(value => value.toLocaleString('en-US')).join('/');
const pointPair = pair => pair.map(value => '+' + value.toFixed(1)).join(' / ');

export function humanAuditAnswer({scope, humanAgreement: human, judgeComparison: judge}) {
  return [
    `R1 and R2 each scored ${scope.explanations} explanations post hoc, including ${scope.heldOut} held out. They agreed on ${count(human.heldOutItems)} held-out checklist items.`,
    `On the held-out set, frozen scoring produced ${count(judge.frozen.valid)} usable replies and replay produced ${count(judge.replay.valid)}. Their scores were identical on all ${judge.shared} shared explanations.`,
    `At the item level, replay agreed with each rater on ${count(judge.replay.r1)} and ${count(judge.replay.r2)} items respectively. Frozen scoring agreed on ${count(judge.frozen.r1)} with each rater. The procedures cover different sets of items.`,
    `The human reviewers’ blinded display masked spans in ${judge.maskedAnswers} of the ${scope.heldOut} held-out explanations. The automated judge saw the full answers. Archived AI-generated development ratings informed the frozen scoring format.`,
  ].join(' ');
}

export function humanEffectsAnswer({rq1, rq2}) {
  return [
    `On the ${rq1.familiesPerModel} families with transcripts per model, both human raters found higher image-based coverage on every dimension. All paired 95% intervals were positive.`,
    'Estimated gains in knowledge and reasoning are positive on the full human-rated set. Every paired interval for reasoning gain minus knowledge gain spans zero.',
    `Excluding the ${rq2.flaggedFamilies} families flagged by either reviewer for directly stating the relation attenuates the gains. The knowledge-over-reasoning pattern under citation replay describes that scorer and its available pairs.`,
  ].join(' ');
}

export function addHumanAuditAppendix(ui, audit, original, replay) {
  const {page, blue, serif, body, rule, rect, table} = ui;
  const {scope, humanAgreement: human, judgeComparison: judge, rq1, rq2} = audit;
  if (scope.dev + scope.heldOut !== scope.explanations || rq2.allFamiliesPerModel - rq2.flaggedFamilies !== rq2.filteredFamiliesPerModel
      || judge.frozen.valid[0] !== original?.scorable || judge.replay.valid[0] !== replay?.scorable
      || scope.heldOut !== original.planned || scope.heldOut !== replay.planned) {
    throw new Error('Human audit inputs and sealed judge counts must match.');
  }

  const audited = page('Human audit of reference coverage', 'MEMEBENCH · POST-HOC HUMAN REVIEW',
    'Two teammates rated explanations after the freeze. Archived AI development ratings informed the original scoring format.');
  blue(audited, 'Audit scope label', 'Explanations per reviewer', 58, 177, 546, 30, 17);
  serif(audited, 'Audit scope count', String(scope.explanations), 55, 210, 548, 63, 35);
  body(audited, 'Audit split count', `${scope.dev} development · ${scope.heldOut} held-out`, 58, 277, 520, 32, 17);
  rect(audited, 'Audit column rule', 629, 175, 1, 128, '#CAD5D9');
  blue(audited, 'Human agreement label', 'R1–R2 held-out checklist agreement', 656, 177, 558, 30, 17);
  serif(audited, 'Held-out human agreement', count(human.heldOutItems), 653, 210, 557, 63, 35);
  body(audited, 'All-split human agreement', `${count(human.allItems)} item decisions agree across both splits`, 656, 277, 555, 32, 16);

  table(audited, [
    ['Held-out judge procedure', 'Scorable explanations', 'Item agreement with R1', 'Item agreement with R2'],
    ['Frozen', count(judge.frozen.valid), count(judge.frozen.r1), count(judge.frozen.r2)],
    ['Citation replay', count(judge.replay.valid), count(judge.replay.r1), count(judge.replay.r2)],
  ], [300, 277, 293, 294], 58, 328, 54);
  rect(audited, 'Shared agreement band', 58, 505, 1164, 59, '#EAF0F2');
  blue(audited, 'Same score comparison', `All ${judge.shared} shared explanation scores are identical under both procedures.`, 72, 515, 1136, 24, 15);
  body(audited, 'Comparison denominator', 'Replay increases scorable coverage. The two procedures score different sets of items.', 72, 539, 1136, 20, 13);
  body(audited, 'Raw differences', `Across splits, ${human.explanationDecisions} explanation-rating decisions differ across ${human.explanationsAffected} answers and ${human.materialDecisions} material decisions differ across ${human.materialCases} cases.`,
    58, 583, 1164, 28, 14);
  body(audited, 'Single-image validation differences', `In the separate ${human.singleImageQuestions}-case single-image task, ${human.singleImageDecisions} decisions differ across ${human.singleImageCases} held-out target-edit images.`,
    58, 612, 1164, 25, 14);
  audited.speakerNotes.text = [
    humanAuditAnswer(audit),
    'Raw cross-split differences comprise 45 explanation-rating decisions across 17 answers and 11 material decisions across 10 cases.',
    'The separate Task B disagreements are one cue judgment and two relation judgments on two held-out target-edit images. These do not count as explanation differences.',
    'The automated procedures score the same saved replies. This is a retrospective check, and no GPU inference was repeated.',
    'A separate issue audit suggested 88 field corrections across 17 answers, including 15 grounding and 73 checklist corrections. The agreement and paired effects presented here use the submitted raw forms.',
  ].join('\n\n');

  const effects = page('Human ratings support the image–OCR gap', 'MEMEBENCH · HUMAN EFFECT ESTIMATES',
    'Values show R1 / R2 in percentage points. Interval statements use paired 95% family-bootstrap intervals.');
  blue(effects, 'Image OCR reasoning contrast', `RQ1 · Reasoning coverage, image minus OCR (${rq1.familiesPerModel} families per model)`, 58, 167, 1164, 32, 18);
  rect(effects, 'Image OCR effects band', 58, 207, 1164, 63, '#EAF0F2');
  blue(effects, '4B reasoning effect', `4B   ${pointPair(rq1.reasoning['4B'])} pp`, 76, 221, 524, 38, 19);
  blue(effects, '8B reasoning effect', `8B   ${pointPair(rq1.reasoning['8B'])} pp`, 659, 221, 530, 38, 19);
  body(effects, 'All dimensions', 'Both raters find positive differences and intervals for visual detail, identity, knowledge and reasoning.', 58, 282, 1164, 28, 14);

  blue(effects, 'Hint gain contrast', 'RQ2 · Coverage gain after adding a factual hint', 58, 320, 1164, 30, 18);
  table(effects, [
    ['Held-out families per model', 'Model', 'Knowledge gain (pp)', 'Reasoning gain (pp)'],
    [`All ${rq2.allFamiliesPerModel}`, '4B', pointPair(rq2.all['4B'].K), pointPair(rq2.all['4B'].R)],
    [`All ${rq2.allFamiliesPerModel}`, '8B', pointPair(rq2.all['8B'].K), pointPair(rq2.all['8B'].R)],
    [`${rq2.filteredFamiliesPerModel} after excluding ${rq2.flaggedFamilies} flagged`, '4B', pointPair(rq2.filtered['4B'].K), pointPair(rq2.filtered['4B'].R)],
    [`${rq2.filteredFamiliesPerModel} after excluding ${rq2.flaggedFamilies} flagged`, '8B', pointPair(rq2.filtered['8B'].K), pointPair(rq2.filtered['8B'].R)],
  ], [382, 132, 325, 325], 58, 361, 41);
  rule(effects, 58, 583, 1164, '#B8A065', 2);
  blue(effects, 'Hint result', 'Gains are smaller after excluding flagged families. All paired reasoning-minus-knowledge intervals span zero.',
    58, 599, 1164, 37, 15);
  effects.speakerNotes.text = [
    humanEffectsAnswer(audit),
    'The four RQ1 dimensions have positive intervals for each model and rater. The smallest estimated point difference is +53.3 pp.',
    'On the filtered RQ2 cohort, 4B reasoning intervals remain positive and 4B knowledge intervals span zero. For 8B both components span zero. This sensitivity analysis still uses the same explanations.',
  ].join('\n\n');
}
