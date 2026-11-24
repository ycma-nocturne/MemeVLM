import path from 'node:path';

const models = ['4B', '8B'];
const dimensions = ['V', 'I', 'K', 'R'];
const hintContrasts = ['K', 'R', 'R_minus_K'];
const developmentProvenance = 'Archived AI-generated development ratings informed the scoring format.';
const replayRule = 'Replay lifts the two-citation cap so a covered item can draw on several parts of an answer. It still requires a complete checklist and distinct, valid verbatim citations, with at least one per covered item.';
const testAuditResults = 'On the fixed 20-case test audit, 11/20 were scorable under frozen scoring and 17/20 on replay. All 11 shared scores were identical.';
const dimensionLabels = { V: 'visual detail', I: 'identity', K: 'knowledge', R: 'reasoning',
  R_minus_K: 'reasoning gain minus knowledge gain' };
const imageHeadlines = [
  ['R', 'Reasoning · image minus OCR'],
  ['V', 'Visual detail · image minus OCR'],
  ['I', 'Identity · image minus OCR'],
  ['K', 'Knowledge · image minus OCR'],
];
const hintHeadlines = [
  ['R_minus_K', 'Reasoning gain minus knowledge gain'],
  ['K', 'Knowledge coverage after adding a factual hint'],
  ['R', 'Reasoning coverage after adding a factual hint'],
];

function verifiedMetric(metric, label, expected, extent = 1) {
  const { denominator: n, n_expected: planned, estimate, ci95: interval } = metric ?? {};
  if (!Number.isInteger(n) || !Number.isInteger(planned) || planned !== expected || n < 0 || n > planned
      || metric.n_unavailable !== planned - n || metric.n_clusters !== n) {
    throw new Error('Invalid matched family counts for ' + label);
  }
  const inRange = value => Number.isFinite(value) && Math.abs(value) <= extent + 1e-12;
  if (n === 0 ? estimate !== null || interval !== null : !inRange(estimate)) {
    throw new Error('Invalid coverage estimate for ' + label);
  }
  if (interval !== null && (n < 2 || !Array.isArray(interval) || interval.length !== 2
      || !interval.every(inRange) || interval[0] > interval[1])) {
    throw new Error('Invalid paired family interval for ' + label);
  }
  return { n, planned, estimate, interval };
}

export function coverageResults(analysis) {
  if (!analysis) return null;
  if (analysis.bootstrap_unit !== 'family_id') throw new Error('Coverage intervals must resample families.');
  const test = analysis.splits?.test?.explanation;
  if (!test) throw new Error('Coverage appendix requires held-out test analysis.');
  const counts = test.counts;
  const planned = counts?.planned_explanations;
  const candidateOutcomes = counts?.judge_run_status ?? {};
  const scoreOutcomes = counts?.score_status ?? {};
  const returned = candidateOutcomes.ok ?? 0;
  const scorable = scoreOutcomes.ok ?? 0;
  const countSum = (outcomes, allowed) => Object.entries(outcomes).reduce((sum, [key, n]) => {
    if (!allowed.includes(key) || !Number.isInteger(n) || n < 0) throw new Error('Invalid held-out outcome counts.');
    return sum + n;
  }, 0);
  if (planned !== 240 || countSum(candidateOutcomes, ['ok', 'failed', 'interrupted']) !== planned
      || countSum(scoreOutcomes, ['ok', 'unavailable', 'not_scored']) !== planned || scorable > returned) {
    throw new Error('Coverage appendix requires terminal held-out judge counts.');
  }
  const found = models.map(label => {
    const size = 'Qwen/Qwen3-VL-' + label + '-Instruct';
    const result = test.models?.[size];
    const image = result?.ocr_subgroups?.available?.condition_differences?.image_only_minus_ocr_only;
    const hint = result?.hint_gains;
    const conditionCounts = result?.counts_by_condition ?? {};
    const cells = Object.values(conditionCounts);
    const scoring = cells.length ? {
      planned: cells.reduce((sum, cell) => sum + cell.planned_explanations, 0),
      passed: cells.reduce((sum, cell) => sum + (cell.score_status.ok ?? 0), 0),
      image: conditionCounts.image_only?.score_status.ok ?? 0,
      hinted: conditionCounts.hinted?.score_status.ok ?? 0,
    } : null;
    const checks = (keys, block, name, expected) => Object.fromEntries(keys.map(key =>
      [key, verifiedMetric(block?.[key], label + ' ' + name + ' ' + key, expected, key === 'R_minus_K' ? 2 : 1)]));
    return { label, scoring, image: checks(dimensions, image, 'image minus OCR', 35),
      hint: checks(hintContrasts, hint, 'factual hint', 40) };
  });
  const byModel = found.every(({ scoring }) => scoring) ? found.map(({ label, scoring }) => {
    if (scoring.planned !== 120 || !Number.isInteger(scoring.passed) || scoring.passed < 0
        || scoring.passed > scoring.planned) throw new Error('Invalid per-model held-out scoring counts.');
    return `${label} ${scoring.passed}/${scoring.planned}`;
  }).join(' and ') : '';
  if (byModel && found.reduce((sum, model) => sum + model.scoring.passed, 0) !== scorable) {
    throw new Error('Per-model and total held-out scoring counts differ.');
  }
  const firstWithPairs = (block, candidates) => candidates.find(([key]) =>
    found.some(model => model[block][key].n > 0));
  const imageHeadline = firstWithPairs('image', imageHeadlines);
  const hintHeadline = firstWithPairs('hint', hintHeadlines);
  return { found, planned, returned, scorable, byModel, imageHeadline, hintHeadline };
}

function points(rate) {
  const number = Number((100 * rate).toFixed(1));
  return (number === 0 ? '' : number < 0 ? '−' : '+') + Math.abs(number).toFixed(1);
}

function estimateLabel({ n, planned, estimate, interval }, includeInterval = false) {
  const families = n + '/' + planned;
  if (n === 0) return families + ' matched';
  return points(estimate) + ' pp (' + families + ')'
    + (includeInterval && interval ? ', 95% CI ' + interval.map(points).join(' to ') + ' pp' : '');
}

function matchedCounts(found, block, key) {
  return found.map(model => {
    const { n, planned } = model[block][key];
    return `${model.label} ${n}/${planned}`;
  }).join(' and ');
}

function unavailableHintSummary(result) {
  const four = result.found[0];
  const { n, planned } = four.hint.R_minus_K;
  if (n !== 0 || !four.scoring) return '';
  return `For 4B, ${four.scoring.image} image-only and ${four.scoring.hinted} hinted judgments passed on separate families, yielding ${n}/${planned} matched pairs.`;
}

export function coverageAnswer(original, replay = null) {
  const result = replay ?? original;
  const { found, planned, returned, scorable, byModel, imageHeadline, hintHeadline } = result;
  const imageKey = (imageHeadline ?? imageHeadlines[0])[0];
  const hintKey = (hintHeadline ?? hintHeadlines[0])[0];
  const comparison = (block, key) => found.map(model =>
    `${model.label} ${estimateLabel(model[block][key], true)}`).join(' and ');
  const hintComponents = found.filter(model => model.hint.K.n && model.hint.R.n).map(model =>
    `${model.label} knowledge ${estimateLabel(model.hint.K)} and reasoning ${estimateLabel(model.hint.R)}`).join('. ');
  const replayHint = found.filter(model => model.hint.K.n && model.hint.R.n).map(model =>
    `${model.label} knowledge ${points(model.hint.K.estimate)} pp and reasoning ${points(model.hint.R.estimate)} pp`).join(', ');
  const scoring = replay
    ? `Original scoring accepted ${original.scorable}/${original.planned}. Post-hoc CPU replay accepted ${scorable}/${planned}${byModel ? ` (${byModel})` : ''} from the same replies, with no new GPU calls. ${replayRule}`
    : `The judge returned ${returned}/${planned} judgments. ${scorable}/${planned} passed scoring checks${byModel ? ` (${byModel})` : ''}.`;
  const answer = [
    scoring,
    unavailableHintSummary(original),
    `For image versus OCR, the ${dimensionLabels[imageKey]} estimates were ${comparison('image', imageKey)}.`,
    `For the factual hint, ${dimensionLabels[hintKey]} yielded ${comparison('hint', hintKey)}.`,
    replay && replayHint && `The separate hint gains were ${replayHint}.`,
    !replay && hintComponents && `The hint gains by dimension were ${hintComponents}.`,
    developmentProvenance,
    imageHeadline || hintHeadline
      ? 'These exploratory estimates describe judged coverage on the scorable pairs.'
      : '',
  ].filter(Boolean).join(' ');
  return replay ? `${answer}\n\n${testAuditResults}` : answer;
}

export async function addCoverageAppendix(ui, { result, resultFigure, replayOf = null }) {
  if (!result) return null;
  const { found, planned, returned, scorable, byModel, imageHeadline, hintHeadline } = result;
  const hasPairs = Boolean(imageHeadline || hintHeadline);
  const { page, picture, blue, body, rect, rule } = ui;
  const slide = replayOf
    ? page('How sensitive is coverage to the citation limit?', 'MEMEBENCH · POST-HOC REPLAY',
      `${scorable}/${planned} passed on CPU${byModel ? ` (${byModel})` : ''}. Original scoring passed ${replayOf.scorable}/${replayOf.planned}. Same replies and zero new GPU calls.`)
    : page('How do images and hints change explanations?', 'MEMEBENCH · ORIGINAL SCORING',
      `Exploratory AI-assisted scoring. ${byModel ? `${byModel} passed. ` : ''}OCR has 35 eligible families. The hint comparison has 40 planned families.`);
  const context = replayOf
    ? 'Complete checklist. Each covered item needs at least one distinct, valid span from the original answer. The two-citation cap is lifted.'
    : 'V  Visual detail   ·   I  Identity   ·   K  Knowledge   ·   R  Reasoning';
  body(slide, 'Scoring context', context,
    56, 159, 1167, 23, 12);
  if (hasPairs) {
    if (!resultFigure) throw new Error('Add the plot rebuilt from the same coverage-test aggregate.');
    await picture(slide, path.basename(resultFigure),
      'Paired changes in assessed checklist coverage. The left panel compares image and OCR. The right panel compares image with and without a factual hint.',
      56, 173, 1167, 391, path.dirname(resultFigure));
  } else {
    rect(slide, 'Judge outcome field', 56, 215, 1167, 305, '#E8EFF1');
    rule(slide, 56, 215, 1167, '#7092AB', 2);
    blue(slide, 'Held-out scoring', `${scorable}/${planned} passed scoring checks`, 87, 276, 1110, 62, 30);
    body(slide, 'Judge responses', `${returned}/${planned} judgments returned.`, 87, 367, 1110, 34, 17);
    body(slide, 'Coverage availability', 'Both comparisons yielded zero scorable family pairs.',
      87, 409, 1110, 38, 17);
  }

  for (const [index, [block, headline]] of [['image', imageHeadline], ['hint', hintHeadline]].entries()) {
    const x = 56 + index * 594;
    rect(slide, 'Coverage finding band ' + (index + 1), x, 567, 572, 73, index === 0 ? '#E8EFF1' : '#F1EADB');
    rule(slide, x, 567, 572, index === 0 ? '#7092AB' : '#B8A065', 2);
    const title = headline?.[1] ?? (block === 'image' ? 'Image minus OCR' : 'Factual hint added');
    blue(slide, 'Coverage finding ' + (index + 1), title, x + 14, 578, 544, 26, 15);
    const metricKey = (block === 'image' ? imageHeadlines : hintHeadlines)[0][0];
    const detail = headline
      ? found.map(model => model.label + ' ' + estimateLabel(model[block][headline[0]])).join('   ·   ')
      : matchedCounts(found, block, metricKey) + ' matched';
    body(slide, 'Matched case estimates ' + (index + 1), detail, x + 14, 608, 544, 25, 13);
  }
  if (replayOf) {
    body(slide, 'Fixed test audit',
      '20-case technical check · 11/20 scorable frozen · 17/20 replay · 11 shared scores identical',
      58, 691, 1164, 21, 12);
  }

  slide.speakerNotes.text = [
    replayOf
      ? `Original scoring accepted ${replayOf.scorable}/${replayOf.planned}. Post-hoc CPU replay accepted ${scorable}/${planned} of the same judge replies, with no new model calls. ${replayRule} ${developmentProvenance}`
      : `The held-out judge returned ${returned}/${planned} judgments. ${scorable}/${planned} passed scoring checks. ${developmentProvenance}`,
    hasPairs
      ? 'The frozen 4B and 8B models explain the same images under different inputs. The left panel subtracts source OCR coverage from image coverage. The right panel subtracts image-only coverage from coverage with an added factual hint. Each plotted point averages the within-family differences in assessed checklist coverage.'
      : 'The two comparisons had no scorable family pairs. The counts are the observed outcome of the held-out run.',
    ...found.map(model => {
      const imageDetail = (replayOf ? ['R'] : dimensions).map(key => key + ' ' + estimateLabel(model.image[key], true)).join(' · ');
      const hintDetail = hintContrasts.map((key, index) =>
        ['Knowledge gain', 'Reasoning gain', 'Reasoning gain minus knowledge gain'][index]
          + ' ' + estimateLabel(model.hint[key], true)).join(' · ');
      return model.label + ' image minus OCR: ' + imageDetail + '. ' + model.label + ' factual hint: ' + hintDetail + '.';
    }),
    hasPairs
      ? 'The reasoning-minus-knowledge contrast uses families scorable on both dimensions under both inputs. Interpret each exploratory estimate using its matched family count and interval.'
      : 'A paired estimate requires a usable score under both inputs for the same family.',
    !replayOf && unavailableHintSummary(result),
    replayOf && 'This sensitivity analysis measures dependence on the citation limit. Semantic relevance requires separate verification.',
    replayOf && `${testAuditResults}`,
    'Wang et al. (2026), MemeBench. https://arxiv.org/abs/2607.27798',
  ].filter(Boolean).join('\n\n');
  return slide;
}
