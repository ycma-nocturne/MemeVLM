import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Presentation, PresentationFile, SpreadsheetFile, Workbook } from '@oai/artifact-tool';
import JSZip from 'jszip';
import { addChineseAppendix } from './chinese-appendix.mjs';
import { addCoverageAppendix, coverageAnswer, coverageResults } from './coverage-appendix.mjs';
import { addHumanAuditAppendix, humanAuditAnswer, humanEffectsAnswer } from './human-audit-appendix.mjs';
import { addMemorySchematic, addSarcasmSchematic } from './public-examples.mjs';

const source = path.dirname(fileURLToPath(import.meta.url));
const assets = path.join(source, 'assets');
const figures = path.join(source, '..', 'figures');
const narration = JSON.parse(await fs.readFile(path.join(source, 'speaker-plan.json'), 'utf8'));
const [resultPath, pilotPath, outputPath, preview] = process.argv.slice(2);
if (!resultPath || !pilotPath || !outputPath) {
  throw new Error('Usage: node build.mjs <full-results.json|--draft> <pilot evidence> <new .pptx> [preview directory]');
}
const isDraft = resultPath === '--draft';
const humanAudit = isDraft ? null : JSON.parse(await fs.readFile(path.join(source, 'human-audit-public.json'), 'utf8'));
const delivery = await fs.realpath(pilotPath);
const results = isDraft ? null : JSON.parse(await fs.readFile(resultPath, 'utf8'));
const coverageSource = path.join(path.dirname(path.resolve(resultPath)), 'reference-study');
const chineseResultsPath = path.join(coverageSource, 'rq3.json');
const chineseResults = isDraft ? null : await fs.readFile(chineseResultsPath, 'utf8').then(JSON.parse, error => {
  if (error.code === 'ENOENT') return null;
  throw error;
});
async function optionalCoverage(name) {
  if (isDraft) return null;
  return fs.readFile(path.join(coverageSource, name), 'utf8').then(JSON.parse, error => {
    if (error.code === 'ENOENT') return null;
    throw error;
  });
}
const [coverageTest, coverageReplayTest] = await Promise.all([
  optionalCoverage('coverage-test.json'), optionalCoverage('coverage-span-replay.json'),
]);
const coverage = coverageResults(coverageTest);
const coverageReplay = coverageResults(coverageReplayTest);
if (coverageReplay && (!coverage || coverage.planned !== coverageReplay.planned
    || coverage.returned !== coverageReplay.returned || coverageReplay.scorable < coverage.scorable)) {
  throw new Error('Expected compatible original and replay aggregate counts.');
}
const chineseTest = chineseResults ? [4, 8].map(size => {
  const result = chineseResults.splits.test[`Qwen/Qwen3-VL-${size}B-Instruct`];
  const strict = result.triples.strict_pass;
  if (strict.n_expected !== 10 || strict.denominator < 1) {
    throw new Error('Chinese test appendix requires verified scores on the 10-family design.');
  }
  return {strict, edited: result.variants.edited.joint_correct, sham: result.triples.sham_preservation,
    permutation: result.permutation.semantic_relation_agreement};
}) : null;
const metricCount = metric => `${metric.numerator}/${metric.denominator}`;
const synthetic = results ? requireCohort(results.synthetic_strict_triples, 32, 32, 'chinese') : null;
const natural = results ? requireCohort(results.native_memeqa?.unseen_pilot_images, 134, 816, 'chinese', 'base') : null;
const destination = path.resolve(outputPath);
if (path.extname(destination) !== '.pptx') throw new Error('Choose a .pptx output path.');
const outputDir = path.dirname(destination);
const guidePath = path.join(outputDir, 'Speaker_Guide.md');
for (const dir of [outputDir, preview].filter(Boolean)) {
  const outsidePilot = candidate => {
    const relative = path.relative(delivery, candidate);
    return relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative);
  };
  if (!outsidePilot(path.resolve(dir))) throw new Error('Choose an output directory outside the pilot delivery.');
  await fs.mkdir(dir, { recursive: true });
  if (!outsidePilot(await fs.realpath(dir))) throw new Error('Choose an output directory outside the pilot delivery.');
}
for (const file of [destination, guidePath]) {
  const exists = await fs.lstat(file).then(() => true, error => {
    if (error.code !== 'ENOENT') throw error;
    return false;
  });
  if (exists) throw new Error('Choose a new output directory. File already exists: ' + file);
}
const evidence = delivery;
const pilot = await json('english-4b/evaluation/test-scores.json');
const demo = pilot.scores;

async function json(file) {
  return JSON.parse(await fs.readFile(path.join(evidence, file), 'utf8'));
}
const ppt = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const c = {
  white: '#FFFFFF', paper: '#FAF8F2', blue: '#003366', gray: '#4E6476', dark: '#173649',
  gold: '#B8A065', lightGold: '#E7CE93', ochre: '#775421', line: '#CAD5D9',
  pale: '#EAF0F2', lightBlue: '#7092AB', midnight: '#09273F',
};
const headline = 'Georgia';
const gradient = (from, to, angle = 0) => ({
  type: 'gradient', gradientKind: 'linear', angleDeg: angle,
  stops: [{ offset: 0, color: from }, { offset: 100000, color: to }],
});
const hatch = (background, ink) => ({
  type: 'solid', color: background, pattern: { type: 'downwardDiagonal', color: ink },
});
const labels = { base: '4B base', adapter: 'Mixed-corpus seed 42' };
const refs = {
  qwen: 'Qwen3-VL-4B-Instruct · https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17',
  memeqa: 'Nguyen et al., ACL 2025 · https://aclanthology.org/2025.acl-long.927/',
  memebench: 'Wang et al., MemeBench · Table 3 uses paired repair and damage · https://arxiv.org/abs/2607.27798',
  results: 'Seed-42 mixed-corpus evaluation · project-rendered Fitness Watch example',
  expanded: 'Expanded 4B evaluation · released aggregate results',
  mer: 'Nie et al., MER-Bench, Figure 6 · https://aclanthology.org/2026.emnlp-main.940/',
};
const names = ['Speaker 1', 'Speaker 2', 'Speaker 3'];
const plan = [];
const schedule = {
  cover: [0, '0:00', '0:25'],
  demo: [0, '0:25', '1:15'],
  families: [0, '1:15', '2:24'],
  evaluation: [1, '2:28', '3:10'],
  architecture: [1, '3:10', '3:47'],
  budget: [1, '3:47', '4:27'],
  checkpoint: [1, '4:27', '5:19'],
  synthetic: [2, '5:23', '6:10'],
  natural: [2, '6:10', '7:00'],
  example: [2, '7:00', '7:58'],
  conclusion: [2, '7:58', '8:45'],
};

function plannedNote(slide, key, sources = [], draftScript, finalScript) {
  const [speaker, start, end] = schedule[key];
  const { cue, script } = narration[key];
  note(slide, names[speaker], start, end, cue, isDraft && draftScript ? draftScript : finalScript ?? script, sources);
}
function text(slide, name, value, x, y, w, h, style = {}) {
  const { size = 17, color = c.dark, bold = false, align = 'left', font = 'Arial', italic = false } = style;
  const item = slide.shapes.add({ name, geometry: 'textbox',
    position: { left: x, top: y, width: w, height: h },
    fill: 'none', line: { fill: 'none', width: 0 } });
  item.text = value;
  item.text.style = { typeface: font, fontSize: size + 'pt', color, bold, italic,
    alignment: align, verticalAlignment: 'top', wrap: 'square', autoFit: 'none', insets: 0 };
  return item;
}
function rect(slide, name, x, y, w, h, fill) {
  slide.shapes.add({ name, geometry: 'rect', position: { left: x, top: y, width: w, height: h },
    fill, line: { fill: 'none', width: 0 } });
}
function rule(slide, x, y, w, fill = c.line, h = 2) {
  rect(slide, 'Divider', x, y, w, h, fill);
}
function page(title, category, caption, {theme = 'method'} = {}) {
  const slide = ppt.slides.add();
  const dark = theme === 'finding';
  const narrative = theme === 'case';
  slide.background.fill = dark || narrative ? '#F9F6ED' : c.white;
  rect(slide, 'Title field', 0, 0, 1280, 145, dark ? gradient('#09273F', '#174A67')
    : narrative ? gradient('#F0E6CA', '#FAF6E9') : gradient('#DFEAEE', '#F4F6F5'));
  rect(slide, 'Spine', 0, 0, 15, 720, dark ? c.gold : (narrative ? '#D7C28C' : '#BACCD1'));
  rect(slide, 'Folio rule', 1180, 35, 42, 1, dark ? c.lightGold : c.ochre);
  text(slide, 'Folio', String(ppt.slides.items.length).padStart(2,'0'), 1180, 14, 42, 20,
    {size: 12, color: dark ? c.lightGold : c.ochre, align:'right', font: headline});
  text(slide, 'Research section', category, 58, 24, 1100, 22,
    {size: 11, bold: true, color: dark ? c.lightGold : c.ochre});
  text(slide, 'Title', title, 55, 56, 1136, 81,
    {size: 32, color: dark ? c.white : c.blue, font: headline});
  rule(slide, 15, 145, 1265, dark ? c.gold : '#C2CFCC', 1);
  if (caption) {
    rect(slide, 'Caption band', 15, 651, 1265, 69, dark ? '#F2EAD6' : narrative ? '#F2ECDB' : '#F3F6F4');
    rule(slide, 58, 658, 1164, '#C7D0CF', 1);
    text(slide, 'Research note', caption, 58, 669, 1164, 35,
      {size: 12, color:c.gray});
  }
  return slide;
}
function serif(slide, name, value, x, y, w, h, size=24, color=c.blue) {
  return text(slide, name, value, x, y, w, h, {size, color, font:headline});
}
function body(slide, name, value, x, y, w, h, size = 18) {
  return text(slide, name, value, x, y, w, h, { size });
}
function blue(slide, name, value, x, y, w, h, size = 20) {
  return text(slide, name, value, x, y, w, h, { size, color: c.blue, bold: true });
}
function accent(slide, name, value, x, y, w, h) {
  return text(slide, name, value, x, y, w, h, { size: 13, color: c.ochre, bold: true });
}
async function picture(slide, file, alt, x, y, width, height, directory = assets) {
  const blob = await fs.readFile(path.join(directory, file));
  const contentType = 'image/png';
  slide.images.add({ blob, contentType, alt, fit: 'contain',
    position: { left: x, top: y, width, height } });
}
async function methodFigure(slide, file, alt) {
  rule(slide, 58, 167, 1164, c.line, 1);
  await picture(slide, file, alt, 58, 172, 1164, 423, figures);
  rule(slide, 58, 600, 1164, c.line, 1);
}
function table(slide, values, widths, x, y, rowHeight, emphasis = []) {
  const columns = widths.length;
  const rows = values.length;
  const item = slide.tables.add({ rows, columns, left: x, top: y,
    width: widths.reduce((sum, w) => sum + w, 0), height: rowHeight * rows, columnWidths: widths, values });
  item.styleOptions = { headerRow: false, bandedRows: false };
  item.borders.assign({ style: 'solid', fill: c.line, width: 1 });
  for (let r = 0; r < rows; r += 1) {
    item.rows[r].height = rowHeight;
    for (let col = 0; col < columns; col += 1) {
      const cell = item.getCell(r, col);
      cell.fill = r === 0 || emphasis.includes(r) ? c.pale : c.white;
      cell.text.style = { typeface: 'Arial', fontSize: r === 0 ? '15pt' : '16pt',
        bold: r === 0 || col === 0 || emphasis.includes(r), color: r === 0 ? c.blue : c.dark,
        alignment: col === 0 ? 'left' : 'center', verticalAlignment: 'middle', autoFit: 'none' };
    }
  }
}
function note(slide, speaker, start, end, cue, script, sources = []) {
  const meta = { slide: ppt.slides.items.length, speaker, start, end, cue, script };
  plan.push(meta);
  slide.speakerNotes.text = [
    speaker + ' · ' + start + '–' + end,
    cue, script, ...sources,
  ].join('\n\n');
}
function requireCohort(cohort, images, questions, ...comparators) {
  if (cohort?.images !== images || cohort.questions !== questions) {
    throw new Error(`Expected ${images} images and ${questions} scoring units.`);
  }
  const same = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-9;
  for (const arm of ['base', ...['chinese', 'bilingual'].flatMap(profile =>
    [42, 43, 44].map(seed => `${profile}-${seed}`))]) {
    const row = cohort.per_arm?.[arm];
    if (!row || row.denominator !== questions || !Number.isInteger(row.correct)
        || row.correct < 0 || row.correct > questions || !same(row.accuracy, row.correct / questions)) {
      throw new Error(`Invalid score in ${images}-image cohort: ${arm}`);
    }
  }
  for (const profile of ['chinese', 'bilingual']) {
    const scores = [42, 43, 44].map(seed => cohort.per_arm[`${profile}-${seed}`].accuracy);
    const mean = scores.reduce((sum, value) => sum + value, 0) / scores.length;
    const sd = Math.sqrt(scores.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (scores.length - 1));
    const actual = cohort.by_profile?.[profile];
    if (!actual || !same(actual.mean, mean) || !same(actual.seed_sample_std, sd)) {
      throw new Error(`Seed summary differs from verified arm scores: ${profile}`);
    }
  }
  for (const comparator of comparators) {
    const contrast = cohort.paired_contrasts?.['bilingual_minus_' + comparator];
    const bounds = contrast?.paired_image_bootstrap_ci95;
    const referenceRate = comparator === 'base' ? cohort.per_arm.base.accuracy : cohort.by_profile.chinese.mean;
    if (!contrast || !same(contrast.delta, cohort.by_profile.bilingual.mean - referenceRate)
        || !Array.isArray(bounds) || bounds.length !== 2
        || bounds.some(value => !Number.isFinite(value) || Math.abs(value) > 1) || bounds[0] > bounds[1]) {
      throw new Error('Invalid paired bootstrap contrast.');
    }
  }
  return cohort;
}
function points(rate, signed = false, decimals = 1) {
  const number = Number((100 * rate).toFixed(decimals));
  const sign = number < 0 ? '−' : (signed && number > 0 ? '+' : '');
  return sign + Math.abs(number).toFixed(decimals);
}
function contrastValue(cohort, comparator) {
  const row = cohort.paired_contrasts['bilingual_minus_' + comparator];
  return points(row.delta, true, 2) + ' pp';
}
function intervalValue(cohort, comparator) {
  return cohort.paired_contrasts['bilingual_minus_' + comparator].paired_image_bootstrap_ci95
    .map(rate => points(rate, true, 2)).join(' to ') + ' pp';
}
function includesZero(cohort, comparator) {
  const [low, high] = cohort.paired_contrasts['bilingual_minus_' + comparator].paired_image_bootstrap_ci95;
  return low <= 0 && 0 <= high;
}
function profileBars(slide, cohort, metric, comparator) {
  body(slide, 'Metric', metric, 56, 172, 1168, 48, 17);
  if (!cohort) {
    blue(slide, 'Pending', 'Results pending', 56, 311, 1100, 56, 31);
    body(slide, 'Pending details', 'The comparison will show the base model and both LoRA profiles.',
      56, 383, 1168, 56, 18);
    return;
  }

  const decimals = comparator === 'base' ? 2 : 1;
  const rateLabel = rate => (100 * rate).toFixed(decimals) + '%';
  const spreadLabel = rate => '± ' + points(rate, false, decimals) + ' pp';
  const rows = [
    ['4B base', cohort.per_arm.base.accuracy, cohort.per_arm.base.correct + '/' + cohort.questions, c.gray],
    ['Replay LoRA', cohort.by_profile.chinese.mean, spreadLabel(cohort.by_profile.chinese.seed_sample_std), c.lightBlue],
    ['Mixed-corpus LoRA', cohort.by_profile.bilingual.mean, spreadLabel(cohort.by_profile.bilingual.seed_sample_std), c.blue],
  ];
  const barLeft = 236;
  const barWidth = 362;
  const scale = [0, 25, 50, 75, 100];

  blue(slide, 'Accuracy heading', 'Accuracy', 56, 232, 660, 40, 20);
  for (const tick of scale) {
    const x = barLeft + barWidth * tick / 100;
    rect(slide, 'Accuracy grid ' + tick, x, 281, 1, 213, tick === 0 ? c.lightBlue : c.line);
    text(slide, 'Accuracy tick ' + tick, tick + '%', x - 26, 504, 52, 22,
      { size: 11, color: c.gray, align: 'center' });
  }
  rect(slide, 'Accuracy axis', barLeft, 493, barWidth, 1, c.lightBlue);

  for (const [index, [label, rate, detail, fill]] of rows.entries()) {
    const y = 289 + index * 72;
    text(slide, label + ' label', label, 56, y + 6, 175, 34,
      { size: 16, color: c.blue, bold: index === 2 });
    if (rate > 0) rect(slide, label + ' bar', barLeft, y + 8, barWidth * rate, 23, fill);
    text(slide, label + ' score', rateLabel(rate), 619, y - 7, 108, 37,
      { size: 21, color: c.blue, bold: index === 2, align: 'right' });
    text(slide, label + ' detail', detail, 612, y + 34, 115, 23,
      { size: 12, color: c.gray, align: 'right' });
  }
  body(slide, 'Seed variation', 'LoRA scores show the mean ± SD across three seeds.',
    56, 552, 682, 40, 14);

  const contrast = cohort.paired_contrasts['bilingual_minus_' + comparator];
  const [lower, upper] = contrast.paired_image_bootstrap_ci95.map(rate => 100 * rate);
  const estimate = 100 * contrast.delta;
  const includes = includesZero(cohort, comparator);
  const min = Math.min(0, lower, estimate);
  const max = Math.max(0, upper, estimate);
  const span = max - min;
  const tickStep = span <= 6 ? 2 : span <= 15 ? 5 : span <= 30 ? 10 : 20;
  const first = Math.floor(min / tickStep) * tickStep;
  const last = Math.ceil((max + 1e-6) / tickStep) * tickStep;
  const axisLeft = 820;
  const axisWidth = 363;
  const axisY = 447;
  const toX = value => axisLeft + axisWidth * (value - first) / (last - first);
  const reference = comparator === 'base' ? 'frozen base' : 'replay';

  rect(slide, 'Results column divider', 758, 234, 2, 354, c.line);
  rect(slide, 'Estimate field', 782, 232, 442, 356, gradient('#E4ECEF','#F0F2E9',90));
  rect(slide, 'Estimate accent', 782, 232, 442, 3, c.gold);
  blue(slide, 'Comparison heading', 'Mixed corpus vs. ' + reference,
    807, 254, 391, 40, 18);
  serif(slide, 'Difference in percentage points', contrastValue(cohort, comparator),
    807, 294, 391, 75, 37);
  body(slide, 'Interval label', 'Paired 95% bootstrap interval',
    807, 372, 391, 28, 14);
  text(slide, 'Interval values', intervalValue(cohort, comparator), 807, 399, 391, 28,
    { size: 15, color: c.dark, bold: true });

  rect(slide, 'Difference axis', axisLeft, axisY, axisWidth, 1, c.lightBlue);
  rect(slide, 'Zero difference reference', toX(0) - 1, axisY - 13, 2, 27, c.gray);
  const intervalLeft = toX(lower);
  const intervalRight = toX(upper);
  rect(slide, 'Bootstrap confidence interval', intervalLeft, axisY - 2,
    Math.max(1, intervalRight - intervalLeft), 4, c.blue);
  rect(slide, 'Lower confidence bound', intervalLeft - 1, axisY - 5, 2, 10, c.blue);
  rect(slide, 'Upper confidence bound', intervalRight - 1, axisY - 5, 2, 10, c.blue);
  slide.shapes.add({ name: 'Paired estimate', geometry: 'ellipse',
    position: { left: toX(estimate) - 6, top: axisY - 6, width: 12, height: 12 },
    fill: c.blue, line: { fill: c.white, width: 1 } });

  for (let tick = first; tick <= last; tick += tickStep) {
    const x = toX(tick);
    rect(slide, 'Difference tick mark ' + tick, x, axisY + 6, 1, 5, c.lightBlue);
    const label = tick < 0 ? '−' + Math.abs(tick) : String(tick);
    text(slide, 'Difference tick ' + tick, label, x - 23, axisY + 17, 46, 24,
      { size: 11, color: c.gray, align: 'center' });
  }
  text(slide, 'Difference units', 'Difference (percentage points)', 807, 493, 391, 25,
    { size: 11, color: c.gray, align: 'center' });
  text(slide, 'Interpretation', includes ? 'The interval includes zero.' : 'The interval lies above zero.',
    807, 540, 391, 31, { size: 15, color: includes ? c.ochre : c.blue, bold: true });
}

// Counts paired by question ID in the frozen MemeQA evaluation ledgers.
const transitions = [
  { seed: 42, corrected: 46, errors: 44 },
  { seed: 43, corrected: 54, errors: 42 },
  { seed: 44, corrected: 47, errors: 39 },
];
const transferSeries = ['Correct → wrong', 'Wrong → correct'];
const transferChartRows = transitions.slice().reverse();

function naturalTransitions(slide, cohort) {
  const cohortLabel = 'Pooled accuracy across 816 questions on 134 images outside the pilot';
  if (!cohort) {
    profileBars(slide, cohort, cohortLabel, 'chinese');
    return;
  }
  body(slide, 'MemeQA cohort', cohortLabel, 56, 174, 1168, 35, 17);

  for (const { seed, corrected, errors } of transitions) {
    const bilingual = cohort.per_arm['bilingual-' + seed].correct;
    const replay = cohort.per_arm['chinese-' + seed].correct;
    if (bilingual - replay !== corrected - errors || corrected > cohort.questions - replay || errors > replay) {
      throw new Error('MemeQA answer transitions differ from the verified score for seed ' + seed);
    }
  }

  rule(slide, 56, 226, 720, c.line, 1);
  blue(slide, 'Transition heading', 'Answer changes by matched seed', 56, 244, 707, 32, 18);
  body(slide, 'Transition comparator', 'Replay → Mixed-corpus LoRA', 56, 279, 707, 25, 14);
  const dataLabels = color => ({ showValue: true, position: 'center', numberFormatCode: '0;0;0',
    textStyle: { fill: color, fontSize: 17, bold: true } });
  slide.charts.add('bar', {
    position: { left: 49, top: 312, width: 726, height: 254 },
    categories: transferChartRows.map(({ seed }) => 'Seed ' + seed),
    series: [
      { name: transferSeries[0], values: transferChartRows.map(({ errors }) => -errors), fill: c.gold,
        dataLabels: dataLabels(c.midnight) },
      { name: transferSeries[1], values: transferChartRows.map(({ corrected }) => corrected), fill: c.blue,
        dataLabels: dataLabels(c.white) },
    ],
    barOptions: { direction: 'bar', grouping: 'stacked', gapWidth: 55, overlap: 100 },
    hasLegend: false,
    chartFill: 'none',
    plotAreaFill: 'none',
    xAxis: { tickLabelPosition: 'low', textStyle: { fill: c.gray, fontSize: 16 } },
    yAxis: { min: -60, max: 60, majorUnit: 20, numberFormatCode: '0;0;0',
      textStyle: { fill: c.gray, fontSize: 13 }, majorGridlines: { fill: c.line, width: 1 },
      line: { fill: c.gray, width: 1 } },
    dataLabels: dataLabels(c.white),
  });
  rect(slide, 'New errors key', 111, 579, 16, 16, c.gold);
  body(slide, 'New errors', 'Correct → wrong', 138, 577, 235, 31, 14);
  rect(slide, 'Corrected answers key', 414, 579, 16, 16, c.blue);
  body(slide, 'Corrected answers', 'Wrong → correct', 441, 577, 246, 31, 14);

  rect(slide, 'Contrast field', 812, 226, 414, 397, gradient('#E5EEEF', '#F3F0E2', 90));
  rule(slide, 812, 226, 414, c.gold, 2);
  for (const [index, comparator] of ['chinese', 'base'].entries()) {
    const y = 248 + index * 139;
    blue(slide, comparator + ' contrast', comparator === 'chinese' ? 'Mixed corpus − replay' : 'Mixed corpus − frozen 4B base',
      838, y, 357, 28, 17);
    serif(slide, comparator + ' estimate', contrastValue(cohort, comparator), 835, y + 32, 356, 53, 30);
    body(slide, comparator + ' interval', '95% CI  ' + intervalValue(cohort, comparator), 838, y + 88, 360, 30, 14);
    rule(slide, 838, y + 123, 360, c.line, 1);
  }
  const rate = value => points(value, false, 2) + '%';
  const accuracy = [
    ['4B base', rate(cohort.per_arm.base.accuracy)],
    ['Replay LoRA', rate(cohort.by_profile.chinese.mean)],
    ['Mixed-corpus LoRA', rate(cohort.by_profile.bilingual.mean)],
  ];
  for (const [index, [arm, value]] of accuracy.entries()) {
    const x = 833 + index * 125;
    text(slide, arm + ' accuracy', value, x, 541, 125, 35, { size: 17, bold: true, color: c.blue, align: 'center' });
    text(slide, arm + ' label', arm, x, 579, 125, 22, { size: 10, color: c.gray, align: 'center' });
  }
}

async function publicPresentation(file, hasChart) {
  const archive = await JSZip.loadAsync(file.data);
  const corePath = 'docProps/core.xml';
  const core = archive.file(corePath);
  if (core) {
    const metadata = (await core.async('string')).replace(
      /<(dc:creator|(?:cp:)?lastModifiedBy)>[^<]*<\/\1>/g,
      (_, tag) => '<' + tag + '>MemeVLM contributors</' + tag + '>')
      .replace(/<dc:title>[^<]*<\/dc:title>/, '<dc:title>MemeVLM Final Presentation</dc:title>');
    archive.file(corePath, metadata);
  }
  if (!hasChart) return archive.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
  const chartParts = Object.keys(archive.files).filter(name => /\/charts\/chart\d+\.xml$/.test(name));
  const chartPart = chartParts.find(name => name.endsWith('/chart1.xml'));
  if (chartParts.length !== 1 || !chartPart) throw new Error('Expected one native MemeQA chart.');

  const workbook = Workbook.create();
  workbook.worksheets.add('Transitions').getRange('A1:C4').values = [
    ['Training seed', ...transferSeries],
    ...transferChartRows.map(({ seed, errors, corrected }) => ['Seed ' + seed, -errors, corrected]),
  ];
  const spreadsheet = await SpreadsheetFile.exportXlsx(workbook);
  const xlsxName = 'memeqa-transitions.xlsx';
  archive.file('ppt/embeddings/' + xlsxName, spreadsheet.data);

  let seriesIndex = 0;
  const chartXml = (await archive.file(chartPart).async('string')).replace(/<c:ser>.*?<\/c:ser>/gs, series => {
    const column = ['B', 'C'][seriesIndex++];
    return series.replace(/<c:tx><c:v>([^<]*)<\/c:v><\/c:tx>/, (_, label) =>
      '<c:tx><c:strRef><c:f>Transitions!$' + column + '$1</c:f><c:strCache><c:ptCount val="1"/>'
      + '<c:pt idx="0"><c:v>' + label + '</c:v></c:pt></c:strCache></c:strRef></c:tx>')
      .replace(/<c:strLit>(.*?)<\/c:strLit>/s, (_, cache) =>
        '<c:strRef><c:f>Transitions!$A$2:$A$4</c:f><c:strCache>' + cache + '</c:strCache></c:strRef>')
      .replace(/<c:numLit>(.*?)<\/c:numLit>/s, (_, cache) =>
        '<c:numRef><c:f>Transitions!$' + column + '$2:$' + column + '$4</c:f>'
        + '<c:numCache>' + cache + '</c:numCache></c:numRef>');
  });
  if (seriesIndex !== transferSeries.length || /<c:(strLit|numLit)>/.test(chartXml)) {
    throw new Error('MemeQA chart series could not be linked to embedded data.');
  }
  const relation = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
  archive.file(chartPart, chartXml.replace('</c:chartSpace>',
    '<c:externalData xmlns:r="' + relation + '" r:id="rIdWorkbook"><c:autoUpdate val="0"/></c:externalData></c:chartSpace>'));
  const relsPart = path.posix.join(path.posix.dirname(chartPart), '_rels', path.posix.basename(chartPart) + '.rels');
  const target = path.posix.relative(path.posix.dirname(chartPart), 'ppt/embeddings/' + xlsxName);
  archive.file(relsPart, '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    + '<Relationship Id="rIdWorkbook" Type="' + relation + '/package" Target="' + target + '"/></Relationships>');
  const contentTypes = '[Content_Types].xml';
  const typesXml = await archive.file(contentTypes).async('string');
  if (!typesXml.includes('Extension="xlsx"')) {
    archive.file(contentTypes, typesXml.replace('</Types>', '<Default Extension="xlsx" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"/></Types>'));
  }
  return archive.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
}

const arms = Object.keys(labels);
if (!pilot.summary.complete || arms.some(arm => pilot.summary[arm].planned !== 96)) {
  throw new Error('Expected the complete bilingual seed-42 pilot.');
}
if (synthetic && (synthetic.per_arm.base.correct !== pilot.summary.base.strict_triple_success
    || synthetic.per_arm['bilingual-42'].correct !== pilot.summary.adapter.strict_triple_success)) {
  throw new Error('Expanded scores differ from the two reused synthetic pilot ledgers.');
}

const cover = ppt.slides.add();
cover.background.fill = gradient('#071E35', '#184B68', 12);
rect(cover, 'Outer image mat', 754, 0, 526, 720, gradient('#D5C79D', '#ECE4CB', 90));
rect(cover, 'Mat rule', 752, 0, 3, 720, c.gold);
rect(cover, 'Fine pattern on mat', 757, 0, 35, 720, hatch('#AB9660','#B6A472'));
rect(cover, 'Inner image mat', 796, 41, 442, 638, '#FBF9EF');
addMemorySchematic(cover, {rect, text}, [811, 61, 412, 598]);
text(cover, 'Research project name', 'MemeVLM', 64, 64, 635, 32,
  {size: 18, color: c.lightGold, font: headline});
rule(cover, 65, 124, 620, '#7C9297', 1);
serif(cover, 'Research title', 'Does AI Get\nthe Punchline?', 59, 167, 669, 198, 54, c.white);
text(cover, 'Research subtitle', 'Learning from Controlled Edits and\nTesting Transfer to Natural Memes',
  65, 403, 655, 108, {size:23, color: '#E3EBE9'});
rule(cover, 64, 600, 620, '#7C9297', 1);
text(cover, 'Authors', 'MemeVLM contributors', 64, 625, 668, 40, {size:16, color:c.white});
plannedNote(cover, 'cover', [refs.memebench]);

const walkthrough = page('What makes the caption funny?', 'MEME INTERPRETATION',
  'Original schematics illustrating cases from MemeBench and MER-Bench.', {theme:'case'});
const examples = [
  [addMemorySchematic,'Scale reversal','Why would 8 GB fear 4 KB?'],
  [addSarcasmSchematic,'Sarcasm','Does the cat look excited?'],
];
for (const [i,[draw,label,question]] of examples.entries()) {
  const x = 56 + i*602;
  accent(walkthrough, label+' label', label.toUpperCase(),x,171,565,26);
  rect(walkthrough, label+' mat',x,207,566,353,i ? '#E7ECEB' : '#EDE8D9');
  draw(walkthrough, {rect, text}, [x+7, 216, 552, 335]);
  serif(walkthrough, label+' question',question,x,581,568,54,21);
}
plannedNote(walkthrough,'demo',[refs.memebench, refs.mer]);

const families=page('Which edit changes the joke?','CONTROLLED MEME FAMILIES',
  'We created 240 controlled families across eight mechanisms. The family split used 168 for training, 40 for validation and 32 for testing.', {theme:'case'});
const cloud=[
  ['original','Original','The intern puts paper files in the sky.'],
  ['edited','Target edit','The intern uses remote storage.'],
  ['sham','Sham edit','The corner mark changes. The joke stays the same.'],
];
for (const [i,[variant,label,meaning]] of cloud.entries()){
  const x=57+i*397;
  accent(families,label,label.toUpperCase(),x,182,370,30);
  rect(families,label+' mount',x,225,371,235,'#E9EEF0');
  await picture(families,'cloud-'+variant+'.png','Cloud Backup: '+meaning,x+8,231,355,223);
  serif(families,label+' reading',meaning,x,488,371,118,20);
}
plannedNote(families,'families');

const split = page('How do we measure interpretation?','EVALUATION',
  '10,000 paired bootstrap draws resample whole synthetic families or whole MemeQA images with their questions.');
await methodFigure(split,'evaluation-protocol.png','Original, target and sham images have separate keys. All seven model arms answer the same items in each test.');
plannedNote(split,'evaluation',[refs.memeqa]);

const architecture = page('Where does LoRA change Qwen3-VL?','MODEL',
  'The visual encoder and base weights stay frozen. Answer-token loss updates rank-8 LoRA adapters in the language decoder.');
await methodFigure(architecture,'qwen3vl-lora.png','Visual tokens and the prompt enter the decoder. LoRA updates the attention and feed-forward projections in the language layers.');
plannedNote(architecture,'architecture',[refs.qwen,'Hu et al. (2022), LoRA. https://openreview.net/forum?id=nZeVKeeFYf9']);

const budget = page('What changes under the same training budget?','TRAINING',
  'The recipes also differ in example diversity and task mix. Their accuracy gap estimates the combined recipe effect.');
await methodFigure(budget,'matched-training.png','Two recipes each process 770 rows per epoch: 672 English plus 98 Chinese unique rows, or 98 unique Chinese rows replayed.');
plannedNote(budget,'budget',[refs.expanded]);

const checkpoint = page('Step 97 minimized observed validation loss','CHECKPOINT SELECTION',
  'Every run completed 291 updates. The first of three measured checkpoints supplied all six evaluated adapters.', {theme:'finding'});
await methodFigure(checkpoint,'training-diagnostics.png','In six runs, training loss declines. Shared-set validation loss is lowest at step 97, before the checks at steps 194 and 291.');
plannedNote(checkpoint,'checkpoint');

const result = page('Mixed-corpus LoRA improved edit accuracy','CONTROLLED EDIT FAMILIES',
  'All seven arms identified the cue correctly on all 96 variants. Differences in relation choices account for the accuracy gap.', {theme:'finding'});
profileBars(result,synthetic,'A family passes when both the cue and the intended relation are correct on all three images.','chinese');
plannedNote(result,'synthetic',[refs.expanded]);

const transfer = page('Does the gain carry over to natural memes?', 'MEMEQA SUBSET',
  natural ? 'Both paired confidence intervals span zero. The bootstrap resamples memes with all their questions.'
    : 'The matched contrast compares mixed-corpus and replay adapters.', {theme:'finding'});
naturalTransitions(transfer,natural);
plannedNote(transfer,'natural',[refs.memeqa,refs.memebench,refs.expanded],
  'The primary natural evaluation uses eligible multiple-choice questions on images outside the pilot. The same paired seeds provide the matched recipe comparison.');

const example = page('The base model read the edit and kept the old joke','FITNESS WATCH',
  'The two models selected the correct relation on the original and the sham. The target edit separates their answers.', {theme:'case'});
const sample = (arm, variant) => {
  const row = demo[arm].find(row => row.id === 'watch/' + variant);
  if (!row || row.status !== 'ok') throw new Error('Missing watch example: ' + arm + '/' + variant);
  return row.score;
};
const targetBase = sample('base','edited');
const targetAdapter = sample('adapter','edited');
if (targetBase.cue_correct !== 1 || targetBase.relation_correct !== 0 || targetAdapter.joint_correct !== 1)
  throw new Error('Changed watch example outcomes');
rect(example,'Watch mount',55,177,455,438,'#E5EFE8');
await picture(example,'watch-edited.png','Edited watch face reading three hours of walking',72,199,421,390);
accent(example,'Variant','TARGET EDIT',554,185,640,28);
serif(example,'Ground truth','“Three hours of walking”',551,219,667,66,27);
rule(example,554,301,646,c.line,1);
text(example,'Frozen base','Frozen 4B base',554,326,620,30,{size:15,color:c.ochre,bold:true});
body(example,'Base interpretation','Read the cue correctly, then chose the explanation that the watch counted clapping as exercise.',554,366,649,94,19);
text(example,'Bilingual adapter','Mixed-corpus LoRA, seed 42',554,475,620,30,{size:15,color:c.ochre,bold:true});
body(example,'Bilingual interpretation','Read the cue correctly and chose the explanation that the watch recorded walking.',554,514,643,99,19);
plannedNote(example,'example',[refs.results]);

const close=ppt.slides.add();
close.background.fill=gradient('#09273F','#0E4665',9);
text(close,'Section','INTERPRETATION',61,44,1080,28,{size:12,bold:true,color:c.lightGold});
serif(close,'Title','What the matched comparison establishes',57,88,1167,90,36,c.white);
rule(close,60,193,1160,'#758E94',1);
const outcomes=[
  ['Controlled edit families',synthetic,'Difference in strict triple accuracy'],
  ['Natural MemeQA subset',natural,'Difference in pooled choice accuracy'],
];
for(const [i,[label,cohort,outcome]] of outcomes.entries()){
  const x=61+i*605;
  text(close,'Outcome '+i,label,x,231,555,45,{size:21,color:c.white,bold:true});
  serif(close,'Matched estimate '+i,cohort ? contrastValue(cohort,'chinese') : 'Pending',x-2,290,554,101,48,c.lightGold);
  text(close,'Endpoint '+i,outcome,x,405,558,38,{size:16,color:'#E5ECE9'});
  text(close,'Interval '+i,cohort?'95% CI  '+intervalValue(cohort,'chinese'):'',x,455,555,45,{size:16,color:c.white});
}
rule(close,60,521,1160,'#758E94',1);
text(close,'Judgment','Against the same replay control, the authored edit task shows a clear gain. Transfer to independently authored questions remains uncertain.',61,554,1122,77,{size:22,color:c.white});
if (chineseTest) {
  text(close,'Chinese controls',`Frozen-model controls: complete triples 4B ${metricCount(chineseTest[0].strict)} · 8B ${metricCount(chineseTest[1].strict)}.`,61,640,1122,27,{size:15,color:c.lightGold});
}
if (coverage) {
  const coverageFooter = humanAudit
    ? `Reference coverage: ${coverage.scorable}/${coverage.planned} frozen · ${coverageReplay.scorable}/${coverageReplay.planned} replay · two human ratings of all held-out explanations.`
    : coverageReplay
    ? `Reference coverage: ${coverage.scorable}/${coverage.planned} original · ${coverageReplay.scorable}/${coverageReplay.planned} post-hoc replay of the same replies.`
    : `Reference coverage: ${coverage.scorable}/${coverage.planned} judgments passed scoring checks · paired results in the appendix.`;
  text(close,'Chinese judge coverage',coverageFooter,61,670,1122,27,{size:15,color:c.lightGold});
}
const spokenCoverage = humanAudit
  ? ' Two teammates later rated all 240 held-out explanations. Human ratings support the image–OCR result. Hint ordering varies.'
  : coverageReplay
  ? ` Coverage scoring accepted ${coverage.scorable} of ${coverage.planned} replies. Post-hoc replay accepted ${coverageReplay.scorable} from the same replies, with no new GPU calls.`
  : coverage ? ` ${coverage.scorable} of ${coverage.planned} judge responses passed scoring checks, with paired results in the appendix.` : '';
plannedNote(close,'conclusion',[refs.expanded],
  'We compare the full training recipes on the authored edits and the independently authored MemeQA questions. The final conclusions depend on both matched comparisons.',
  `${narration.conclusion.script}${spokenCoverage} Thank you.`);

const bySeed=page('Accuracy across the three paired training seeds','MODEL VARIABILITY',
  'Accuracy is displayed in percent. The 4B base scored 37.5% on synthetic triples and 53.55% on the primary MemeQA subset.');
const rows=[['Training recipe','Seed','Strict triple accuracy','MemeQA subset accuracy']];
for (const profile of ['chinese','bilingual']) for (const seed of [42,43,44]) {
  const arm = profile+'-'+seed;
  rows.push([profile==='chinese'?'Replay LoRA':'Mixed-corpus LoRA',String(seed),
    synthetic?points(synthetic.per_arm[arm].accuracy,false,1):'Pending',natural?points(natural.per_arm[arm].accuracy,false,2):'Pending']);
}
table(bySeed,rows,[315,140,338,368],58,190,56,[4,5,6]);
bySeed.speakerNotes.text='The synthetic endpoint has 32 held-out families, and the natural endpoint has 816 questions on 134 images. Each row is a separately trained adapter.';

const failure=page('The adapter read the cue and reversed the roles','ROLE REVERSAL',
  'The same error appeared on the original and sham in two held-out families (mixed-corpus seed 42).', {theme:'case'});
await picture(failure,'role-original.png','Dog says SIT to trainer.',56,188,515,410);
accent(failure,'Reference','REFERENCE INTERPRETATION',618,208,572,36);
serif(failure,'Gold relation','The dog takes the trainer’s role.',615,250,584,111,25);
rule(failure,618,383,570,c.line,1);
accent(failure,'Response','MIXED-CORPUS LORA CHOICE',618,410,593,36);
body(failure,'Error','The trainer gives the command in the usual direction.',618,458,577,112,22);
failure.speakerNotes.text='The adapter correctly identified the cue, then assigned the action to the conventional role. It chose the correct relation after the target edits in both cases.';

if (chineseTest) {
  const [four, eight] = chineseTest;
  await addChineseAppendix({page, picture, text, serif, rect, rule, body}, {
    resultFigure: path.join(figures, 'rq3-test.png'),
    resultTakeaway: `Complete triples: 4B ${metricCount(four.strict)} · 8B ${metricCount(eight.strict)}.`,
    resultNotes: `Both models answer all three variants correctly on the underlying development example. On the ten-family test, 4B and 8B answer ${metricCount(four.edited)} and ${metricCount(eight.edited)} target edits correctly. Semantic agreement is ${metricCount(four.sham)} and ${metricCount(eight.sham)} between originals and shams, and ${metricCount(four.permutation)} and ${metricCount(eight.permutation)} after the option-order repeat. Agreement includes pairs of identical wrong interpretations.`,
  });
}

await addCoverageAppendix({page, picture, blue, body, rect, rule}, {
  result: coverage, resultFigure: path.join(figures, 'reference-coverage-test.png'),
});
await addCoverageAppendix({page, picture, blue, body, rect, rule}, {
  result: coverageReplay, resultFigure: path.join(figures, 'reference-coverage-replay.png'), replayOf: coverage,
});
if (humanAudit) {
  addHumanAuditAppendix({page, blue, serif, body, rule, rect, table}, humanAudit, coverage, coverageReplay);
}
const qaAnswer = ({evidence, answer}) => {
  if (evidence === 'reference-coverage' && coverage) {
    return humanAudit ? humanEffectsAnswer(humanAudit) : coverageAnswer(coverage, coverageReplay);
  }
  return evidence === 'human-audit' && humanAudit ? humanAuditAnswer(humanAudit) : answer;
};

const guide=[
  '# MemeVLM', '',
  '| Speaker | Time | Slides |','|---|---|---|',
  `| ${names[0]} | 0:00–2:28 | 1–3, including the handoff |`,
  `| ${names[1]} | 2:28–5:23 | 4–7, including the handoff |`,
  `| ${names[2]} | 5:23–8:45 | 8–11 |`,'',
  'The main talk takes 8:45 including two four-second handoffs. Use the appendix for the separate 3:00 Q&A.','',
  ...plan.flatMap(row=>['## '+row.slide+' · '+row.speaker+' · '+row.start+'–'+row.end,'',row.cue,'',row.script,'']),
  '## Questions · 3:00','',
  ...narration.qa.flatMap(entry=>['**'+entry.question+'** · '+names[entry.speaker],'',qaAnswer(entry), '']),
].join('\n');
const pptx = await PresentationFile.exportPptx(ppt);
await fs.writeFile(destination, await publicPresentation(pptx, Boolean(natural)), { flag: 'wx' });
await fs.writeFile(guidePath,guide,{flag:'wx'});
if(preview){
  for(const [index,slide] of ppt.slides.items.entries()){
    const blob=await ppt.export({slide,format:'png',scale:1});
    await fs.writeFile(path.join(preview,'slide-'+String(index+1).padStart(2,'0')+'.png'),Buffer.from(await blob.arrayBuffer()),{flag:'wx'});
  }
}
