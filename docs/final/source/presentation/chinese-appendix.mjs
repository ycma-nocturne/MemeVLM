import path from 'node:path';
const examples = [
  ['original', 'Original', 'Spiny handle', 'The spikes make an unlocked door hard to approach.', '#2D6757', '#E5ECE3'],
  ['edited', 'Target edit', 'Smooth knob', 'The visual cue for the barrier disappears.', '#846226', '#F1E9D4'],
  ['sham', 'Sham edit', 'Spines retained', 'The handle changes appearance while the barrier remains.', '#315876', '#E4EDF0'],
];

function handleIcon(slide, key, x, y, color) {
  slide.shapes.add({
    name: key + ' handle icon', geometry: 'ellipse',
    position: { left: x + 7, top: y + 7, width: 16, height: 16 },
    fill: key === 'sham' ? '#CBDADE' : 'none', line: { fill: color, width: 2 },
  });
  if (key === 'edited') return;
  const ends = [[15,1,15,6],[24,5,21,9],[29,15,24,15],[25,25,21,21],
    [15,29,15,24],[5,25,9,21],[1,15,6,15],[5,5,9,9]];
  slide.shapes.add({
    name: key + ' cactus spines', geometry: 'custom',
    position: { left: x, top: y, width: 30, height: 30 },
    fill: 'none', line: { fill: color, width: 1.5 },
    customPaths: [{
      width: 30, height: 30,
      commands: ends.flatMap(([x1, y1, x2, y2]) =>
        [{ moveTo: { x: x1, y: y1 } }, { lineTo: { x: x2, y: y2 } }]),
    }],
  });
}

export async function addChineseAppendix(ui, { resultFigure, resultTakeaway, resultNotes = '' } = {}) {
  const { page, picture, text, serif, rect, rule, body } = ui;
  const slide = page('Does interpretation follow the visual evidence?', 'MEMEBENCH',
    resultFigure ? 'Original schematic of a development case. The measured results use separate held-out families.' : '',
    { theme: 'case' });
  serif(slide, 'Caption in English', '“Why won’t you open up?”   “It’s not locked.”', 55, 179, 642, 43, 20);
  body(slide, 'Reference relationship', 'A physical barrier stands in for emotional distance.', 56, 226, 610, 31, 13);

  for (const [index, [key, label, cue, meaning, color, mat]] of examples.entries()) {
    const x = 55 + index * 212;
    rect(slide, label + ' mat', x, 268, 195, 353, mat);
    rule(slide, x, 268, 195, color, 2);
    handleIcon(slide, key, x + 9, 276, color);
    text(slide, label, label, x + 46, 279, 135, 28, { size: 15, color, bold: true });
    rect(slide, label + ' schematic field', x + 12, 319, 171, 183, '#F6F2EA');
    rect(slide, label + ' door frame', x + 48, 338, 98, 136, '#D4BE95');
    rect(slide, label + ' door panel', x + 55, 344, 84, 123, key === 'sham' ? '#EDF1ED' : '#F9F5EB');
    rect(slide, label + ' hinge', x + 55, 363, 4, 18, color);
    rect(slide, label + ' lower hinge', x + 55, 427, 4, 18, color);
    handleIcon(slide, key, x + 103, 391, color);
    text(slide, label + ' schematic label', 'Schematic', x + 12, 479, 171, 21,
      {size: 9, color, align: 'center'});
    text(slide, cue, cue, x + 11, 516, 173, 26, { size: 14, color, bold: true });
    body(slide, label + ' reference meaning', meaning, x + 11, 550, 173, 61, 12);
  }

  rect(slide, 'Example and benchmark divider', 690, 174, 1, 448, '#C2CFCC');
  serif(slide, 'Held-out evaluation', 'Held-out meme families', 713, 179, 508, 45, 21);
  rule(slide, 715, 229, 502, '#C2CFCC', 1);
  if (resultFigure) {
    await picture(slide, path.basename(resultFigure),
      'Correctness and consistency by model on held-out families.', 714, 247, 508, 347, path.dirname(resultFigure));
    if (resultTakeaway) body(slide, 'Benchmark interpretation', resultTakeaway, 716, 599, 502, 48, 13);
  }
  slide.speakerNotes.text = [
    'The same caption stays in all three images. Opening up refers to becoming emotionally accessible. In the original and the sham, the spiny handle supports an “unlocked but hard to approach” reading. The target edit replaces that handle with a smooth knob, removing visual support for this specific metaphor.',
    'This schematic illustrates a development case. Scoring uses the source inputs. Under the held-out protocol, the model must read the visual cue and select the supported interpretation on each version. Strict accuracy requires all three variants to pass.',
    resultNotes,
    'Wang et al. (2026), MemeBench. https://arxiv.org/abs/2607.27798',
  ].filter(Boolean).join('\n\n');
  return slide;
}
