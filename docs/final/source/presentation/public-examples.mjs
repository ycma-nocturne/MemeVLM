const colors = {
  ink: '#18394B',
  navy: '#143A54',
  gold: '#EAD8A8',
  paper: '#F5F7F5',
  sand: '#F3EDE0',
  lightRule: '#C4D2D5',
};

export function addMemorySchematic(slide, {rect, text}, [x, y, width, height]) {
  const pad = 22;
  const span = (width - 3 * pad) / 2;
  const baseline = y + Math.round(height * .78);
  const tall = Math.round(height * .51);
  const short = Math.round(height * .28);
  rect(slide, 'Memory schematic background', x, y, width, height, colors.paper);
  rect(slide, 'Schematic visual baseline', x + pad, baseline, width - 2 * pad, 1, colors.lightRule);
  text(slide, 'Schematic identifier', 'CASE SCHEMATIC', x + pad, y + 20, width - 2 * pad, 23,
    {size: 12, color: colors.ink, bold: true});

  for (const [index, [label, size, visualHeight, fill, ink]] of [
    ['Apollo-era machine', '4 KB', tall, colors.navy, '#FFFFFF'],
    ['Modern browser', '8 GB', short, colors.gold, colors.ink],
  ].entries()) {
    const left = x + pad + index * (span + pad);
    rect(slide, label + ' block', left, baseline - visualHeight, span, visualHeight, fill);
    const sizeTop = baseline - Math.round(visualHeight * .50) - 26;
    text(slide, label + ' size', size, left + 3, sizeTop, span - 6, 56,
      {size: height > 400 ? 32 : 28, color: ink, align: 'center', font: 'Georgia'});
    text(slide, label + ' legend', label, left, baseline + 10, span, 45,
      {size: 12, color: colors.ink, align: 'center'});
  }
}

export function addSarcasmSchematic(slide, {rect, text}, [x, y, width, height]) {
  const pad = 22;
  const center = x + width / 2;
  const faceY = y + 150;
  const ellipse = (name, left, top, w, h, fill) => slide.shapes.add({
    name, geometry: 'ellipse', position: {left, top, width: w, height: h},
    fill, line: {fill: 'none', width: 0},
  });
  rect(slide, 'Sarcasm schematic background', x, y, width, height, colors.sand);
  text(slide, 'Schematic identifier', 'CASE SCHEMATIC', x + pad, y + 20, width - 2 * pad, 23,
    {size: 12, color: colors.ink, bold: true});
  rect(slide, 'Classroom prompt mat', x + pad, y + 57, width - 2 * pad, 61, '#FFFFFF');
  text(slide, 'Classroom prompt', 'A fun class project', x + pad, y + 72, width - 2 * pad, 40,
    {size: 21, color: colors.ink, align: 'center', font: 'Georgia'});
  ellipse('Schematic cat left ear', center - 54, faceY - 8, 28, 54, colors.navy);
  ellipse('Schematic cat right ear', center + 26, faceY - 8, 28, 54, colors.navy);
  ellipse('Schematic cat head', center - 63, faceY + 9, 126, 96, colors.navy);
  rect(slide, 'Level left eyelid', center - 38, faceY + 49, 26, 3, colors.gold);
  rect(slide, 'Level right eyelid', center + 12, faceY + 49, 26, 3, colors.gold);
  ellipse('Small nose', center - 4, faceY + 62, 8, 5, colors.gold);
  rect(slide, 'Flat expression', center - 12, faceY + 73, 24, 3, colors.gold);
  text(slide, 'Illustrated expression', 'Flat expression', x + pad, y + 274, width - 2 * pad, 25,
    {size: 13, color: colors.ink, align: 'center'});
}
