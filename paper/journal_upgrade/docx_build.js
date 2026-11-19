// Build the Word manuscript from docx_assets/content.json.
// Single-column Letter layout: Word cannot reproduce the journal's two-column
// typesetting, and a single-column manuscript is what submission systems expect.
const fs = require("fs");
const path = require("path");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, HeadingLevel, BorderStyle, ShadingType,
  ExternalHyperlink, LevelFormat, PageOrientation,
} = d;

const HERE = __dirname;
const A = path.join(HERE, "docx_assets");
const DATA = JSON.parse(fs.readFileSync(path.join(A, "content.json"), "utf8"));
const FIGSIZE = DATA.figsizes;

const TEXT_W_DXA = 9360;          // 6.5in at 1440 dxa/in
const EMU = 914400;
const SERIF = "Times New Roman";

const nodef = { top: 0, bottom: 0, left: 0, right: 0 };

function runs(list, opts = {}) {
  return (list || []).map(r => new TextRun({
    text: r.text, bold: !!r.bold, italics: !!r.italic,
    font: SERIF, size: opts.size || 22,
  }));
}

function para(list, opts = {}) {
  return new Paragraph({
    children: runs(list, opts),
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    spacing: { before: opts.before || 0, after: opts.after === undefined ? 120 : opts.after,
               line: opts.line || 276 },
  });
}

function caption(prefix, number, list) {
  const kids = [new TextRun({ text: `${prefix} ${number}. `, bold: true, font: SERIF, size: 18 })];
  kids.push(...runs(list, { size: 18 }));
  return new Paragraph({
    children: kids, alignment: AlignmentType.LEFT,
    spacing: { before: 60, after: 200, line: 240 }, keepLines: true,
  });
}

// ----------------------------------------------------------------- images
function imagePara(file, maxWin, maxHin, after = 60) {
  const sz = FIGSIZE[file];
  let w = maxWin, h = maxWin;
  if (sz) {
    const [pw, ph] = sz;
    const scale = Math.min(maxWin / pw, maxHin / ph);
    w = pw * scale; h = ph * scale;
  }
  return new Paragraph({
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(path.join(A, file)),
      transformation: { width: Math.round(w * 96), height: Math.round(h * 96) },
    })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after },
  });
}

function eqPara(b) {
  const scale = Math.min(5.6 / b.w_in, 1);
  return new Paragraph({
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(path.join(A, b.png)),
      transformation: {
        width: Math.round(b.w_in * scale * 96),
        height: Math.round(b.h_in * scale * 96),
      },
    })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
  });
}

// ----------------------------------------------------------------- tables
function buildTable(b) {
  const ncol = Math.max(
    ...b.header.map(r => r.spans.reduce((a, c) => a + c, 0)),
    ...b.rows.map(r => r.length));
  const wide = ncol >= 7;
  const fs_ = wide ? 13 : 16;                     // half-points: 6.5pt / 8pt
  const firstW = Math.round(TEXT_W_DXA * (wide ? 0.235 : 0.30));
  const restW = Math.round((TEXT_W_DXA - firstW) / (ncol - 1));
  const widths = [firstW, ...Array(ncol - 1).fill(restW)];
  widths[ncol - 1] = TEXT_W_DXA - firstW - restW * (ncol - 2);

  const cell = (txt, i, opt = {}) => new TableCell({
    columnSpan: opt.span && opt.span > 1 ? opt.span : undefined,
    width: {
      size: widths.slice(i, i + (opt.span || 1)).reduce((a, c) => a + c, 0),
      type: WidthType.DXA,
    },
    shading: opt.shade
      ? { type: ShadingType.CLEAR, fill: "E2EDF7", color: "auto" } : undefined,
    borders: {
      top: opt.topLine
        ? { style: BorderStyle.SINGLE, size: 6, color: "000000" } : { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      bottom: opt.botLine
        ? { style: BorderStyle.SINGLE, size: 6, color: "000000" } : { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    margins: { top: 20, bottom: 20, left: 60, right: 60 },
    children: [new Paragraph({
      children: [new TextRun({
        text: txt || "", bold: !!opt.bold, font: SERIF, size: fs_,
      })],
      alignment: (i === 0 && !opt.centerSpan) ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { before: 0, after: 0, line: 200 },
    })],
  });

  const rows = [];
  b.header.forEach((hr, hi) => {
    const kids = [];
    let col = 0;
    hr.cells.forEach((t, k) => {
      const span = hr.spans[k] || 1;
      kids.push(cell(t, col, {
        bold: true, span,
        topLine: hi === 0, botLine: hi === b.header.length - 1,
        centerSpan: span > 1,
      }));
      col += span;
    });
    while (col < ncol) {
      kids.push(cell("", col, {
        bold: true, topLine: hi === 0, botLine: hi === b.header.length - 1,
      }));
      col += 1;
    }
    rows.push(new TableRow({ tableHeader: true, children: kids }));
  });
  b.rows.forEach((rr, ri) => {
    const padded = Array.from({ length: ncol }, (_, i) => rr[i] || "");
    const shade = b.shaded.includes(ri);
    rows.push(new TableRow({
      children: padded.map((t, i) => cell(t, i, {
        shade, bold: shade, botLine: ri === b.rows.length - 1,
      })),
    }));
  });

  return new Table({
    columnWidths: widths,
    width: { size: TEXT_W_DXA, type: WidthType.DXA },
    rows,
  });
}

// ------------------------------------------------------------------ build
const children = [];
for (const b of DATA.blocks) {
  switch (b.type) {
    case "title":
      children.push(new Paragraph({
        children: [new TextRun({ text: b.text, bold: true, font: SERIF, size: 32 })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 360, line: 320 },
      }));
      break;

    case "heading":
      children.push(new Paragraph({
        heading: b.level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
        children: [new TextRun({
          text: b.text, bold: true, italics: b.level === 2,
          font: SERIF, size: b.level === 1 ? 26 : 23,
        })],
        spacing: { before: b.level === 1 ? 320 : 240, after: 120 },
        keepNext: true,
      }));
      break;

    case "para":
      children.push(para(b.runs));
      break;

    case "bullets":
      b.items.forEach(it => children.push(new Paragraph({
        children: runs(it), numbering: { reference: "bullets", level: 0 },
        spacing: { before: 0, after: 60, line: 276 },
      })));
      break;

    case "numbered":
      b.items.forEach(it => children.push(new Paragraph({
        children: runs(it), numbering: { reference: "numbers", level: 0 },
        spacing: { before: 0, after: 100, line: 276 },
        alignment: AlignmentType.JUSTIFIED,
      })));
      break;

    case "equation":
      children.push(eqPara(b));
      break;

    case "callout":
      b.paras.forEach((p, i) => children.push(new Paragraph({
        children: runs(p, { size: 20 }),
        alignment: AlignmentType.LEFT,
        spacing: { before: i === 0 ? 160 : 0, after: 80, line: 260 },
        indent: { left: 360, right: 360 },
        border: i === 0
          ? { top: { style: BorderStyle.SINGLE, size: 6, color: "808080" } } : undefined,
      })));
      children.push(new Paragraph({
        text: "", spacing: { after: 160 },
        border: { top: { style: BorderStyle.SINGLE, size: 6, color: "808080" } },
      }));
      break;

    case "figure": {
      const multi = b.images.length > 1;
      b.images.forEach((im, i) => {
        children.push(imagePara(im, multi ? 4.3 : 6.1, multi ? 2.6 : 4.2,
                                multi && b.subcaptions[i] ? 0 : 60));
        if (multi && b.subcaptions[i]) {
          children.push(new Paragraph({
            children: [new TextRun({
              text: `(${String.fromCharCode(97 + i)}) ${b.subcaptions[i]}`,
              font: SERIF, size: 17, italics: true,
            })],
            alignment: AlignmentType.CENTER,
            spacing: { before: 20, after: 120 },
          }));
        }
      });
      children.push(caption("Fig.", b.number, b.caption));
      break;
    }

    case "table":
      children.push(caption("Table", b.number, b.caption));
      children.push(buildTable(b));
      if (b.note && b.note.length) {
        children.push(new Paragraph({
          children: runs(b.note, { size: 17 }),
          spacing: { before: 60, after: 200 },
        }));
      } else {
        children.push(new Paragraph({ text: "", spacing: { after: 200 } }));
      }
      break;

    case "refs":
      b.items.forEach(r => {
        const kids = [new TextRun({ text: `[${r.n}] `, font: SERIF, size: 19 }),
                      new TextRun({ text: r.text + " ", font: SERIF, size: 19 })];
        if (r.url) {
          kids.push(new ExternalHyperlink({
            link: r.url,
            children: [new TextRun({
              text: r.url.replace("https://doi.org/", "doi:"),
              font: SERIF, size: 19, color: "007398", underline: {},
            })],
          }));
        }
        children.push(new Paragraph({
          children: kids, alignment: AlignmentType.LEFT,
          spacing: { before: 0, after: 80, line: 240 },
          indent: { left: 360, hanging: 360 },
        }));
      });
      break;

    default:
      console.log("UNHANDLED BLOCK", b.type);
  }
}

const doc = new Document({
  creator: "", title: "A Leakage-Audited Benchmark of Granular-Ball Twin Support Vector Machines",
  description: "",
  styles: {
    default: {
      document: { run: { font: SERIF, size: 22 } },
      heading1: {
        run: { font: SERIF, size: 26, bold: true },
        paragraph: { spacing: { before: 320, after: 120 }, keepNext: true },
      },
      heading2: {
        run: { font: SERIF, size: 23, bold: true, italics: true },
        paragraph: { spacing: { before: 240, after: 120 }, keepNext: true },
      },
    },
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
                   alignment: AlignmentType.LEFT,
                   style: { paragraph: { indent: { left: 420, hanging: 260 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.LOWER_ROMAN, text: "(%1)",
                   alignment: AlignmentType.LEFT,
                   style: { paragraph: { indent: { left: 520, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children,
  }],
});

const out = path.join(HERE, "EF-GBTSVM_revised_0_6_6p8.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024 / 1024).toFixed(2), "MB");
});
