// PDF writer and fit report: file structure (header, xref offsets, trailer, page count), Flate streams that inflate
// (zlib format) to the expected sizes, the uncompressed fallback, and the report with all / no sections. The full
// report is also handed to run_tests.py (window.__testArtifacts) and saved as test/out/fit_report_test.pdf for an
// external check (pdfinfo / pdftoppm / mutool).
import { test, check } from './harness';
import { PdfDoc, textWidth } from '../src/io/pdf';
import { makeFitReport } from '../src/report/fitreport';
import { syntheticReportData } from './report_data';

const latin1 = (b: Uint8Array) => { let s = ''; for (let i = 0; i < b.length; i += 8192) s += String.fromCharCode(...b.subarray(i, i + 8192)); return s; };

async function inflate(b: Uint8Array): Promise<Uint8Array> {
  const s = new Blob([b]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

/** Structural check of a PDF written by PdfDoc; returns the page count and stream statistics. */
async function checkPdf(b: Uint8Array) {
  const s = latin1(b);
  check(s.startsWith('%PDF-1.4\n'), 'header %PDF-1.4');
  check(s.endsWith('%%EOF\n'), 'ends with %%EOF');
  const sx = /startxref\n(\d+)\n%%EOF\n$/.exec(s);
  check(!!sx, 'startxref');
  const xo = +sx![1];
  check(s.startsWith('xref\n', xo), `xref at the startxref offset ${xo}`);
  const hd = /^xref\n0 (\d+)\n/.exec(s.slice(xo))!;
  const n = +hd[1], e0 = xo + hd[0].length;
  check(s.slice(e0, e0 + 20) === '0000000000 65535 f \n', 'xref entry 0');
  for (let i = 1; i < n; i++) {
    const ent = s.slice(e0 + 20 * i, e0 + 20 * (i + 1));
    check(/^\d{10} 00000 n \n$/.test(ent), `xref entry ${i} format: ${JSON.stringify(ent)}`);
    const off = +ent.slice(0, 10);
    check(s.startsWith(`${i} 0 obj\n`, off), `object ${i} at its xref offset ${off}`);
  }
  const tr = /trailer\n<< \/Size (\d+) \/Root 1 0 R/.exec(s.slice(xo));
  check(!!tr && +tr[1] === n, 'trailer /Size');
  const count = +/\/Type \/Pages \/Count (\d+)/.exec(s)![1];
  const pages = (s.match(/\/Type \/Page /g) ?? []).length;
  check(count === pages, `/Count ${count} vs ${pages} page objects`);
  // every stream: /Length matches, Flate streams inflate; images to width x height x 3 bytes
  let nStream = 0, nFlate = 0, nImg = 0;
  const re = /(\d+) 0 obj\n<< ([^\n]*?)\/Length (\d+)( \/Filter \/FlateDecode)? >>\nstream\n/g;
  for (let m; (m = re.exec(s));) {
    const len = +m[3], start = m.index + m[0].length;
    check(s.startsWith('\nendstream\nendobj\n', start + len), `object ${m[1]}: /Length ${len} matches the stream`);
    let body = b.subarray(start, start + len);
    if (m[4]) { body = await inflate(body); nFlate++; }
    const im = /\/Width (\d+) \/Height (\d+)/.exec(m[2]);
    if (im) { check(body.length === +im[1] * +im[2] * 3, `image ${m[1]}: ${body.length} bytes for ${im[1]} x ${im[2]} RGB`); nImg++; }
    else check(/BT|re|Tj/.test(latin1(body.subarray(0, 4000))), `content stream ${m[1]} has drawing operators`);
    nStream++;
  }
  check(nStream === pages + nImg, `${nStream} streams = ${pages} pages + ${nImg} images`);
  return { pages, nStream, nFlate, nImg, objects: n };
}

test('pdf: writer structure, xref, Flate and uncompressed', async () => {
  const doc = new PdfDoc({ title: 'test λ' });
  for (let p = 0; p < 3; p++) {
    const pg = doc.addPage();
    pg.text(50, 60, `page ${p + 1}: (parens) back\\slash µm ± σ − €`, { size: 12 });
    pg.rect(50, 80, 100, 20, { fill: '#3b6fb6', stroke: { color: '#000', width: 1 } });
    pg.polyline([50, 150, 100, 120, NaN, NaN, 150, 150, 200, 110], { color: [200, 40, 40], width: 1.5, dash: [3, 2] });
    const rgb = new Uint8Array(4 * 3 * 3).map((_, i) => (i * 37) & 255);
    pg.image(doc.addImage(rgb, 4, 3), 50, 200, 80, 60);
  }
  const z = await checkPdf(await doc.save());
  check(z.pages === 3 && z.nImg === 3 && z.nFlate === 6, `pages ${z.pages}, images ${z.nImg}, flate ${z.nFlate}`);
  const u = await checkPdf(await doc.save({ compress: false }));
  check(u.pages === 3 && u.nFlate === 0, 'uncompressed');
  const w = textWidth('Hello', 10);
  check(Math.abs(w - 0.01 * (722 + 556 + 222 + 222 + 556)) < 1e-9, `Helvetica width of "Hello" ${w}`);
  return `3 pages, ${z.objects} objects; uncompressed ok`;
});

test('pdf: fit report from synthetic data; empty and partial reports', async () => {
  const t0 = performance.now();
  const bytes = await makeFitReport(syntheticReportData());
  const ms = performance.now() - t0;
  const r = await checkPdf(bytes);
  check(r.pages >= 3 && r.pages <= 6, `report pages ${r.pages}`);
  check(r.nImg >= 30, `report images ${r.nImg}`);
  const s = latin1(bytes);
  check(s.includes('/BaseFont /Helvetica-Bold') && s.includes('/Encoding /WinAnsiEncoding'), 'standard fonts, WinAnsi');
  const empty = await checkPdf(await makeFitReport({}));
  check(empty.pages === 1 && empty.nImg === 0, `empty report: ${empty.pages} page, ${empty.nImg} images`);
  const part = await checkPdf(await makeFitReport({ title: 'Zernikes only', zernike: { noll: [5, 6, 7], values: [0.01, -0.02, 0.005] } }));
  check(part.pages === 1 && part.nImg === 0, 'partial report');
  let b64 = '';
  for (let i = 0; i < bytes.length; i += 8192) b64 += String.fromCharCode(...bytes.subarray(i, i + 8192));
  ((window as any).__testArtifacts ??= {})['fit_report_test.pdf'] = btoa(b64);
  return `${r.pages} pages, ${r.nImg} images, ${(bytes.length / 1024).toFixed(0)} kB, ${ms.toFixed(0)} ms`;
});
