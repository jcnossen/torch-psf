// PDF writer and fit report: file structure (header, xref offsets, trailer, page count), Flate streams that inflate
// (zlib format) to the expected sizes, the uncompressed fallback, and the report with all / no sections. The full
// report is also handed to run_tests.py (window.__testArtifacts) and saved as test/out/fit_report_test.pdf for an
// external check (pdfinfo / pdftoppm / mutool).
import { test, check } from './harness';
import { PdfDoc, textWidth } from '../src/io/pdf';
import { makeFitReport } from '../src/report/fitreport';
import { syntheticReportData } from './report_data';
import { getDevice } from '../src/gpu/device';
import { simulateDemo, ArrayMovie, DEMO, DEMO_ZERNIKE } from '../src/smlm/simulate';
import { estimateInsitu } from '../src/smlm/estimate';
import { insituReportData, beadQuality } from '../src/report/fromapp';
import { DEFAULT_NOLL } from '../src/psf/vectorial';
import { StopToken } from '../src/core/task';
import type { FitSummary } from '../src/ui/protocol';

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
  check(r.pages >= 4 && r.pages <= 7, `report pages ${r.pages}`);
  check(r.nImg >= 30, `report images ${r.nImg}`);
  const s = latin1(bytes);
  check(s.includes('/BaseFont /Helvetica-Bold') && s.includes('/Encoding /WinAnsiEncoding'), 'standard fonts, WinAnsi');
  const empty = await checkPdf(await makeFitReport({}));
  check(empty.pages === 1 && empty.nImg === 0, `empty report: ${empty.pages} page, ${empty.nImg} images`);
  const part = await checkPdf(await makeFitReport({ title: 'Zernikes only', zernike: { noll: [5, 6, 7], values: [0.01, -0.02, 0.005] } }));
  check(part.pages === 1 && part.nImg === 0, 'partial report');
  artifact('fit_report_test.pdf', bytes);
  return `${r.pages} pages, ${r.nImg} images, ${(bytes.length / 1024).toFixed(0)} kB, ${ms.toFixed(0)} ms`;
});

function artifact(name: string, bytes: Uint8Array) {
  let b64 = '';
  for (let i = 0; i < bytes.length; i += 8192) b64 += String.fromCharCode(...bytes.subarray(i, i + 8192));
  ((window as any).__testArtifacts ??= {})[name] = btoa(b64);
}

test('pdf: ROI quality range page (with / without rejected ROIs, reference line, no q)', async () => {
  const d = syntheticReportData(), Q = d.quality!;
  const txt = async (x: typeof d) => latin1(await makeFitReport(x, { compress: false }));
  const s = await txt(d);
  check(s.includes('(ROI quality range: best, last accepted, first rejected)') && s.includes('(First rejected: just above the cutoff)') &&
    s.includes('(Last accepted: just below the cutoff)') && s.includes('rejection threshold 1.25'), 'quality page with all three bands');
  check(Q.best.every((r, i) => r.rank === i) && Q.lastAccepted.every((r) => r.q < 1.25) && Q.firstRejected.every((r) => r.q >= 1.25), 'band selection');
  const noRej = { ...d, quality: { ...Q, firstRejected: [] } };
  const s2 = await txt(noRej);
  check(s2.includes('(ROI quality range') && !s2.includes('First rejected'), 'no rejected ROIs: two bands');
  const ref = { ...d, quality: { ...Q, cutoffKind: 'reference' as const } };
  const s3 = await txt(ref);
  check(s3.includes('no rejection') && s3.includes('Just above the reference line'), 'reference line labelled');
  const s4 = await txt({ ...d, quality: { ...Q, q: new Float32Array(0) } });
  check(!s4.includes('ROI quality range'), 'no q: no section');
  const full = await checkPdf(await makeFitReport(d)), without = await checkPdf(await makeFitReport({ ...d, quality: null }));
  check(full.pages === without.pages + 1 && full.nImg === without.nImg + 9 * 3 + 18, `one page more (${full.pages} vs ${without.pages}), 27 images + 18 colour bars (${full.nImg - without.nImg})`);
  return `${full.pages} pages with the quality page, ${without.pages} without`;
});

test('pdf: quality range of the bead fit (chi2 per bead, reference line)', async () => {
  const R = 11, nz = 5, nb = 6, R2 = R * R;
  let seed = 3; const u = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const data: Float32Array[] = [], model: Float32Array[] = [];
  for (let k = 0; k < nb; k++) {
    const m = Float32Array.from({ length: nz * R2 }, (_, i) => 10 + (500 / (1 + (Math.floor(i / R2) - 2) ** 2)) * Math.exp(-(((i % R) - 5) ** 2 + ((Math.floor(i / R) % R) - 5) ** 2) / (2 + (Math.floor(i / R2) - 2) ** 2)));
    model.push(m); data.push(Float32Array.from(m, (v, i) => v + Math.sqrt(v) * (2 * u() - 1) * 1.7 + (k === 4 && i % 7 === 0 ? 200 : 0)));
  }
  const r = { R, data, model, beads: Array.from({ length: nb }, (_, k) => ({ stack: 0, y0: 0, x0: 0, x: 0, y: 0, zoff: 0, heldout: k === 5, nz })) } as unknown as FitSummary;
  const Q = beadQuality(r)!;
  check(Q.cutoffKind === 'reference' && Q.q.length === nb && Q.firstRejected.length === 1 && Q.firstRejected[0].label!.startsWith('bead 5'), `bead 5 above the reference: ${Q.firstRejected.map((x) => x.label)}`);
  check(Q.best.length === 3 && Q.lastAccepted.length === 2 && Q.best[0].data.length === R2 && Q.best[0].label!.includes('plane 3 of 5'), `bands, sharpest plane: best ${Q.best.length}, last accepted ${Q.lastAccepted.length}, ${Q.best[0].label}`);
  const raw = await makeFitReport({ title: 'Bead fit quality range (test)', quality: Q }, { compress: false }), s = latin1(raw);
  check(s.includes('Just above the reference line') && s.includes('not a rejection'), 'reference labelled');
  artifact('quality_range_beads.pdf', raw);
  return `q ${Array.from(Q.q, (v) => v.toFixed(2)).join(' ')}, reference ${Q.cutoff!.toFixed(3)}`;
});

test('pdf: quality range of an in-situ estimate (demo movie, worker data -> report)', async () => {
  const { device } = await getDevice();
  const p = { ...DEMO, T: 300, nsites: 600 };
  const d = await simulateDemo(device, p);
  const form = {
    optics: d.optics, gain: p.gain, offset: p.offset, readnoise_e: p.readnoise, focus_um: p.focus_um, zlo: p.zlo, zhi: p.zhi,
    roisize: 15, threshold: 6, isolation: 0, chunk: 50, maxFrames: 0, npupil: 32, oversample: 2, noll: DEFAULT_NOLL,
    zcoef: DEFAULT_NOLL.map((j) => (j === 6 ? 0.08 : 0)), nspots: 800, nheldout: 200, iters: 8, tile: 16,
  };
  const r = await estimateInsitu(device, new ArrayMovie(d.adu, d.T, d.H, d.W), form, new StopToken(), () => {});
  const Q = r.quality!;
  check(!!Q && Q.q.length === r.n_train && Q.cutoff === r.deviance!.threshold && Q.cutoffKind === 'rejection', `q of the ${r.n_train} training spots, cutoff = threshold`);
  let nRej = 0; for (const v of Q.q) if (!(v < Q.cutoff!)) nRej++;
  check(nRej === r.n_rejected, `rejected ${nRej} = n_rejected ${r.n_rejected}`);
  const shown = [...Q.best, ...Q.lastAccepted, ...Q.firstRejected];
  check(shown.length <= 9 && Q.firstRejected.length === Math.min(3, nRej) && Q.firstRejected.every((x) => x.q >= Q.cutoff!) && Q.lastAccepted.every((x) => x.q < Q.cutoff!), 'bands');
  // the model is the first-pass fit that gave q: the Poisson deviance of data vs model reproduces q
  const rv = form.readnoise_e ** 2;
  let md = 0;
  for (const x of shown) {
    let s = 0;
    for (let i = 0; i < x.data.length; i++) { const dd = x.data[i] + rv, mu = x.model[i] + rv; s += 2 * ((dd > 0 ? dd * Math.log(Math.max(dd, 1e-9) / Math.max(mu, 1e-6)) : 0) - (dd - mu)); }
    md = Math.max(md, Math.abs(s / x.data.length - x.q) / x.q);
  }
  check(md < 0.02, `deviance of the shown data / model vs q: max rel. difference ${md.toExponential(1)}`);
  const bytes = await makeFitReport(insituReportData(r, { form, results: [['spots', String(r.n_spots)]], source: 'simulated demo (test)' }));
  const c = await checkPdf(bytes);
  artifact('quality_range_insitu.pdf', bytes);
  return `${r.n_train} training spots, ${nRej} rejected (threshold ${Q.cutoff!.toFixed(3)}); deviance check ${md.toExponential(1)}; report ${c.pages} pages`;
});
