// preview.png of a psf.zip (docs/PSF_FORMAT.md; torch_psf.psfzip.psf_figure): one row per channel, five xy slices
// through z (each normalized to its own maximum) and an xz section through the centre row (z up). Drawn on an
// OffscreenCanvas (UI thread or worker), encoded with convertToBlob('image/png').
import type { ZStack } from '../psf/zstack';

function gray(ctx: OffscreenCanvasRenderingContext2D, a: Float32Array, w: number, h: number, x0: number, y0: number, W: number, H: number, flipY = false) {
  let mx = 0;
  for (let i = 0; i < w * h; i++) if (a[i] > mx) mx = a[i];
  const im = ctx.createImageData(w, h);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const v = Math.round(255 * Math.max(0, a[r * w + c]) / (mx || 1)), o = ((flipY ? h - 1 - r : r) * w + c) * 4;
    im.data[o] = im.data[o + 1] = im.data[o + 2] = v; im.data[o + 3] = 255;
  }
  const tmp = new OffscreenCanvas(w, h);
  tmp.getContext('2d')!.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, x0, y0, W, H);
}

export async function previewPng(zs: ZStack, names: string[], title: string): Promise<Uint8Array> {
  const { Z, C, H, W } = zs;
  const zi = [...new Set(Array.from({ length: Math.min(5, Z) }, (_, i) => Math.round(Z === 1 ? 0 : (i * (Z - 1)) / Math.max(1, Math.min(5, Z) - 1))))];
  const s = 120, gap = 10, left = 46, top = 34, rowH = s + 34;
  const hasXz = Z > 1, xzW = hasXz ? 2 * s : 0;
  const width = left + zi.length * (s + gap) + (hasXz ? 30 + xzW : 0) + 16, height = top + C * rowH + 8;
  const cv = new OffscreenCanvas(width, height), ctx = cv.getContext('2d')!;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#222222'; ctx.font = '14px sans-serif'; ctx.fillText(title, left, 20);
  ctx.font = '11px sans-serif';
  const plane = (z: number, c: number) => zs.stack.subarray(((z * C) + c) * H * W, ((z * C) + c + 1) * H * W);
  for (let c = 0; c < C; c++) {
    const y0 = top + c * rowH + 14;
    ctx.fillStyle = '#222222'; ctx.save(); ctx.translate(16, y0 + s / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(names[c] ?? `ch${c}`, -12, 0); ctx.restore();
    zi.forEach((z, j) => {
      const x0 = left + j * (s + gap);
      gray(ctx, plane(z, c), W, H, x0, y0, s, s);
      ctx.fillStyle = '#52514e'; ctx.fillText(`z ${zs.z_um[z].toFixed(2)} um`, x0 + 22, y0 - 4);
    });
    if (hasXz) {
      const xz = new Float32Array(Z * W), row = H >> 1;
      for (let z = 0; z < Z; z++) xz.set(plane(z, c).subarray(row * W, (row + 1) * W), z * W);
      const x0 = left + zi.length * (s + gap) + 30;
      gray(ctx, xz, W, Z, x0, y0, xzW, s, true);
      ctx.fillStyle = '#52514e';
      ctx.fillText('xz section (y = 0), z up', x0 + 40, y0 - 4);
      ctx.fillText(`${zs.z_um[Z - 1].toFixed(2)}`, x0 - 28, y0 + 10); ctx.fillText(`${zs.z_um[0].toFixed(2)}`, x0 - 28, y0 + s);
    }
  }
  const blob = await cv.convertToBlob({ type: 'image/png' });
  return new Uint8Array(await blob.arrayBuffer());
}
