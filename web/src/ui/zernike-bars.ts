import { nollModeName } from '../psf/vectorial';

/** Zernike coefficients (waves rms) by Noll index; several sets side by side (e.g. start, fitted, truth). */
export function drawZernikeSets(cv: HTMLCanvasElement, noll: number[], sets: { z: number[]; color: string; label: string }[]) {
  const dpr = window.devicePixelRatio || 1;
  const cw = cv.clientWidth || cv.width, ch = cv.clientHeight || cv.height;
  cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
  const ctx = cv.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cw, H = ch, pad = 38;
  ctx.clearRect(0, 0, W, H);
  const m = Math.max(0.02, ...sets.flatMap((s) => s.z.map(Math.abs)));
  const bw = (W - pad - 6) / noll.length, y0 = H / 2 + 6, k = sets.length;
  cv.onpointermove = (event) => {
    const rect = cv.getBoundingClientRect();
    const x = (event.clientX - rect.left) * W / rect.width;
    const y = (event.clientY - rect.top) * H / rect.height;
    const i = Math.floor((x - pad) / bw);
    if (i < 0 || i >= noll.length || y < 20) { cv.removeAttribute('title'); return; }
    cv.title = `${nollModeName(noll[i])} (Noll ${noll[i]})\n` + sets.map((s) => `${s.label}: ${s.z[i].toFixed(4)} waves rms`).join('\n');
  };
  cv.onpointerleave = () => cv.removeAttribute('title');
  ctx.strokeStyle = '#aeb6bf'; ctx.beginPath(); ctx.moveTo(pad, y0 + .5); ctx.lineTo(W - 4, y0 + .5); ctx.stroke();
  ctx.font = "10px Inter, 'Segoe UI', system-ui, sans-serif"; ctx.fillStyle = '#8c959f';
  ctx.fillText(`+${m.toFixed(3)}`, 0, 30); ctx.fillText(`-${m.toFixed(3)}`, 0, H - 16);
  const amp = H / 2 - 24;
  noll.forEach((j, i) => {
    sets.forEach((s, q) => {
      const h = (s.z[i] / m) * amp;
      ctx.fillStyle = s.color;
      ctx.fillRect(pad + i * bw + 2 + q * (bw - 4) / k, h >= 0 ? y0 - h : y0, Math.max(1, (bw - 4) / k - 1), Math.abs(h));
    });
    ctx.fillStyle = '#8c959f';
    ctx.fillText(String(j), pad + i * bw + bw / 2 - 5, H - 2);
  });
  let x = pad + 2;
  ctx.font = "11px Inter, 'Segoe UI', system-ui, sans-serif";
  for (const s of sets) { ctx.fillStyle = s.color; ctx.fillRect(x, 4, 10, 10); ctx.fillStyle = '#656d76'; ctx.fillText(s.label, x + 14, 13); x += 24 + ctx.measureText(s.label).width; }
}
