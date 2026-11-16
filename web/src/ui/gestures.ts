// Two-finger pinch / pan on a canvas (pointer events, for touch screens; the mouse wheel zooms separately).
// The views keep their transform as scale = nm per device pixel and (cx, cy) = view centre; trackPinch reports, per
// move of a two-pointer gesture, the factor f for that scale (f < 1 zooms in), the midpoint (mx, my) in device px
// relative to the canvas centre, and the midpoint shift (dx, dy) in device px. Single-pointer handlers of the view
// check pinching() and follow only their own pointer, so lifting one finger does not make the view jump.

export interface Pinch { pinching(): boolean }

export function trackPinch(cv: HTMLCanvasElement, onPinch: (f: number, mx: number, my: number, dx: number, dy: number) => void): Pinch {
  const pts = new Map<number, { x: number; y: number }>();
  let prev: { d: number; x: number; y: number } | null = null;
  const pair = () => {
    const [a, b] = [...pts.values()];
    return { d: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  };
  cv.addEventListener('pointerdown', (e) => {
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    prev = pts.size === 2 ? pair() : null;
  });
  cv.addEventListener('pointermove', (e) => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size !== 2 || !prev) return;
    const s = pair(), r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const f = prev.d / Math.max(s.d, 1);
    onPinch(f, (s.x - r.left) * dpr - cv.width / 2, (s.y - r.top) * dpr - cv.height / 2, (s.x - prev.x) * dpr, (s.y - prev.y) * dpr);
    prev = s;
  });
  const up = (e: PointerEvent) => { pts.delete(e.pointerId); prev = pts.size === 2 ? pair() : null; };
  cv.addEventListener('pointerup', up);
  cv.addEventListener('pointercancel', up);
  return { pinching: () => pts.size >= 2 };
}
