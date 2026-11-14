// Minimal in-browser test harness: tests register with test(), results go to window.__testResults.
export interface TestResult { name: string; pass: boolean; ms: number; msg?: string; }
type Fn = () => Promise<string | void>;
const tests: [string, Fn][] = [];
export function test(name: string, fn: Fn) { tests.push([name, fn]); }

export class Fail extends Error {}
export function check(cond: boolean, msg: string) { if (!cond) throw new Fail(msg); }

export async function runAll(gpu: string) {
  const only = new URLSearchParams(location.search).get('only') || '';
  const log = document.getElementById('log')!;
  const res: TestResult[] = [];
  log.textContent = '';
  for (const [name, fn] of tests) {
    if (only && !name.includes(only)) continue;
    const t0 = performance.now();
    let r: TestResult;
    try {
      const msg = await fn();
      r = { name, pass: true, ms: performance.now() - t0, msg: msg || '' };
    } catch (e) {
      r = { name, pass: false, ms: performance.now() - t0, msg: String((e as Error).message ?? e) };
      if (!(e instanceof Fail)) console.error((e as Error).stack);
    }
    res.push(r);
    log.textContent += `${r.pass ? 'PASS' : 'FAIL'} ${name} (${r.ms.toFixed(0)} ms) ${r.msg}\n`;
    console.log(`# ${r.pass ? 'PASS' : 'FAIL'} ${name} ${r.msg}`);
  }
  (window as any).__testResults = { gpu, tests: res };
  (window as any).__testsDone = true;
}

export async function fetchBuf(path: string): Promise<ArrayBuffer> {
  const r = await fetch('/testdata/' + path);
  if (!r.ok) throw new Error('fetch ' + path + ': ' + r.status);
  return r.arrayBuffer();
}
export async function fetchJson(path: string): Promise<any> { return JSON.parse(new TextDecoder().decode(await fetchBuf(path))); }

export function maxAbs(a: ArrayLike<number>, off = 0, n = a.length) {
  let m = 0; for (let i = off; i < off + n; i++) m = Math.max(m, Math.abs(a[i])); return m;
}
export function maxAbsDiff(a: ArrayLike<number>, ao: number, b: ArrayLike<number>, bo: number, n: number) {
  let m = 0; for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[ao + i] - b[bo + i])); return m;
}
