import { exampleMovie } from '../src/dense/field-example';
import { TiffMovie } from '../src/io/tiff';
import { check, test } from './harness';

test('Dense example frame count follows the memory budget', async () => {
  const root = document.createElement('section'), anchor = document.createElement('p');
  root.append(anchor);
  const files: File[] = [];
  exampleMovie(anchor, () => ({
    optics: {pixel_um:.108,NA:1.49,n_imm:1.518,n_med:1.33,wavelength_um:.68},
    diameterNm:40,stepNm:20,offset:0,gain:1
  }), f => files.push(f));
  const frames = root.querySelector<HTMLInputElement>('.frames')!;
  const button = root.querySelector('button')!;
  root.querySelector<HTMLInputElement>('.density')!.value = '.01';
  root.querySelector<HTMLInputElement>('.strength')!.value = '0';
  for (const n of [1,101]) {
    frames.value = String(n);
    await button.onclick!.call(button,new PointerEvent('click'));
    const movie = await TiffMovie.open([files[files.length-1]]);
    check(movie.n === n, 'TIFF frame count '+movie.n);
    const last = await movie.read(n-1,n);
    check(last.length===160*160 && last.every(Number.isFinite), 'read final TIFF frame');
    if (n>1) {
      const first = await movie.read(0,1);
      check(first.some((v,i)=>v!==last[i]),'distinct snapshots of reused render buffer');
    }
  }
  frames.value = '1000000000';
  await button.onclick!.call(button,new PointerEvent('click'));
  check(files.length===2 && root.textContent!.includes('budget allows up to'), 'over-budget request rejected');
  URL.revokeObjectURL(root.querySelector('a')!.href);
});
