import { test, check } from './harness';
import { getDevice } from '../src/gpu/device';
import { fieldSplineRenderer } from '../src/dense/field-spline';
import ref from './field_spline_reference.json';

test('Dense movie: bilinear finite-bead B-spline vs PyTorch', async () => {
  const { device } = await getDevice();
  const model = { optics: ref.optics, roisize: 7, npupil: 16, oversample: 2, noll: ref.modes,
    zcoef: ref.coef.map(row=>row[0]), focus_um: .05, z_range_um: null };
  const renderer = await fieldSplineRenderer(device,model,{type:'zernike_poly',order:2,zernike_noll:ref.modes,
    feats_norm:{cx:40,cy:40,sx:40,sy:40},coef:ref.coef},100,7,[-.04,.08]);
  let parity = 0, physical = 0;
  try {
    for (const [t,drift] of ref.drift.entries()) {
      await renderer.render(ref.xy,drift,(j,x,y,stamp)=>{
        check(x===Math.round(ref.xy[j][0]+drift[0]) && y===Math.round(ref.xy[j][1]+drift[1]),'bead centre');
        const expected = ref.frames[t][j].flat(), direct = ref.direct[t][j].flat();
        const peak = Math.max(...expected), directPeak = Math.max(...direct);
        stamp.forEach((v,k)=>{
          parity = Math.max(parity,Math.abs(v-expected[k])/peak);
          physical = Math.max(physical,Math.abs(v-direct[k])/directPeak);
        });
      });
    }
  } finally { renderer.destroy(); }
  check(parity < 5e-5, 'WebGPU vs PyTorch spline '+parity);
  check(physical < .04, 'spline vs direct vectorial '+physical);
  return 'max peak-normalized pixel differences: PyTorch spline '+parity+', direct optics '+physical;
});
