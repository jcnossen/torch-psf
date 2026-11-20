"""Check the pattern localizer with the TRUE PSF: errors against the simulated truth."""
import sys, runpy, torch
sys.argv = ['run_pattern.py', sys.argv[1] if len(sys.argv) > 1 else '0.9', '']       # no scenarios: definitions only
ns = runpy.run_path('study/insitu/run_pattern.py')
C, F = ns['C'], ns['F']
for scn in 'A':
    spl = F.new_spline()
    spl.knots = spl.sample(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
    d6, corner, tt = ns['simulate'](scn)
    loc = ns['PatternLoc'](d6, corner)
    for rep in range(3):
        loc(spl, None, None)
        e = loc.th - tt
        print(f'M {ns["M"]} call {rep}: rms x {e[:,0].std()*108:.2f} y {e[:,1].std()*108:.2f} nm z {e[:,2].std()*1000:.1f}  '
              f'|y err| > 50 nm: {(e[:,1].abs()*108 > 50).float().mean():.3f}  |x err| > 50 nm: {(e[:,0].abs()*108 > 50).float().mean():.3f}  '
              f'median |y| {e[:,1].abs().median()*108:.2f}  N ratio {(loc.th[:,3]/tt[:,3]).median():.3f}')
    z = tt[:, 2]
    for lo in (0.0, 0.2, 0.4, 0.6):
        m = (z >= lo) & (z < lo + 0.2)
        print(f'   z {lo:.1f}: mean y err {e[m,1].mean()*108:.2f} nm, frac |y err| near one pitch (170-270 nm) {((e[m,1].abs()*108 > 170) & (e[m,1].abs()*108 < 270)).float().mean():.3f}')
