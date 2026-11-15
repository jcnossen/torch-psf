"""Sensitivity of the profiled CRLB to the rank cut-off of the density projector (base case, P = 12 / 16)."""
from common import *
import fisher_sweep as fs
from torch_psf.sofi import fisher as F
orig = F._proj_fisher
rows = {}
for tol in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
    F._proj_fisher = lambda Aw, Bw, _t, tol=tol: orig(Aw, Bw, tol)
    r = fs.run(fs.config(), [12, 16])
    rows[tol] = r
    print(f"tol {tol:.0e} rank {r['patch'][16]['rank']:5d}  arb " + fs.fmt(r), flush=True)
save_json(rows, 'fisher_tolcheck.json')
