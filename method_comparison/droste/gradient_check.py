"""Check all 13 physical and cached-spline score components at fixed emitter parameters."""
import argparse
import json
from pathlib import Path

import torch
from torch_psf import PSF
from torch_psf.estimation import nat_field, nat_polynomial
from torch_psf.estimation.knots import knot_gradient
from torch_psf.estimation.nat import NAT_MODES
from torch_psf.fit import poisson_nll


def run(calibration, fitted):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit = fitted[0]["fit"]
    base = PSF.load(calibration, roisize=15, angle=0, device=device)
    base.z_range = (0., .4)
    with torch.no_grad():
        base.zcoef.zero_()
    nat = nat_field(base, fit["gamma"], fit["norm"])
    vec = nat.base
    cols = [vec.zidx.index(j) for j in NAT_MODES]
    tiled = nat.to_tiled(tile=30., kernel="bspline", integrated=True, oversample=3, dz_um=.025, xy_limit=3.)
    tile = tiled.spline(0)
    center = tiled.centers[0].tolist()
    big = nat.at(*center).resized(tile.Pl)
    g = torch.tensor(fit["gamma"], dtype=vec.dtype, device=device, requires_grad=True)
    pos = g.new_tensor([[.15, -.25, z] for z in (.02, .1, .2, .3, .38)])
    theta = torch.cat((pos, g.new_tensor([[2000., 8.]]).expand(len(pos), -1)), dim=1)

    def pupil(point):
        c = (nat.field.features([point]) @ nat_polynomial(g).T)[0]
        full = torch.zeros_like(vec.zcoef)
        full[cols] = c
        return full

    c_center = pupil(center)
    knots_ad = tile.sample(big, params={"zcoef": c_center}, differentiable=True, chunk=2)
    mu_spline = 2000 * tile.render_splines(pos, knots=knots_ad) + 8
    rng = torch.Generator(device=device).manual_seed(1809)
    out = []
    for offset in (0, 8):
        xy = [v + offset for v in center]
        mu_direct = 2000 * vec(pos, vec.focus_um, zcoef_offset=pupil(xy)-vec.zcoef.detach()) + 8
        data = torch.poisson(mu_direct.detach(), generator=rng)
        direct, = torch.autograd.grad(poisson_nll(data, mu_direct).sum(), g)
        full_ad, = torch.autograd.grad(poisson_nll(data, mu_spline).sum(), g, retain_graph=True)
        adj, _ = knot_gradient(tile, data, theta)
        grad_z = tile.vjp_vectorial(big, None, adj)["zcoef"]
        cache, = torch.autograd.grad(c_center, g, grad_z, retain_graph=True)
        def rel(a,b):
            return float((a-b).norm()/b.norm())
        out.append(dict(offset_px_per_axis=offset, cached_vs_spline_autograd=rel(cache, full_ad),
                        cached_vs_direct=rel(cache, direct), direct=direct.tolist(), cached=cache.tolist(),
                        spline_autograd=full_ad.tolist()))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("calibration", "fitted", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.calibration, json.loads(a.fitted.read_text())), indent=2) + "\n")
