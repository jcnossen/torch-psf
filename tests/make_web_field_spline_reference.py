"""Independent PyTorch reference for the dense finite-bead WebGPU field spline."""
import json
import math
from pathlib import Path

import torch
from torch_psf import FieldPSF, FieldZernikes, VectorialPSF
from torch_psf.estimation import FieldBeadSpline


@torch.no_grad()
def reference():
    roi, padded = 7, 11
    optics = dict(NA=1.45, n_med=1.33, n_imm=1.518, pixel_um=.108, wavelength_um=.68)
    modes = [5, 6, 8]
    coef = torch.tensor([[.04, 0, .008, .065, -.006, 0], [.08, -.07, 0, 0, 0, .006], [-.025, .015, 0, -.02, 0, 0]])
    xy = torch.tensor([[0., 0.], [12.25, 69.7], [39.51, 40.5], [55.8, 32.4], [83.4, -3.7]])
    drift = [[0., 0., -40.], [.2, -.3, 17.], [-.1, .15, 80.]]
    radius, lo, hi = .05, -.04, .08
    a = radius * math.sqrt(.6)
    nodes = torch.tensor([[a/.108,0,0],[-a/.108,0,0],[0,a/.108,0],[0,-a/.108,0],[0,0,a],[0,0,-a]])
    nodes[:, 2] += radius
    vec = VectorialPSF(padded, **optics, zernike_indices=modes, npupil=16, oversample=2,
                       pixel_integration='midpoint', device='cpu')
    def pupil(u, v):
        vec.set_zernike(dict(zip(modes, (coef @ torch.tensor([1,v,v*v,u,u*v,u*u], dtype=coef.dtype)).tolist())))
    pupil(0,0)
    field = FieldZernikes(modes, 2, norm=(40,40,40,40), coef=coef[:,1:], device='cpu')
    spline = FieldBeadSpline(FieldPSF(vec.resized(roi),field),2*radius,(lo,hi))
    frames, direct = [], []
    for dx,dy,dz in drift:
        field_xy = xy + torch.tensor([dx,dy])
        offsets = field_xy - (field_xy+.5).floor()
        frames.append(spline.render(field_xy,dz/1000).tolist())
        exact = []
        for offset, (u,v) in zip(offsets, (field_xy/40-1).clamp(-1,1)):
            pupil(u,v)
            p = nodes.clone()
            p[:,:2] += offset
            exact.append(vec.render(p,focus=radius+dz/1000).mean(0)[2:-2,2:-2])
        direct.append(torch.stack(exact).tolist())
    return dict(optics=optics,modes=modes,coef=coef.tolist(),xy=xy.tolist(),drift=drift,frames=frames,direct=direct)


if __name__ == '__main__':
    torch.set_num_threads(1)
    path = Path(__file__).parents[1] / 'web/test/field_spline_reference.json'
    path.write_text(json.dumps(reference(), separators=(',', ':'))+'\n')
