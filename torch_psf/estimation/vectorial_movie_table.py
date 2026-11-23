"""Tabulated finite-bead vectorial PSF with differentiable camera-coordinate interpolation."""
import numpy as np
import torch
from torch.nn import functional as F
from .bead_movie import FiniteBeadRenderer


class VectorialMovieTable:
    """Tabulate four optical coordinates, retaining the six sphere nodes for simulation."""
    def __init__(self, optics, radius=7):
        self.optic = FiniteBeadRenderer(dict(optics), radius)
        self.radius = radius
        self.side = 2*radius+1
        self.device = self.optic.device
        span = optics.get('focus_span_um', .045)
        calibrated = optics.get('calibrated_psf') is not None
        self.focus = np.linspace(-span, span, 2 * min(50, max(3, int(np.ceil(span / .025)))) + 1)
        self.wave_limit = .18
        grid = np.linspace(-.52,.52,7)
        a,y,x = np.meshgrid(([0.] if calibrated else np.linspace(-self.wave_limit,self.wave_limit,15)),grid,grid,indexing="ij")
        offsets = self.tensor(np.stack([x.ravel(),y.ravel()],axis=1))
        wave = self.tensor(a.ravel())
        banks = []
        with torch.no_grad():
            for z in self.focus:
                self.optic.cfg["focus_um"] = optics["focus_um"]+float(z)
                images = self.optic.render(offsets,wave)
                bank = images.reshape(*a.shape,6*self.side**2).permute(3,0,1,2).contiguous()
                banks.append(bank)
        self.optic.cfg["focus_um"] = optics["focus_um"]
        self.nodes_bank = torch.stack(banks)
        self.mean_bank = self.nodes_bank.reshape(len(banks),6,self.side**2,*a.shape).mean(1).contiguous()

    def tensor(self,value):
        return self.optic.tensor(value)

    def kernels(self, offset, waves, focus, *, nodes=False):
        """All emitters in one exposure share a focus offset in um."""
        bank = self.nodes_bank if nodes else self.mean_bank
        where = ((float(focus)-self.focus[0])/(self.focus[1]-self.focus[0]))
        if where < -1e-4 or where > len(bank)-1+1e-4:
            raise ValueError("Focus outside vectorial table")
        where = np.clip(where,0,len(bank)-1)
        low = min(int(where),len(bank)-2)
        mix = float(where-low)
        table = torch.lerp(bank[low],bank[low+1],mix) if mix else bank[low]
        grid = torch.cat([offset/.52,waves[:,None]/self.wave_limit],1)
        vals = F.grid_sample(table[None],grid.reshape(1,1,1,-1,3),mode="bilinear",
                             padding_mode="border",align_corners=True)[0,:,0,0].T
        return vals.reshape(len(grid),-1,self.side,self.side) if nodes else vals
