"""Finite-bead movie rendering on a bilinear camera field and cubic focus/XY spline."""
import math

import torch

from torch_psf.optics import pixel_xy
from torch_psf.spline import SplinePSF, bspline_prefilter


class FieldBeadSpline:
    """Cache a FieldPSF for rendering a bead movie with a fixed optical calibration.

    diameter_um describes a solid fluorescent bead touching the coverslip. The third
    local coordinate is objective displacement from bead-centre focus (micrometres).
    Field nodes span the full field.norm box, including its edges.
    """

    @torch.no_grad()
    def __init__(self, field_psf, diameter_um, focus_range_um, field_nodes=5, dz_um=.04):
        self.lo, self.hi = map(float, focus_range_um)
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo <= self.hi
                and math.isfinite(dz_um) and dz_um > 0 and isinstance(field_nodes, int) and field_nodes >= 2
                and math.isfinite(diameter_um) and diameter_um >= 0):
            raise ValueError("invalid focus interval, bead diameter, or spline sampling")
        roi = field_psf.roisize
        if roi % 2 == 0:
            raise ValueError('expected an odd stamp size')
        padded = roi + 4
        self.n = field_nodes
        self.norm = field_psf.field.norm
        cx, cy, sx, sy = self.norm
        dev = field_psf.dev
        self.device = dev
        radius = diameter_um / 2
        a = radius * math.sqrt(.6)
        px, py = pixel_xy(field_psf.pixel_um)
        sphere = [[a/px,0,0],[-a/px,0,0],[0,a/py,0],[0,-a/py,0],[0,0,a],[0,0,-a]]
        nodes = torch.tensor(sphere if radius else [[0.,0.,0.]], device=dev, dtype=torch.float32)
        nodes[:, 2] += radius
        shifts = torch.tensor([[x,y,0.] for y in (-.25,.25) for x in (-.25,.25)], device=dev)
        points = (shifts[:, None]+nodes).reshape(-1,3)
        nz = math.ceil((self.hi-self.lo)/dz_um) + 5
        z0 = self.lo-2*dz_um
        volumes = []
        for j in range(field_nodes):
            for i in range(field_nodes):
                x = cx + sx*(2*i/(field_nodes-1)-1)
                y = cy + sy*(2*j/(field_nodes-1)-1)
                model = field_psf.at(x,y).resized(padded)
                planes = []
                for k in range(nz):
                    frame = model.render(points, focus=radius+z0+k*dz_um)
                    image = frame.reshape(2,2,len(nodes),padded,padded).mean(2)
                    planes.append(image.flip((0,1)).permute(2,0,3,1).reshape(2*padded,2*padded))
                volumes.append(bspline_prefilter(torch.stack(planes)))
        self.spline = SplinePSF(roi,(self.lo,self.hi),2,dz_um,device=dev,kernel='bspline',integrated=False,
                                knots=torch.stack(volumes),knot_z0_um=z0,knot_u0_px=-(2*padded-1)/4)

    def render(self, xy, focus_um):
        """One stamp per absolute bead position [N,2] px; scalar or per-bead focus drift in µm.

        The pixel centre of each returned stamp is floor(xy + 0.5).
        """
        xy = torch.as_tensor(xy,device=self.device,dtype=self.spline.dtype).reshape(-1,2)
        focus = torch.as_tensor(focus_um,device=self.device,dtype=xy.dtype).expand(len(xy))
        if ((focus < self.lo-1e-8) | (focus > self.hi+1e-8)).any():
            raise ValueError("focus outside cached field spline interval")
        if not len(xy):
            return xy.new_empty((0,self.spline.roisize,self.spline.roisize))
        pos = torch.column_stack((xy-(xy+.5).floor(),focus))
        cx, cy, sx, sy = self.norm
        grid = (((xy-xy.new_tensor([cx,cy]))/xy.new_tensor([sx,sy])+1)*((self.n-1)/2)).clamp(0,self.n-1)
        gx, gy = grid.floor().long().T
        fx, fy = (grid-grid.floor()).T
        hx, hy = (gx+1).clamp_max(self.n-1), (gy+1).clamp_max(self.n-1)
        indices = torch.stack((gy*self.n+gx,gy*self.n+hx,hy*self.n+gx,hy*self.n+hx),1)
        weights = torch.stack(((1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy),1)
        images = self.spline.evaluate_splines(pos[:,None],indices)
        return (images * weights[:,:,None,None]).sum(1).clamp_min(0)
