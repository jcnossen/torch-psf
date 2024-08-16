
from torch_convspline import eval_image_spline3D, eval_image_spline3D_deriv

from torch import nn
import torch
from torch import Tensor
import numpy as np

@torch.jit.script
def poisson_crlb(mu:Tensor, jac:Tensor):
    """
    Compute crlb from expected value and per pixel derivatives, where pixel noise is assumed have Poisson distribution
    mu: [N, H, W]
    jac: [N, H,W, coords]
    """
    fisher = torch.matmul(jac[...,None], jac[...,None,:])
    fisher = fisher / mu[...,None,None]  
    fisher = fisher.flatten(1, len(mu.shape)-1).sum(1)

    fisher_inv, info = torch.linalg.inv_ex(fisher, check_errors=False)

    return torch.sqrt( torch.diagonal(fisher_inv, dim1=1, dim2=2) )


class SplinePSF(nn.Module):
    """
    Input params:
    [x,y,z,intensity,bg]

    0,0,0 creates a 3D PSF sampled from the center of the zstack
    """
    def __init__(self, zstack, output_shape, zstep=1, device=None):
        super().__init__()

        self.device = device

        if type(zstack) is np.ndarray:
            zstack = torch.from_numpy(zstack.astype(np.float32)).to(device)

        self.zstack = zstack
        self.output_shape = output_shape
        self.zstep = zstep # um per layer
    
    def crlb(self, params):
        r = self.forward_with_deriv(params)
        intensities = params[:, 3]
        bg = params[:, 4]

        # calculate the CRLB
        mu = intensities[:, None, None] * r[:,0] + bg[:, None, None]
        jac = r[:, 1:].permute((0, 2, 3, 1)) * intensities[:, None, None, None]
        crlb = poisson_crlb(mu, jac)

        return crlb
    
    def _prepare(self, params):
        """ params order is x,y,z """
        return torch.stack(
            [
                params[:, 2] / self.zstep + self.zstack.shape[0]//2,
                self.zstack.shape[1]//2 - params[:, 1],
                self.zstack.shape[2]//2 - params[:, 0]
            ],
            dim=1
        )

    def forward(self, params):
        zyx = self._prepare(params)

        intensities = params[:, 3]
        bg = params[:, 4]

        # repeat zstack
        #zstack = self.zstack[None].repeat(len(params), 1, 1, 1)

        # expand zstack
        zstack = self.zstack[None].expand(len(params), -1, -1, -1)

        return eval_image_spline3D(zyx, self.output_shape, zstack) * intensities[:, None, None] + bg[:, None, None]

    def forward_with_deriv(self, xyz):
        """
        returns [batchsize, 4, output_shape[0], output_shape[1]],
        where the 4 channels are expected value, and the derivatives in x, y, and z
        """
        zyx = self._prepare(xyz)

        zstack = self.zstack[None].expand(len(xyz), -1, -1, -1)

        r = eval_image_spline3D_deriv(zyx, self.output_shape, zstack, use_fast_slice=False)
        return r[:, [0, 3, 2, 1]]
    

