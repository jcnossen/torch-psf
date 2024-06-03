
from torch_convspline import eval_image_spline3D, eval_image_spline3D_deriv

from torch import nn
import torch

class SplinePSF(nn.Module):
    """
    Input params:
    [x,y,z,intensity,bg]

    0,0,0 creates a 3D PSF sampled from the center of the zstack
    """
    def __init__(self, zstack, output_shape):
        super().__init__()

        self.zstack = zstack
        self.output_shape = output_shape
    
    def crlb(self, params):
        r = eval_image_spline3D_deriv(params, self.output_shape, self.zstack[None])

    def forward(self, params):
        x = self.zstack.shape[2]//2 - params[:, 0]
        y = self.zstack.shape[1]//2 - params[:, 1]
        z = params[:, 2]# + self.zstack.shape[0]//2
        zyx = torch.stack([z, y, x], dim=1)

        intensities = params[:, 3]
        bg = params[:, 4]

        # repeat zstack
        #zstack = self.zstack[None].repeat(len(params), 1, 1, 1)

        # expand zstack
        zstack = self.zstack[None].expand(len(params), -1, -1, -1)

        return eval_image_spline3D(zyx, self.output_shape, zstack)  * intensities[:, None, None] + bg[:, None, None]

