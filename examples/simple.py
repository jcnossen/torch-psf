
import torch

# Create a 2D Gaussian zstack with some sort of astigmatism
from torch_convspline.gaussian import gaussian2D
from torch_convspline.psf import SplinePSF


W = 32
D = 20
A = 0.01
focal_x = D//2+3
focal_y = D//2-3

params = torch.zeros((D, 6))
params[:, :2] = W/2
params[:, 2] = 1
params[:, 3] = 0
params[:, 4] = 1 + (torch.arange(D)-focal_x)**2 * A # sigma_x
params[:, 5] = 1 + (torch.arange(D)-focal_y)**2 * A # sigma_y

zstack = gaussian2D(params, W, W)

roisize = 16
psf = SplinePSF(zstack, (roisize,roisize))
evals = torch.zeros((100, 5))
evals[:,:2] = roisize/2
evals[:,2] = torch.linspace(0, D-4, len(evals))
evals[:,3] = 500
evals[:,4] = 3

#zscan = psf.forward(evals)
#zscan = torch.poisson(zscan)

#deriv = psf.forward_with_deriv(evals)

#from smlmtorch.ui.array_view_pyqt import image_view
#image_view(deriv)

print(psf.crlb(evals))

