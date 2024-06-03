import torch
from torch import erf
import math

def gaussian2D(params, width: int, height: int):
    device = params.device

    sqrt2 = math.sqrt(2)

    # Unpack parameters
    x0 = params[:, 0]
    y0 = params[:, 1]
    sigma_x = params[:, 4]
    sigma_y = params[:, 5]

    x = torch.arange(0, width, dtype=torch.float32, device=device)
    y = torch.arange(0, height, dtype=torch.float32, device=device)

    # Use [None] instead of meshgrid
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    xx = xx[None]
    yy = yy[None]
    
    # Calculate error function
    erf_x1 = erf((xx - x0[:,None,None] + 0.5) / (sqrt2 * sigma_x[:,None,None]))
    erf_x2 = erf((xx - x0[:,None,None] - 0.5) / (sqrt2 * sigma_x[:,None,None]))
    erf_y1 = erf((yy - y0[:,None,None] + 0.5) / (sqrt2 * sigma_y[:,None,None]))
    erf_y2 = erf((yy - y0[:,None,None] - 0.5) / (sqrt2 * sigma_y[:,None,None]))

    # Integrate Gaussian over pixel
    gaussian = 0.25 * (erf_x1 - erf_x2) * (erf_y1 - erf_y2)
    return gaussian

