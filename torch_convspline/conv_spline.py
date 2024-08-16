"""
3D catmull rom splines evaluated using conv2d pytorch operations

author: Jelmer Cnossen
license: MIT
"""
import os
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from typing import Tuple
#from torch.func import jacrev, jacfwd
#from block_copy import BlockCopy



def _S(X):
    return X.detach().cpu().numpy()

@torch.jit.script
def geometry_matrix(device: torch.device, tau:float=0.5):
    return torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [-tau, 0.0, tau, 0.0],
        [2*tau, tau-3,3-2*tau,-tau],
        [-tau, 2-tau,tau-2,tau]
    ], device=device, dtype=torch.float32)

@torch.jit.script
def spline_weights(x, tau:float=0.5):
    G = geometry_matrix(device=x.device, tau=tau)
    w = x[:,None]**torch.arange(4, device=x.device)[None] @ G
    return w

@torch.jit.script
def spline_weights_deriv(x, tau:float=0.5):
    G = geometry_matrix(device=x.device, tau=tau)
    exp = torch.arange(4, device=x.device)[None]
    w = (exp * x[:,None]**( (exp-1).clamp_min(0) )) @ G
    return w


def gauss1D(x, mu, sigma):
    # eval gaussian intntegrated over the pixel
    return 0.5*(torch.erf((x+0.5-mu)/(sigma*np.sqrt(2))) - torch.erf((x-0.5-mu)/(sigma*np.sqrt(2))))


#@torch.jit.script
def slice_images(images, roipos, shape:Tuple[int,int]):
    ry = torch.arange(shape[0], device=images.device, dtype=torch.long)
    rx = torch.arange(shape[1], device=images.device, dtype=torch.long)
    I = torch.arange(images.shape[0], device=images.device, dtype=torch.long)
    Y = roipos[:,0,None,None] + ry[None,:,None]
    X = roipos[:,1,None,None] + rx[None,None,:]
    Y = torch.clamp(Y, 0, images.shape[1]-1)
    X = torch.clamp(X, 0, images.shape[2]-1)
    I = I[:,None,None]
    return images[I,Y,X]

#@torch.jit.script
def slice_volumes(images, roipos, shape:Tuple[int,int,int]):
    rz = torch.arange(shape[0], device=images.device, dtype=torch.long)
    ry = torch.arange(shape[1], device=images.device, dtype=torch.long)
    rx = torch.arange(shape[2], device=images.device, dtype=torch.long)
    I = torch.arange(images.shape[0], device=images.device, dtype=torch.long)
    Z = roipos[:,0,None,None,None] + rz[None,:,None,None]
    Y = roipos[:,1,None,None,None] + ry[None,None,:,None]
    X = roipos[:,2,None,None,None] + rx[None,None,None,:]
    Z = torch.clamp(Z, 0, images.shape[1]-1)
    Y = torch.clamp(Y, 0, images.shape[2]-1)
    X = torch.clamp(X, 0, images.shape[3]-1)
    I = I[:,None,None,None]
    return images[I,Z,Y,X]


def eval_image_spline2D(position, output_shape : Tuple[int,int] , images, deriv:bool =True):
    # Evaluates a 2D X,Y spline-interpolated image in a set of 2D images
    # Images: (N, H, W)
    # position: (N, 2) coordinates, indicating corner position within the source images
    # position (0,0) means the center of the source image is at pixel (0,0) of the output image

    assert len(images.shape) == 3, "Images should be 3D"
    assert len(position.shape) == 2, "Position should be 2D"
    assert len(images) == len(position)

    pixel_ix = torch.floor(position)
    offsets = position - pixel_ix
    pixel_ix = pixel_ix.long()

    subimages = slice_images(images, pixel_ix - 1, 
        (output_shape[0] + 3, output_shape[1] + 3))

    # weights are per-image, which is incompatible with a batched convolution
    # however, we can do a batched convolution by using groups instead, 
    # batchsize is now 1, and the number of groups is the batchsize
    # in_channels = len(images)
    weights_eval = spline_weights(offsets[:,0])[:,:,None] * spline_weights(offsets[:,1])[:,None,:]

    if deriv:
        weights_deriv_0 = spline_weights_deriv(offsets[:,0])[:,:,None] * spline_weights(offsets[:,1])[:,None,:]
        weights_deriv_1 = spline_weights(offsets[:,0])[:,:,None] * spline_weights_deriv(offsets[:,1])[:,None,:]

        # 3 output_chan per image: image, deriv0, deriv1
        weights = torch.stack([weights_eval, weights_deriv_0, weights_deriv_1], dim=1).view(-1,4,4)

        r = torch.nn.functional.conv2d(subimages[None], 
                weights[:,None], padding=0, groups=len(images))

        return r.view(-1, 3, output_shape[0], output_shape[1])
    else:
        # weight shape: (output_chan, in_chan/groups, kernel_size, kernel_size)
        return torch.nn.functional.conv2d(subimages[None], 
                weights_eval[:,None], padding=0, groups=len(images))[0]


class ConvSplineInterp3D(torch.nn.Module):
    def __init__(self, output_shape, tau=0.5):
        super().__init__()
        self.tau = tau
        self.output_shape = output_shape

    def forward(self, position, images):
        # Evaluates a 2D X,Y spline-interpolated image in a 3D volume
        # Images: (N, D, H, W)
        pixel_ix = torch.floor(position)
        offsets = position - pixel_ix
        pixel_ix = pixel_ix.long()

        # currently the slicing takes 5x as much time as the convolution...
        subimages = slice_volumes(images, pixel_ix - 1, 
            (4, self.output_shape[0] + 3, self.output_shape[1] + 3))

        h,w = subimages.shape[-2:]

        # 4 channels per image
        subimages = subimages.view(1,4*len(images),h,w)
    
        weights_eval = (
            spline_weights(offsets[:,0])[:,:,None,None] *
            spline_weights(offsets[:,1])[:,None,:,None] *
            spline_weights(offsets[:,2])[:,None,None,:])

        # For the semi-3D version (3D LUT with 2D image output),
        # we have 4 channels (the neigboring 4 slices) per group
        return torch.nn.functional.conv2d(subimages, weights_eval, 
                    groups=len(images), stride=1)[0]

    @torch.jit.export
    def forward_deriv(self, position, images):
        # Evaluates a 2D X,Y spline-interpolated image in a 3D volume
        # Images: (N, D, H, W)
        pixel_ix = torch.floor(position)
        offsets = position - pixel_ix
        pixel_ix = pixel_ix.long()

        # currently the slicing takes 5x as much time as the convolution...
        subimages = slice_volumes(images, pixel_ix - 1, 
            (4, self.output_shape[0] + 3, self.output_shape[1] + 3))

        h,w = subimages.shape[-2:]

        # 4 channels per image
        subimages = subimages.view(1,4*len(images),h,w)
    
        axis_w = [ spline_weights(offsets[:,i]) for i in range(3) ]

        weights_eval = (
            axis_w[0][:,:,None,None] *
            axis_w[1][:,None,:,None] *
            axis_w[2][:,None,None,:])

        weights_deriv_0 = (
            spline_weights_deriv(offsets[:,0])[:,:,None,None] *
            axis_w[1][:,None,:,None] *
            axis_w[2][:,None,None,:])

        weights_deriv_1 = (
            axis_w[0][:,:,None,None] *
            spline_weights_deriv(offsets[:,1])[:,None,:,None] *
            axis_w[2][:,None,None,:])

        weights_deriv_2 = (
            axis_w[0][:,:,None,None] *
            axis_w[1][:,None,:,None] *
            spline_weights_deriv(offsets[:,2])[:,None,None,:])

        weights = torch.cat([weights_eval, weights_deriv_0, weights_deriv_1, weights_deriv_2], dim=1).view(-1, 4, 4, 4)
        # weight shape: (output_chan, in_chan/groups, kernel_size, kernel_size)
        return torch.nn.functional.conv2d(subimages, weights, 
                    groups=len(images), stride=1).view(len(images),4, *self.output_shape)
        #return weights



#@torch.jit.script
def eval_image_spline3D(position, output_shape:Tuple[int,int], images:torch.Tensor):
    # Evaluates a 2D X,Y spline-interpolated image in a 3D volume
    # Images: (N, D, H, W)

    pixel_ix = torch.floor(position)
    offsets = position - pixel_ix
    pixel_ix = pixel_ix.long()

    # currently the slicing takes 5x as much time as the convolution...
    subimages = slice_volumes(images, pixel_ix - 1, 
        (4, output_shape[0] + 3, output_shape[1] + 3))

    h,w = subimages.shape[-2:]

    # 4 channels per image
    subimages = subimages.view(1,4*images.shape[0],h,w)
   
    axis_w = [ spline_weights(offsets[:,i]) for i in range(3) ]

    weights_eval = (
        axis_w[0][:,:,None,None] *
        axis_w[1][:,None,:,None] *
        axis_w[2][:,None,None,:])

    # weight shape: (output_chan, in_chan/groups, kernel_size, kernel_size)
    return torch.nn.functional.conv2d(subimages, weights_eval.view(-1, 4, 4, 4), 
                groups=images.shape[0], stride=1).view(images.shape[0], *output_shape)
    #return weights

@torch.jit.script
def compute_weights_deriv(offsets):
    axis_w = [ spline_weights(offsets[:,i]) for i in range(3) ]

    weights_eval = (
        axis_w[0][:,:,None,None] *
        axis_w[1][:,None,:,None] *
        axis_w[2][:,None,None,:])

    weights_deriv_0 = (
        spline_weights_deriv(offsets[:,0])[:,:,None,None] *
        axis_w[1][:,None,:,None] *
        axis_w[2][:,None,None,:])

    weights_deriv_1 = (
        axis_w[0][:,:,None,None] *
        spline_weights_deriv(offsets[:,1])[:,None,:,None] *
        axis_w[2][:,None,None,:])

    weights_deriv_2 = (
        axis_w[0][:,:,None,None] *
        axis_w[1][:,None,:,None] *
        spline_weights_deriv(offsets[:,2])[:,None,None,:])

    weights = torch.cat([weights_eval, weights_deriv_0, weights_deriv_1, weights_deriv_2], dim=1).view(-1, 4, 4, 4)
    return weights

#@torch.jit.script
def eval_image_spline3D_deriv(position, output_shape:Tuple[int,int], images:torch.Tensor, use_fast_slice=True):
    # Evaluates a 2D X,Y spline-interpolated image in a 3D volume
    # Images: (N, D, H, W)

    pixel_ix = torch.floor(position)
    offsets = position - pixel_ix
    pixel_ix = pixel_ix.to(torch.int32, non_blocking=True)
    images = images.contiguous()

    # currently the slicing takes 5x as much time as the convolution...
    #subimages = slice_volumes(images, pixel_ix - 1, (4, output_shape[0] + 3, output_shape[1] + 3))
    #subimages = cuda_slice(images, pixel_ix - 1, (4, output_shape[0] + 3, output_shape[1] + 3))

    #assert images.dtype == torch.float32
    if use_fast_slice:
        from torch_convspline.fast_slice import fast_slice
        subimages = fast_slice(images, pixel_ix - 1, (4, output_shape[0] + 3, output_shape[1] + 3))
    else:
        subimages = slice_volumes(images, pixel_ix - 1, (4, output_shape[0] + 3, output_shape[1] + 3))
    
    h,w = subimages.shape[-2:]

    # 4 channels per image
    subimages = subimages.view(1,4*images.shape[0],h,w)

    weights = compute_weights_deriv(offsets)
    weights = weights.to(images.dtype, non_blocking=True)

    # weight shape: (output_chan, in_chan/groups, kernel_size, kernel_size)
    return torch.nn.functional.conv2d(subimages, weights, 
                groups=images.shape[0], stride=1).view(images.shape[0],4, *output_shape)
    #return weights


if __name__ == '__main__':
    # random 3D volume

    src = torch.rand(1, 10, 10, 10)
    output = eval_image_spline3D(torch.tensor([[5.5,0,0]]), (10,10), src)

    plt.imshow(_S(output[0]))
