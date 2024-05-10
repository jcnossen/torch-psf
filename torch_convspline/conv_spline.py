"""
3D catmull rom splines evaluated using conv2d pytorch operations

author: Jelmer Cnossen
"""
import os
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
#from pytweezer.tirf.torch_splines import CatmullRomSpline1D, CatmullRomSpline2D
from typing import Tuple
#from torch.func import jacrev, jacfwd
#from block_copy import BlockCopy
from pytweezer.tracking.torch_levmar import lm_update

from pytweezer.tracking.torch_levmar import LM_MLE

#from pytweezer.cuda.pytwcuda import slice_volumes as cuda_slice


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

@torch.jit.script
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
        from pytweezer.utils.fast_slice import fast_slice
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


def calc_deriv(pos, output_shape, images):
    def non_batched(p, image):
        return eval_image_spline2D(p[None], output_shape, image[None]).flatten()

    def calc_deriv(p, image):
        ev = non_batched(p, image)
        return ev, jacfwd(non_batched)(p, image)

    ev, deriv = torch.vmap(calc_deriv)(pos, images)
    return (ev.view(len(ev), *output_shape), 
            deriv.view(len(ev), *output_shape, -1).permute(0,3,1,2))



#@torch.jit.script
def lm_alphabeta(mu, jac, smp):
    """
    mu: [numsamples]
    jac: [numsamples, numparams]
    smp: [numsamples]
    """
    invmu = 1.0 / torch.clip(mu, min=1e-9)
    af = smp*invmu**2

    jacm = torch.matmul(jac[:,:,None], jac[:,None,:])
    alpha = jacm * af[:,None,None]
    alpha = alpha.sum(0)

    beta = (jac * (smp*invmu-1)[:,None]).sum(0)
    return alpha, beta


def levmar_fit_spline3D(initial_pos, lut, iterations, lambda_, samples):
    pos = initial_pos.clone()
    output_shape = samples.shape[1:]

    for i in range(iterations):
        images = eval_image_spline3D_deriv(pos, output_shape, lut, deriv=True)
        ev = images[:, 0]
        jac = images[:, 1:].permute(0,2,3,1).reshape(len(images), -1, 3)

        alpha, beta = lm_alphabeta(ev.flatten(), jac, samples.view(len(images),-1))
        
        pos = lm_update(pos, alpha, beta, lambda_)
        print(pos)
    return pos

"""
def lm_alphabeta(pos, images, samples):
    output_shape = samples.shape[1:]
    def non_batched(p, image):
        return eval_image_spline2D(p[None], output_shape, image[None]).flatten()

    def calc_update(p, image, smp):
        ev = non_batched(p, image)
        jac = jacfwd(non_batched)(p, image)
        alpha, beta = lm_alphabeta(ev, jac, smp.flatten())
        return alpha, beta

    alpha,beta = torch.vmap(calc_update)(pos, images, samples)
    return alpha,beta
"""

def benchmark_evals(zs, pos3D, shape, its):
    with tqdm.tqdm(total=M*its) as pbar:
        
        for i in range(its):
            r = eval_image_spline3D_deriv(position = pos3D, output_shape=shape, images = zs)
            #r = eval_image_spline2D(cornerpos, shape, images, deriv=True)
            #calc_deriv(cornerpos, shape, images)
            #lm_update(cornerpos, images, samples)
            pbar.update(M)




if __name__ == '__main__':
    lut_fn = os.path.split(__file__)[0] + '/../../data/lut000.jpg'
    lut_fn = os.path.abspath(lut_fn)
    lut = plt.imread(lut_fn)[:,:,0].astype(np.float32)
    
    from pytweezer.legacy.pyqtrk.benchmark import generate_image

    print('Generating z-stack')
    N = len(lut)
    roisize = 100
    zstack = np.array( [ generate_image([roisize/2,roisize/2,i], roisize, roisize, lut, 
                                       minradius = 2, maxradius = roisize/3)
                        for i in np.linspace(0,len(lut)-1,N, endpoint=False)])
    
    #plt.figure(); plt.imshow(zstack[len(zstack)//2])

    from pytweezer import image_view
    #image_view(zstack)

    shape = roisize,roisize
    dev = torch.device('cuda:0')

    zstack = torch.tensor(zstack, dtype=torch.float32, device=dev)
    M = 256*2
    pos3D = torch.tensor([[10, roisize/2 - shape[0]/2, roisize/2 - shape[1]/2]], dtype=torch.float32, device=dev)
    pos3D = pos3D.repeat(M,1)
    pos3D[:,0] += torch.linspace(-5,5,M).to(dev)
    pos3D[:,1] += torch.linspace(-5,5,M).to(dev)
    pos3D[:,2] += torch.linspace(-5,5,M).to(dev)
    spl = ConvSplineInterp3D(shape)
    r = eval_image_spline3D(pos3D, shape, zstack.expand(M,-1,-1,-1))
    #r = spl.forward(pos3D, zstack.expand(M,-1,-1,-1))
    image_view(r)

    #images = zstack[len(zstack)//2].repeat(N, 1,1)
    #cornerpos = torch.tensor([[roisize/2 - shape[0]/2, roisize/2 - shape[1]/2]], dtype=torch.float32, device=dev).repeat(len(images),1)
    #cornerpos[:,1] += torch.linspace(-10,10,N).to(dev)
    #cornerpos[:,0] += torch.linspace(-10,10,N).to(dev)
    #r = eval_image_spline2D(cornerpos, shape, images, deriv=True)
    #image_view(r)

    #samples = torch.poisson(images)
    #lm_update(cornerpos, images, samples)

    #if False:

    # create lambda with output_shape baked in

    #zs = zs = zstack.expand(M,-1,-1,-1).contiguous()
    #benchmark_evals(zs, pos3D, shape, its=4000)

    #ev,deriv = calc_deriv(cornerpos, shape, images)
    #image_view(deriv)

#    plt.figure(); plt.imshow(_S(r[0]))
