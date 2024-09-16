"""
Fast blitting of 3D volumes using CUDA / CuPy
Author: Jelmer Cnossen

Results on RTX 4060 (Mobile):

Meshgrid based iterations/s: 104.04. Transfer speed: 12.70 GB/s
Fast slice f32 iterations/s: 880.36. Transfer speed: 107.47 GB/s
Fast slice f16 iterations/s: 1395.27. Transfer speed: 85.16 GB/s
Fast slice pre-alloc f32 iterations/s: 880.66. Transfer speed: 107.50 GB/s
"""
import torch
import torch.nn as nn

# Note if cupy fails to load properly, try setting CUDA_PATH in your environment variables
import cupy
import numpy as np
import time
from typing import Tuple
from tqdm import tqdm


@torch.jit.script
def slice_volumes(images, roipos, shape:Tuple[int,int,int]):
    rz = torch.arange(shape[0], device=images.device, dtype=torch.long)
    ry = torch.arange(shape[1], device=images.device, dtype=torch.long)
    rx = torch.arange(shape[2], device=images.device, dtype=torch.long)
    I = torch.arange(len(images), device=images.device, dtype=torch.long)
    Z = roipos[:,0,None,None,None] + rz[None,:,None,None]
    Y = roipos[:,1,None,None,None] + ry[None,None,:,None]
    X = roipos[:,2,None,None,None] + rx[None,None,None,:]
    Z = torch.clamp(Z, 0, images.shape[1]-1)
    Y = torch.clamp(Y, 0, images.shape[2]-1)
    X = torch.clamp(X, 0, images.shape[3]-1)
    I = I[:,None,None,None]
    return images[I,Z,Y,X]


_block_copy_kernel_code = r'''
#include <cuda_fp16.h>
extern "C" __global__
void block_copy_kernel_PixelType(const PixelType* source, PixelType* output, const int* offsets, int B, int W, int H, int D, int out_W, int out_H, int out_D) {
    int x_out = threadIdx.x + blockDim.x * blockIdx.x;
    int y_out = threadIdx.y + blockDim.y * blockIdx.y;
    int b = blockIdx.z;

    if (x_out < out_W && y_out < out_H) {
        int src_y = y_out + offsets[b * 3 + 1];
        int src_x = x_out + offsets[b * 3 + 2];
        src_y = min(max(src_y, 0), H-1);
        src_x = min(max(src_x, 0), W-1);

        for (int z_out = 0; z_out < out_D; z_out++) {
            int src_z = z_out + offsets[b * 3];
            src_z = min(max(src_z, 0), D-1);
            output[b * (out_W * out_H * out_D) + z_out * (out_H * out_W) + y_out * out_W + x_out] = source[b * (W * H * D) + src_z * (H * W) + src_y * W + src_x];
        }
    }
}
'''


# Compile the custom CUDA kernel
_block_copy_kernel_f16 = cupy.RawKernel(_block_copy_kernel_code.replace('PixelType', '__half'), 'block_copy_kernel___half')
_block_copy_kernel_f32 = cupy.RawKernel(_block_copy_kernel_code.replace('PixelType', 'float'), 'block_copy_kernel_float')

def fast_slice_to(source, offsets, output):
    B, D, H, W = source.shape
    out_D, out_H, out_W = output.shape[1:]
    
    # check that source is float32 and offsets are 32 bit
    assert source.dtype == torch.float32 or source.dtype == torch.float16
    assert offsets.dtype == torch.int32
    assert output.dtype == source.dtype
    assert len(output) == len(source)

    stream_handle = torch.cuda.current_stream().cuda_stream
    cupy_stream = cupy.cuda.ExternalStream(stream_handle)

    with cupy_stream:
        # Convert PyTorch tensors to CuPy arrays
        source_cupy = cupy.asarray(source)
        output_cupy = cupy.asarray(output)
        offsets_cupy = cupy.asarray(offsets)

        # Launch the custom CUDA kernel 
        threads_per_block = (min(out_W, 32), min(out_H, 32), 1)
        blocks_per_grid = ((out_W + threads_per_block[0] - 1) // threads_per_block[0], (out_H + threads_per_block[1] - 1) // threads_per_block[1], B)

        if source.dtype == torch.float32:
            _block_copy_kernel_f32 (blocks_per_grid, threads_per_block, (source_cupy, output_cupy, offsets_cupy, B, W, H, D, out_W, out_H, out_D))
        else:
            _block_copy_kernel_f16 (blocks_per_grid, threads_per_block, (source_cupy, output_cupy, offsets_cupy, B, W, H, D, out_W, out_H, out_D))

def fast_slice(source, offsets, out_shape):
    output = torch.empty((len(source), *out_shape), device=source.device, dtype=source.dtype)
    fast_slice_to(source, offsets, output)
    return output

def speed_test():
    B = 256
    D, H, W = 40, 128, 128
    out_D, out_H, out_W = 20, 80, 80

    source = torch.randn(B, D, H, W).cuda()
    source_f16 = source.half()
    offsets = torch.randint(-4, D - out_D + 5, (B, 3), device=source.device, dtype=torch.int32)
    out_shape = (out_D, out_H, out_W)
    
    n_iterations = 2000

    for _ in range(10):
        fast_slice(source, offsets, out_shape)
        fast_slice(source_f16, offsets, out_shape)
        slice_volumes(source, offsets, out_shape)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in tqdm(range(100)):
        slice_volumes(source, offsets, out_shape)
    torch.cuda.synchronize()
    time_slice_img = time.time() - start_time

    start_time = time.time()
    for _ in tqdm(range(n_iterations)):
        fast_slice(source, offsets, out_shape)
    torch.cuda.synchronize()
    time_fast_slice_f32 = time.time() - start_time

    start_time = time.time()
    for _ in tqdm(range(n_iterations)):
        fast_slice(source_f16, offsets, out_shape)
    torch.cuda.synchronize()
    time_fast_slice_f16 = time.time() - start_time

    output = torch.empty((len(source), *out_shape), device=source.device, dtype=source.dtype)
    start_time = time.time()
    for _ in tqdm(range(n_iterations)):
        fast_slice_to(source, offsets, output)
    torch.cuda.synchronize()
    time_fast_slice_to_f32 = time.time() - start_time

    # compute memory transfer speed
    n_elem = out_shape[0] * out_shape[1] * out_shape[2] * B

    print(f'Input memory size: {source.numel() * 4 / 1024**3:.2f} GB')
    print(f"Meshgrid based iterations/s: {100 / time_slice_img:.2f}. Transfer speed: {100 * n_elem * 4 / time_slice_img / 1024**3:.2f} GB/s")
    print(f"Fast slice f32 iterations/s: {n_iterations / time_fast_slice_f32:.2f}. Transfer speed: {n_iterations * n_elem * 4 / time_fast_slice_f32 / 1024**3:.2f} GB/s")
    print(f"Fast slice f16 iterations/s: {n_iterations / time_fast_slice_f16:.2f}. Transfer speed: {n_iterations * n_elem * 2 / time_fast_slice_f16 / 1024**3:.2f} GB/s")
    print(f"Fast slice pre-alloc f32 iterations/s: {n_iterations / time_fast_slice_to_f32:.2f}. Transfer speed: {n_iterations * n_elem * 4 / time_fast_slice_to_f32 / 1024**3:.2f} GB/s")

def test_fast_slice():
    B = 3
    D, H, W = 5, 10, 10
    out_D, out_H, out_W = 5, 10, 10

    source = torch.randn(B, D, H, W).cuda()
    source_f16 = source.half()
    offsets = torch.tensor([[0,0,3],[-2,0,0],[0,-4,0]], device=source.device, dtype=torch.int32)
    out_shape = (out_D, out_H, out_W)

    out1 = fast_slice(source, offsets, out_shape)
    out2 = slice_volumes(source, offsets, out_shape)
    assert torch.allclose(out1, out2.float())
    print('fast_slice test passed')	

    #from pytweezer import image_view
    #image_view(torch.cat((out1, out2.float()), dim=-1))

    #import matplotlib.pyplot as plt
    #fig,ax=plt.subplots(1,2)
    #ax[0].imshow(out1[2,0].cpu())
    #ax[1].imshow(out2[2,0].cpu())
    #plt.show()    


if __name__ == '__main__':
    import os
    # Hack to make VS Code debugger launch properly
    os.chdir('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin')

    test_fast_slice()
    # Run the speed test
    speed_test()
    
    #simple_test()
