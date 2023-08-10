from typing import Any
import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
import random
kernel_path = './kernels/'


def AddGaussianNoise(tensor):
    noise = torch.randn_like(tensor, device='cuda:3')
    for i in range(noise.shape[0]):
        noise[i,:,:] = noise[i,:,:] * random.uniform(0.01, 0.5)
        # noise[i,:,:] = noise[i,:,:] * 0.01
    return tensor + noise

def loadKernel(size):
    index = int(np.floor((20 * np.random.rand(1)[0])))
    kernel_name = os.path.join(kernel_path, "{}.png".format(str(index)))
    kernel = np.array(imageio.imread(kernel_name))
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.div(255).unsqueeze(0).unsqueeze(0)
    kernel = T.Resize(size=(size, size))(kernel)
    return kernel.clone().detach()


def save_img(sharp_, blur_, kernel_, index, dir_sharp_, dir_blur_, dir_kernel_):
    sharp_ = sharp_.data.mul(255)
    blur_ = blur_.data.mul(255)
    kernel_ = kernel_.data.mul(255)

    for i in range(sharp_.shape[0]):

        filename = str(index).zfill(4) + "_" + "{}.png".format(str(i))

        dir_sharp = os.path.join(dir_sharp_, filename)
        dir_blur = os.path.join(dir_blur_, filename)
        dir_kernel = os.path.join(dir_kernel_, filename)

        sharp = sharp_[i,:,:,:].squeeze(0)
        blur = blur_[i,:,:,:].squeeze(0)
        kernel = kernel_.squeeze(0).squeeze(0).cpu().numpy().astype('uint8')

        sharp = np.transpose(sharp.cpu().numpy(), (1, 2, 0)).astype('uint8')
        blur = np.transpose(blur.cpu().numpy(), (1, 2, 0)).astype('uint8')

        imageio.imwrite(dir_sharp, sharp)
        imageio.imwrite(dir_blur, blur)
        imageio.imwrite(dir_kernel, kernel)


def AddUniformBlur(tensor, kernel):
    kernel_size = kernel.shape[-1]
    kernel = kernel.div(torch.sum(kernel))
    blur_ = nn.functional.conv2d(tensor, kernel.repeat(3,1,1,1), padding=((kernel_size)//2), groups=3)
    return AddGaussianNoise(blur_).clamp(0,1)

