from typing import Any
import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from scipy.io import savemat

kernel_path = './kernels/'

def GaussianKernel(kernel_size, sigma):
    x = torch.arange(0, kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = torch.outer(kernel, kernel)   
    kernel = kernel / torch.sum(kernel)
    return kernel.clone().detach()

def AddGaussianNoise(tensor, percent):
    return tensor + torch.randn(tensor.size(), device='cuda:3') * percent

def loadKernel(size):
    index = int(np.floor((8 * np.random.rand(1)[0])+1))
    kernel_name = os.path.join(kernel_path, "{}.png".format(str(index)))
    kernel = np.array(imageio.imread(kernel_name))
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.div(255).unsqueeze(0).unsqueeze(0)
    kernel = T.Resize(size=(size, size))(kernel)
    return kernel.clone().detach()

def save_kernel(kernel, filename):
    kernel = kernel.squeeze(0).squeeze(0).squeeze(0)
    # print(kernel.size())
    kernel = kernel.data.mul(255)
    mdic = {'kernel': kernel.cpu().numpy()}
    savemat(filename, mdic)

def save_img(img, filename):
    img = img.data.mul(255)
    img = img.squeeze(0)
    if img.shape[0] == 3:
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype('uint8')
        imageio.imwrite(filename, img)
    else:
        img = img.squeeze(0)
        img = img.cpu().numpy().astype('uint8')
        imageio.imwrite(filename, img)


    
class BlurDataset:
    def __init__(self, dataset, transform=None, device="cuda:3"):
        self.dataset = dataset
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clear, _ = self.dataset[idx]
        if self.transform is not None:
            clear = clear.to(self.device)
            blur, kernel = self.transform(clear)
            return clear, blur.squeeze(0), kernel
        else: 
            return clear

    

class AddGaussianBlur:
    def __init__(self):
        pass

    def __call__(self, tensor):
        kernel_sizes = np.arange(35,95,2)
        np.random.shuffle(kernel_sizes)
        kernel_size = kernel_sizes[0]

        sigmas = np.random.uniform(1,2,50)
        np.random.shuffle(sigmas)
        sigma = sigmas[0]

        kernel = GaussianKernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).to("cuda:3")
        blur_ = nn.functional.conv2d(tensor.unsqueeze(0), kernel.repeat(3,1,1,1), padding=((kernel_size)//2), groups=3)
        return AddGaussianNoise(blur_, 0.01).clamp(0,1), kernel


class AddUniformBlur:
    def __init__(self):
        pass

    def __call__(self, tensor):        
        kernel_sizes = np.arange(13,27,2)
        np.random.shuffle(kernel_sizes)
        kernel_size = kernel_sizes[0]
        kernel = loadKernel(kernel_size).to("cuda:3")
        blur_ = nn.functional.conv2d(tensor.unsqueeze(0), kernel.repeat(3,1,1,1), padding=((kernel_size)//2), groups=3)
        return AddGaussianNoise(blur_.mul(torch.max(tensor)/torch.max(blur_)),0.01).clamp(0,1), kernel