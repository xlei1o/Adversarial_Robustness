import torch
import imageio
import numpy as np
import torch.nn as nn
from scipy.io import savemat

def GaussianKernel(kernel_size, sigma):
    x = torch.arange(0, kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = torch.outer(kernel, kernel)   
    kernel = kernel / torch.sum(kernel)
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
    # print(img.size())
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype('uint8')
    imageio.imwrite(filename, img)


class BlurDataset:
    def __init__(self, dataset, transform=None, device="cpu"):
        self.dataset = dataset
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sharp, _ = self.dataset[idx]
        sharp = sharp.to(self.device)
        blur, kernel = self.transform(sharp)

        return sharp, blur.squeeze(0), kernel
    

class AddGaussianBlur:
    def __init__(self):
        kernel_sizes = np.arange(5,75,2)
        np.random.shuffle(kernel_sizes)
        kernel_size = kernel_sizes[0]
        sigma = np.random.uniform(0.5,1.5,1)[0]
        self.kernel_size = kernel_size
        self.kernel = GaussianKernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)  
        # self.kernel = kernel.repeat(3,1,1,1)

    def __call__(self, tensor):
        # print(tensor.size())
        return nn.functional.conv2d(tensor.unsqueeze(0), self.kernel.repeat(3,1,1,1), padding=((self.kernel_size)//2), groups=3), self.kernel


