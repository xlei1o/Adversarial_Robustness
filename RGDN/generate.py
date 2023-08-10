import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import os
import imageio
from scipy.io import savemat
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as T
import random

# kernel_path = './kernels'
# sharp_path = './datasets'

kernel_path = './KERNEL'
sharp_path = './PASCAL'

def AddGaussianNoise(tensor):
    noise = torch.randn_like(tensor, device='cuda:2')
    for i in range(noise.shape[0]):
        noise[i,:,:] = noise[i,:,:] * random.uniform(0.01, 0.05)
    return tensor + noise

def loadKernel(size):
    index = int(np.floor((5 * np.random.rand(1)[0])))
    kernel_name = os.path.join(kernel_path, "{}.png".format(str(index)))
    kernel = np.array(imageio.imread(kernel_name))
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.div(255).unsqueeze(0).unsqueeze(0)
    kernel = T.Resize(size=(size, size))(kernel)
    kernel = kernel.div(torch.sum(kernel))
    return kernel.clone().detach()


def to_numpy(sharp_, blur_, kernel_):

    sharp = sharp_.squeeze(0)
    blur = blur_.squeeze(0)

    kernel = kernel_.squeeze(0).squeeze(0).cpu().numpy()
    kernel_T = torch.flip(kernel_.squeeze(0).squeeze(0), [0, 1]).cpu().numpy()

    sharp = np.transpose(sharp.cpu().numpy(), (1, 2, 0))
    blur = np.transpose(blur.cpu().numpy(), (1, 2, 0))

    return sharp, blur, kernel, kernel_T


def AddUniformBlur(tensor, kernel):
    kernel_size = kernel.shape[-1]
    # kernel = kernel.div(torch.sum(kernel))
    blur_ = nn.functional.conv2d(tensor, kernel.repeat(3,1,1,1), padding=((kernel_size)//2), groups=3)
    hks = (kernel_size) // 2
    blur = AddGaussianNoise(blur_).clamp(0,1)
    blur = torch.nn.functional.pad(blur_, (hks, hks, hks, hks), mode='replicate')
    return blur


class Data(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_filenames = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.data_dir, image_filename)
        image = np.array(imageio.imread(image_path)).transpose(2,0,1)
        image = image[:3,:,:]
        tensor_image = torch.from_numpy(image).float()/255
        tensor_image = T.RandomCrop(size = (256,256))(tensor_image)
        return tensor_image
    

class GenerateData:
    def __init__(self):
        dataset_test_base = Data(sharp_path)
        
        dataset_test_loader = torch.utils.data.DataLoader(dataset_test_base,
                                                            batch_size=1, shuffle=False,
                                                            num_workers=0)

        apath = os.path.join('./rgdn_dataset')

        tqdm_test = tqdm(dataset_test_loader, desc=f"Generating Test Data", position=0, ncols=80)
        for index, (sharp) in enumerate(tqdm_test):
            filename = os.path.join(apath, "{}.mat".format(str(index).zfill(4)))

            kernel_sizes = np.arange(13,27,2)
            np.random.shuffle(kernel_sizes)
            kernel_size = kernel_sizes[0]
            kernel = loadKernel(kernel_size)

            sharp = sharp.to('cuda:2')
            kernel = kernel.to('cuda:2')

            blur = AddUniformBlur(sharp, kernel)


            sharp, blur, kernel, kernel_T = to_numpy(sharp, blur, kernel)

            mdic = {'y': blur,
                    'k': kernel,
                    'x_gt': sharp,
                    'kt': kernel_T
                    }
            savemat(filename, mdic)


GenerateData()