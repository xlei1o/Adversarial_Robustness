import torch
import torchvision
import torchvision.transforms as T
from TrainingData import common
from tqdm import tqdm
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import imageio
import random

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


class TrainingData:
    def __init__(self, args):
        self.args = args
        data_dir = './datasets'
        # data_dir = './test'
        dataset_train_base = Data(data_dir)
        
        dataset_train_loader = torch.utils.data.DataLoader(dataset_train_base,
                                                            batch_size=4, shuffle=False,
                                                            num_workers=0)

        apath = os.path.join(self.args.dir_data)
        train_dir_blur = os.path.join(apath, "InputBlurredImage/")
        train_dir_sharp = os.path.join(apath, "InputTargetImage/")
        train_dir_kernel = os.path.join(apath, "psfMotionKernel/")      
        if not os.path.exists(os.path.dirname(train_dir_blur)):
                    os.makedirs(train_dir_blur)
                    os.makedirs(train_dir_sharp)
                    os.makedirs(train_dir_kernel)

        tqdm_train = tqdm(dataset_train_loader, desc=f"Generating Training Data", position=0, ncols=80)
        for index, (sharp) in enumerate(tqdm_train):

            kernel_sizes = np.arange(13,35,2)
            np.random.shuffle(kernel_sizes)
            kernel_size = kernel_sizes[0]
            kernel = common.loadKernel(kernel_size)
            
            sharp = sharp.to('cuda:3')
            kernel = kernel.to('cuda:3')

            blur = common.AddUniformBlur(sharp, kernel)

            common.save_img(sharp, blur, kernel, index, train_dir_sharp, train_dir_blur, train_dir_kernel)


