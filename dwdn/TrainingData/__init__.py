import torch
import torchvision
import torchvision.transforms as T
from TrainingData import common
from tqdm import tqdm
import os


class TrainingData:
    def __init__(self, args):
        self.args = args
        train_size = 2
        vaild_size = 2
        # generate training dataset
        dataset_train_base = torchvision.datasets.CelebA(root='.', download=True, split='train',
                                                         transform=T.ToTensor())
        dataset_valid_base = torchvision.datasets.CelebA(root='.', download=True, split='valid',
                                                         transform=T.ToTensor())

        dataset_train_base, _ = torch.utils.data.random_split(dataset_train_base,
                                                              [train_size,
                                                               len(dataset_train_base) - train_size])
        dataset_valid_base = torch.utils.data.Subset(dataset_valid_base, range(vaild_size))

        dataset_train = common.BlurDataset(dataset=dataset_train_base,
                                           transform=common.AddGaussianBlur())
                                        #    device='cuda')
        dataset_valid = common.BlurDataset(dataset=dataset_valid_base,
                                           transform=common.AddGaussianBlur())
                                        #    device='cuda')
        
        dataset_train_loader = torch.utils.data.DataLoader(dataset_train,
                                                            batch_size=1, shuffle=False,
                                                            num_workers=0)
        dataset_vaild_loader = torch.utils.data.DataLoader(dataset_valid,
                                                            batch_size=1, shuffle=False,
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
        for index, (sharp, blur, kernel) in enumerate(tqdm_train):

            blur_name = os.path.join(train_dir_blur, "{}.png".format(str(index)))
            kernel_name = os.path.join(train_dir_kernel, "{}.mat".format(str(index)))
            sharp_name = os.path.join(train_dir_sharp, "{}.png".format(str(index)))

            common.save_img(sharp, sharp_name)
            common.save_img(blur, blur_name)
            common.save_kernel(kernel, kernel_name)


        vaild_dir_blur = os.path.join(apath, "blurredImage/")
        vaild_dir_sharp = os.path.join(apath, "GTImage/")
        vaild_dir_kernel = os.path.join(apath, "kernelImage/")
        if not os.path.exists(os.path.dirname(vaild_dir_blur)):
                os.makedirs(os.path.dirname(vaild_dir_kernel))
                os.makedirs(os.path.dirname(vaild_dir_sharp))
                os.makedirs(os.path.dirname(vaild_dir_blur))
        tqdm_vaild = tqdm(dataset_vaild_loader, desc=f"Generating Validation Data", position=0, ncols=80)
        for index, (sharp, blur, kernel) in enumerate(tqdm_vaild):
            blur_name = os.path.join(vaild_dir_blur, "{}.png".format(str(index)))
            kernel_name = os.path.join(vaild_dir_kernel, "{}.mat".format(str(index)))
            sharp_name = os.path.join(vaild_dir_sharp, "{}.png".format(str(index)))

            common.save_img(sharp, sharp_name)
            common.save_img(blur, blur_name)
            common.save_kernel(kernel, kernel_name)



