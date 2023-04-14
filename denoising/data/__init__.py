import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.transforms import transforms
# from importlib import import_module
from data.trans_to_noising import NoiseReconstructionDataset, AddGaussianNoise
from data.tensor_to_list import TensorListDataset


class Data:
    def __init__(self, args):
        self.args = args
        self.train_size = args.train_size
        self.noise_level_test = args.noise_level_test
        self.test_size = args.test_size
        self.noise_level_train = args.noise_level_train
        self.device = args.device
        self.data_as_tensorlist = args.data_as_tensorlist
        self.dataloader_batch_size = args.dataloader_batch_size
        self.dataloader_num_workers = args.dataloader_num_workers

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        normalize_inverse = transforms.Normalize((-0.5, -0.5, -0.5), (1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5))
        transform = transforms.Compose(
            [
                transforms.RandomCrop((80, 80)),
                transforms.ToTensor(),
                normalize])

        dataset_train_base = torchvision.datasets.CelebA(root='./data', download=True, split='train',
                                                         transform=transform)
        dataset_test_base = torchvision.datasets.CelebA(root='./data', download=True, split='valid',
                                                        transform=transform)

        dataset_train_base, _ = torch.utils.data.random_split(dataset_train_base,
                                                              [self.train_size,
                                                               len(dataset_train_base) - self.train_size])
        dataset_test_base = torch.utils.data.Subset(dataset_test_base, range(self.test_size))

        print(f"Dataset sizes: train={len(dataset_train_base)} and test={len(dataset_test_base)}")
        x, y = dataset_test_base[0]
        print(f"Dataset returns tuple with types: {type(x)}, {type(y)}")
        self.n = np.prod(x.shape)  # the dimensionality is later used to scale the adversarial perturbations.
        print(f"Images have shape: {x.shape} with dimension: n={self.n}")

        self.dataset_train = NoiseReconstructionDataset(dataset=dataset_train_base,
                                                   transform=AddGaussianNoise(mean=0, std=self.noise_level_train),
                                                   device=self.device)
        self.dataset_test = NoiseReconstructionDataset(dataset=dataset_test_base,
                                                  transform=AddGaussianNoise(mean=0, std=self.noise_level_test),
                                                  device=self.device)

        # for efficiency we preload the dataset (takes some time to process); deactivate if gpu memory is not large
        # enough
        self.dataset_train_preload = TensorListDataset(self.dataset_train,
                                                       device=self.device) if self.data_as_tensorlist else self.dataset_train
        self.dataset_test_preload = TensorListDataset(self.dataset_test,
                                                      device=self.device) if self.data_as_tensorlist else self.dataset_test

        self.dataset_train_loader = torch.utils.data.DataLoader(self.dataset_train_preload,
                                                                batch_size=self.dataloader_batch_size, shuffle=True,
                                                                num_workers=self.dataloader_num_workers)

        self.dataset_test_loader = torch.utils.data.DataLoader(self.dataset_test_preload,
                                                               batch_size=self.dataloader_batch_size, shuffle=False,
                                                               num_workers=self.dataloader_num_workers)

        y, x = self.dataset_test_preload[7]
        self.y_plt = normalize_inverse(y.cpu()).permute(1,2,0).clamp(0,1)
        self.x_plt = normalize_inverse(x.cpu()).permute(1,2,0).clamp(0,1)