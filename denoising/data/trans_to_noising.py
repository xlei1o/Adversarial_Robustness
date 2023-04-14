import torch


class NoiseReconstructionDataset(object):
    def __init__(self, dataset, transform=None, target_transform=None, device="cpu"):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        x = x.to(self.device)
        xr = self.transform(x) if self.transform is not None else x
        yr = self.target_transform(x) if self.target_transform is not None else x
        return xr, yr


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # return tensor + torch.randn(tensor.size(), device=tensor.get_device()) * self.std + self.mean
        return tensor + torch.randn(tensor.size(), device='cpu') * self.std + self.mean
