import torch


class TensorListDataset(object):
    def __init__(self, base_dataset, device):
        base_dataset_list = list(base_dataset)
        tensors_x = [torch.unsqueeze(x, dim=0) for x, _ in base_dataset_list]
        tensors_y = [torch.unsqueeze(y, dim=0) for _, y in base_dataset_list]
        self.tensor_list_x = torch.cat(tensors_x).to(device)
        self.tensor_list_y = torch.cat(tensors_y).to(device)

    def __len__(self):
        return len(self.tensor_list_x)

    def __getitem__(self, idx):
        return self.tensor_list_x[idx, ...], self.tensor_list_y[idx, ...]
