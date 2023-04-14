import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from attack import AttackerModel
import data


class Trainer(object):
    def __init__(self, args, dataloader, model, **attack_kwargs):
        # super().__init__()

        self.device = args.device
        self.dataloader = dataloader
        self.model = AttackerModel(model)
        self.loss_fn = nn.MSELoss()
        self.epochs = args.epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.loss_train_values = []
        self.loss_train_values_adv = []
        self.loss_test_values = []
        self.loss_test_values_adv = []

        for t in trange(self.epochs, desc=f"Training", unit="epoch", position=0):
            # one could do experiment tracking here (e.g. tensorboard)
            self.loss_train_values.append(self.train(make_adv=False, **attack_kwargs))
            self.loss_train_values_adv.append(self.train(make_adv=True, **attack_kwargs))
            self.loss_test_values.append(self.test(make_adv=False, **attack_kwargs))
            self.loss_test_values_adv.append(self.test(make_adv=True, **attack_kwargs))

    def train(self, make_adv, **attack_kwargs):
        self.model.train()
        loss_total = 0
        for _, (X, y) in enumerate(self.dataloader.dataset_train_loader):
            X, y = X.to(self.device), y.to(self.device)

            _, pred = self.model(X.float(), target=y.float(), make_adv=make_adv, **attack_kwargs)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
        return loss_total / len(self.dataloader.dataset_train_loader)

    def test(self, make_adv, **attack_kwargs):
        self.model.eval()
        test_loss = 0
        for X, y in self.dataloader.dataset_test_loader:
            X, y = X.to(self.device), y.to(self.device)
            _, pred = self.model(X, target=y, make_adv=make_adv, **attack_kwargs)
            test_loss += self.loss_fn(pred, y).item()
        return test_loss / len(self.dataloader.dataset_test_loader)
