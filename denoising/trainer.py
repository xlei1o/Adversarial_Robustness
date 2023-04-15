import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from attack import AttackerModel
import math


class Trainer:
    def __init__(self, args, dataloader, model):
        # super().__init__()
        self.args = args

        self.dataloader = dataloader
        self.model_adv = AttackerModel(model)
        self.model_std = AttackerModel(model)
        self.loss_fn = nn.MSELoss()
        self.optimizer_std = optim.Adam(self.model_std.parameters(), lr=1e-3)
        self.optimizer_adv = optim.Adam(self.model_adv.parameters(), lr=1e-3)
        #
        # self.loss_train_values_std = []
        # self.loss_train_values_adv = []
        # self.loss_test_values = []
        # self.loss_test_values_adv = []
        self.test_loss_std_std = []
        self.test_loss_std_adv = []
        self.test_loss_adv_std = []
        self.test_loss_adv_adv = []
        std_kwargs = {'optimizer': self.optimizer_std,
                      'model': self.model_std,
                      'make_adv': False}
        adv_kwargs = {'optimizer': self.optimizer_adv,
                      'model': self.model_adv,
                      'make_adv': False}

        attack_kwargs = self.attack_settings()
        if self.args.test_only:
            self.load()

        else:
            if self.args.train_model:

                for t in trange(self.args.epochs, desc=f"Standard Training", unit="epoch", position=0):
                    # one could do experiment tracking here (e.g. tensorboard)
                    # self.loss_train_values.append(self.train(make_adv=False, **attack_kwargs))
                    # self.loss_train_values_adv.append(self.train(make_adv=True, **attack_kwargs))
                    # self.loss_test_values.append(self.test(make_adv=False, **attack_kwargs))
                    # self.loss_test_values_adv.append(self.test(make_adv=True, **attack_kwargs))
                    self.train(**std_kwargs, **attack_kwargs)
                    self.test_loss_std_std.append(self.test(self.model_std, make_adv=False, **attack_kwargs))
                    self.test_loss_std_adv.append(self.test(self.model_std, make_adv=True, **attack_kwargs))
                for t in trange(self.args.epochs, desc=f"Adversarial Training", unit="epoch", position=0):
                    self.train(**adv_kwargs, **attack_kwargs)
                    self.test_loss_adv_std.append(self.test(self.model_adv, make_adv=False, **attack_kwargs))
                    self.test_loss_adv_adv.append(self.test(self.model_adv, make_adv=True, **attack_kwargs))
                self.save()

            else:
                self.load()


    def attack_settings(self):
        eps = self.args.eps_rel * math.sqrt(self.dataloader.n)
        attack_kwargs = {
            'constraint': "2",
            'eps': eps,
            'step_size': 2.5 * (eps / self.args.adv_iterations),
            'iterations': self.args.adv_iterations,
            'random_start': True,
            'random_restarts': 0,
            'use_best': False,
            'random_mode': "uniform_in_sphere"}
        return attack_kwargs

    def train(self, model, optimizer, make_adv, **attack_kwargs):
        model.train()
        loss_total = 0
        for _, (X, y) in enumerate(self.dataloader.dataset_train_loader):
            X, y = X.to(self.args.device), y.to(self.args.device)

            _, pred = model(X.float(), target=y.float(), make_adv=make_adv, **attack_kwargs)
            loss = self.loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
        return loss_total / len(self.dataloader.dataset_train_loader)

    def test(self, model, make_adv, **attack_kwargs):
        model.eval()
        test_loss = 0
        for X, y in self.dataloader.dataset_test_loader:
            X, y = X.to(self.args.device), y.to(self.args.device)
            _, pred = model(X, target=y, make_adv=make_adv, **attack_kwargs)
            test_loss += self.loss_fn(pred, y).item()
        return test_loss / len(self.dataloader.dataset_test_loader)

    def save(self):

        filename = 'model_{}'.format(self.args.model)
        torch.save({
            'model_std': self.model_std,
            'model_adv': self.model_adv,
            # 'model_std': self.model_std.state_dict(),
            # 'model_adv': self.model_adv.state_dict(),
            # 'optimizer_std': self.optimizer_std.state_dict(),
            # 'optimizer_adv': self.optimizer_adv.state_dict()
        },
            os.path.join(self.args.model_path, '{}.pt'.format(filename))
        )
        print('The model has been saved.')

    def load(self):
        filename = 'model_{}'.format(self.args.model)
        file_path = os.path.join(self.args.model_path, '{}.pt'.format(filename))
        kwargs = {'map_location': lambda storage, loc: storage}
        print('Loading model from {}.'.format(file_path))
        # self.model.load_state_dict(
        #     torch.load(file_path, **kwargs),
        #     strict=False)

        checkpoint = torch.load(file_path)
        self.model_std.load_state_dict(checkpoint['model_std'])
        self.model_adv.load_state_dict(checkpoint['model_adv'])
        # self.optimizer_std.load_state_dict(checkpoint['optimizer_std'])
        # self.optimizer_adv.load_state_dict(checkpoint['optimizer_adv'])
        self.model_std.eval()
        self.model_adv.eval()
        print('Finish loading.')
