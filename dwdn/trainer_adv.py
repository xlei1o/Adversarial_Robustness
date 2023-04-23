import os
import decimal
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from tqdm import tqdm
import math

import utils_deblur
import torch.nn.functional as F
import torch.nn as nn

from attacker import AttackerModel


class Trainer_adv:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args

        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss  ## MSELoss
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp

        self.error_last = 1e8

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step()

    def set_loader(self, new_loader):
        self.loader_train = new_loader.loader_train
        self.loader_test = new_loader.loader_test

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def clip_gradient(self, optimizer, grad_clip):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def set_adv(self):
        # eps = self.args.eps * math.sqrt(self.args.n_colors)
        attack_kwargs = {
            'constraint': self.args.constraint,
            'eps': self.args.eps,
            'step_size': 2.5 * (self.args.eps / self.args.adv_iterations),
            'iterations': self.args.adv_iterations,
            'random_start': False,
            'random_restarts': 0,
            'use_best': False,
            'random_mode': "uniform_in_sphere"
        }
        return attack_kwargs

    def train(self):
        print("Image Deblur Training")
        # self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        attack_kwargs = self.set_adv()
        for batch, (blur, sharp, kernel, filename) in enumerate(self.loader_train):

            blur = torch.squeeze(blur, 1)
            sharp = torch.squeeze(sharp, 1)
            kernel = torch.squeeze(kernel, 1)

            blur = blur.to(self.device)
            sharp = sharp.to(self.device)

            self.optimizer.zero_grad()

            _, deblur = self.model(blur, kernel, target=None, make_adv=True, **attack_kwargs)
            self.n_levels = 2
            self.scale = 0.5
            loss = 0
            for level in range(self.n_levels):
                scale = self.scale ** (self.n_levels - level - 1)
                n, c, h, w = sharp.shape
                hi = int(round(h * scale))
                wi = int(round(w * scale))
                sharp_level = F.interpolate(sharp, (hi, wi), mode='bilinear')
                loss = loss + self.loss(deblur[level], sharp_level)

            self.ckp.report_log(loss.item())
            loss.backward()
            self.clip_gradient(self.optimizer, self.args.grad_clip)
            self.optimizer.step()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {}'.format(
                    (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                    self.loss.display_loss(batch)))

        self.scheduler.step()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch
        self.model.eval()
        self.ckp.start_log(train=False)
        attack_kwargs = self.set_adv()
        # with torch.no_grad():
        if self.args.targeted:
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (blur, sharp, kernel, target, filename) in enumerate(tqdm_test):

                blur = torch.squeeze(blur, 0)
                target = torch.squeeze(target, 0)
                kernel = torch.squeeze(kernel, 0)

                blur = blur.to(self.device)
                target = target.to(self.device)

                deblur, deblur_adv = self.model(blur, kernel, target=target, make_adv=True, **attack_kwargs)

                if self.args.save_images:
                    deblur_adv = utils_deblur.postprocess(deblur_adv[-1], rgb_range=self.args.rgb_range)
                    deblur = utils_deblur.postprocess(deblur[-1], rgb_range=self.args.rgb_range)
                    save_list_adv = [deblur_adv[0]]
                    save_list = [deblur[0]]
                    self.ckp.save_images(filename, save_list_adv, make_adv=True)
                    self.ckp.save_images(filename, save_list)
        else:
            tqdm_test = tqdm(self.loader_test, desc=f"Test Progress", position=0, ncols=80)
            for idx_img, (blur, sharp, kernel, target, filename) in enumerate(tqdm_test):
                blur = torch.squeeze(blur, 0)
                kernel = torch.squeeze(kernel, 0)
                blur = blur.to(self.device)

                deblur, deblur_adv = self.model(blur, kernel, make_adv=True, **attack_kwargs)

                if self.args.save_images:
                    deblur_adv = utils_deblur.postprocess(deblur_adv[-1], rgb_range=self.args.rgb_range)
                    deblur = utils_deblur.postprocess(deblur[-1], rgb_range=self.args.rgb_range)
                    save_list_adv = [deblur_adv[0]]
                    save_list = [deblur[0]]
                    self.ckp.save_images(filename, save_list_adv, make_adv=True)
                    self.ckp.save_images(filename, save_list)

        print('Save Path for Standard Test: {}'.format('./results/Standard'))
        print('Save Path for Adversarial Test: {}'.format('./results/Adversarial'))
        self.ckp.end_log(len(self.loader_test), train=False)


    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
