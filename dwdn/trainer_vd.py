import os
import decimal
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from tqdm import tqdm

import utils_deblur
import torch.nn.functional as F
import torch.nn as nn

att_kwargs = {'constraint': 'inf',
              'eps': 0.5,
              'step_size': 0.1,
              'iterations': 1,
              # 'do_tqdm': True
              }


class Trainer_VD:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args

        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
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

    def train(self):
        print("Image Deblur Training")
        # self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()

        for batch, (blur, sharp, kernel, filename) in enumerate(self.loader_train):

            blur = torch.squeeze(blur, 1)
            sharp = torch.squeeze(sharp, 1)
            kernel = torch.squeeze(kernel, 1)

            blur = blur.to(self.device)
            sharp = sharp.to(self.device)

            self.optimizer.zero_grad()

            deblur = self.model(blur, kernel)
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

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (blur, sharp, kernel, filename) in enumerate(tqdm_test):

                blur = torch.squeeze(blur, 0)
                # targeted = torch.squeeze(targeted, 0)
                kernel = torch.squeeze(kernel, 0)
                blur = blur.to(self.device)
                # targeted = targeted.to(self.device)

                sigma = self.startGaussian(blur)
                # deblur = self.model(blur, kernel)
                # print(type(deblur[0]))
                # print(sigma.size())
                # print(blur.size())
                # a = sigma+blur
                # print(a.size())
                deblur = self.projected_gradient_descent(blur=blur, kernel=kernel, sigma=sigma, target=None, make_adv=True, **att_kwargs)

                if self.args.save_images:
                    deblur = utils_deblur.postprocess(deblur[-1], rgb_range=self.args.rgb_range)
                    save_list = [deblur[0]]
                    self.ckp.save_images(filename, save_list)

            self.ckp.end_log(len(self.loader_test), train=False)

    def startGaussian(self, tensor, mean=0, std=0.1):
        return torch.randn(tensor.size(), device='cpu') * std + mean

    def projected_gradient_descent(self, blur, kernel, sigma, target=None, make_adv=False, constraint='inf', eps=0.5,
                                   step_size=0.1,
                                   iterations=10):
        # self.model = model
        # self.sigma = sigma
        loss_fn = nn.MSELoss()
        # self.input = input
        # clamp = (0, 1)
        orig_output = self.model(blur, kernel)
        sigma_adv = sigma.clone().detach().requires_grad_(True).to('cpu')

        if make_adv:
            """Performs the projected gradient descent attack on a images."""

            targeted = target is not None
            num_channels = self.args.n_colors
            for i, _ in enumerate(orig_output):
                for _ in range(iterations):
                    _sigma_adv = sigma_adv.clone().detach().requires_grad_(True)
                    adv_input = blur + _sigma_adv
                    prediction = self.model(adv_input, kernel)[i]
                    loss = loss_fn(prediction, target[i] if targeted else orig_output[i])
                    loss.requires_grad = True
                    loss.backward()

                    with torch.no_grad():
                        # Force the gradient step to be a fixed size in a certain norm
                        if constraint == 'inf':
                            gradients = _sigma_adv.grad.sign() * step_size
                        else:
                            # Note .view() assumes batched image data as 4D tensor
                            gradients = _sigma_adv.grad * step_size / _sigma_adv.grad.view(_sigma_adv.shape[0], -1).norm(
                                constraint, dim=-1).view(-1, num_channels, 1, 1)

                        if targeted:
                            # Targeted: Gradient descent with on the loss of the (incorrect) target label
                            # w.r.t. the image data
                            sigma_adv -= gradients
                        else:
                            # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                            # the model parameters
                            sigma_adv += gradients

                    # Project back into l_norm ball and correct range
                    if constraint == 'inf':
                        # Workaround as PyTorch doesn't have elementwise clip
                        sigma_adv = torch.max(torch.min(sigma_adv, sigma + eps), sigma - eps)
                    else:
                        delta = sigma_adv - sigma

                        # Assume x and x_adv are batched tensors where the first dimension is
                        # a batch dimension
                        mask = delta.view(delta.shape[0], -1).norm(constraint, dim=1) <= eps

                        scaling_factor = delta.view(delta.shape[0], -1).norm(constraint, dim=1)
                        scaling_factor[mask] = eps

                        # .view() assumes batched images as a 4D Tensor
                        delta *= eps / scaling_factor.view(-1, 1, 1, 1)

                        sigma_adv = sigma + delta

                # sigma_adv = sigma_adv.clamp(*clamp)
                # sigma_adv.detach()
            return self.model(blur+sigma_adv, kernel)

        else:
            return orig_output

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
