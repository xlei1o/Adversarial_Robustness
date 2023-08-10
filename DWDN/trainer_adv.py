import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from tqdm import tqdm
import scipy.io as sio
import utils_deblur



# def PSNR(output, source):
#     mse = F.mse_loss(output, source)
#     return 10*torch.log10(torch.max(source)**2/mse)

def PSNR(output, source):
    mse = (source-output.clamp(0, 1)).pow(2).mean()
    return -10 * torch.log10(mse)

def NCC(output, source):
    output -= torch.mean(output)
    source -= torch.mean(source)
    ncc_ = F.conv2d(output, source)/(torch.sqrt(torch.sum(output ** 2)) * torch.sqrt(torch.sum(source ** 2)))
    return ncc_.item()

def write(data, targeted, path, constraint, eps, step_size, iterations, random_start, random_mode):
    if targeted:
        mdic = {'eps': eps, 'constraint': constraint, 'PGD_iterations': iterations, 'Targeted': targeted, 
                'SourceStandardPSNR': data[0], 'SourceAdversarialPSNR': data[1], 'SourceStandardNCC':data[2], 'SourceAdversarialNCC': data[3], 'TargetPSNR': data[4], 'TargetNCC': data[5]}
        apath = os.path.join(path, "targeted/")
    else:
        mdic = {'eps': eps, 'constraint': constraint, 'PGD_iterations': iterations, 'Targeted': targeted, 
                'StandardPSNR': data[0], 'AdversarialPSNR': data[1], 'StandardNCC':data[2], 'AdversarialNCC': data[3]}
        apath = os.path.join(path, "untargeted/")
    filename = os.path.join(apath, "{}.mat".format('NoiseModel_'+str(eps)+'_'+constraint))
    sio.savemat(filename, mdic)

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
        attack_kwargs = {
            'constraint': self.args.constraint,
            'eps': self.args.eps,
            'step_size': self.args.adv_step,
            'iterations': self.args.adv_iterations,
            'random_start': False,
            'random_mode': "uniform_on_sphere"
        }
        if self.args.constraint == '2':
            attack_kwargs['step_size'] = np.linspace(1,0.001,500)
            
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

        psnr_adv, psnr_std = [], []
        ncc_adv, ncc_std = [], []
        target_psnr, target_ncc = [], []

        if self.args.targeted:
            tqdm_test = tqdm(self.loader_test, desc=f"Test Progress", position=0, ncols=80)
            for idx_img, (blur, clear, kernel, target, filename) in enumerate(tqdm_test):
                blur = torch.squeeze(blur, 0)
                target = torch.squeeze(target, 0)
                kernel = torch.squeeze(kernel, 0)
                clear = torch.squeeze(clear, 0)

                blur = blur.to(self.device)
                target = target.to(self.device)

                deblur, deblur_adv = self.model(blur, kernel, target=target, make_adv=True, **attack_kwargs)

                with torch.no_grad():
                    deblur_ = deblur[-1].to('cpu')
                    deblur_adv_ = deblur_adv[-1].to('cpu')
                    target_ = target.to('cpu')

                    psnr_adv.append(PSNR(deblur_adv_, clear))
                    psnr_std.append(PSNR(deblur_, clear))
                    ncc_adv.append(NCC(deblur_adv_, clear))
                    ncc_std.append(NCC(deblur_, clear))

                    target_psnr.append(PSNR(deblur_adv_, target_))
                    target_ncc.append(NCC(deblur_adv_, target_))



                if self.args.save_images:
                    deblur_adv = utils_deblur.postprocess(deblur_adv[-1], rgb_range=self.args.rgb_range)
                    deblur = utils_deblur.postprocess(deblur[-1], rgb_range=self.args.rgb_range)
                    save_list_adv = [deblur_adv[0]]
                    save_list = [deblur[0]]
                    self.ckp.save_images(filename, save_list_adv, make_adv=True)
                    self.ckp.save_images(filename, save_list)

        else:
            tqdm_test = tqdm(self.loader_test, desc=f"Test Progress", position=0, ncols=80)
            for idx_img, (blur, clear, kernel, target, filename) in enumerate(tqdm_test):
                blur = torch.squeeze(blur, 0)
                kernel = torch.squeeze(kernel, 0)
                clear = torch.squeeze(clear, 0)

                blur = blur.to(self.device)

                deblur, deblur_adv = self.model(blur, kernel, make_adv=True, **attack_kwargs)
                
                with torch.no_grad():
                    deblur_ = deblur[-1].to('cpu')
                    deblur_adv_ = deblur_adv[-1].to('cpu')
                    
                    psnr_adv.append(PSNR(deblur_adv_, clear))
                    psnr_std.append(PSNR(deblur_, clear))
                    ncc_adv.append(NCC(deblur_adv_, clear))
                    ncc_std.append(NCC(deblur_, clear))


                if self.args.save_images:
                    deblur_adv = utils_deblur.postprocess(deblur_adv[-1], rgb_range=self.args.rgb_range)
                    deblur = utils_deblur.postprocess(deblur[-1], rgb_range=self.args.rgb_range)
                    save_list_adv = [deblur_adv[0]]
                    save_list = [deblur[0]]
                    self.ckp.save_images(filename, save_list_adv, make_adv=True)
                    self.ckp.save_images(filename, save_list)

        measurements = [np.array(psnr_std), np.array(psnr_adv), np.array(ncc_std), np.array(ncc_adv), np.array(target_psnr), np.array(target_ncc)]
        write(measurements, self.args.targeted, self.args.measure_path, **attack_kwargs)

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
