import torch
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import trange


class AttackerStep:
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        raise NotImplementedError

    def step(self, x, g, i):
        raise NotImplementedError

    def random_perturb(self, x):
        raise NotImplementedError

    def get_best(self, x, loss, x_new, loss_new):
        loss_per_ex = torch.sum(loss.view(x.shape[0], -1), dim=1)
        loss_new_per_ex = torch.sum(loss_new.view(x_new.shape[0], -1), dim=1)
        replace = loss_per_ex < loss_new_per_ex
        replace_e = replace.view(-1, *([1] * (len(x.shape) - 1))).expand(x.shape)
        return torch.where(replace_e, x_new, x), torch.where(replace_e, loss_new, loss)


class L2Step(AttackerStep):

    def project(self, x):
        diff = x - self.orig_input
        eps = torch.norm(self.orig_input) * self.eps
        diff = diff.renorm(p=2, dim=0, maxnorm=eps)
        return self.orig_input + diff

    def step(self, x, g, i):
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))
        scaled_g = g / (g_norm + 1e-10)
        return x - scaled_g * self.step_size[i]

    def random_perturb(self, x, mode="uniform_in_sphere"):
        if mode == "uniform_in_sphere":
            l = len(x.shape) - 1
            rp = torch.randn_like(x)
            rp2 = rp.view(x.shape[0], -1)
            u = torch.rand(rp2.shape[0], device=x.device)
            n = rp2.shape[-1]
            perturb = rp * torch.pow(u, 1.0 / n).view(-1, *([1] * l)) / torch.norm(rp2, dim=1).view(-1, *([1] * l))
            return x + self.eps * perturb
        elif mode == "uniform_on_sphere":
            l = len(x.shape) - 1
            rp = torch.randn_like(x)
            rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1] * l))
            return x + self.eps * rp / (rp_norm + 1e-10)
        else:
            raise Exception(f"mode: {mode} not supported for L2-type perturbations.")


class LinfStep(AttackerStep):
    def project(self, x):
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return self.orig_input + diff

    def step(self, x, g, i):
        return x - torch.sign(g) * self.step_size

    def random_perturb(self, x, mode):
        return x + 2 * (torch.rand_like(x) - 0.5) * self.eps


STEPS = {
    '2': L2Step,
    'inf': LinfStep
}


class AttackerModel(torch.nn.Module):
    def __init__(self, model):
        super(AttackerModel, self).__init__()
        self.model = model

    def forward(self, x, kernel, kernel_T, make_adv=False, constraint="inf", eps=0.3, step_size=0.1, iterations=2,
                random_start=False, random_mode="uniform_in_sphere"):

        if make_adv:
            prev_training = bool(self.training)
            self.eval()

            ###### (adv. training begin)
            orig_input = x.clone()
            criterion = torch.nn.MSELoss(reduction='mean')

            target = self.model(orig_input, kernel, kernel_T)[-1]


            step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
            step = step_class(eps=eps, orig_input=orig_input, step_size=step_size)  ## to attackerstep

            def calc_loss(input):
                adv_deblur = self.model(input, kernel, kernel_T)
                loss = -criterion(adv_deblur[-1], target)
                return loss
 
                
            # Main function for making adversarial examples
            def get_adv_examples(x):
                # keeping track of best candidates

                # Random start (to escape certain types of gradient masking)
                if random_start:
                    x = step.random_perturb(x, random_mode)

                # PGD iterates
                for i in trange(iterations, desc=f"Adversarial Iteration for Current Image", position=1, ncols=120, leave=False):

                    x = x.clone().detach().requires_grad_(True)
                    loss = calc_loss(x)

                    grad, = torch.autograd.grad(loss, x)
                    
                    with torch.no_grad():

                        x = step.step(x, grad,i)
                        x = step.project(x)

                return x.clone().detach()

            x_init = x.clone().detach()
            adv_ret= get_adv_examples(x_init)

            ###### (adv. training end)
            if (prev_training):
                self.train()
            inp = adv_ret
        else:
            inp = x

        return self.model(x, kernel, kernel_T), self.model(inp, kernel, kernel_T)