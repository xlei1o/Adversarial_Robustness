import torch.nn as nn
import torch
import torch.nn.functional as F


def projected_gradient_descent(model, args, input, sigma, target=None, constraint='inf', eps=0.5, step_size=0.1, iterations=10):
    # self.model = model
    # self.sigma = sigma
    loss_fn = nn.MSELoss()
    # self.input = input
    clamp=(0, 1)

    """Performs the projected gradient descent attack on a images."""
    sigma_adv = sigma.clone().detach().requires_grad_(True).to('cpu')
    targeted = target is not None
    num_channels = args.n_colors

    orig_output = model(input)

    for i in range(iterations):
        _sigma_adv = sigma_adv.clone().detach().requires_grad_(True)

        prediction = model(input+_sigma_adv)
        loss = loss_fn(prediction, target if targeted else orig_output)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if constraint == 'inf':
                gradients = _sigma_adv.grad.sign() * step_size
            else:
                # Note .view() assumes batched image data as 4D tensor
                gradients = _sigma_adv.grad * step_size / _sigma_adv.grad.view(_sigma_adv.shape[0], -1).norm(constraint, dim=-1).view(-1, num_channels, 1, 1)

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
    return model(input+sigma_adv)
#
#
# import torch
# class AttackerStep:
#     def __init__(self, orig_input, eps, step_size, use_grad=True):
#         self.orig_input = orig_input
#         self.eps = eps
#         self.step_size = step_size
#         self.use_grad = use_grad
#
#     def project(self, x):
#         raise NotImplementedError
#
#     def step(self, x, g):
#         raise NotImplementedError
#
#     def random_perturb(self, x):
#         raise NotImplementedError
#
#     def get_best(self, x, loss, x_new, loss_new):
#         loss_per_ex = torch.sum(loss.view(x.shape[0], -1), dim=1)
#         loss_new_per_ex = torch.sum(loss_new.view(x_new.shape[0], -1), dim=1)
#         replace = loss_per_ex < loss_new_per_ex
#         replace_e = replace.view(-1, *([1]*(len(x.shape)-1))).expand(x.shape)
#         return torch.where(replace_e, x_new, x), torch.where(replace_e, loss_new, loss)
#
# class L2Step(AttackerStep):
#     """
#     Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
#     and :math:`\epsilon`, the constraint set is given by:
#     .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
#     """
#     def project(self, x):
#         """
#         """
#         diff = x - self.orig_input
#         diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
#         return self.orig_input + diff
#
#     def step(self, x, g):
#         l = len(x.shape) - 1
#         g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
#         scaled_g = g / (g_norm + 1e-10)
#         return x + scaled_g * self.step_size
#
#     def random_perturb(self, x, mode="uniform_in_sphere"):
#         if mode == "uniform_in_sphere":
#             l = len(x.shape) - 1
#             rp = torch.randn_like(x)
#             rp2 = rp.view(x.shape[0], -1)
#             u = torch.rand(rp2.shape[0], device=x.device)
#             n = rp2.shape[-1]
#             perturb = rp * torch.pow(u, 1.0/n).view(-1, *([1]*l)) / torch.norm(rp2, dim=1).view(-1, *([1]*l))
#             return x + self.eps*perturb
#         elif mode == "uniform_on_sphere":
#             l = len(x.shape) - 1
#             rp = torch.randn_like(x)
#             rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
#             return x + self.eps * rp / (rp_norm + 1e-10)
#         else:
#             raise Exception(f"mode: {mode} not supported for L2-type perturbations.")
#
#
# class LinfStep(AttackerStep):
#     def project(self, x):
#         diff = x - self.orig_input
#         diff = torch.clamp(diff, -self.eps, self.eps)
#         return self.orig_input + diff
#
#     def step(self, x, g):
#         l = len(x.shape) - 1
#         g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
#         scaled_g = g / (g_norm + 1e-10)
#         return x + scaled_g * self.step_size
#
#     def random_perturb(self, x, mode):
#         return x + 2 * (torch.rand_like(x) - 0.5) * self.eps
#
# STEPS = {
#     '2': L2Step,
#     'inf' : LinfStep
# }
#
# class AttackerModel(torch.nn.Module):
#     def __init__(self, model):
#         super(AttackerModel, self).__init__()
#         self.model = model
#
#     def forward(self, x, target=None, make_adv=False, constraint="2", eps=0.1, step_size=0.1, iterations=10,
#                 random_start=False, random_restarts=5, random_mode="uniform_in_sphere", use_best=False, restart_use_best=True):
#
#         if make_adv:
#             prev_training = bool(self.training)
#             self.eval()
#
#             ###### (adv. training begin)
#             orig_input = x.clone()
#             criterion = torch.nn.MSELoss(reduction='none')
#
#             step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
#             step = step_class(eps=eps, orig_input=orig_input, step_size=step_size)
#
#             def calc_loss(inp, target):
#                 output = self.model(inp)
#                 return criterion(output, target), output
#
#             # Main function for making adversarial examples
#             def get_adv_examples(x, losses):
#                 # keeping track of best candidates
#                 best_x = x.clone().detach()
#                 best_loss = losses.clone().detach()
#
#                 # Random start (to escape certain types of gradient masking)
#                 if random_start:
#                     x = step.random_perturb(x, random_mode)
#                     losses, _ = calc_loss(x, target)
#                     best_x, best_loss = step.get_best(best_x, best_loss, x, losses) if use_best else (x, losses)
#
#                 # PGD iterates
#                 for i in range(iterations):
#                     x = x.clone().detach().requires_grad_(True)
#                     losses, _ = calc_loss(x, target)
#                     assert losses.shape[0] == x.shape[0], \
#                             'Shape of losses must match input!'
#
#                     loss = torch.mean(losses)
#
#                     grad, = torch.autograd.grad(loss, [x])
#
#                     with torch.no_grad():
#                         #best_buffer.update_best(losses.clone().detach(), x.clone().detach())
#                         best_x, best_loss = step.get_best(best_x, best_loss, x, losses) if use_best else (x, losses)
#
#                         x = step.step(x, grad)
#                         x = step.project(x)
#
#                 losses, _ = calc_loss(x, target)
#                 best_x, best_loss = step.get_best(best_x, best_loss, x, losses) if use_best else (x, losses)
#                 if torch.any(losses != best_loss):
#                     print(f"Loss improved from {losses.sum()} to {best_loss.sum()}")
#
#                 return best_x.clone().detach(), best_loss.clone().detach()
#
#             x_init = x.clone().detach()
#             loss_init, _ = calc_loss(x_init, target)
#
#             if random_restarts:
#                 best_x = x_init
#                 best_loss = loss_init
#
#                 for _ in range(random_restarts):
#                     old_loss = best_loss.clone()
#                     new_x, new_loss = get_adv_examples(x_init, loss_init)
#                     best_x, best_loss = step.get_best(best_x, best_loss, new_x, new_loss) if restart_use_best else (new_x, new_loss)
#
#                 adv_ret = best_x
#             else:
#                 adv_ret,_ = get_adv_examples(x_init, loss_init)
#
#             ###### (adv. training end)
#             if(prev_training):
#                 self.train()
#             inp = adv_ret
#         else:
#             inp = x
#
#         return inp, self.model(inp)