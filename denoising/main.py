from option import args
import matplotlib.pyplot as plt
import math
import data
import logger
import model
from trainer import Trainer
import attack

chkp = logger.Logger(args)

print("Adversarial Attacks for Image Denoising")

loader = data.Data(args)
# the dataset now returns pairs images (x,y)
fig, ax = plt.subplots(ncols=2, figsize=(5,5*2))
ax[0].imshow(loader.y_plt); ax[0].set_title(f"Noisy image (level: {args.noise_level_train})")
ax[1].imshow(loader.x_plt); ax[1].set_title("Original image")
fig.tight_layout()

# needs to be adapted for different settings (e.g. for classification one usually needs restarts and more iterations)
eps = args.eps_rel* math.sqrt(loader.n)
attack_kwargs = {
    'constraint': "2",
    'eps': eps,
    'step_size': 2.5 * (eps / args.adv_iterations),
    'iterations': args.adv_iterations,
    'random_start': True,
    'random_restarts': 0,
    'use_best': False,
    'random_mode': "uniform_in_sphere"
}

model = model.Model(args, chkp)
t = Trainer(args, loader, model, **attack_kwargs)
