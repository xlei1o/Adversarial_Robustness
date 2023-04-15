from option import args
import matplotlib.pyplot as plt

import numpy as np
import data
import model
from trainer import Trainer

print("Adversarial Attacks for Image Denoising")

loader = data.Data(args)
# the dataset now returns pairs images (x,y)
fig, ax = plt.subplots(ncols=2, figsize=(5,5*2))
ax[0].imshow(loader.y_plt); ax[0].set_title(f"Noisy image (level: {args.noise_level_train})")
ax[1].imshow(loader.x_plt); ax[1].set_title("Original image")
fig.tight_layout()

model = model.Model(args)
t = Trainer(args, loader, model)
if not args.test_only:
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("adversarial error")
    ax[0].scatter(np.arange(0,args.epochs), np.array(t.test_loss_std_adv))
    ax[0].scatter(np.arange(0,args.epochs), np.array(t.test_loss_adv_adv))
    ax[0].plot(np.arange(0,args.epochs), np.array(t.test_loss_std_adv))
    ax[0].plot(np.arange(0,args.epochs), np.array(t.test_loss_adv_adv))
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("standard error")
    ax[1].scatter(np.arange(0,args.epochs), np.array(t.test_loss_std_std))
    ax[1].scatter(np.arange(0,args.epochs), np.array(t.test_loss_adv_std))
    ax[1].plot(np.arange(0,args.epochs), np.array(t.test_loss_std_std), label="Standard Training")
    ax[1].plot(np.arange(0,args.epochs), np.array(t.test_loss_adv_std), label="Adversarial Training")
    fig.legend()
    fig.tight_layout()
    plt.show()