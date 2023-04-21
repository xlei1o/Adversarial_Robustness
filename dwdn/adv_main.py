import torch
import data
import model
import loss
from option import args
from trainer_adv import Trainer_adv
from logger import logger
from attacker import AttackerModel
import warnings
warnings.filterwarnings('ignore')


torch.manual_seed(args.seed)
chkp = logger.Logger(args)

print("Deep Wiener Deconvolution Network")
loader = data.Data(args)
model = model.Model(args, chkp)
model = AttackerModel(model)
loss = loss.Loss(args, chkp) if not args.test_only else None
t = Trainer_adv(args, loader, model, loss, chkp)
while not t.terminate():
    t.train()
    t.test()


chkp.done()
