import torch
import data
import model
import loss
from option import args
from trainer_vd import Trainer_VD
from logger import logger
from attacker import AttackerModel

torch.manual_seed(args.seed)
chkp = logger.Logger(args)

print("Deep Wiener Deconvolution Network")
loader = data.Data(args)
model = model.Model(args, chkp)
model = AttackerModel(model)
loss = loss.Loss(args, chkp) if not args.test_only else None
t = Trainer_VD(args, loader, model, loss, chkp)
while not t.terminate():
    t.train()
    t.test()


chkp.done()
