import torch
import gc
import data
import model
import loss
from option import args
from trainer_adv import Trainer_adv
from logger import logger
from TrainingData import TrainingData
from attacker import AttackerModel
import warnings
warnings.filterwarnings('ignore')

# _ = TrainingData(args)  

torch.manual_seed(args.seed)
chkp = logger.Logger(args)


print("Deep Wiener Deconvolution Network")
if args.training_data_generation:
    _ = TrainingData(args)  

gc.collect()
torch.cuda.empty_cache()

loader = data.Data(args)
model = model.Model(args, chkp)
model = AttackerModel(model)
loss = loss.Loss(args, chkp) if not args.test_only else None
t = Trainer_adv(args, loader, model, loss, chkp)
while not t.terminate():
    t.train()
    t.test()


chkp.done()