# script for testing a training model
# Please custumize the cropping and padding operations and stopping conditions as demanded.

from __future__ import absolute_import, print_function
import models
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import data
import scipy.misc
import time
import scipy.io as sio
from options.running_options import Options
import torch.nn.functional as F
import utils
import imageio
import numpy as np
from attacker import AttackerModel
import warnings
warnings.filterwarnings('ignore')
import os
#
opt_parser = Options()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda:1" if use_cuda else "cpu")
crop_size = opt.CropSize # set as 0 if input is not padded in advance

def PSNR(output, source):
    mse = (source-output.clamp(0, 1)).pow(2).mean()
    return -10 * torch.log10(mse)

def NCC(output, source):
    output -= torch.mean(output)
    source -= torch.mean(source)
    ncc_ = F.conv2d(output, source)/(torch.sqrt(torch.sum(output ** 2)) * torch.sqrt(torch.sum(source ** 2)))
    return ncc_.item()

def write(data, constraint, eps):
    mdic = {'eps': eps, 'constraint': constraint,
            'SourceStandardPSNR': data[0], 'SourceAdversarialPSNR': data[1], 'SourceStandardNCC':data[2], 'SourceAdversarialNCC': data[3]}
    sio.savemat("{}.mat".format('StdModel_'+str(eps)), mdic)
    
    
attack_kwargs = {
    'constraint': '2',
    'eps': 0.04,
    'step_size': np.linspace(1,0.001,500),
    'iterations': 100,
    'random_start': False,
    'random_mode': "uniform_on_sphere"
}

# model
model = models.OptimizerRGDN(
        opt.StepNum,
        use_grad_adj=opt.UseGradAdj,
        use_grad_scaler=opt.UseGradScaler,
        use_reg=opt.UseReg,
        stop_epsilon=opt.StopEpsilon)

model_para = torch.load(opt.TrainedModelPath, map_location=device)
model.load_state_dict(model_para)
model = AttackerModel(model)


# model = nn.DataParallel(model)
model.to(device)
# 
model.eval()
##
model_name = opt.ModelName

# data path
# data_root = '../'
# dataset_name = 'rgdn_dataset'
data_path = opt.DataPath #data_root + dataset_name
outpath = opt.OutPath #data_root + dataset_name + '_results_' + model_name + '/'
utils.mkdir(outpath)

##
Dataset = data.BlurryImageDataset(data_path)
test_data_loader = DataLoader(Dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=0)

sample_num = test_data_loader.__len__()

psnr_adv, psnr_std, ncc_adv, ncc_std = [], [], [], []
for batch_idx, ( (y, k, kt, x_gt), sample_name) in enumerate(test_data_loader):
    print('%d / %d, %s' % (batch_idx+1, sample_num, sample_name[0]))
    y, kt, k = y.to(device), k.to(device), kt.to(device)
    if(opt.ImgPad):
        k_size = k.size()[2]
        padding_size = int((k_size / 2) * 1.5)
        y = torch.nn.functional.pad(y, [padding_size, padding_size, padding_size, padding_size], mode='replicate')

    start = time.time()
    output_, output_seq = model(y, k, kt, make_adv=True, **attack_kwargs)
    # output_len = len(output_seq)
    x_final = output_seq[-1]
    x_ = output_[-1]
    # print('Time {}'.format(time.time() - start))
    ##
    if (opt.ImgPad):
        y = utils.truncate_image(y, padding_size)
        x_final = utils.truncate_image(x_final, padding_size)
        x_ = utils.truncate_image(x_, padding_size)

    if (crop_size>0):
        x_est_np = utils.truncate_image(x_final, crop_size)
        x_np = utils.truncate_image(x_, crop_size)
    elif(crop_size==0):
        x_est_np = x_final.cpu()
        x_np = x_.cpu()
    else:
        crt_crop_size = int(k.size()[2] /2)
        x_est_np = utils.truncate_image(x_final, crt_crop_size)
        x_np = utils.truncate_image(x_, crt_crop_size)

    x_est_np = utils.tensor_to_np_img(x_est_np)
    x_np = utils.tensor_to_np_img(x_np)
    #

    x_est_np = utils.box_proj(x_est_np)
    x_np = utils.box_proj(x_np)
    
    sample_name_full = sample_name[0]
    sample_name = sample_name_full[0:len(sample_name_full) - 4]
    # imageio.imwrite(outpath + sample_name + '_estx.png', (x_est_np * 255).astype(np.uint8))

    with torch.no_grad():
        psnr_adv.append(PSNR(torch.from_numpy(x_est_np).reshape(1,3,256,256), x_gt.reshape(3,256,256).unsqueeze(0)))
        psnr_std.append(PSNR(torch.from_numpy(x_np).reshape(1,3,256,256), x_gt.reshape(3,256,256).unsqueeze(0)))
        ncc_adv.append(NCC(torch.from_numpy(x_est_np).reshape(1,3,256,256), x_gt.reshape(3,256,256).unsqueeze(0)))
        ncc_std.append(NCC(torch.from_numpy(x_np).reshape(1,3,256,256), x_gt.reshape(3,256,256).unsqueeze(0)))
        
    torch.cuda.empty_cache()

    write([psnr_std, psnr_adv, ncc_std, ncc_adv], attack_kwargs['constraint'], attack_kwargs['eps'])