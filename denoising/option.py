import argparse
import template

parser = argparse.ArgumentParser(description='AAD')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='AAD',
                    help='You can set various templates in option.py')
parser.add_argument('--task', type=str, default='Denoising',
                    help='Type of task (Denoising)')

# Hardware specifications
parser.add_argument('--device', default="cuda",
                    help='use cpu or cuda-suported gpu')
parser.add_argument('--data_as_tensorlist', action='store_true',
                    help='if true loads all data into gpu before training/eval')

# noise levels (=sigma above)
parser.add_argument('--noise_level_train', default=0.1)
parser.add_argument('--noise_level_test', default=0.1)

# UNet model architecture
parser.add_argument('--enc_chs', default=(8, 16, 32, 64))
parser.add_argument('--dec_chs', default=(64, 32, 16, 8))
parser.add_argument('--unet_classes', default=3,
                    help='RGB')

# dataloader / training
parser.add_argument('--dataloader_batch_size', type=int, default=256)
parser.add_argument('--dataloader_num_workers', type=int, default=0)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--train_size', type=int, default=200000)
parser.add_argument('--test_size', type=int, default=2000)

# Model
parser.add_argument('--train_model', action='store_true',
                    help='if true, training, if false, loading')

parser.add_argument('--model_path', type=str, default='./model')

parser.add_argument('--model', type=str, default='UNET',
                    help='model name')
parser.add_argument('--test_only', action='store_true')


# adversarial attacks / training
parser.add_argument("--eps_rel", default=0.1,
                    help='will be multiplied by math.sqrt(n), where n is the image dimension (e.g. 3*32*32)')
parser.add_argument('--adv_iterations', type=int, default=1,
                    help='number of PGD iterations; with value 1 it is a variant of FastFGSM (not necessarily '
                         'sufficiently strong attack')

args = parser.parse_args()
template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
