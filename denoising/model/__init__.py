import os
import torch
import torch.nn as nn
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.args = args
        self.device = args.device
        self.train_model = args.train_model
        self.save_model = args.save_model
        self.model = args.model
        self.pre_train = args.pre_train
        self.resume = args.resume

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
        )

    def forward(self, *args):
        return self.model(*args)

    def get_model(self):
        return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False, filename=''):
        target = self.get_model()
        filename = 'model_{}'.format(filename)
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', '{}latest.pt'.format(filename))
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', '{}best.pt'.format(filename))
            )

        if self.save_models:
            # if epoch>self.args.epochs-15:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=False):

        kwargs = {'map_location': lambda storage, loc: storage}
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )

        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        # elif self.args.test_only:
        #     self.get_model().load_state_dict(
        #         torch.load(
        #             os.path.join(apath, 'model', 'model_best.pt'),
        #             **kwargs
        #         ),
        #         strict=False
        #     )