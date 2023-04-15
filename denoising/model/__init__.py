import os
import torch
import torch.nn as nn
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')

        self.args = args

        module = import_module('model.' + self.args.model.lower())
        self.model = module.make_model(args).to(self.args.device)

    def forward(self, *args):
        return self.model(*args)

    def get_model(self):
        return self.model.module
    #
    # def save(self):
    #
    #     filename = 'model_{}'.format(self.args.model)
    #     torch.save(
    #         self.model.state_dict(),
    #         os.path.join(self.args.model_path, 'model', '{}latest.pt'.format(filename))
    #     )
    #     print('The model have been saved.')
    #
    # def load(self, pre_train='.'):
    #     kwargs = {'map_location': lambda storage, loc: storage}
    #     if pre_train != '.':
    #         print('Loading model from {}'.format(pre_train))
    #         self.get_model().load_state_dict(
    #             torch.load(pre_train, **kwargs),
    #             strict=False
    #         )

        # elif resume:
        #     print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
        #     self.get_model().load_state_dict(
        #         torch.load(
        #             os.path.join(apath, 'model', 'model_latest.pt'),
        #             **kwargs
        #         ),
        #         strict=False
        #     )
        # elif self.args.test_only:
        #     self.get_model().load_state_dict(
        #         torch.load(
        #             os.path.join(apath, 'model', 'model_best.pt'),
        #             **kwargs
        #         ),
        #         strict=False
        #     )
