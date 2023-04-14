import torch.nn as nn
import torch.nn.functional as F
from model.common import Encoder, Decoder


def make_model(args, parent=False):
    return UNet(args)


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        self.enc_chs = (args.unet_classes, ) + args.enc_chs
        self.dec_chs = args.dec_chs
        self.num_class = args.unet_classes

        self.encoder = Encoder(self.enc_chs)
        self.decoder = Decoder(self.dec_chs)
        self.head = nn.Conv2d(self.dec_chs[-1], self.num_class, 1)
        self.retain_dim = False
        self.out_sz = (572, 572)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out
