#!/usr/bin/env python
# -*- coding: utf-8 -*-

def set_template(args):

    if args.template == 'AAD':

        args.task = 'Denoising'

        if args.task == "Denoising":
            args.noise_level_train = 0.1
            args.noise_level_test = 0.1

            args.enc_chs = (8, 16, 32, 64)
            args.dec_chs = (64, 32, 16, 8)
            args.unet_classes = 3

            args.dataloader_batch_size = 256
            args.dataloader_num_workers = 0
            args.device = "cpu"
            args.data_as_tensorlist = False
            args.epochs = 2

            args.eps_rel = 0.1
            args.adv_iterations = 1

            args.train_size = 100
            args.test_size = 10

            args.train_model = True
            args.test_only = False
