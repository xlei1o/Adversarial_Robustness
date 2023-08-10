#!/usr/bin/env python
# -*- coding: utf-8 -*-

def set_template(args):

    if args.template == 'DWDN':

        args.task = 'Deblurring'

        if args.task == "Deblurring":
            args.data_train = 'BLUR_IMAGE'
            args.dir_data = './TrainingData'
            args.data_test = 'BLUR_IMAGE'
            args.dir_data_test = './TestData'
            args.reset = False
            args.model = "deblur"
            args.test_only = True
            args.pre_train = "./model/model_B.pt"
            args.measure_path = "./measurements"
            args.save = "deblur"
            args.loss = "1*L1"
            args.patch_size = 256
            args.batch_size = 8
            args.grad_clip = 0.5
            if args.test_only:
                args.save = "deblur_test"
            args.save_results = True
            args.save_models = True
            args.no_augment = True
            args.save_images = False
            
            args.blur_type = 'uniform'
            args.targeted = False
            # For L2 pertubations, here eps denotes the relative adversarial noise level.
            args.eps = 0.08
            args.adv_iterations = 100
            # For L2 pertubations, we use open loop step-size control rule. 
            args.adv_step = 0.001
            args.constraint = '2'
            
            args.test_size = 1000

            args.cuda_device = 'cuda:1'
            args.n_GPUs = 1