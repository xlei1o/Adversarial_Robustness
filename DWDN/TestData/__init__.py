import torch
import torchvision
import torchvision.transforms as T
from TestData import common
from tqdm import tqdm
import os


class TestData:
    def __init__(self, args):
        self.args = args

        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

        dataset_test_base = torchvision.datasets.CelebA(root='.', download=True, split='test',
                                                         transform=transform)
        dataset_target_base = torchvision.datasets.CelebA(root='.', download=True, split='valid',
                                                         transform=transform)

        dataset_test_base, _ = torch.utils.data.random_split(dataset_test_base,
                                                              [self.args.test_size,
                                                               len(dataset_test_base) - self.args.test_size])
        dataset_target = torch.utils.data.Subset(dataset_target_base, range(self.args.test_size))

        dataset_target = common.BlurDataset(dataset=dataset_target)

        dataset_target_loader = torch.utils.data.DataLoader(dataset_target,
                                                            batch_size=1, shuffle=True,
                                                            num_workers=0)
        apath = os.path.join(self.args.dir_data_test)
        test_dir_clear = os.path.join(apath, "clearImage/")
        test_dir_blur = os.path.join(apath, "blurredImage/")
        test_dir_target = os.path.join(apath, "targetedImage/")
        test_dir_kernel = os.path.join(apath, "kernelImage/")      
        if not os.path.exists(os.path.dirname(test_dir_kernel)):
                    os.makedirs(test_dir_blur)
                    os.makedirs(test_dir_target)
                    os.makedirs(test_dir_kernel)
                    os.makedirs(test_dir_clear)

        if self.args.blur_type == 'gaussian':
            dataset_test = common.BlurDataset(dataset=dataset_test_base,
                                            transform=common.AddGaussianBlur())
            
            dataset_test_loader = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=1, shuffle=True,
                                                                num_workers=0)

            
            tqdm_test = tqdm(dataset_test_loader, desc=f"Generating Test Data", position=0, ncols=80)
            for index, (clear, blur, kernel) in enumerate(tqdm_test):

                blur_name = os.path.join(test_dir_blur, "{}.png".format(str(index)))
                clear_name = os.path.join(test_dir_clear, "{}.png".format(str(index)))
                kernel_name = os.path.join(test_dir_kernel, "{}.mat".format(str(index)))

                common.save_img(blur, blur_name)
                common.save_kernel(kernel, kernel_name)
                common.save_img(clear, clear_name)

        else:
                
                dataset_test = common.BlurDataset(dataset=dataset_test_base,
                                    transform=common.AddUniformBlur())
        
                
                dataset_test_loader = torch.utils.data.DataLoader(dataset_test,
                                                                    batch_size=1, shuffle=True,
                                                                    num_workers=0)
                
                tqdm_test = tqdm(dataset_test_loader, desc=f"Generating Test Data", position=0, ncols=80)
                for index, (clear, blur, kernel) in enumerate(tqdm_test):

                    blur_name = os.path.join(test_dir_blur, "{}.png".format(str(index)))
                    clear_name = os.path.join(test_dir_clear, "{}.png".format(str(index)))
                    kernel_name = os.path.join(test_dir_kernel, "{}.png".format(str(index)))

                    common.save_img(blur, blur_name)
                    common.save_img(kernel.squeeze(0), kernel_name)
                    common.save_img(clear, clear_name)


        tqdm_adv = tqdm(dataset_target_loader, desc=f"Generating Adversarial Data", position=0, ncols=80)
        for index, (target) in enumerate(tqdm_adv):
            target_name = os.path.join(test_dir_target, "{}.png".format(str(index)))
            common.save_img(target, target_name)



