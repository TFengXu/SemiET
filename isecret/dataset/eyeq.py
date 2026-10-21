# -*- coding: utf-8 -*-
from isecret.dataset.common import BaseDataset, CustomDataSet, OfflineDegradeCustomDataSet
from isecret.dataset.build import DATASET_REGISTRY
import os
from torchvision import transforms


@DATASET_REGISTRY.register('eyeq')
class Eyeq(BaseDataset):
    '''
    EyeQ Dataset, which is an unpaired dataset
    '''
    def __init__(self, args, task):
        self.task = task
        self.args = args
        self._init_dataset()
        BaseDataset.__init__(self, args, self.good_dataset, self.bad_dataset)

    def _init_dataset(self):
        # Initialize pre-processing
        if self.task == 'train':
            transform = transforms.Compose([transforms.Resize((self.args.data.size, self.args.data.size)),
                                            transforms.RandomCrop((self.args.data.crop_size, self.args.data.crop_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.args.data.mean,
                                                                 self.args.data.std)])
            mask_transform = transforms.Compose([transforms.Resize((self.args.data.size, self.args.data.size), transforms.InterpolationMode.NEAREST),
                                            transforms.RandomCrop((self.args.data.crop_size, self.args.data.crop_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.Resize((self.args.data.size, self.args.data.size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.args.data.mean,
                                                                 self.args.data.std)])
            mask_transform = transforms.Compose([transforms.Resize((self.args.data.size, self.args.data.size), transforms.InterpolationMode.NEAREST),
                                           transforms.RandomCrop((self.args.data.crop_size, self.args.data.crop_size)),
                                            transforms.ToTensor()])
        # Initialize directory
        dataset = OfflineDegradeCustomDataSet
        good_dir = os.path.join(self.args.data.data_dir, self.task, 'crop_good')
        degrade_dir = os.path.join(self.args.data.data_dir, self.task, 'degrade_good')
        if self.args.train.lambda_lesion_dice > 0:
            good_lesion_dir = os.path.join(self.args.data.data_dir, self.task, 'crop_good_pseudo_lesion')
            bad_lesion_dir = os.path.join(self.args.data.data_dir, self.task, 'crop_usable_pseudo_lesion')
            good_saliency_dir = None
            saliency_ignore_mask_dir = None
        elif self.args.train.lambda_saliency_mse > 0:
            good_saliency_dir = os.path.join(self.args.data.data_dir, self.task, 'saliency_good')
            saliency_ignore_mask_dir = os.path.join(self.args.data.data_dir, self.task, 'saliency_ignore_mask')
            good_lesion_dir = None
            bad_lesion_dir = None
        else:
            good_lesion_dir = None
            bad_lesion_dir = None
            good_saliency_dir = None
            saliency_ignore_mask_dir = None
        self.good_dataset = dataset(self.args, good_dir, degrade_dir, transform, 
                                    mask_transform=mask_transform, lesion_dir=good_lesion_dir, 
                                    saliency_dir=good_saliency_dir, saliency_ignore_mask_dir=saliency_ignore_mask_dir,
                                    saliency_transform=transform, including_idt=True)
        bad_dir = os.path.join(self.args.data.data_dir, self.task, 'crop_usable')
        self.bad_dataset = CustomDataSet(self.args, bad_dir, transform)



@ DATASET_REGISTRY.register('minieyeq')
class MiniEyeq(Eyeq):
    def __init__(self, args, task):
        super().__init__(args, task)