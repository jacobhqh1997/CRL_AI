import os
import random
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import glob
from data_loaders.transformscopy import get_transform
from monai.data import CacheDataset, DataLoader,DistributedSampler

def get_dataloaders(config):
    train_images = sorted(glob.glob(os.path.join(config.data.data_dir, "train", "imagesTr", "*.nii.gz")))
    pnet_masks =sorted (glob.glob(os.path.join(config.data.data_dir, "train", "pnet_masks", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(config.data.data_dir, "train", "labelsTr", "*.nii.gz")))
    val_images = sorted(glob.glob(os.path.join(config.data.data_dir, "valid", "imagesTr", "*.nii.gz")))
    val_masks = sorted(glob.glob(os.path.join(config.data.data_dir, "valid", "pnet_masks", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(config.data.data_dir, "valid", "labelsTr", "*.nii.gz")))    
    data_dicts_train = [{"image": image_name, "P_mask": mask_name,  "label": label_name} for image_name, mask_name,label_name in zip(train_images, pnet_masks,train_labels)]
    data_dicts_valid = [{"image": image_name, "P_mask": mask_name,  "label": label_name} for image_name, mask_name,label_name in zip(val_images, val_masks,val_labels)]
    dataloaders = {}
    for split in ["train", "valid"]:
        if split == "train":
            train_dataset = CacheDataset(data=data_dicts_train, transform=get_transform(split),cache_num=50, num_workers=config.data.num_workers)
            if config.exp.multi_gpu:
                train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True,  shuffle=True)
            else:
                train_sampler = None
            dataloaders[split] = DataLoader(
                dataset=train_dataset,
                batch_size=config.data.batch_size, 
                num_workers=config.data.num_workers,
                shuffle=False,
                pin_memory=False,
                sampler=train_sampler
            )
        elif split == "valid":
            valid_dataset = CacheDataset(data=data_dicts_valid, transform=get_transform(split),cache_num=50, num_workers=config.data.num_workers)
            if config.exp.multi_gpu:
                valid_sampler = DistributedSampler(dataset=valid_dataset, even_divisible=True,  shuffle=False)
            else:
                valid_sampler = None
            dataloaders[split] = DataLoader(
                dataset=valid_dataset,
                batch_size=1, 
                shuffle=False, 
                num_workers=config.data.num_workers,
                pin_memory=False,
                sampler=valid_sampler         
            )       
    return dataloaders