import numpy as np
import random
import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data.sampler import  WeightedRandomSampler
def ImageFolder(data_dir, img_size, batch_size, model):
    std = 1. / 255.
    means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
    if model == 'train':

        train_data = torchvision.datasets.ImageFolder(data_dir,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(img_size),
                                                        transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.CenterCrop(size=img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean = means, std = [std]*3)
                                                    ]))
        weights = [1,10,8,20,21]
        sampler = WeightedRandomSampler(weights,\
                                num_samples=32000,\
                                replacement=True)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,sampler = sampler)
        
        

        return train_data,train_data_loader
    elif model == 'test':

        test_data = torchvision.datasets.ImageFolder(data_dir,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(img_size),
                                                        torchvision.transforms.CenterCrop(size=img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=means, std=[std]*3)
                                                        ]))
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        return test_data, test_data_loader