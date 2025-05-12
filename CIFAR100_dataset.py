# source: https://github.com/LJY-HY/cifar_pytorch-lightning/blob/master/datasets/CIFAR.py
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, num_workers, batch_size=128):
        super().__init__()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2470, 0.2435, 0.2616]
        # transforms.RandomCrop(32,padding=4),
        #                                     transforms.RandomHorizontalFlip(),
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.CIFAR10(root='./workspace/datasets/cifar10',train=True,download=True, transform=self.transform)
        datasets.CIFAR10(root='./workspace/datasets/cifar10',train=False,download=True, transform=self.transform)

    def setup(self, stage):
        cifar_train = datasets.CIFAR10(root='./workspace/datasets/cifar10',train=True,download=True, transform=self.transform)
        self.cifar_test = datasets.CIFAR10(root='./workspace/datasets/cifar10',train=False,download=True, transform=self.transform)
        self.cifar_train = cifar_train

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return cifar_val

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, num_workers, batch_size=128):
        super().__init__()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.CIFAR100(root='./workspace/datasets/cifar100', train=True, download=True, transform=self.transform)
        datasets.CIFAR100(root='./workspace/datasets/cifar100', train=False, download=True, transform=self.transform)

    def setup(self, stage):
        cifar_train = datasets.CIFAR100(root='./workspace/datasets/cifar100', train=True, download=True, transform=self.transform)
        self.cifar_test = datasets.CIFAR100(root='./workspace/datasets/cifar100', train=False, download=True, transform=self.transform)
        self.cifar_train = cifar_train

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
        return cifar_val

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)