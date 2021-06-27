import torch
import torchvision.transforms as T
import torchvision.datasets
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Subset, DataLoader


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)


def get_train_val_subsets(train_set, valid_set):
    num_train = len(train_set)
    indices = torch.randperm(num_train).tolist()
    valid_size = int(np.floor(0.05 * num_train))

    train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

    train_set = Subset(train_set, train_idx)
    valid_set = Subset(valid_set, valid_idx)
    return train_set, valid_set


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)]
        )
        self.test_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)]
        )
        self.cfg = cfg

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.cfg.dataset.path, train=True, download=True)

    def setup(self, stage=None):
        train_set = torchvision.datasets.CIFAR10(self.cfg.dataset.path,
                                                 train=True,
                                                 transform=self.train_transforms)
        valid_set = torchvision.datasets.CIFAR10(self.cfg.dataset.path,
                                                 train=True,
                                                 transform=self.test_transforms)
        self.train_set, self.valid_set = get_train_val_subsets(train_set, valid_set)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_set,
                                  shuffle=True,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  **self.cfg.dataloaders.train)

        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_set,
                                  shuffle=False,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  **self.cfg.dataloaders.validation)
        return valid_loader

    def test_dataloader(self):
        return self.val_dataloader()


class CELEBADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_transforms = T.Compose([
            CropTransform((25, 50, 25 + 128, 50 + 128)),
            T.Resize(128),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)]
        )
        self.test_transforms = T.Compose([
            CropTransform((25, 50, 25 + 128, 50 + 128)),
            T.Resize(128),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)]
        )
        self.cfg = cfg

    def prepare_data(self):
        torchvision.datasets.CelebA(self.cfg.dataset.path, split='train', download=True)

    def setup(self, stage=None):
        train_set = torchvision.datasets.CelebA(self.cfg.dataset.path,
                                                train=True,
                                                transform=self.train_transforms)
        valid_set = torchvision.datasets.CelebA(self.cfg.dataset.path,
                                                train=True,
                                                transform=self.test_transforms)
        self.train_set, self.valid_set = get_train_val_subsets(train_set, valid_set)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_set,
                                  batch_size=self.cfg.training.dataloader.batch_size,
                                  shuffle=True,
                                  num_workers=self.cfg.training.dataloader.num_workers,
                                  pin_memory=True,
                                  drop_last=self.cfg.training.dataloader.drop_last,
                                  persistent_workers=True)

        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_set,
                                  batch_size=self.cfg.validation.dataloader.batch_size,
                                  shuffle=False,
                                  num_workers=self.cfg.validation.dataloader.num_workers,
                                  pin_memory=True,
                                  drop_last=self.cfg.validation.dataloader.drop_last,
                                  persistent_workers=True)
        return valid_loader

    def test_dataloader(self):
        return self.val_dataloader()


def get_datamodule(cfg):
    if cfg.dataset.name.upper() == "CIFAR10":
        return CIFAR10DataModule(cfg)
    elif cfg.dataset.name.upper() == "CELEBA":
        return CELEBADataModule(cfg)
    else:
        raise ValueError(cfg.dataset.name)
