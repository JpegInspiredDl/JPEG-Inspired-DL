from __future__ import print_function

import os
import numpy as np
import random
from torchvision import datasets, transforms
import torchvision
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Sampler
from .cub2011 import *
from .dogs import *
from .flowers import *
from .food101 import *
from .pets import *
#

# More dataset are available here
# https://github.com/OscarXZQ/weight-selection/blob/main/datasets.py

class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]

class BalancedSampler(Sampler):
    def __init__(self, dataset, num_samples_per_class=10):
        self.dataset = dataset
        self.num_samples_per_class = num_samples_per_class

        # Create a mapping from class index to list of indices
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.base_dataset.imgs):
            self.class_to_indices[label].append(idx)

        # Ensure we have enough samples
        for label in self.class_to_indices:
            if len(self.class_to_indices[label]) < num_samples_per_class:
                raise ValueError(f"Not enough samples for class {label}")

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

    def __iter__(self):
        indices = []
        for label in self.classes:
            indices.extend(random.sample(self.class_to_indices[label], self.num_samples_per_class))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_classes * self.num_samples_per_class

def load_dataset(name, batch_size=32, num_workers=8, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), root='', **kwargs):
    # Dataset 
    if name in ['ImageNet','tinyimagenet', 'CUB200', 'STANFORD120', 'Flowers', 'Food101', 'Pets']:
        # TODO
        if name == 'tinyimagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        elif name == 'ImageNet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

            # get_train_sampler = lambda d: BatchSampler(BalancedSampler(d, num_samples_per_class=10), batch_size, False)
            get_train_sampler = lambda d: BatchSampler(RandomSampler(d), batch_size, False)
            get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), batch_size, False)

            trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=16)
            valloader   = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=16)

            return trainloader, valloader


        elif name == 'STANFORD120':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_set = Dogs('./data/dogs', transform=transform_train, train=True, download=True)
            val_set  = Dogs('./data/dogs', transform=transform_test, train=False, download=True)
            trainset = DataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

            valset = DataLoader(val_set,
                                    batch_size=int(batch_size/2),
                                    shuffle=False,
                                    num_workers=int(num_workers/2))
            
        elif name == 'Flowers':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_set = Flowers102('./data/flowers', split="train", transform=transform_train, download=True)
            val_set  = Flowers102('./data/flowers', split="val", transform=transform_test, download=True)
            
            trainset = DataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

            valset = DataLoader(val_set,
                                    batch_size=int(batch_size/2),
                                    shuffle=False,
                                    num_workers=int(num_workers/2))
        elif name == 'Food101':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_set = Food101('./data/food101', split="train", transform=transform_train, download=True)
            val_set  = Food101('./data/food101', split="test", transform=transform_test, download=True)
            
            trainset = DataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

            valset = DataLoader(val_set,
                                    batch_size=int(batch_size/2),
                                    shuffle=False,
                                    num_workers=int(num_workers/2))

        elif name == 'stanford_cars':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_set = torchvision.datasets.stanford_cars('./data/stanford_cars', split="train", transform=transform_train, download=True)
            val_set = torchvision.datasets.stanford_cars('./data/stanford_cars', split="val", transform=transform_test, download=False)
            
            trainset = DataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

            valset = DataLoader(val_set,
                                    batch_size=int(batch_size/2),
                                    shuffle=False,
                                    num_workers=int(num_workers/2))
        
        elif name == 'CUB200':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_set = Cub2011('./data/cub2011', transform=transform_train, train=True, download=True)
            val_set  = Cub2011('./data/cub2011', transform=transform_test, train=False, download=True)
            trainset = DataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

            valset = DataLoader(val_set,
                                    batch_size=int(batch_size/2),
                                    shuffle=False,
                                    num_workers=int(num_workers/2))
            
        elif name == 'Pets':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_set = Pets('./data/pets', transform=transform_train, train=True, download=True)
            val_set  = Pets('./data/pets', transform=transform_test, train=False, download=True)
            trainset = DataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

            valset = DataLoader(val_set,
                                    batch_size=int(batch_size/2),
                                    shuffle=False,
                                    num_workers=int(num_workers/2))
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            train_val_dataset_dir = os.path.join(root, name, "train")
            test_dataset_dir = os.path.join(root, name, "test")

    elif name.startswith('cifar'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if name == 'cifar10':
            CIFAR = datasets.CIFAR10
        else:
            CIFAR = datasets.CIFAR100

        trainset = DatasetWrapper(CIFAR(root, train=True,  download=True, transform=transform_train))
        valset   = DatasetWrapper(CIFAR(root, train=False, download=True, transform=transform_test))
    else:
        raise Exception('Unknown dataset: {}'.format(name))
    

    

    return trainset, valset

    # get_train_sampler = lambda d: BatchSampler(RandomSampler(d), kwargs['batch_size'], False)
    # get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    # trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=4)
    # valloader   = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=4)

    # return trainloader, valloader


