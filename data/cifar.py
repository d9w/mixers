import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout


class Cifar:
    def __init__(self, batch_size, threads, data_dir='./data'):
        self.data_dir = data_dir
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomResizedCrop(size=(64, 64), scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
