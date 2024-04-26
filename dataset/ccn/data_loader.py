import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .cifar import CIFAR100, CIFAR10
from randaugment import CIFAR10Policy, ImageNetPolicy, RandAugment

# 用对比损失需要两张不同增强的图
class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w, x_s

def load_ccn_data(dataset, noise_type, noise_rate, batch_size, num_workers):
    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=CLDataTransform(  transforms.Compose([
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomCrop(32, 4),
                                                            transforms.ToTensor(),
                                                            normalize,
                                                            ]),
                                                            transforms.Compose([
                                                            transforms.RandomCrop(size=32, padding=4),
                                                            transforms.RandomHorizontalFlip(),
                                                            CIFAR10Policy(),
                                                            transforms.ToTensor(),
                                                            normalize,
                                                            ])   
                                                         ) ,                            
                                noise_type=noise_type,
                                noise_rate=noise_rate
                                )

        test_dataset = CIFAR10(root='./data/',
                               download=True,
                               train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   normalize,
                               ]),
                               noise_type=noise_type,
                               noise_rate=noise_rate
                               )
    if dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

        train_dataset = CIFAR100(root='./data/',
                                 download=True,
                                 train=True,
                                transform=CLDataTransform(  transforms.Compose([
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomCrop(32, 4),
                                                            transforms.ToTensor(),
                                                            normalize,
                                                            ]),
                                                            transforms.Compose([
                                                            transforms.RandomCrop(size=32, padding=4),
                                                            transforms.RandomHorizontalFlip(),
                                                            CIFAR10Policy(),
                                                            transforms.ToTensor(),
                                                            normalize,
                                                            ])   
                                                         ) , 
                                 noise_type=noise_type,
                                 noise_rate=noise_rate
                                 )

        test_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize,
                                ]),
                                noise_type=noise_type,
                                noise_rate=noise_rate
                                )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              drop_last=True,
                              shuffle=True
                              )
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             drop_last=True,
                             shuffle=False
                             )

    return train_loader, test_loader, train_dataset, test_dataset
