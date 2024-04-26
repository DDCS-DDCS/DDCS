import torch
import torchvision
import torchvision.transforms as transforms
from .subset import Subset
from dataset import idn
from dataset.idn.data.data_loader.utils import create_train_val
from .dataloader import DataLoader_noise
import numpy as np
from randaugment import CIFAR10Policy, ImageNetPolicy, RandAugment
__all__ = ["load_noisydata"]

data_info_dict = {
    "CIFAR10":{
        "mean":(0.4914, 0.4822, 0.4465),
        "std":(0.2023, 0.1994, 0.2010),
        "root": "./data/",
        'random_crop':32
    },
    "CIFAR100":{
        "mean":(0.4914, 0.4822, 0.4465),
        "std":(0.2023, 0.1994, 0.2010),
        "root": "./data/",
        'random_crop':32
    },
    "SVHN":{
        "mean":(0.5, 0.5, 0.5),
        "std":(0.5, 0.5, 0.5),
        "root": "~/.torchvision/datasets/SVHN",
        'random_crop':32
    },
    "MNIST":{
        "mean":(0.1306604762738429,),
        "std":(0.30810780717887876,),
        "root": "~/.torchvision/datasets/MNIST",
        'random_crop':28
    },
    "FASHIONMNIST":{
        "mean":(0.286,),
        "std":(0.353,),
        "root": "~/.torchvision/datasets/FashionMNIST",
        'random_crop':28
    }
}
        
# 用对比损失需要两张不同增强的图
class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w, x_s

def build_transform(rescale_size=512, crop_size=448, s=1):
    cifar_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])
    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])
    cifar_train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        # RandAugment(),
        # ImageNetPolicy(),
        # Cutout(size=crop_size // 16),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])
    train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        RandAugment(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return {'train': train_transform, 'test': test_transform, 'train_strong_aug': train_transform_strong_aug,
            'cifar_train': cifar_train_transform, 'cifar_test': cifar_test_transform, 'cifar_train_strong_aug': cifar_train_transform_strong_aug}


def load_noisydata(dataset = "CIFAR10", num_workers = 0, batch_size = 128, add_noise = False, noise_type = None, flip_rate_fixed = None, random_state = 1, trainval_split=None, train_frac = 1, augment = True):
  
    def transform_target(label):
        label = np.array(label)
        target = torch.from_numpy(label).long()
        return target  

    print('=> preparing data..')
 
    dataset = dataset.upper()
    info = data_info_dict[dataset]

    root = info["root"]
    random_crop = info["random_crop"]
    normalize = transforms.Normalize(info["mean"], info["std"])
    # if augment != False:
    #     transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor() ])
    # else:
    #     transform_train = transforms.Compose([transforms.ToTensor()])

    # transform_test = transforms.Compose([transforms.ToTensor()])

    # 一个增强变成两个
    transform = build_transform(rescale_size=32, crop_size=32)
    transform_train = CLDataTransform(transform['cifar_train'], transform['cifar_train_strong_aug'])

            
    if dataset=='CIFAR10':           
        transform_test=transforms.Compose([ transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                            ])
    if dataset=='CIFAR100':           
        transform_test=transforms.Compose([ transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
                                            ])


    test_dataset = idn.data.dataset.__dict__[dataset + "_noise"](root=root, train=False, transform=transform_test, transform_eval=transform_test, add_noise = False, target_transform = transform_target)

    train_val_dataset = idn.data.dataset.__dict__[dataset + "_noise"](
        root = root, 
        train = True, 
        transform = transform_train, 
        transform_eval = transform_test,
        target_transform = transform_target,
        add_noise = True,
        noise_type = noise_type, 
        flip_rate_fixed = flip_rate_fixed,
        random_state = random_state
    )

    train_dataset, val_dataset = create_train_val(train_val_dataset,trainval_split,train_frac)
    train_val_dataset = Subset(train_val_dataset, list(range(0, len(train_val_dataset), 1))) 
    train_val_loader = DataLoader_noise(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader_noise(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    est_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    test_loader = DataLoader_noise(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    

    return train_val_loader, train_loader, val_loader, est_loader, test_loader



