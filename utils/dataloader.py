from torchvision import datasets
import torch
import numpy as np
# import albumentations as A
import matplotlib.pyplot as plt

from .transform import get_transforms

class Cifar10Dataset(datasets.CIFAR10):
    
    def __init__(self, root = "./data", train = True, download = True, transform = None, viz = False):
        super().__init__(root = root, train = train, download = download, transform = transform)
        self.viz = viz
    
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]
            if self.viz:
                return image,label
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
        
        return image,label

class MNISTDataset(datasets.MNIST):
    def __init__(self, root="./data", train=True, download=True, transform=None, viz=False):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.viz = viz

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]
            if self.viz:
              return image, label
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
        return image, label

def dataloaders(data, train_batch_size = None, val_batch_size = None, seed=42):
    train_transforms, test_transforms = get_transforms()

    if data == "CIFAR10":
        train_ds = Cifar10Dataset('./data', train=True, download=True, transform=train_transforms)
        test_ds = Cifar10Dataset('./data', train=False, download=True, transform=test_transforms)
    elif data == "MNIST":
        train_ds = MNISTDataset('./data', train=True, download=True, transform=train_transforms)
        test_ds = MNISTDataset('./data', train=False, download=True, transform=test_transforms)

    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
    
    train_batch_size = train_batch_size or (128 if cuda else 64)
    val_batch_size = val_batch_size or (128 if cuda else 64)
    print(train_batch_size, val_batch_size)

    train_dataloader_args = dict(shuffle=True, batch_size=train_batch_size, num_workers=4, pin_memory=True)
    val_dataloader_args = dict(shuffle=True, batch_size=val_batch_size, num_workers=4, pin_memory=True) 

    train_loader = torch.utils.data.DataLoader(train_ds, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_ds, **val_dataloader_args)

    return train_loader, test_loader


def data_details(data_name, cols = 5, rows = 2, train_data = True, transform = False, vis=True):
    train_transforms, test_transforms = get_transforms()
    if transform and train_data:
        transform = train_transforms
    elif transform and not(train_data):
        transform = test_transforms
    else:
        transform = None    

    if data_name == "CIFAR10":
        data = Cifar10Dataset('./data', train=train_data, download=True, transform=transform, viz=True )
    elif data_name == "MNIST":
        data = MNISTDataset('./data', train=train_data, download=True, transform=transform, viz=True )

    if vis:
        figure = plt.figure(figsize=(cols*1.5, rows*1.5))
        for i in range(1, cols * rows + 1):
            img, label = data[i]
            print(img.shape)
            figure.add_subplot(rows, cols, i)
            plt.title(data.classes[label])
            plt.axis("off")
            plt.imshow(img, cmap="gray")

        plt.tight_layout()
        plt.show() 
    if (transform is None) and vis:
        print(' - mean:', np.mean(data.data, axis=(0,1,2)) / 255.)
        print(' - std:', np.std(data.data, axis=(0,1,2)) / 255.)
        print(' - var:', np.var(data.data, axis=(0,1,2)) / 255.)

    return data.classes
