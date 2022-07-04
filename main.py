import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary

import torchvision

import os
import argparse
from tqdm import tqdm

from models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--b', default=128, type=int, help='batch size')  
parser.add_argument('--e', default=5, type=int, help='no of epochs') 
parser.add_argument('--norm', default="batch", type=str, help='Normalization Type')  
parser.add_argument('--n', default=10, type=int, help='No of Images to be displayed after prediction (should be multiple of 5)') 
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')                  
args = parser.parse_args()

best_acc = 0
start_epoch = 0

def main(lr = args.lr, batch_size = args.b, epochs = args.e, norm = args.norm, n = args.n, resume = args.resume):
# def main(lr = 0.007, batch_size = 128, epochs = 5, norm = "batch", n = 10, resume = False):
    global best_acc, start_epoch
    SEED = 69
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")
    if norm == "batch":
        model = ResNet18(use_batchnorm=True).to(device)
    elif norm == "layer":
        model = ResNet18(use_layernorm=True).to(device)
    elif norm == "group":
        model = ResNet18(use_groupnorm=True).to(device) 
    else:
        print("Please enter a valid Normalization Type")

    print("\n\n****************************************************************************\n")
    print("*****Model Summary*****")
    summary(model, input_size=(3, 32, 32))
    print("\n****************************************************************************\n")

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['net'])
        print('==> Model loaded from checkpoint..')
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    model, best_acc = train_model(model, device, norm, start_epoch, best_acc, epochs = epochs, batch_size = batch_size, learning_rate = lr)

    print("\n****************************************************************************\n")

    print("*****Correctly Classified Images*****\n")

    image_prediction("CIFAR10", model, "Correctly Classified Images", n=n,r=int(n/5),c=5, misclassified = False)

    print("\n****************************************************************************\n")

    # print("*****Correctly Classified GradCam Images*****\n")

    # image_prediction("CIFAR10", model, "Correctly Classified GradCam Images", n=n,r=int(n/5),c=5, misclassified = False, gradcam=True)

    # print("\n****************************************************************************\n")

    print("*****Misclassified Images*****\n")

    image_prediction("CIFAR10", model, "Misclassified Images", n=n,r=int(n/5),c=5, misclassified = True)

    print("\n****************************************************************************\n")

    # print("*****Misclassified GradCam Images*****\n")

    # image_prediction("CIFAR10", model, "Misclassified GradCam Images", n=n,r=int(n/5),c=5, misclassified = True, gradcam=True)

    # print("\n****************************************************************************\n")

if __name__ == "__main__":
    main()