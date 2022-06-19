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

def train(model, device, train_loader, optimizer, scheduler, criterion, l1_reg = None):
    model.train()

    # collect stats - for accuracy calculation
    correct = 0
    processed = 0
    epoch_loss = 0
    epoch_accuracy = 0
    pbar = tqdm(train_loader)

    for batch_id, (data, target) in enumerate(pbar):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Gather prediction and calculate loss + backward pass + optimize weights
        label_pred = model(data)
        label_loss = criterion(label_pred, target)

        # L1 regularization
        if l1_reg is not None:
            l1_criterion = nn.L1Loss(size_average=False)
            l1_reg_loss = 0
            for param in model.parameters():
                l1_reg_loss += l1_criterion(param, torch.zeros_like(param))
                # print("L1 reg loss: ", l1_reg_loss)
            label_loss += l1_reg * l1_reg_loss

        # Calculate gradients
        label_loss.backward()
        # Optimizer
        optimizer.step()

        # Metrics calculation- For epoch Accuracy(total correct pred/total items) and loss 
        pred = label_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        epoch_loss += label_loss.item()
        pbar.set_description(desc=f'Training Set: Loss={epoch_loss/len(train_loader)}, Batch_id={batch_id}, Train Accuracy={100*correct/processed:0.2f}')
    
    epoch_accuracy = (100*correct/processed)
    epoch_loss /= len(train_loader)
    # scheduler.step(epoch_loss/len(train_loader))
    
    scheduler.step(epoch_loss)

    return epoch_accuracy, epoch_loss

def test(model, device, test_loader, criterion, epoch):
    global best_acc
    model.eval()

    correct = 0
    processed = 0
    epoch_loss = 0
    epoch_accuracy = 0
    pbar = tqdm(test_loader)

    with torch.no_grad():        
        for batch_id, (data, target) in enumerate(pbar):
            data = data.to(device)
            target = target.to(device)

            label_pred = model(data)
            label_loss = criterion(label_pred, target)

            # Metrics calculation
            pred = label_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            epoch_loss += label_loss.item()
            pbar.set_description(desc=f'Test Set: Loss={epoch_loss/len(test_loader)}, Batch_id={batch_id}, Test Accuracy={100*correct/processed:0.2f}')

    epoch_accuracy = (100*correct)/processed
    epoch_loss /= len(test_loader)

     # Save checkpoint.
    if epoch_accuracy > best_acc:
        print("\n*****Saving Model*****")
        state = {
            'net': model.state_dict(),
            'acc': epoch_accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = epoch_accuracy
    
    return epoch_accuracy, epoch_loss

def train_model(model, device, norm, start_epoch, epochs, batch_size, learning_rate):
    print("\n\n****************************************************************************\n")
    print("*****Training Parameters*****\n")
    print(f"Normalization Type: {norm} normalization\n")
    print(f"No Of Epochs: {epochs}\n")
    print(f"Batch size: {batch_size}\n")
    print(f"Initial Learning Rate: {learning_rate}")
    print("\n****************************************************************************\n")

    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []

    train_loader, test_loader = dataloaders("CIFAR10", train_batch_size=batch_size,
                                                    val_batch_size=batch_size,)


    # Optimization algorithm from torch.optim
    optimizer = get_optimizer(model.parameters(), lr=learning_rate, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.05, patience=1)
    # Loss condition
    criterion = nn.CrossEntropyLoss()
    print("\n****************************************************************************\n")
    print("*****Training Starts*****\n")

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        print(f"Training Epoch: {epoch}")
        train_acc_delta, train_loss_delta = train(model, device, train_loader, optimizer, scheduler, criterion)
        test_acc_delta, test_loss_delta = test(model, device, test_loader, criterion, epoch)
        # print(f"Learning Rate: {scheduler._last_lr[0]},{optimizer.param_groups[0]['lr']}")
        train_accuracy.append(round(train_acc_delta, 2))
        train_loss.append(round(train_loss_delta, 4))
        test_accuracy.append(round(test_acc_delta, 2))
        test_loss.append(round(test_loss_delta, 4))

    print("*****Training Stops*****\n")
    print("\n****************************************************************************\n")

    # print("\n****************************************************************************\n")
    # print("*****Loss and Accuracy Details*****\n")
    # plot_single("model_1", train_loss, train_accuracy, test_loss, test_accuracy)
    # print("\n****************************************************************************\n")

    return model

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

    model = train_model(model, device, norm, start_epoch, epochs = epochs, batch_size = batch_size, learning_rate = lr)

    print("\n****************************************************************************\n")

    # print("*****Correctly Classified Images*****\n")

    # image_prediction("CIFAR10", model, "Correctly Classified Images", n=n,r=int(n/5),c=5, misclassified = False, gradcam=False)

    # print("\n****************************************************************************\n")

    # print("*****Correctly Classified GradCam Images*****\n")

    # image_prediction("CIFAR10", model, "Correctly Classified GradCam Images", n=n,r=int(n/5),c=5, misclassified = False, gradcam=True)

    # print("\n****************************************************************************\n")

    # print("*****Misclassified Images*****\n")

    # image_prediction("CIFAR10", model, "Misclassified Images", n=n,r=int(n/5),c=5, misclassified = True, gradcam=False)

    # print("\n****************************************************************************\n")

    # print("*****Misclassified GradCam Images*****\n")

    # image_prediction("CIFAR10", model, "Misclassified GradCam Images", n=n,r=int(n/5),c=5, misclassified = True, gradcam=True)

    # print("\n****************************************************************************\n")

if __name__ == "__main__":
    main()