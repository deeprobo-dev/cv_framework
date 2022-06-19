import torch.optim as optim

def get_optimizer(params, lr=0.01, momentum=0.9, weight_decay=0):
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

