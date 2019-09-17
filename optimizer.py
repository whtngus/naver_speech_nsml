import math
import torch
from torch import optim
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def build_optimizer(args, model, lr):
    optimizer_config = args['train_param']['optimizer']
    if optimizer_config == "Adam":    
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_config == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=lr, weight_decay=0.000025)
    elif optimizer_config == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.000025)   
    elif optimizer_config == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.025)
    return optimizer

def build_scheduler(args, optimizer):
    train_args = args['train_param']
    if train_args['scheduler'] == 'cosine':
        t_max = train_args['cosine']['t_max']
        eta_min = train_args['cosine']['eta_min']
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif train_args['scheduler'] == 'steplr':
        step_size = train_args['steplr']['step_size']
        gamma = train_args['steplr']['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return
    return scheduler
