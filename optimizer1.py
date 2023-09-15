"""
优化器构建，默认采用adam优化器
"""

import torch
from torch import optim
import math
import logging
from config import cfg

def get_optimizer(args, net):

    param_groups = net.parameters()
    print("optimizer is Adam-")
    optimizer = optim.Adam(param_groups,lr=args.lr)
    if args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / args.max_epoch, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return optimizer, scheduler





