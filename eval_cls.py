from __future__ import print_function,division

import os
import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from kdnet import KDNet
from data_utils import pts_cls_dataset,pts_collate


is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='KD-network')
parser.add_argument('--data-eval', metavar='DIR',default='/home/gaoyuzhe/Downloads/PointCNN/data/modelnet/test_files.txt',
                    help='txt file to validate dataset')
parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--resume', default='cls_modelnet40.pth',type=str, metavar='PATH',help='path to latest checkpoint ')

args=parser.parse_args()

net=KDNet()
if is_GPU:
    net=net.cuda()

def eval():
    net.eval()
    total_correct=0

    data_eval = pts_cls_dataset(datalist_path=args.data_eval)
    eval_loader = torch.utils.data.DataLoader(data_eval,batch_size=32, shuffle=True, collate_fn=pts_collate)
    print("dataset size:", len(eval_loader.dataset))

    if os.path.exists(args.resume):
        if is_GPU:
            checkoint = torch.load(args.resume)
        else:
            checkoint = torch.load(args.resume, map_location=lambda storage, loc: storage)

        start_epoch = checkoint['epoch']
        net.load = net.load_state_dict(checkoint['model'])
        num_iter= checkoint['iter']
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("Warining! No resume checkpoint to load")
        exit()

    for batch_idx, (split_dims, pts, label) in enumerate(eval_loader):
        if is_GPU:
            pts = Variable(pts.cuda())
            label = Variable(label.cuda())
        else:
            pts = Variable(pts)
            label = Variable(label)
        pred = net(pts, split_dims)

        _, pred_index = torch.max(pred, dim=1)
        num_correct = (pred_index.eq(label)).data.cpu().sum().item()

        print ('in batch{} correct:{}/32'.format(batch_idx,num_correct))
        total_correct +=num_correct

    print ('the average correct rate:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))

eval()