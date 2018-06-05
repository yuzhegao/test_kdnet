import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

## test
import torch.utils.data as data
from data_utils import pts_cls_dataset,pts_collate


class KDNet(nn.Module):
    def __init__(self, num_class=40):
        super(KDNet, self).__init__()
        self.fc1=nn.Linear(3,32)
        self.conv1 = nn.Conv1d(32, 32 * 3, 1, 1)
        self.conv2 = nn.Conv1d(32, 64 * 3, 1, 1)
        self.conv3 = nn.Conv1d(64, 64 * 3, 1, 1)
        self.conv4 = nn.Conv1d(64, 128 * 3, 1, 1)
        self.conv5 = nn.Conv1d(128, 128 * 3, 1, 1)
        self.conv6 = nn.Conv1d(128, 256 * 3, 1, 1)
        self.conv7 = nn.Conv1d(256, 256 * 3, 1, 1)
        self.conv8 = nn.Conv1d(256, 512 * 3, 1, 1)
        self.conv9 = nn.Conv1d(512, 512 * 3, 1, 1)
        self.conv10 = nn.Conv1d(512, 128 * 3, 1, 1)
        self.fc8 = nn.Linear(128, num_class)

        self.bn1 = nn.BatchNorm1d(32 * 3)
        self.bn2 = nn.BatchNorm1d(64 * 3)
        self.bn3 = nn.BatchNorm1d(64 * 3)
        self.bn4 = nn.BatchNorm1d(128 * 3)
        self.bn5 = nn.BatchNorm1d(128 * 3)
        self.bn6 = nn.BatchNorm1d(256 * 3)
        self.bn7 = nn.BatchNorm1d(256 * 3)
        self.bn8 = nn.BatchNorm1d(512 * 3)
        self.bn9 = nn.BatchNorm1d(512 * 3)
        self.bn10 = nn.BatchNorm1d(128 * 3)

    def kdconv(self,x, num_pts, featdim, sel, conv, bn):
        ## x [N,8,2048]  sel [N,2048]

        batchsize = x.size(0)
        x = F.relu(bn(conv(x)))  ## [N,3*32,2048]
        x = x.view(-1, featdim, 3, num_pts)  ## [N,32,3,2048]
        x = x.view(-1, featdim, 3 * num_pts)  ## [N,32,3*2048]
        x = x.transpose(1, 0).contiguous()  ##[32,N,3*2048]
        x = x.view(featdim, 3 * num_pts * batchsize)  ##[32,N*3*2048]

        sel = Variable(sel + (torch.arange(0, num_pts) * 3).repeat(batchsize, 1).long()).view(-1, 1)
        ## (torch.arange(0, num_pts) * 3).repeat(batchsize, 1)  ->[N,2048]
        # print sel.size()
        offset = Variable(
            (torch.arange(0, batchsize) * num_pts * 3).repeat(num_pts, 1).transpose(1, 0).contiguous().long().view(-1,
                                                                                                                   1))
        sel = sel + offset

        if x.is_cuda:
            sel = sel.cuda()
        sel = sel.squeeze()

        x = torch.index_select(x, dim=1, index=sel)  ##[32,N*2048]
        x = x.view(featdim, batchsize, num_pts)  ##[32,N,2048]
        x = x.transpose(1, 0).contiguous()  ##[N,32,2048]
        x = x.view(batchsize, featdim, num_pts / 2, 2)  ##[N,32,1024,2]
        x = torch.max(x, dim=-1)[0]
        return x

    def forward(self, x, c): ## x [N,3,2048]
        init_num_pts=x.size()[-1]

        x = torch.transpose(x,dim0=2,dim1=1).contiguous().view(-1,3) ##[N*2048,3]
        x = self.fc1(x) ##[N*2048,32]
        x = torch.transpose(x.view(-1,init_num_pts,32),dim0=2,dim1=1) ##[N,32,2048]

        x1 = self.kdconv(x, 1024, 32, c[-1], self.conv1, self.bn1)
        x2 = self.kdconv(x1, 512, 64, c[-2], self.conv2, self.bn2)
        x3 = self.kdconv(x2, 256, 64, c[-3], self.conv3, self.bn3)
        x4 = self.kdconv(x3, 128, 128, c[-4], self.conv4, self.bn4)
        x5 = self.kdconv(x4, 64, 128, c[-5], self.conv5, self.bn5)
        x6 = self.kdconv(x5, 32, 256, c[-6], self.conv6, self.bn6)
        x7 = self.kdconv(x6, 16, 256, c[-7], self.conv7, self.bn7)
        x8 = self.kdconv(x7, 8, 512, c[-8], self.conv8, self.bn8)
        x9 = self.kdconv(x8, 4, 512, c[-9], self.conv9, self.bn9)
        x10 = self.kdconv(x9, 2, 128, c[-10], self.conv10, self.bn10) ##[N,128,1]

        scores=self.fc8(torch.squeeze(x10)) ##[N,40]
        pred = F.log_softmax(scores,dim=1)


        return pred


"""
my_dataset=pts_cls_dataset(datalist_path='/home/gaoyuzhe/Downloads/PointCNN/data/modelnet/test_files.txt')
data_loader = torch.utils.data.DataLoader(my_dataset,
            batch_size=8, shuffle=True, collate_fn=pts_collate)

net=KDNet()

for batch_idx, (split_dims,pts,label) in enumerate(data_loader):
    print (pts.size())
    print (label.size())

    pts=Variable(pts)

    output=net(pts,split_dims)
    print (output.size())
    break
"""