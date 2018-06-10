import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class KDNet(nn.Module):
    def __init__(self, num_class=40):
        super(KDNet, self).__init__()
        self.fc1=nn.Linear(3,8)
        self.conv1 = nn.Conv1d(8*2, 32 * 3, 1, 1)
        self.conv2 = nn.Conv1d(32*2, 64 * 3, 1, 1)
        self.conv3 = nn.Conv1d(64*2, 64 * 3, 1, 1)
        self.conv4 = nn.Conv1d(64*2, 128 * 3, 1, 1)
        self.conv5 = nn.Conv1d(128*2, 128 * 3, 1, 1)
        self.conv6 = nn.Conv1d(128*2, 256 * 3, 1, 1)
        self.conv7 = nn.Conv1d(256*2, 256 * 3, 1, 1)
        self.conv8 = nn.Conv1d(256*2, 512 * 3, 1, 1)
        self.conv9 = nn.Conv1d(512*2, 512 * 3, 1, 1)
        self.conv10 = nn.Conv1d(512*2, 1024 * 3, 1, 1)
        self.fc8 = nn.Linear(1024, num_class)

        self.bn1 = nn.BatchNorm1d(32 * 3)
        self.bn2 = nn.BatchNorm1d(64 * 3)
        self.bn3 = nn.BatchNorm1d(64 * 3)
        self.bn4 = nn.BatchNorm1d(128 * 3)
        self.bn5 = nn.BatchNorm1d(128 * 3)
        self.bn6 = nn.BatchNorm1d(256 * 3)
        self.bn7 = nn.BatchNorm1d(256 * 3)
        self.bn8 = nn.BatchNorm1d(512 * 3)
        self.bn9 = nn.BatchNorm1d(512 * 3)
        self.bn10 = nn.BatchNorm1d(1024 * 3)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    m.bias.data.zero_()


    def kdconv(self,x, sel, conv, bn, dropout=False):   ##[N,C,W] in pytorch
        ## x [N,8,1024]  sel [N,512]
        ## featdim=x,size()[1]=8

        batchsize = x.size(0)  ## N
        featdim = x.size(1) ## 8
        num_pts = x.size(2) ##1024

        ## concat two points feature
        x=torch.transpose(x,dim0=1,dim1=2).contiguous() ## [N,1024,8]
        x=x.view(batchsize,num_pts/2,featdim*2) ## [N,512,16]
        x=torch.transpose(x,dim0=1,dim1=2).contiguous() ## [N,16,512]


        ## fc -> [N,3*32,512]
        if dropout:
            x = F.relu(F.dropout(bn(conv(x)),p=0.5))
        else:
            x = F.relu(bn(conv(x)))

        num_pts = num_pts/2 ## 512
        featdim_out= x.size(1)/3 ## 32
        #print ('featdim {}'.format(featdim_out))

        ## index_select ->[N,512,32]
        x = x.view(-1, 3, featdim_out, num_pts)  ## [N,3,32,512]
        x = torch.transpose(x,dim0=1,dim1=2).contiguous() ## [N,32,3,512]

        x = x.view(-1, featdim_out, 3 * num_pts)## [N,32,3*512]
        x = torch.transpose(x,dim0=0,dim1=1).contiguous()## [32,N,3*512]
        x = x.view(featdim_out, 3 * num_pts * batchsize)  ## [32,N*3*512]

        if sel.ndim !=2:
            sel=sel.reshape(sel.shape[0],1)
        sel = Variable(sel + (torch.arange(0, num_pts) * 3).repeat(batchsize, 1).long()).view(-1, 1)
        ## (torch.arange(0, num_pts) * 3).repeat(batchsize, 1)  ->[N,512]
        offset = Variable((torch.arange(0, batchsize) * num_pts * 3).repeat(num_pts, 1).transpose(1, 0).contiguous().long().view(-1,1))
        #print offset.size(),sel.size()
        sel = sel + offset
        if x.is_cuda:
            sel = sel.cuda()
        sel = sel.squeeze() ##[N,512] ->select N*512 from N*3*512

        x = torch.index_select(x, dim=1, index=sel)  ##[32,N*512]
        x = x.view(featdim_out, batchsize, num_pts)  ##[32,N,512]
        x = x.transpose(1, 0).contiguous()  ##[N,32,512]

        return x

    def forward(self, x, c): ## x [N,3,2048]
        init_num_pts=x.size()[-1]

        x = torch.transpose(x,dim0=2,dim1=1).contiguous().view(-1,3) ##[N*1024,3]
        x = F.relu(self.fc1(x)) ##[N*1024,32]
        x = torch.transpose(x.view(-1,init_num_pts,8),dim0=2,dim1=1) ##[N,32,1024]

        x1 = self.kdconv(x, c[-1], self.conv1, self.bn1)
        x2 = self.kdconv(x1, c[-2], self.conv2, self.bn2)
        x3 = self.kdconv(x2, c[-3], self.conv3, self.bn3)
        x4 = self.kdconv(x3, c[-4], self.conv4, self.bn4)
        x5 = self.kdconv(x4, c[-5], self.conv5, self.bn5)
        x6 = self.kdconv(x5, c[-6], self.conv6, self.bn6)
        x7 = self.kdconv(x6, c[-7], self.conv7, self.bn7)
        x8 = self.kdconv(x7, c[-8], self.conv8, self.bn8)
        x9 = self.kdconv(x8, c[-9], self.conv9, self.bn9)
        x10 = self.kdconv(x9, c[-10], self.conv10, self.bn10) ##[N,128,1]

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