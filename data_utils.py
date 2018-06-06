from __future__ import print_function,division

import os
import h5py
import time
import numpy as np

import torch
import torchvision
import torch.utils.data as data

import my_kdtree

max_depth=11


## use to load point clouds and class_label from .h5
def load_cls(filelist,use_extra_feature=False):
    points = []
    labels = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))
        #print (data['data'][...].shape) ##(2048, 2048, 3)
        if 'normal' in data and use_extra_feature:
            points.append(np.concatenate([data['data'][...], data['normal'][...]], axis=-1).astype(np.float32))
        else:
            points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int64))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0))


class pts_cls_dataset(data.Dataset):
    def __init__(self,datalist_path,num_points=1024,max_depth=10,use_extra_feature=False):
        super(pts_cls_dataset, self).__init__()
        self.depth=max_depth
        self.num_points=num_points
        self.extra_feature=use_extra_feature
        self.pts, self.label = load_cls(datalist_path,use_extra_feature=use_extra_feature)
        print ('data size:{} label size:{}'.format(self.pts.shape,self.label.shape))

    def __getitem__(self, index):
        pts,label=self.pts[index],self.label[index]
        if len(pts) > self.num_points:
            choice=np.random.choice(len(pts),self.num_points,replace=False)
            pts=pts[choice]

        ## data argument
        pts = pts + np.random.uniform(low=-0.1,high=0.1,size=[1,3]) ## translation [-0.1,0.1]
        scale_param=np.random.uniform(low=0.66,high=1.5)
        pts[:,2]=pts[:,2]*scale_param
        pts[:,0]=pts[:,0]*scale_param

        split_dims,tree_pts=my_kdtree.make_cKDTree(pts,depth=self.depth)

        tree_pts=torch.from_numpy(tree_pts.astype(np.float32))
        split_dims = [(np.array(item).astype(np.int64)) for item in split_dims]

        return split_dims,tree_pts,label

    def __len__(self):
        return len(self.label)


def pts_collate(batch):
    pts_batch=[]
    label_batch=[]
    depth=len(batch[0][0]) ## 11

    split_dims_batch=[[] for i in range(depth)]

    for sample in batch:
        pts_batch.append(sample[1])
        label_batch.append(sample[2])
        for j in range(depth):
            split_dims_batch[j].append(sample[0][j])

    for k in range(depth):
        split_dims_batch[k]=np.squeeze(split_dims_batch[k])

    pts_batch,label_batch=torch.transpose(torch.stack(pts_batch,dim=0),dim0=1,dim1=2),\
                          torch.from_numpy(np.squeeze(label_batch))

    return split_dims_batch,pts_batch,label_batch

"""
t1=time.time()
my_dataset=pts_cls_dataset(datalist_path='/home/gaoyuzhe/Downloads/PointCNN/data/modelnet/test_files.txt')
a=my_dataset[1]

t2=time.time()
print (t2-t1,'s')

data_loader = torch.utils.data.DataLoader(my_dataset,batch_size=4, shuffle=True, collate_fn=kdtree_collate)

for batch_idx, (split_dims,pts,label) in enumerate(data_loader):
    print ('\n')
    print (len(split_dims))
    print (pts.size())
    print (label.size())

    print (split_dims[2])
    break
"""