from __future__ import print_function,division

import os
import h5py
import time
import numpy as np

import torch
import torchvision
import torch.utils.data as data

import my_kdtree2

## use to load point clouds and class_label from .h5
def load_cls(filelist,use_extra_feature=False):
    points = []
    labels = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))
        if 'normal' in data and use_extra_feature:
            points.append(np.concatenate([data['data'][...], data['normal'][...]], axis=-1).astype(np.float32))
        else:
            points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int64))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0))

#########################
# modelnet40
#########################
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
        if len(pts) > (self.num_points*2):
            choice=np.random.choice(len(pts),(self.num_points*2),replace=False)
            pts=pts[choice]

        ## data argument
        """
        scale_param=np.random.uniform(low=0.66,high=1.5)
        pts[:, 2] = pts[:, 2] * scale_param
        pts[:, 0] = pts[:, 0] * scale_param ## scale horizonal plane

        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        pts = pts.dot(rotation_matrix)  ## rotate

        pts = pts + np.random.uniform(low=-0.1,high=0.1,size=[1,3]) ## translation [-0.1,0.1]

        sigma = 0.01
        clip = 0.05
        jittered_data = np.clip(sigma * np.random.randn(self.num_points, 3), -1 * clip, clip)
        pts+=jittered_data ## jitter
        """

        tree=my_kdtree2.kdtree2(pts)

        tree_pts=(np.squeeze(tree.pts_set)).astype(np.float32)
        split_dims=[(np.array(item).astype(np.int64)) for item in tree.cut_dims]

        return split_dims,tree_pts,label

    def __len__(self):
        return len(self.label)


def pts_collate(batch):
    pts_batch=[]
    label_batch=[]
    depth=len(batch[0][0]) ## 11

    split_dims_batch=[[] for i in range(depth)]

    for sample in batch:
        pts_batch.append(torch.from_numpy(sample[1]))
        label_batch.append(sample[2])
        for j in range(depth):
            split_dims_batch[j].append(sample[0][j])

    for k in range(depth):
        split_dims_batch[k]=np.squeeze(split_dims_batch[k])

    pts_batch=torch.transpose(torch.stack(pts_batch,dim=0),dim0=1,dim1=2)
    label_batch =torch.from_numpy(np.squeeze(label_batch))

    return split_dims_batch,pts_batch,label_batch


###########################
# shapenet_core
###########################
class shapenet_dataset(data.Dataset):
    def __init__(self, root, npoints=1024, classification=False, class_choice=None, train=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')  ## dir of each category
            # print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]  ## all filename of this category

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
                ## self.meta -> dict of all category list

        self.datapath = []
        for item in self.cat:  ## item ->class
            for fn in self.meta[item]:  ## fn ->file
                self.datapath.append((item, fn[0], fn[1]))  ## all files: list of (class_name,file.pts,file.seg)

        self.classes = dict(zip(self.cat, range(len(self.cat))))  ## dict of {class_name:class_index}
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath) / 50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l  ## max num_seg_class in a single 3d shape
                    # print(self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)  ## [num_points,3]
        seg = np.loadtxt(fn[2]).astype(np.int64)  ## [num_points,1]
        # print(point_set.shape, seg.shape)


        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  ## -mean [num_points,3]
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)  ## scalar
        dist = np.expand_dims(np.expand_dims(dist, 0), 1)  ## [1,1]
        point_set = point_set / dist

        if len(point_set)>(self.npoints*2):
            choice = np.random.choice(len(seg), (self.npoints) * 2, replace=True)
            point_set = point_set[choice, :]
            point_set = point_set + 1e-5 * np.random.rand(*point_set.shape)

        tree = my_kdtree2.kdtree2(point_set)

        tree_pts = (np.squeeze(tree.pts_set)).astype(np.float32)
        split_dims = [(np.array(item).astype(np.int64)) for item in tree.cut_dims]

        if self.classification:
            return split_dims,tree_pts, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)



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