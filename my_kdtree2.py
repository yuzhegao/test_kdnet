import numpy as np
from collections import defaultdict

def computeVariance(arrayList):
    for ele in arrayList:
        ele = float(ele)
    LEN = len(arrayList)
    array = np.array(arrayList)
    sum1 = array.sum()
    array2 = array * array
    sum2 = array2.sum()
    mean = sum1 / LEN
    # D[X] = E[x^2] - (E[x])^2
    variance = sum2 / LEN - mean ** 2
    return variance

class KD_node(object):
    def __init__(self, point=None, split=None, LL = None, RR = None):
        self.point = point
        self.split = split
        self.left = LL
        self.right = RR

class kdtree2(object):
    def __init__(self,points):
        self.cut_dims=defaultdict(list) ## design for num_points=1024
        self.pts_set=[]
        self.createKDTree(None,points,level=0)
        self.cut_dims=list(self.cut_dims.values())[1:-1]

    def createKDTree(self,root, data_list, level=0):
        LEN = len(data_list)
        if LEN == 0:
            return

        dimension = data_list.shape[1]
        num_points = data_list.shape[0]
        max_var = 0
        split = 0
        for i in range(dimension):
            ll = []
            for t in data_list:
                ll.append(t[i])
            var = computeVariance(ll)
            if var > max_var:
                max_var = var
                split = i
        self.cut_dims[level].append(split)  ## level=0,then LEN=2048=2^(11-level)

        sort_index = np.argsort(data_list[:, split].reshape(num_points, ))
        # left_index=sort_index[LEN/2:]
        # right_index=sort_index[:LEN/2]
        data_list = data_list[sort_index]

        point = data_list[LEN / 2]
        if level == 10:
            self.pts_set.append(point)
        root = KD_node(point, split)
        root.left = self.createKDTree(root.left, data_list[:(LEN / 2)], level=level + 1)  ## 0~1023
        root.right = self.createKDTree(root.right, data_list[(LEN / 2 + 1):], level=level + 1)  ##1025~2047
        return root

"""
import h5py
data = h5py.File('/home/gaoyuzhe/Downloads/3d_data/modelnet/ply_data_test0.h5')
print (data['data'].shape)

idx=np.random.randint(0,2047)
pts=data['data'][idx]
print (pts.shape) ## when the input of tree1 is 1024,the tree2 is 2048 !!!

tree=kdtree2(pts)


print (tree.cut_dims)
print (len(tree.cut_dims))
print (len(tree.cut_dims[-1]))
"""