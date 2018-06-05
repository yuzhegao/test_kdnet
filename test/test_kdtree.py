import numpy as np
import h5py
import scipy.spatial


data = h5py.File('/home/gaoyuzhe/Downloads/PointCNN/data/modelnet/ply_data_test0.h5')
print (data['data'].shape)

idx=np.random.randint(0,2047)
pts=data['data'][idx]

kdtree2 = scipy.spatial.cKDTree(pts,leafsize=1,balanced_tree=False)
print len(kdtree2.tree.indices)
print len(kdtree2.tree.lesser.indices)
print len(kdtree2.tree.lesser.lesser.indices)
print len(kdtree2.tree.lesser.lesser.lesser.indices)
print len(kdtree2.tree.lesser.lesser.lesser.lesser.indices)
print len(kdtree2.tree.lesser.lesser.lesser.lesser.lesser.indices)
print len(kdtree2.tree.lesser.lesser.lesser.lesser.lesser.lesser.indices)
print len(kdtree2.tree.lesser.lesser.lesser.lesser.lesser.lesser.lesser.indices)
print len(kdtree2.tree.lesser.lesser.lesser.lesser.lesser.lesser.lesser.lesser.indices)
print len(kdtree2.tree.lesser.lesser.lesser.lesser.lesser.lesser.lesser.lesser.lesser.indices)
print len(kdtree2.tree.lesser.lesser.lesser.lesser.lesser.lesser.lesser.lesser.lesser.lesser.indices)


