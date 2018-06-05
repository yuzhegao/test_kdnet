import numpy as np
import  torch

a=[np.array([2,2]),np.array([1,1,2,2]),np.array([1,1,2,2,3,3]),np.array([1,1,2,2,3,3])]
b=[np.array([0,0]),np.array([0,0,1,1]),np.array([1,1,1,1,2,2]),np.array([1,1,2,2,3,3])]
c=[np.array([0,0]),np.array([0,0,0,0]),np.array([1,1,1,1,2,2]),np.array([1,1,2,2,3,3])]


batch=zip(a,b)
batch=zip(batch,c)

print batch
print '\n'
batch=[np.squeeze(item) for item in batch]
print batch
print len(batch)

x = torch.from_numpy(np.array([[1, 2, 3],[4,5,6]]))
print x.repeat(4, 2)

#x = torch.from_numpy(np.array([1, 2, 3]))
#print (x + np.array([4,5,6]))
