import torch
import sparseconvnet as scn
import uproot
import matplotlib.pyplot as plt
import numpy as np
from model import ResNet
from model import Hello

from timeit import default_timer as timer

file = uproot.open('data/nue_6350_26_1313.root')
tblob = file['T_rec_charge_blob']

x = tblob.array('x')
y = tblob.array('y')
z = tblob.array('z')
q = tblob.array('q')
print(len(x))
print('x: {} {}'.format(np.min(x), np.max(x)))
print('y: {} {}'.format(np.min(y), np.max(y)))
print('z: {} {}'.format(np.min(z), np.max(z)))
print('q: {} {}'.format(np.min(q), np.max(q)))

max_pos = 10000
x = np.linspace(0, max_pos-1, num=max_pos)
y = np.linspace(0, max_pos-1, num=max_pos)
z = np.linspace(0, max_pos-1, num=max_pos)
q = np.linspace(42., 42., num=max_pos)

pos = np.stack((x,y,z), axis=1)
q = np.expand_dims(q, axis=1)
# q = np.stack((q,q,q), axis=1)

# input visualize
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# img = ax.scatter(x, y, z, c=q, cmap=plt.jet())
# fig.colorbar(img)
# plt.show()

# Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print("Using CUDA.")
else:
    print("Not using CUDA.")

model = ResNet(dimension=3, device=device)

# check the SparseTenSor
input_layer = model.inputLayer
st = input_layer([torch.LongTensor(pos),torch.FloatTensor(q).to(device)])
sl = st.get_spatial_locations()
print('spatial_locations: ', sl.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# img = ax.scatter(sl[:,0], sl[:,1], sl[:,2], cmap=plt.jet())
# fig.colorbar(img)
# plt.show()

nreapeat = 100
# leave warmup outside the timer
output = model([torch.LongTensor(pos),torch.FloatTensor(q).to(device)])
start = timer()
for rep in range(nreapeat):
    output = model([torch.LongTensor(pos),torch.FloatTensor(q).to(device)])
end = timer()
print('Forward time: {0:.1f} ms'.format((end-start)/nreapeat*1000))

print('Output SparseConvNetTensor:', output)
