import torch
import sparseconvnet as scn
import uproot
import matplotlib.pyplot as plt
import numpy as np
from model import ResNet
from model import Hello

file = uproot.open('data/nue_6350_26_1313.root')
tblob = file['T_rec_charge_blob']

x = tblob.array('x')
y = tblob.array('y')
z = tblob.array('z')
q = tblob.array('q')

# x = np.linspace(0, 4, num=10)
# y = np.linspace(0, 9, num=10)
# z = np.linspace(0, 4, num=10)
# q = np.linspace(42., 42., num=10)

pos = np.stack((x,y,z), axis=1)
q = np.expand_dims(q, axis=1)
# q = np.stack((q,q,q), axis=1)

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

model = Hello()

output = model([torch.LongTensor(pos),torch.FloatTensor(q).to(device)])

print('Output SparseConvNetTensor:', output.shape)
