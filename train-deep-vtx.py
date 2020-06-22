import torch
import torch.nn as nn
import torch.optim as optim
import sparseconvnet as scn
import uproot
import matplotlib.pyplot as plt
import numpy as np
from model import Hello
from model import ResNet
from model import DeepVtx

from timeit import default_timer as timer

def vtype_encodeing_func(type) :
    if type == 0:
        return 1
    elif type == 2:
        return 2
    elif type == 6:
        return 3
    return 0
vtype_encodeing = np.vectorize(vtype_encodeing_func)

file = uproot.open('data/nue_7054_173_8677.root')
tblob = file['T_rec_charge_blob']
tvtx = file['T_vtx']

x = tblob.array('x')
y = tblob.array('y')
z = tblob.array('z')
q = tblob.array('q')
blob_coords = np.stack((x,y,z), axis=1)
blob_ft = np.stack((q,np.zeros_like(q),np.zeros_like(q)), axis=1)
print(blob_coords.shape)

vx = tvtx.array('x')
vy = tvtx.array('y')
vz = tvtx.array('z')
vtype = vtype_encodeing(tvtx.array('type'))
vmain = tvtx.array('flag_main')
vtx_coords = np.stack((vx,vy,vz), axis=1)
vtx_ft = np.stack((np.zeros_like(vtype),vtype,vmain), axis=1)
# sort by vtype by decreasing order
vtx_coords = vtx_coords[np.argsort(vtx_ft[:, 1])[::-1]]
vtx_ft = vtx_ft[np.argsort(vtx_ft[:, 1])[::-1]]
# print(vtx_ft)

coords = np.concatenate((vtx_coords, blob_coords), axis=0)
ft = np.concatenate((vtx_ft, blob_ft), axis=0)
print(coords.shape, ', ', ft.shape)

# input visualize
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
coords = coords[0:len(vtype),]
ft = ft[0:len(vtype),]
img = ax.scatter(x, y, z, cmap="Greys", alpha=0.05)
img = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=ft[:,1], cmap=plt.jet())
ax = fig.add_subplot(122)
img = ax.scatter(y, z, cmap="Greys", alpha=0.05)
img = ax.scatter(coords[:,1], coords[:,2], c=ft[:,1], cmap=plt.jet(), marker='*')
plt.xlim(-100,0)
plt.ylim(300,500)
fig.colorbar(img)
plt.show()
exit()

# Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print("Using CUDA.")
else:
    print("Not using CUDA.")
device = 'cpu'
model = DeepVtx(dimension=3, device=device)

# check the SparseTenSor
# input_layer = model.inputLayer
# st = input_layer([torch.LongTensor(pos),torch.FloatTensor(q).to(device)])
# sl = st.get_spatial_locations()
# print('spatial_locations: ', sl.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# img = ax.scatter(sl[:,0], sl[:,1], sl[:,2], cmap=plt.jet())
# fig.colorbar(img)
# plt.show()
# exit()


criterion = nn.BCELoss().to(device)
# criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

nepoch = 50
# leave warmup outside the timer
prediction = model([torch.LongTensor(coords),torch.FloatTensor(ft).to(device)])
model.train()
start = timer()
for epoch in range(nepoch):
    if epoch%10==0 :
        print('epoch: ', epoch)
    prediction = model([torch.LongTensor(coords),torch.FloatTensor(ft).to(device)])
    # truth = torch.unsqueeze(torch.LongTensor(ft[:,2]).to(device), 1)
    truth = torch.FloatTensor(ft[:,2]).to(device)
    # print(prediction.shape, ', ', truth.shape)
    # print(truth)
    loss = criterion(prediction[0:10,0].view(-1),truth[0:10].view(-1))
    loss.backward()
    optimizer.step()
end = timer()
if nepoch > 0:
    print('Forward time: {0:.1f} ms'.format((end-start)/nepoch*1000))

# print('Output SparseConvNetTensor:', prediction)

# test segmentation results
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ncand = np.count_nonzero(ft[:,1]>1)
ncand = 10
img = ax.scatter(coords[0:ncand,0], coords[0:ncand,1], coords[0:ncand,2], c=prediction.cpu().detach().numpy()[0:ncand], cmap=plt.jet())
ax = fig.add_subplot(122)
img = ax.scatter(y, z, cmap="Greys", alpha=0.05)
img = ax.scatter(coords[0:ncand,1], coords[0:ncand,2], c=prediction.cpu().detach().numpy()[0:ncand], cmap=plt.jet(), marker='*')
plt.xlim(-100,0)
plt.ylim(300,500)
fig.colorbar(img)
plt.show()
