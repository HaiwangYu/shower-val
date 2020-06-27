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
import csv
import util

# Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print("Using CUDA.")
else:
    print("Not using CUDA.")
# device = 'cpu'
model = DeepVtx(dimension=3, device=device)
model.train()

model_path = 'checkpoints/CP80.pth'
model.load_state_dict(torch.load(model_path))

start_sample = 0
max_sample = 100 + start_sample
start = timer()
with open('list1-val.csv') as f:
    reader = csv.reader(f, delimiter=' ')
    isample = 0
    for row in reader:
        isample = isample + 1
        if isample < start_sample :
            continue
        if isample > max_sample :
            break
        print('isample: {} : {}'.format(isample,row[0]))
        coords, ft = util.load_vtx(row, vis=False)
        # get truth from ft
        truth = torch.FloatTensor(ft[:,-1]).to(device)
        # remove the truth from ft
        ft = ft[:,0:-1]
        prediction = model([torch.LongTensor(coords),torch.FloatTensor(ft).to(device)])
        pred_np = prediction.cpu().detach().numpy()
        pred_np = pred_np[:,1] - pred_np[:,0]
        # if np.count_nonzero(pred_np>0.9) <= 0: continue
        util.vis_prediction(coords, pred_np, ft[:,2], -0.1)
        # util.vis_prediction(coords, ft[:,2], ft[:,1], 0.99)
end = timer()
print('time: {0:.1f} ms'.format((end-start)/1*1000))