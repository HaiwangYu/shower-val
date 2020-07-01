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

model_path = 'checkpoints/CP19.pth'
model.load_state_dict(torch.load(model_path))

start_sample = 0
max_sample = 1000 + start_sample
resolution = 2.0
start = timer()
with open('list1-val.csv') as f:
    reader = csv.reader(f, delimiter=' ')
    isample = 0
    stat = {}
    for row in reader:
        isample = isample + 1
        if isample < start_sample :
            continue
        if isample > max_sample :
            break
        print('isample: {} : {}'.format(isample,row[0]))
        
        coords_np, ft_np = util.load(row, vis=False, res=resolution)
        coords = torch.LongTensor(coords_np)
        truth = torch.LongTensor(ft_np[:,-1]).to(device)
        ft = torch.FloatTensor(ft_np[:,0:-1]).to(device)
        prediction = model([coords,ft[:,0:1]])
        
        pred_np = prediction.cpu().detach().numpy()
        pred_np = pred_np[:,1] - pred_np[:,0]
        truth_np = truth.cpu().detach().numpy()
        key = util.vis_prediction(coords_np, ft_np, pred_np, truth_np, ref=ft_np[:,2], threshold=0)
        if key in stat :
            stat[key] += 1
        else :
            stat[key] = 1
    for key in stat :
        print(key, ': ', stat[key])
end = timer()
print('time: {0:.1f} ms'.format((end-start)/1*1000))