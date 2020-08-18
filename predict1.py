import torch
import torch.nn as nn
import torch.optim as optim
import sparseconvnet as scn
import uproot
import matplotlib.pyplot as plt
import numpy as np
from model import DeepVtx

from timeit import default_timer as timer
import csv
import util

# Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
# use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
use_cuda = False
torch.set_num_threads(1)

device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print("Using CUDA.")
else:
    print("Using CPU.")

nIn = 1
model = DeepVtx(dimension=3, nIn=nIn, device=device)
model.train()

model_path = 't48k/m16-l5-lr5d-res0.5/CP24.pth'
model.load_state_dict(torch.load(model_path))

start_sample = 0
max_sample = 1000 + start_sample
resolution = 0.5
loose_cut = 1.0
# val_list = 'list/numucc-24k-val.csv'
val_list = 'list/nuecc-21k-val.csv'
results = []
start = timer()
with open(val_list) as f:
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
        
        coords_np, ft_np = util.load(row, vis=False, resolution=resolution)
        coords = torch.LongTensor(coords_np)
        truth = torch.FloatTensor(ft_np[:,-1]).to(device)
        ft = torch.FloatTensor(ft_np[:,0:-1]).to(device)
        prediction = model([coords,ft[:,0:1]])
        
        pred_np = prediction.cpu().detach().numpy()
        pred_np = pred_np[:,1] - pred_np[:,0]
        truth_np = truth.cpu().detach().numpy()
        
        # prediction and vis
        result = util.vis_prediction_regseg(
            np.column_stack((coords_np, pred_np)),
            np.column_stack((coords_np, truth_np)),
            cand=np.column_stack((coords_np, ft_np[:,1])),
            vis=True
            )
        results.append(result)

end = timer()
print('time: {0:.1f} ms'.format((end-start)/1*1000))