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
device = 'cpu'
model = DeepVtx(dimension=3, device=device)
model.train()

criterion = nn.BCELoss().to(device)
# criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

dir_checkpoint = 'checkpoints/'
max_samples = 100
nepoch = 10
start = timer()
for epoch in range(nepoch):
    if epoch%10==0 :
        print('epoch: ', epoch)
    sample = []
    with open('list1.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        isample = 0
        for row in reader:
            isample = isample + 1
            if isample > max_samples :
                break
            coords, ft = util.load_vtx(row, vis=False)
            # get truth from ft
            truth = torch.FloatTensor(ft[:,-1]).to(device)
            # remove the truth from ft
            ft = ft[:,0:-1]
            prediction = model([torch.LongTensor(coords),torch.FloatTensor(ft).to(device)])
            loss = criterion(prediction[0:10,0].view(-1),truth[0:10].view(-1))
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch))
end = timer()
if nepoch > 0:
    print('Training time: {0:.1f} ms'.format((end-start)/nepoch*1000))

