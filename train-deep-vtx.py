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

import sys
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
torch.set_num_threads(1)

model = DeepVtx(dimension=3, device=device)
model.train()

criterion = nn.BCELoss().to(device)
# criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

dir_checkpoint = 'checkpoints/'
outfile_loss = open(dir_checkpoint+'/loss.txt','w')
ntrain = 1000
nval = 200
nepoch = 50
start = timer()
for epoch in range(nepoch):

    # setup toolbar
    toolbar_width = 20
    sys.stdout.write("epoch %d : [%s]" % (epoch, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    train_loss = 0
    with open('list1-train.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        isample = 0
        for row in reader:
            if isample > ntrain :
                break
            coords, ft = util.load_vtx(row, vis=False)
            truth = torch.FloatTensor(ft[:,-1]).to(device)
            # remove the truth from ft
            ft = ft[:,0:-1]
            prediction = model([torch.LongTensor(coords),torch.FloatTensor(ft).to(device)])
            if isample%(ntrain/toolbar_width) == 0 :
                sys.stdout.write("=")
                sys.stdout.flush()
                # print('\tisample: {}'.format(isample))
                # util.vis_prediction(coords, prediction)
            loss = criterion(prediction[0:10,0].view(-1),truth[0:10].view(-1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            isample = isample + 1
    torch.save(model.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch))
    train_loss = train_loss / isample
    sys.stdout.write("]\n")
    
    # validation
    val_loss = 0
    with open('list1-val.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        isample = 0
        for row in reader:
            if isample > nval :
                break
            coords, ft = util.load_vtx(row, vis=False)
            truth = torch.FloatTensor(ft[:,-1]).to(device)
            # remove the truth from ft
            ft = ft[:,0:-1]
            prediction = model([torch.LongTensor(coords),torch.FloatTensor(ft).to(device)])
            loss = criterion(prediction[0:10,0].view(-1),truth[0:10].view(-1))
            val_loss += loss.item()
            isample = isample + 1
    val_loss = val_loss / isample
    print('epoch {} finished - tloss: {:.6f} vloss: {:.6f}'.format(epoch, train_loss, val_loss))
    print('{}, {:.6f}, {:.6f}'.format(epoch, train_loss, val_loss), file=outfile_loss, flush=True)
end = timer()
if nepoch > 0:
    print('time/epoch: {0:.1f} ms'.format((end-start)/nepoch*1000))

