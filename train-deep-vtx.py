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

def balance_BCE(criterion, prediction, truth, sig_len = 1):
    if torch.isnan(prediction).any() or torch.isnan(truth).any() :
        return None
    if len(prediction.shape) != 1 or len(truth.shape) != 1 :
        raise Exception('input needs to have 1 dim')
    if prediction.shape[0] != truth.shape[0] :
        raise Exception('input needs to have same length')
    tot_len = prediction.shape[0]
    if tot_len < 1 or tot_len < sig_len or sig_len < 0 :
        raise Exception('length err')

    bkg_len = tot_len - sig_len
    loss_sig = criterion(prediction[0:sig_len], truth[0:sig_len]) * bkg_len / tot_len
    loss_bkg = criterion(prediction[sig_len:], truth[sig_len:]) * sig_len / tot_len

    # print(truth.shape[0], ': ', sig_len, ', ', bkg_len)
    return loss_sig + loss_bkg
    


# Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
# use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
use_cuda = False
torch.set_num_threads(1)

device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print("Using CUDA.")
else:
    print("Not using CUDA.")

model = DeepVtx(dimension=3, device=device)
model.train()
start_epoch = 34
if start_epoch > 0 :
    model.load_state_dict(torch.load('checkpoints/CP{}.pth'.format(start_epoch-1)))

# criterion = nn.BCELoss().to(device)
weight = torch.tensor([1, 400], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weight).to(device)
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

dir_checkpoint = 'checkpoints/'
outfile_loss = open(dir_checkpoint+'/loss.txt','a+')
ntrain = 4000
nval = 1000
nepoch = 50
start = timer()
for epoch in range(start_epoch, start_epoch+nepoch):

    # setup toolbar
    toolbar_width = 50
    epoch_time = timer()
    sys.stdout.write("epoch %d : [%s]" % (epoch, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    train_loss = 0
    train_hits = 0
    with open('list1-train.csv') as f:
        optimizer.zero_grad()
        reader = csv.reader(f, delimiter=' ')
        isample = 0
        for row in reader:
            if isample > ntrain :
                break
            coords, ft = util.load_vtx(row, vis=False)
            truth = torch.LongTensor(ft[:,-1]).to(device)
            # remove the truth from ft
            ft = ft[:,0:-1]
            prediction = model([torch.LongTensor(coords),torch.FloatTensor(ft).to(device)])
            if isample%(ntrain/toolbar_width) == 0 :
                sys.stdout.write("=")
                sys.stdout.flush()
                # print('\tisample: {}'.format(isample))
                # util.vis_prediction(coords, prediction)
            
            pred_np = prediction.cpu().detach().numpy()
            # class 1 - class 0 and exclude the 1st point
            pred_np = pred_np[1:,1] - pred_np[1:,0]
            if np.argmax(pred_np) == np.argmax(truth.cpu().detach().numpy()) :
                train_hits = train_hits + 1
            
            # loss = criterion(prediction[0:10,0].view(-1),truth[0:10].view(-1))
            # loss = balance_BCE(criterion, prediction.view(-1), truth.view(-1))
            loss = criterion(prediction,truth)
            if(loss is None) :
                continue
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            isample = isample + 1

    torch.save(model.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch))

    if isample > 0 :
        train_loss = train_loss / isample
        train_hits = train_hits / isample

    sys.stdout.write("]\n")
    
    # validation
    val_loss = 0
    val_hits = 0
    with open('list1-val.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        isample = 0
        for row in reader:
            if isample > nval :
                break
            coords, ft = util.load_vtx(row, vis=False)
            truth = torch.LongTensor(ft[:,-1]).to(device)
            # remove the truth from ft
            ft = ft[:,0:-1]
            prediction = model([torch.LongTensor(coords),torch.FloatTensor(ft).to(device)])
            
            pred_np = prediction.cpu().detach().numpy()
            pred_np = pred_np[:,1] - pred_np[:,0]
            if np.argmax(pred_np) == np.argmax(truth.cpu().detach().numpy()) :
                val_hits = val_hits + 1

            # loss = criterion(prediction[0:10,0].view(-1),truth[0:10].view(-1))
            # loss = balance_BCE(criterion, prediction.view(-1), truth.view(-1))
            loss = criterion(prediction,truth)
            if(loss is None) :
                continue
            val_loss += loss.item()
            isample = isample + 1

    if isample > 0 :
        val_loss = val_loss / isample
        val_hits = val_hits / isample

    epoch_time = timer() - epoch_time
    print('tloss: {:.6f} vloss: {:.6f} thits: {:.6f} vhits: {:.6f} time: {}'.format(train_loss, val_loss, train_hits, val_hits, epoch_time))
    print('{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, train_loss, val_loss, train_hits, val_hits), file=outfile_loss, flush=True)
end = timer()
if nepoch > 0:
    print('time/epoch: {0:.1f} ms'.format((end-start)/nepoch*1000))

