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
start_epoch = 40
if start_epoch > 0 :
    model.load_state_dict(torch.load('checkpoints/CP{}.pth'.format(start_epoch-1)))

w = 400
lr = 1e-5
# criterion = nn.BCELoss().to(device)
weight = torch.tensor([1, w], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weight).to(device)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
optimizer = optim.Adam(model.parameters(), lr=lr)

dir_checkpoint = 'checkpoints/'
outfile_loss = open(dir_checkpoint+'/loss.csv','a+')
outfile_log  = open(dir_checkpoint+'/log','a+')
ntrain = 1000
nval = 250
nepoch = 50

print('lr: {} weight: {} start: {} ntrain: {} nval: {} device: {}'.format(
    lr, w, start_epoch, ntrain, nval, device
), file=outfile_log, flush=True)

start = timer()
for epoch in range(start_epoch, start_epoch+nepoch):

    # setup toolbar
    toolbar_width = 50
    epoch_time = timer()
    sys.stdout.write("epoch %d : [%s]" % (epoch, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    epoch_loss = 0
    epoch_crt = np.zeros([2,2,2])
    hit = 0
    with open('list1-train.csv') as f:
        optimizer.zero_grad()
        reader = csv.reader(f, delimiter=' ')
        ntry = 0
        npass =  0
        nfail = np.zeros(10)
        for row in reader:
            ntry = ntry + 1
            if ntry > ntrain :
                break
            if ntry%(ntrain/toolbar_width) == 0 :
                sys.stdout.write("=")
                sys.stdout.flush()
            
            coords_np, ft_np = util.load_vtx(row, vis=False)
            
            if ft_np[np.argmax(ft_np[:,-1]), 0] <= 0 :
                nfail[0] = nfail[0] + 1
                # if epoch == start_epoch :
                #     print('no charge for {}'.format(ntry))
                #     util.load_vtx(row, vis=True)
                continue
            
            coords = torch.LongTensor(coords_np)
            truth = torch.LongTensor(ft_np[:,-1]).to(device)
            ft = torch.FloatTensor(ft_np[:,0:-1]).to(device)

            prediction = model([coords,ft])
            
            # debug section
            # input = model.inputLayer([torch.LongTensor(coords_np),torch.FloatTensor(ft_np).to(device)])
            # print(torch.FloatTensor(ft_np).to(device)[:,3]-input.features[:,3])
            # exit()

            # if True :
            #     pred_np = prediction.cpu().detach().numpy()
            #     pred_np = pred_np[:,1] - pred_np[:,0]
            #     truth_np = truth.cpu().detach().numpy()
            #     util.vis_prediction(coords_np, pred_np, truth_np, ref=ft_np[:,2], threshold=0)
            #     exit()
            
            pred_np = prediction.cpu().detach().numpy()
            # class 1 - class 0 and exclude the 1st point
            pred_np = pred_np[:,1] - pred_np[:,0]
            truth_np = truth.cpu().detach().numpy()
            vtx_id_truth = np.argmax(truth_np)
            
            c = 0; r = 0; t = 0
            if ft[vtx_id_truth,1] > 0 :
                c = 1
            if ft[vtx_id_truth,2] > 0 :
                r = 1
            if vtx_id_truth == np.argmax(pred_np):
                t = 1
                hit = hit + 1
            epoch_crt[c,r,t] = epoch_crt[c,r,t] + 1
            
            # if ntry == 1:
            #     print(coords_np[coords_np[:,0]==93])
            #     print(ft_np[coords_np[:,0]==93])
            #     print(ntry, ft_np)
            #     exit()
            
            # loss = criterion(prediction[0:10,0].view(-1),truth[0:10].view(-1))
            # loss = balance_BCE(criterion, prediction.view(-1), truth.view(-1))
            loss = criterion(prediction,truth)
            if(loss is None) :
                continue
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            npass = npass + 1

    torch.save(model.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch))

    train_loss = 0
    train_hits = 0
    if npass > 0 :
        train_loss = epoch_loss / npass
        # train_hits = np.sum(epoch_crt[:,:,1]) / npass
        train_hits = hit / npass

    sys.stdout.write("]\n")
    print(epoch_crt)
    print('train: ntry: {} npass: {} vq=0: {}'.format(ntry, npass, nfail[0]))
    
    # validation
    epoch_loss = 0
    epoch_crt = np.zeros([2,2,2])
    with open('list1-val.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        ntry = 0
        npass =  0
        nfail = np.zeros(10)
        for row in reader:
            ntry = ntry + 1
            if ntry > nval :
                break
            
            coords_np, ft_np = util.load_vtx(row, vis=False)
            
            if ft_np[np.argmax(ft_np[:,-1]), 0] <= 0 :
                nfail[0] = nfail[0] + 1
                # if epoch == start_epoch :
                #     print('no charge for {}'.format(ntry))
                continue
            
            coords = torch.LongTensor(coords_np)
            truth = torch.LongTensor(ft_np[:,-1]).to(device)
            ft = torch.FloatTensor(ft_np[:,0:-1]).to(device)

            prediction = model([coords,ft])
            
            pred_np = prediction.cpu().detach().numpy()
            pred_np = pred_np[:,1] - pred_np[:,0]
            truth_np = truth.cpu().detach().numpy()
            vtx_id_truth = np.argmax(truth_np)
            
            c = 0; r = 0; t = 0
            if ft[vtx_id_truth,1] > 0 :
                c = 1
            if ft[vtx_id_truth,2] > 0 :
                r = 1
            if vtx_id_truth == np.argmax(pred_np):
                t = 1
            epoch_crt[c,r,t] = epoch_crt[c,r,t] + 1

            # loss = criterion(prediction[0:10,0].view(-1),truth[0:10].view(-1))
            # loss = balance_BCE(criterion, prediction.view(-1), truth.view(-1))
            loss = criterion(prediction,truth)
            if(loss is None) :
                continue
            epoch_loss += loss.item()
            npass = npass + 1

    val_loss = 0
    val_hits = 0
    if npass > 0 :
        val_loss = epoch_loss / npass
        val_hits = np.sum(epoch_crt[:,:,1]) / npass

    epoch_time = timer() - epoch_time
    
    print(epoch_crt)
    print('val: ntry: {} npass: {} vq=0: {}'.format(ntry, npass, nfail[0]))
    print('tloss: {:.6f} vloss: {:.6f} thits: {:.6f} vhits: {:.6f} time: {:.6f}'.format(train_loss, val_loss, train_hits, val_hits, epoch_time))
    print('{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, train_loss, val_loss, train_hits, val_hits), file=outfile_loss, flush=True)
end = timer()
if nepoch > 0:
    print('time/epoch: {0:.1f} ms'.format((end-start)/nepoch*1000))

