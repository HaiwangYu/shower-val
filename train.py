#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import sparseconvnet as scn
import numpy as np
import csv
import uproot
import matplotlib.pyplot as plt

import argparse
import math
import re
import sys
from timeit import default_timer as timer

from model import Hello
from model import ResNet
from model import DeepVtx
import util


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir-checkpoint', default='checkpoints',
                        metavar='PATH',
                        help="dir for checkpoints"
                             " (default : 'checkpoints')")

    parser.add_argument('--train-list', default='list/nuecc-39k-train.csv',
                        metavar='TRAINLIST',
                        help='train-list')
    parser.add_argument('--val-list', default='list/nuecc-21k-val.csv',
                        metavar='VALLIST',
                        help='val-list')
    parser.add_argument('--ntrain', type=int,
                        help="ntrain",
                        default=100)
    parser.add_argument('--nval', type=int,
                        help="ntrain",
                        default=20)
    parser.add_argument('--nepoch', type=int,
                        help="nepoch",
                        default=2)
    parser.add_argument('--start-epoch', type=int,
                        help="continue from previous checkpoint if > 0",
                        default=0)

    parser.add_argument('--resolution', type=float,
                        help="resolution for voxelization in cm",
                        default=1.0)
    parser.add_argument('--loose_cut', type=float,
                        help="loose_cut used in in-training monitoring in cm",
                        default=2.0)

    parser.add_argument('--lr0', type=float,
                        help="initital learning rate",
                        default=1e-5)
    parser.add_argument('--lrd', type=float,
                        help="learning rate decay lr = lr0*exp(-lrd*epoch)",
                        default=0.05)
    parser.add_argument('--use-cuda', action='store_true',
                        help="Use cuda",
                        default=False)
    parser.add_argument('--vis', action='store_true',
                        help="visualize data",
                        default=False)

    return parser.parse_args()


def scheduler_exp(optimizer, lr0, gamma, epoch):
    lr = lr0*math.exp(-gamma*epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == "__main__":
    args = get_args()
    outfile_log = open(args.dir_checkpoint+'/log','a+')
    print(args, file=outfile_log, flush=True)

    # Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
    # use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
    use_cuda = args.use_cuda
    torch.set_num_threads(1)

    device = 'cuda:0' if use_cuda else 'cpu'
    if use_cuda:
        print("Using CUDA.")
    else:
        print("Using CPU.")

    nIn = 1
    model = DeepVtx(dimension=3, nIn=nIn, device=device)
    model.train()
    start_epoch = args.start_epoch
    if start_epoch > 0 :
        model.load_state_dict(torch.load('{}/CP{}.pth'.format(args.dir_checkpoint, start_epoch-1)))
    
    # loss
    
    # w = 100
    # weight = torch.tensor([1, w], dtype=torch.float32)
    # criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    
    criterion = nn.MSELoss().to(device)

    # optimizer
    lr0 = args.lr0
    lr_decay = args.lrd
    # optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=lr0)

    outfile_loss = open(args.dir_checkpoint+'/loss.csv','a+')
    train_list = args.train_list
    val_list = args.val_list
    ntrain = args.ntrain
    nval = args.nval
    nepoch = args.nepoch
    # batch_size = 1
    resolution = args.resolution
    loose_cut = args.loose_cut
    vertex_assign_cut = 0.0

    print('lr: {:.2e}*exp-{:.2e}*epoch start: {} ntrain: {} nval: {} device: {} nIn: {} resolution:{} loose_cut: {}'.format(
        lr0, lr_decay, start_epoch, ntrain, nval, device, nIn, resolution, loose_cut
    ), file=outfile_log, flush=True)
    print('train: {} val: {}'.format(
        train_list, val_list
    ), file=outfile_log, flush=True)

    start = timer()
    for epoch in range(start_epoch, start_epoch+nepoch):
        optimizer = scheduler_exp(optimizer, lr0, lr_decay, epoch)

        # setup toolbar
        toolbar_width = 100
        epoch_time = timer()
        sys.stdout.write("train %d : [%s]" % (epoch, " " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        
        epoch_loss = 0
        epoch_crt = np.zeros([2,2,2])
        epoch_pur = 0; epoch_eff = 0; epoch_loose = 0
        batch_list = []
        with open(train_list) as f:
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
                
                coords_np, ft_np = util.load(row, vis=args.vis, resolution=resolution, vertex_assign_cut=vertex_assign_cut)
                
                if ft_np[np.argmax(ft_np[:,-1]), 0] <= 0 :
                    nfail[0] = nfail[0] + 1
                    continue
                
                # mini-batch
                # if len(batch_list) < batch_size :
                #     batch_list.append(row)
                #     continue
                # else :
                #     coords_np, ft_np = util.batch_load(batch_list)
                
                coords = torch.LongTensor(coords_np)
                truth = torch.FloatTensor(ft_np[:,-1]).to(device)
                ft = torch.FloatTensor(ft_np[:,0:-1]).to(device)

                prediction = model([coords,ft[:,0:nIn]])
                
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
                if np.isnan(pred_np).any() :
                    continue
                # class 1 - class 0 and exclude the 1st point
                pred_np = pred_np[:,1] - pred_np[:,0]
                truth_np = truth.cpu().detach().numpy()
                truth_idx = np.argmax(truth_np)
                pred_idx = np.argmax(pred_np)
                
                c = 0; r = 0; t = 0
                if ft[truth_idx,1] > 0 :
                    c = 1
                if ft[truth_idx,2] > 0 :
                    r = 1
                if truth_idx == pred_idx:
                    t = 1
                epoch_crt[c,r,t] += 1

                # pred_cand = pred_np >= pred_np[np.argmax(pred_np)]
                pred_cand = pred_np > 0
                if pred_cand[truth_idx] == True :
                    epoch_eff += 1
                    epoch_pur += 1./np.count_nonzero(pred_cand)
                d = np.linalg.norm(coords[pred_idx,:] - coords[truth_idx,:])
                if d*resolution <= loose_cut :
                    epoch_loose += 1
                loss = criterion(prediction[:,1]-prediction[:,0],truth)
                if(loss is None) :
                    continue
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                npass = npass + 1

        sys.stdout.write("]\n")

        torch.save(model.state_dict(), args.dir_checkpoint + '/CP{}.pth'.format(epoch))

        train_loss = 0
        train_hits = 0
        train_pur = 0
        train_eff = 0
        train_loose = 0
        if npass > 0 :
            train_loss = epoch_loss / npass
            train_hits = np.sum(epoch_crt[:,:,1]) / npass
            train_eff = epoch_eff / npass
            train_pur = epoch_pur / npass
            train_loose = epoch_loose / npass
        
        if epoch == start_epoch :
            print('train: ntry: {} npass: {} vq=0: {}'.format(ntry, npass, nfail[0]), file=outfile_log, flush=True)
        print('epoch: {}'.format(epoch), file=outfile_log, flush=True)
        print(epoch_crt, file=outfile_log, flush=True)
        
        # validation
        sys.stdout.write("val   %d : [%s]" % (epoch, " " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        
        epoch_loss = 0
        epoch_crt = np.zeros([2,2,2])
        epoch_pur = 0; epoch_eff = 0; epoch_loose = 0
        with open(val_list) as f:
            reader = csv.reader(f, delimiter=' ')
            ntry = 0
            npass =  0
            nfail = np.zeros(10)
            for row in reader:
                ntry = ntry + 1
                if ntry > nval :
                    break
                if ntry%(nval/toolbar_width) == 0 :
                    sys.stdout.write("=")
                    sys.stdout.flush()
                
                coords_np, ft_np = util.load(row, vis=args.vis, resolution=resolution, vertex_assign_cut=vertex_assign_cut)
                
                if ft_np[np.argmax(ft_np[:,-1]), 0] <= 0 :
                    nfail[0] = nfail[0] + 1
                    # if epoch == start_epoch :
                    #     print('no charge for {}'.format(ntry))
                    continue
                
                coords = torch.LongTensor(coords_np)
                truth = torch.FloatTensor(ft_np[:,-1]).to(device)
                ft = torch.FloatTensor(ft_np[:,0:-1]).to(device)

                prediction = model([coords,ft[:,0:nIn]])
                
                pred_np = prediction.cpu().detach().numpy()
                if np.isnan(pred_np).any() :
                    continue
                pred_np = pred_np[:,1] - pred_np[:,0]
                truth_np = truth.cpu().detach().numpy()
                truth_idx = np.argmax(truth_np)
                pred_idx = np.argmax(pred_np)
                
                c = 0; r = 0; t = 0
                if ft[truth_idx,1] > 0 :
                    c = 1
                if ft[truth_idx,2] > 0 :
                    r = 1
                if truth_idx == pred_idx:
                    t = 1
                epoch_crt[c,r,t] = epoch_crt[c,r,t] + 1

                # pred_cand = pred_np >= pred_np[np.argmax(pred_np)]
                pred_cand = pred_np > 0
                if pred_cand[truth_idx] == True :
                    epoch_eff = epoch_eff + 1
                    epoch_pur = epoch_pur + 1./np.count_nonzero(pred_cand)
                d = np.linalg.norm(coords[pred_idx,:] - coords[truth_idx,:])
                if d*resolution <= loose_cut :
                    epoch_loose += 1

                loss = criterion(prediction[:,1]-prediction[:,0],truth)
                if(loss is None) :
                    continue
                epoch_loss += loss.item()
                npass = npass + 1

        val_loss = 0
        val_hits = 0
        val_pur = 0
        val_eff = 0
        val_loose = 0
        if npass > 0 :
            val_loss = epoch_loss / npass
            val_hits = np.sum(epoch_crt[:,:,1]) / npass
            val_eff = epoch_eff / npass
            val_pur = epoch_pur / npass
            val_loose = epoch_loose / npass

        sys.stdout.write("]\n")
        
        epoch_time = timer() - epoch_time
        
        if epoch == start_epoch :
            print('train: ntry: {} npass: {} vq=0: {}'.format(ntry, npass, nfail[0]), file=outfile_log, flush=True)
        print('epoch: {}'.format(epoch), file=outfile_log, flush=True)
        print(epoch_crt, file=outfile_log, flush=True)
        
        metrics = '{}, '.format(epoch)
        metrics += 'loss: {:.6f}, {:.6f}, '.format(train_loss, val_loss)
        metrics += 'hit: {:.6f}, {:.6f}, '.format(train_hits, val_hits)
        metrics += 'eff: {:.6f}, {:.6f}, '.format(train_eff, val_eff)
        metrics += 'pur: {:.6f}, {:.6f}, '.format(train_pur, val_pur)
        metrics += 'loose: {:.6f}, {:.6f}, '.format(train_loose, val_loose)
        metrics += 'time: {:.6f}, '.format(epoch_time)
        print(metrics)
        print(re.sub(r'[a-z]*: ', r'', metrics), file=outfile_loss, flush=True)
    end = timer()
    if nepoch > 0:
        print('time/epoch: {0:.1f} ms'.format((end-start)/nepoch*1000))

