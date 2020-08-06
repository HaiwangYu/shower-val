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

nsample = 100
input_tag = 'val'
input_list = 'hybrid.csv'

dists2 = np.loadtxt('dist-nuecc-16k-{}.csv'.format(input_tag), delimiter=',') # from file

def gen_dist() :
    input_list='list/nuecc-21k-{}.csv'.format(input_tag)
    input_list='list/numucc-24k-{}.csv'.format(input_tag)
    
    model_path = 'CELoss/t16k/m16-l5-lrd-res1.0/CP35.pth'
    model_path = 't48k/m16-l5-lr5d-res0.5/CP24.pth'

    # TODO tune these cuts
    resolution = 0.5
    dnn_trad_dist_cut = 2.0 # cm

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

    model.load_state_dict(torch.load(model_path))

    start_sample = 0
    max_sample = nsample + start_sample
    dists = []
    start = timer()
    with open(input_list) as f:
        reader = csv.reader(f, delimiter=' ')
        isample = 0
        for row in reader:
            isample = isample + 1
            if isample < start_sample :
                continue
            if isample > max_sample :
                break
            print('isample: {} : {}'.format(isample,row[0]))
            
            coords_np, ft_np = util.load(row, vis=True, resolution=resolution, mode='vox')
            Truth_shower_KE_MeV = float(row[5])
            
            # vertex charge cut
            # if ft_np[np.argmax(ft_np[:,-1]), 0] <= 0 :
            #     continue

            coords = torch.LongTensor(coords_np)
            truth = torch.LongTensor(ft_np[:,-1]).to(device)
            ft = torch.FloatTensor(ft_np[:,0:-1]).to(device)
            prediction = model([coords,ft[:,0:1]])
            
            pred_np = prediction.cpu().detach().numpy()
            pred_np = pred_np[:,1] - pred_np[:,0]
            truth_np = truth.cpu().detach().numpy()
            
            ################
            # hybrid alg.
            ################

            # point based coords and ft
            coords_p_np, ft_p_np = util.load(row, vis=False, vox=False, mode='vox')
            
            # vox -> point
            coords_np = coords_np.astype(float)
            coords_np *= resolution
            for i in range(3) :
                coords_np[:,i] += np.min(coords_p_np[:,i]) + 0.5*resolution

            # TODO handle multiple candidates
            dnn_pred_idx = np.argmax(pred_np)
            coords_p_dnn = coords_np[dnn_pred_idx]

            trad_pred_filter = ft_p_np[:,1] > 0
            coords_p_tp = coords_p_np[trad_pred_filter]
            ft_p_tp = ft_p_np[trad_pred_filter]
            coords_p_trad = coords_p_np[np.argmax(ft_p_np[:,2])]

            d_dnn_trad, i = util.closest(coords_p_tp, coords_p_dnn)
            coords_p_hybrid = coords_p_tp[i]
            
            if d_dnn_trad > dnn_trad_dist_cut :
                print('dnn_trad_dist_cut fail')
                coords_p_hybrid = coords_p_trad

            truth_p_idx = np.argmax(ft_p_np[:,-1])
            coords_p_truth = coords_p_np[truth_p_idx]

            d_dnn = np.linalg.norm(coords_p_dnn - coords_p_truth)
            d_trad = np.linalg.norm(coords_p_trad - coords_p_truth)
            d_hybrid = np.linalg.norm(coords_p_hybrid - coords_p_truth)
            dists.append([d_dnn, d_trad, d_hybrid, d_dnn_trad, Truth_shower_KE_MeV, np.max(pred_np)])
            # print('hybrid dist: {}'.format(d_dnn_trad))
            
            # voxel based vis
            # if d_hybrid < 0.1 :
            #     print('debug: ', d_hybrid)
            #     ret = util.vis_prediction(coords_np, ft_np, pred_np, truth_np, ref1=ft_np[:,1], ref2=ft_np[:,2], resolution=resolution, loose_cut=1., vis=True)
            
    end = timer()
    print('time: {0:.1f} ms'.format((end-start)/1*1000))

    dists = np.array(dists)
    np.savetxt('hybrid.csv', dists, delimiter=',')

    return dists

if __name__ == '__main__' :
    dists = gen_dist()
    # dists = np.loadtxt('hybrid.csv', delimiter=',')

    fontsize = 24
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_title(input_tag, fontsize=fontsize)
    ax.hist(dists[0:nsample,0], 5000, range=(-0.05, 499.5), density=True, histtype='step', linewidth=2, cumulative=True, label='DNN')
    ax.hist(dists[0:nsample,1], 5000, range=(-0.05, 499.5), density=True, histtype='step', linewidth=2, cumulative=True, label='Tradition')
    ax.hist(dists[0:nsample,2], 5000, range=(-0.05, 499.5), density=True, histtype='step', linewidth=2, cumulative=True, label='Hybrid')
    ax.hist(dists2[0:nsample,1], 5000, range=(-0.05, 499.5), density=True, histtype='step', linewidth=2, cumulative=True, label='Closest Charge')
    ax.hist(dists2[0:nsample,2], 5000, range=(-0.05, 499.5), density=True, histtype='step', linewidth=2, cumulative=True, label='Closest Candidate')
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.xlabel('Distance [cm]', fontsize=fontsize)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.xlim(-1,5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid()

    match_cut = 1.0
    nbins = 10
    x_vals = dists[:,4]
    x_bins = np.linspace(0, 2000, nbins)
    y = []
    x = []
    for i in range(nbins-1) :
        x_filter = np.logical_and(x_vals>=x_bins[i], x_vals<x_bins[i+1])
        den = np.count_nonzero(x_filter)
        eff0 = np.count_nonzero(dists[x_filter,0]<match_cut) / den
        eff1 = np.count_nonzero(dists[x_filter,1]<match_cut) / den
        eff2 = np.count_nonzero(dists[x_filter,2]<match_cut) / den
        x.append(0.5*(x_bins[i]+x_bins[i+1]))
        y.append([eff0, eff1, eff2])
    
    x = np.array(x)
    y = np.array(y)

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    
    # Truth Shower KE hist
    # ax.hist(x_vals, 100, range=(0, 2000), density=True, label='Truth Shower KE')
    # plt.xlabel('Truth Shower KE [MeV]', fontsize=fontsize)
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.show()
    
    ax.set_title(input_tag, fontsize=fontsize)
    plt.plot(x, y[:,0], '-o', linewidth=2, label='DNN')
    plt.plot(x, y[:,1], '-o', linewidth=2, label='Tradition')
    plt.plot(x, y[:,2], '-o', linewidth=2, label='Hybrid')
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel('Truth Shower KE [MeV]', fontsize=fontsize)
    plt.ylabel('Eff.', fontsize=fontsize)
    plt.xlim(0, 2000)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid()



    
    
    
    plt.show()