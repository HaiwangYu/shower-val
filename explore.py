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

def gen_dist(input_list='list/list-val.csv', nsample = 10) :
    start_sample = 0
    max_sample = nsample + start_sample
    dists = []
    start = timer()
    with open(input_list) as f:
        reader = csv.reader(f, delimiter=' ')
        ntry = 0
        for row in reader:
            ntry = ntry + 1
            if ntry < start_sample :
                continue
            if ntry > max_sample :
                break
            print('ntry: {} : {}'.format(ntry,row[0]))
            
            coords, ft = util.load(row, vis=False, vox=False)
            
            rec_idx = np.argmax(ft[:,2])
            tru_idx = np.argmax(ft[:,-1])
            t = coords[tru_idx,:]
            r = coords[rec_idx,:]
            d_trad = np.linalg.norm(t-r)

            qcoords = coords[ft[:,0]>0]
            d_charge, i = util.closest(qcoords, t)

            coords_p_np, ft_p_np = util.load(row, vis=False, vox=False, mode='vox')
            trad_pred_filter = ft_p_np[:,1] > 0
            coords_p_tp = coords_p_np[trad_pred_filter]
            ft_p_tp = ft_p_np[trad_pred_filter]
            coords_p_trad = coords_p_np[np.argmax(ft_p_np[:,2])]

            d_cand, i = util.closest(coords_p_tp, t)

            dists.append([d_trad, d_charge, d_cand])

            # if d_cand > 1:
            #     fontsize = 24
            #     fig = plt.figure(0)
            #     ax = fig.add_subplot(111)
            #     title = '{} : {:.1f}, {:.1f}'.format(row[0].split('/')[-1].split('.')[0], d_cand, dists[ntry-1, 0])
            #     ax.set_title(title, fontsize=fontsize)
            #     charge_filter = ft[:,0] > 0
            #     img = ax.scatter(coords[charge_filter,2], coords[charge_filter,1], c=ft[charge_filter,0], cmap="jet", alpha=0.1)
            #     plt.colorbar(img)
            #     cand_filter = ft[:,1] > 0
            #     ax.scatter(coords[cand_filter,2], coords[cand_filter,1], marker='*', facecolors='none', edgecolors='y', label='candidate')
            #     ax.scatter(coords[rec_idx,2], coords[rec_idx,1], marker='s', facecolors='none', edgecolors='g', label='rec. vtx.')
            #     truth_fiter = ft[:,3] > 0
            #     ax.scatter(coords[truth_fiter,2], coords[truth_fiter,1], marker='s', facecolors='none', edgecolors='r', label='truth vtx.')
                
            #     plt.legend(loc='best', fontsize=fontsize)
            #     plt.xlabel('Z [cm]', fontsize=fontsize)
            #     plt.ylabel('Y [cm]', fontsize=fontsize)
            #     plt.xticks(fontsize=fontsize)
            #     plt.yticks(fontsize=fontsize)
            #     plt.grid()
            #     plt.show()

    end = timer()
    print('time: {0:.1f} ms'.format((end-start)/1*1000))
    dists = np.array(dists)
    np.savetxt('dist.csv', dists, delimiter=',')
    return dists

if __name__ == '__main__' :
    input_list='list/numucc-24k-val.csv'
    dists = gen_dist(input_list, 1000) # from gen
    # dists = np.loadtxt('dist.csv', delimiter=',') # from file

    nsample = dists.shape[0]

    closest_v = np.count_nonzero(dists[:,0]<1) / nsample
    closest_q = np.count_nonzero(dists[:,1]<1) / nsample
    print('closest_v < 1: {}'.format(closest_v))
    print('closest_q < 1: {}'.format(closest_q))

    fontsize = 24
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    # ax.hist(dists[:,0], 600, range=(-0.05, 59.05), density=True, label='Rec. Vtx. PDF')
    # ax.hist(dists[:,1], 600, range=(-0.05, 59.05), density=True, histtype='step', label='Closest Charge PDF')
    ax.hist(dists[:,0], 5000, range=(-0.05, 499.5), density=True, histtype='step', linewidth=2, cumulative=True, label='Rec. Vtx.')
    ax.hist(dists[:,1], 5000, range=(-0.05, 499.5), density=True, histtype='step', linewidth=2, cumulative=True, label='Closest Charge')
    ax.hist(dists[:,2], 5000, range=(-0.05, 499.5), density=True, histtype='step', linewidth=2, cumulative=True, label='Closest Candidate')
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.xlabel('Distance [cm]', fontsize=fontsize)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(-1,5)
    plt.grid()
    plt.show()