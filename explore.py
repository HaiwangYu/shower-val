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

def closest(a, v) :
    l = a.shape[0]
    d = np.empty(l)
    for i in range(l) :
        d[i] = np.linalg.norm(a[i]-v)
    idx = np.argmin(d)
    return d[idx], idx

def gen_dist(nsample = 10) :
    start_sample = 0
    max_sample = nsample + start_sample
    dist = np.zeros([nsample, 2])
    start = timer()
    with open('list1-val.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        ntry = 0
        for row in reader:
            ntry = ntry + 1
            if ntry < start_sample :
                continue
            if ntry > max_sample :
                break
            # print('ntry: {} : {}'.format(ntry,row[0]))
            
            coords, ft = util.load_vtx(row, vis=False, vox=False)
            
            rec_idx = np.argmax(ft[:,2])
            tru_idx = np.argmax(ft[:,-1])
            t = coords[tru_idx,:]
            r = coords[rec_idx,:]
            dist[ntry-1, 0] = np.linalg.norm(t-r)

            qcoords = coords[ft[:,0]>0]
            d, i = closest(qcoords, t)
            dist[ntry-1, 1] = d

            if d > 1:
                fontsize = 24
                fig = plt.figure(0)
                ax = fig.add_subplot(111)
                title = '{} : {:.1f}, {:.1f}'.format(row[0].split('/')[-1].split('.')[0], d, dist[ntry-1, 0])
                ax.set_title(title, fontsize=fontsize)
                charge_filter = ft[:,0] > 0
                img = ax.scatter(coords[charge_filter,2], coords[charge_filter,1], c=ft[charge_filter,0], cmap="jet", alpha=0.1)
                plt.colorbar(img)
                cand_filter = ft[:,1] > 0
                ax.scatter(coords[cand_filter,2], coords[cand_filter,1], marker='*', facecolors='none', edgecolors='y', label='candidate')
                ax.scatter(coords[rec_idx,2], coords[rec_idx,1], marker='s', facecolors='none', edgecolors='g', label='rec. vtx.')
                truth_fiter = ft[:,3] > 0
                ax.scatter(coords[truth_fiter,2], coords[truth_fiter,1], marker='s', facecolors='none', edgecolors='r', label='truth vtx.')
                
                plt.legend(loc='best', fontsize=fontsize)
                plt.xlabel('Z [cm]', fontsize=fontsize)
                plt.ylabel('Y [cm]', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.grid()
                plt.show()

    end = timer()
    print('time: {0:.1f} ms'.format((end-start)/1*1000))
    return dist

nsample = 100
dist = gen_dist(nsample)
# np.savetxt('dist.csv', dist, delimiter=',')
# dist = np.loadtxt('dist.csv', delimiter=',')

closest_v = np.count_nonzero(dist[:,0]<1) / nsample
closest_q = np.count_nonzero(dist[:,1]<1) / nsample
print('closest_v < 1: {}'.format(closest_v))
print('closest_q < 1: {}'.format(closest_q))

fontsize = 24
fig = plt.figure(1)
ax = fig.add_subplot(111)
# ax.hist(dist[:,0], 600, range=(-0.05, 59.05), density=True, label='Rec. Vtx. PDF')
# ax.hist(dist[:,1], 600, range=(-0.05, 59.05), density=True, histtype='step', label='Closest Charge PDF')
ax.hist(dist[:,0], 1000, density=True, histtype='step', linewidth=2, cumulative=True, label='Rec. Vtx. CDF')
ax.hist(dist[:,1], 1000, density=True, histtype='step', linewidth=2, cumulative=True, label='Closest Charge CDF')
plt.legend(loc='lower right', fontsize=fontsize)
plt.xlabel('Distance [cm]', fontsize=fontsize)
plt.ylabel('Probability', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid()
plt.show()