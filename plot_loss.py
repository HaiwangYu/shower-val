import matplotlib.pyplot as plt
import numpy as np

inputs   = [
    # 'test-m16-l4-lr0.0001-adam-t500v100-e35/loss.txt',
    # 'test-m16-l4-lr0.0001-adam-t1000v100-e27/loss.txt',
    # 'test-m16-l4-lr0.0001-0.00001-adam-t4000v1000-e77/loss.txt',
    'checkpoints/loss.txt',
    ]
in_labels = [
    # 't500v100',
    # 't1000v100',
    # 't4000',
    'lr1e-5',
    ]
col_labels = [
    'Train',
    'Val',
    ]

fontsize = 18
    
ax = plt.subplot(121)
ax.set_title('loss')
for input, in_lable in zip(inputs, in_labels) :
    data = np.genfromtxt(input, delimiter=',')
    for itag in range(len(col_labels)) :
        plt.plot(data[:,0], data[:,itag+1], '-o', label='{}:{}'.format(in_lable, col_labels[itag]))
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Mean Loss", fontsize=fontsize)
    
ax = plt.subplot(122)
ax.set_title('hit rate')
for input, in_lable in zip(inputs, in_labels) :
    data = np.genfromtxt(input, delimiter=',')
    for itag in range(len(col_labels)) :
        plt.plot(data[:,0], data[:,itag+3], '-o', label='{}:{}'.format(in_lable, col_labels[itag]))
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.ylim(0,1)
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Hit rate", fontsize=fontsize)

plt.show()