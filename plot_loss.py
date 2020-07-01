import matplotlib.pyplot as plt
import numpy as np

inputs   = [
    't500-res2/loss.csv',
    't4000-res2/loss.csv',
    'checkpoints/loss.csv',
    ]
in_labels = [
    '500lrd',
    '4000',
    '1000lrd'
    ]
col_labels = [
    'epoch', # 0
    'tloss', # 1
    'vloss', 
    'thit',  # 3
    'vhit',
    'teff',  # 5
    'veff',
    'tpur',  # 7
    'vpur',
    'tloose',  # 9
    'vloose',
    ]
col_symbol = [
    '',
    '-',
    '--',
    '-',
    '--',
    '-o',
    '--o',
    '-^',
    '--^',
    '-s',
    '--s',
    ]

fontsize = 18
    
ax = plt.subplot(121)
ax.set_title('loss')
for input, in_lable in zip(inputs, in_labels) :
    data = np.genfromtxt(input, delimiter=',')
    for icol in [1, 2] :
        plt.plot(data[:,0], data[:,icol], col_symbol[icol], label='{}:{}'.format(in_lable, col_labels[icol]))
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Mean Loss", fontsize=fontsize)
    
ax = plt.subplot(122)
ax.set_title('hit rate')
for input, in_lable in zip(inputs, in_labels) :
    data = np.genfromtxt(input, delimiter=',')
    for icol in [7] :
        plt.plot(data[:,0], data[:,icol], col_symbol[icol], label='{}:{}'.format(in_lable, col_labels[icol]))
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.ylim(0,1)
plt.xlabel("Epoch", fontsize=fontsize)
# plt.ylabel("Hit rate", fontsize=fontsize)

plt.show()