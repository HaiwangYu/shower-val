import matplotlib.pyplot as plt
import numpy as np

in_files   = [
    'scratch/res-test/t1000-res2/loss.csv',
    # 'scratch/res-test/t500-res2//loss.csv',
    'checkpoints/loss.csv',
    ]
in_labels = [
    'res2.0',
    'res0.3',
    # 'test',
    ]
in_colors = [
    # 'k',
    'r',
    'b',
    'y',
    'g',
    'c',
    'm',
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
for in_file, in_lable, in_color in zip(in_files, in_labels, in_colors) :
    data = np.genfromtxt(in_file, delimiter=',')
    for icol in [1, 2] :
        plt.plot(data[:,0], data[:,icol], col_symbol[icol], c=in_color, label='{}:{}'.format(in_lable, col_labels[icol]))
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Mean Loss", fontsize=fontsize)
    
ax = plt.subplot(122)
ax.set_title('hit rate')
for in_file, in_lable, in_color in zip(in_files, in_labels, in_colors) :
    data = np.genfromtxt(in_file, delimiter=',')
    for icol in [3,4, 7, 8] :
        plt.plot(data[:,0], data[:,icol], col_symbol[icol], c=in_color, label='{}:{}'.format(in_lable, col_labels[icol]))
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.ylim(0,1)
plt.xlabel("Epoch", fontsize=fontsize)
# plt.ylabel("Hit rate", fontsize=fontsize)

plt.show()