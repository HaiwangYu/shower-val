import matplotlib.pyplot as plt
import numpy as np

in_files   = [
    't4000/m16-l5-lrd-eval/loss.csv',
    't8000/m16-l5-lrd-res1.0/loss.csv',
    't16000/m16-l5-lrd-res1.0/loss.csv',
    ]
in_labels = [
    '4000',
    '8000',
    '16000',
    ]
in_colors = [
    'r',
    'b',
    'y',
    'g',
    'c',
    'm',
    'k',
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

fontsize = 24
    
ax = plt.subplot(121)
ax.set_title('loss', fontsize=fontsize)
for in_file, in_lable, in_color in zip(in_files, in_labels, in_colors) :
    data = np.genfromtxt(in_file, delimiter=',')
    for icol in [1, 2] :
        plt.plot(data[:,0], data[:,icol], col_symbol[icol], c=in_color, label='{}:{}'.format(in_lable, col_labels[icol]))
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Mean Loss", fontsize=fontsize)
    
ax = plt.subplot(122)
ax.set_title('hit rate @ 2cm', fontsize=fontsize)
for in_file, in_lable, in_color in zip(in_files, in_labels, in_colors) :
    data = np.genfromtxt(in_file, delimiter=',')
    for icol in [9,10] :
        plt.plot(data[:,0], data[:,icol], col_symbol[icol], c=in_color, label='{}:{}'.format(in_lable, col_labels[icol]))
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.ylim(0,1)
plt.xlabel("Epoch", fontsize=fontsize)
# plt.ylabel("Hit rate", fontsize=fontsize)

plt.show()