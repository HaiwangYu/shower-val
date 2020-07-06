import matplotlib.pyplot as plt
import numpy as np

in_files   = [
    't8000/m16-l5-lrd-res1.0/loss.csv',
    # 't4000/m16-l5-lr5-res2.0/loss.csv',
    # 't4000/m16-l5-lrd-res0.3/loss.csv',
    # 't8000/m16-l5-lrd-res1.0/loss.csv',
    't16000/m16-l5-lrd-res1.0/loss.csv',
    't16000/m16-l5-lr6-res1.0-vc4.0/loss.csv',
    ]
in_labels = [
    '8000',
    # 'res2.0',
    # 'res0.3',
    # 't8000-r1',
    '16000',
    '5e-6',
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
    for icol in [7,8,9,10] :
        plt.plot(data[:,0], data[:,icol], col_symbol[icol], c=in_color, label='{}:{}'.format(in_lable, col_labels[icol]))
# plt.legend(loc='best',fontsize=fontsize)
plt.grid()
plt.ylim(0,1)
plt.xlabel("Epoch", fontsize=fontsize)
# plt.ylabel("Hit rate", fontsize=fontsize)

plt.show()