import matplotlib.pyplot as plt
import numpy as np

inputs   = [
    'test-m16-l4-lr0.001-s100-e50/loss.txt',
    'checkpoints/loss.txt',
    ]
in_labels = [
    'Ref',
    'Test',
    ]
col_labels = [
    'Training',
    'Validation',
    ]
ax = plt.subplot(111)
ax.set_title('loss')

for input, in_lable in zip(inputs, in_labels) :
    data = np.genfromtxt(input, delimiter=',')
    for itag in range(len(col_labels)) :
        plt.plot(data[:,0], data[:,itag+1], '-o', label='{}:{}'.format(in_lable, col_labels[itag]))

fontsize = 18
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
# plt.ylim(0.003,0.010)
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Mean Loss", fontsize=fontsize)
plt.show()