import matplotlib.pyplot as plt
import numpy as np

inputs   = 'checkpoints/loss.txt'
labels = [
    'Training',
    'Validation',
    ]
ax = plt.subplot(111)
ax.set_title(inputs)
data = np.genfromtxt(inputs, delimiter=',')
for itag in range(len(labels)) :
    plt.plot(data[:,0], data[:,itag+1], '-o', label=labels[itag])

fontsize = 18
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
# plt.ylim(0.003,0.010)
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Mean Loss", fontsize=fontsize)
plt.show()