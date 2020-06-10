import uproot
import matplotlib.pyplot as plt
import numpy as np

file = uproot.open('data/nue_6350_26_1313.root')
tblob = file['T_rec_charge_blob']

x = tblob.array('x')
y = tblob.array('y')
z = tblob.array('z')
q = tblob.array('q')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x, y, z, c=q, cmap=plt.jet())
fig.colorbar(img)
plt.show()