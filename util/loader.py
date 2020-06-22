import uproot
import numpy as np
import matplotlib.pyplot as plt

def vtype_encodeing_func(type) :
    if type == 0:
        return 1
    elif type == 2:
        return 2
    elif type == 6:
        return 3
    return 0
vtype_encodeing = np.vectorize(vtype_encodeing_func)

def load_vtx(meta, vis=False) :

    root_file = uproot.open(meta[0])
    tblob = root_file['T_rec_charge_blob']
    tvtx = root_file['T_vtx']

    x = tblob.array('x')
    y = tblob.array('y')
    z = tblob.array('z')
    q = tblob.array('q')
    blob_coords = np.stack((x,y,z), axis=1)
    blob_ft = np.stack((q,np.zeros_like(q),np.zeros_like(q),np.zeros_like(q)), axis=1)

    vx = tvtx.array('x')
    vy = tvtx.array('y')
    vz = tvtx.array('z')
    vtype = vtype_encodeing(tvtx.array('type'))
    vmain = tvtx.array('flag_main')
    vtx_coords = np.stack((vx,vy,vz), axis=1)
    vtx_ft = np.stack((np.zeros_like(vtype),vtype,vmain,np.zeros_like(vtype)), axis=1)
    # sort by vtype by decreasing order
    vtx_coords = vtx_coords[np.argsort(vtx_ft[:, 1])[::-1]]
    vtx_ft = vtx_ft[np.argsort(vtx_ft[:, 1])[::-1]]

    tvx = np.array([float(meta[2])])
    tvy = np.array([float(meta[3])])
    tvz = np.array([float(meta[4])])
    tvtx_coords = np.stack((tvx,tvy,tvz), axis=1)
    tvtx_ft = np.stack((np.zeros_like(tvx),np.zeros_like(tvx),np.zeros_like(tvx),np.ones_like(tvx)), axis=1)

    coords = np.concatenate((tvtx_coords, vtx_coords, blob_coords), axis=0)
    ft = np.concatenate((tvtx_ft, vtx_ft, blob_ft), axis=0)

    # input visualize
    if vis:
        vis_coords = coords[0:len(vtype)+1,]
        vis_ft = ft[0:len(vtype)+1,]
        fig = plt.figure()
        
        # 3D
        # ax = fig.add_subplot(121, projection='3d')
        # img = ax.scatter(x, y, z, cmap="Greys", alpha=0.05)
        # img = ax.scatter(vis_coords[:,0], vis_coords[:,1], vis_coords[:,2], c=vis_ft[:,1], cmap=plt.jet())
        
        # 2D
        ax = fig.add_subplot(111)
        img = ax.scatter(y, z, cmap="Greys", alpha=0.05)
        img = ax.scatter(vis_coords[:,2], vis_coords[:,1], c=vis_ft[:,1], cmap=plt.jet(), marker='*', alpha=0.5)
        plt.xlabel('Z [cm]')
        plt.ylabel('Y [cm]')
        
        fig.colorbar(img)
        plt.show()

    return [coords, ft]

def vis_prediction(coords, prediction, quantile=0.99):
    prediction = prediction.cpu().detach().numpy()[:,0]
    quantile = (len(prediction)-1.)/len(prediction)
    quantile_val = np.quantile(prediction, quantile)
    print('{} @ {}'.format(quantile_val, quantile))
    vtx_filter = prediction>=quantile_val
    
    fig = plt.figure(1)
    
    ax = fig.add_subplot(121)
    img = ax.hist(prediction,100)
    
    ax = fig.add_subplot(122)
    img = ax.scatter(coords[:,2], coords[:,1], cmap="Greys", alpha=0.05)
    img = ax.scatter(coords[0,2], coords[0,1], marker='s', facecolors='none', edgecolors='r')
    img = ax.scatter(coords[:,2][vtx_filter], coords[:,1][vtx_filter], c=prediction[vtx_filter], cmap=plt.jet(), marker='*', alpha=0.5)
    plt.xlabel('Z [cm]')
    plt.ylabel('Y [cm]')
    fig.colorbar(img)
    
    plt.show()