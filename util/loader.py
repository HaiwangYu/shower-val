import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def vtype_encodeing_func(type) :
    if type == 0:
        return 1
    elif type == 2:
        return 2
    elif type == 6:
        return 3
    return 0
vtype_encodeing = np.vectorize(vtype_encodeing_func)

def voxelize(coords) :
    if len(coords.shape) != 2:
        raise Exception('coords should have 2 dims')
    
    # all voxel indices needs to be non-negative
    for i in range(coords.shape[1]) :
        coords[:,i] = coords[:,i] - np.min(coords[:,i])
    
    return coords

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
        ax = fig.add_subplot(121, projection='3d')
        img = ax.scatter(x, y, z, cmap="Greys", alpha=0.05)
        img = ax.scatter(vis_coords[:,0], vis_coords[:,1], vis_coords[:,2], c=vis_ft[:,1], cmap=plt.jet())
        
        # 2D
        ax = fig.add_subplot(122)
        img = ax.scatter(z, y, cmap="Greys", alpha=0.05)
        img = ax.scatter(vis_coords[:,2], vis_coords[:,1], c=vis_ft[:,1], cmap=plt.jet(), marker='*', alpha=0.5)
        plt.xlabel('Z [cm]')
        plt.ylabel('Y [cm]')
        
        fig.colorbar(img)
        plt.show()

    # print(coords.shape)
    # print(ft.shape)
    return [voxelize(coords), ft]

def gen_sample() :

    min = -100
    max = -1
    npoints = 389
    x = np.linspace(min, max, npoints)
    y = np.linspace(min, max, npoints)
    z = np.linspace(min, max, npoints)
    q = np.ones_like(x)
    coords = np.stack((x,y,z), axis=1)
    ft = np.stack((np.ones_like(q),np.ones_like(q),np.ones_like(q),np.ones_like(q)), axis=1)
    
    # bias = np.min(coords)
    # if bias < 0 :
    #     coords = coords - bias
    
    print(coords)
    print(ft.shape)
    return [coords, ft]

def type_to_color_func(type):
    if type == 1:
        return 'b'
    elif type == 2:
        return 'g'
    elif type == 3:
        return 'y'
    return 'k'
type_to_color = np.vectorize(type_to_color_func)

def vis_prediction(coords, prediction, ref=None, threshold=0.99):
    print('{} points pass {} threshold'.format(np.count_nonzero(prediction>threshold), threshold))
    print(prediction[np.argmax(prediction)])

    if ref is not None :
        ref_filter = ref>0
    
    fig = plt.figure(1)
    
    ax = fig.add_subplot(121)
    img = ax.hist(prediction,100)
    plt.xlabel('prediction')
    
    ax = fig.add_subplot(122)
    ax.scatter(coords[:,2], coords[:,1], cmap="Greys", alpha=0.05)
    ax.scatter(coords[0,2], coords[0,1], marker='s', facecolors='none', edgecolors='r')
    
    # draw coords pass threshold
    # vtx_filter = prediction>threshold
    # img = ax.scatter(
    #     coords[:,2][vtx_filter],
    #     coords[:,1][vtx_filter],
    #     c=prediction[vtx_filter],
    #     # vmin=0.99, vmax=1.0,
    #     cmap=plt.jet(),
    #     marker='*',
    #     alpha=0.5)
    # fig.colorbar(img)
    
    # draw the best one
    idx = np.argmax(prediction)
    img = ax.scatter(
        coords[:,2][idx],
        coords[:,1][idx],
        marker='*',
        facecolors='y',
        edgecolors='y')
    
    if ref is not None :
        ax.scatter(
            coords[:,2][ref_filter],
            coords[:,1][ref_filter],
            facecolors='none',
            edgecolors=type_to_color(ref[ref_filter]),
            marker='o')

    plt.xlabel('Z [cm]')
    plt.ylabel('Y [cm]')
    
    plt.show()