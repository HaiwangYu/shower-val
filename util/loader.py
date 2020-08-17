import math

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

def closest(a, v) :
    l = a.shape[0]
    d = np.empty(l)
    for i in range(l) :
        d[i] = np.linalg.norm(a[i]-v)
    idx = np.argmin(d)
    return d[idx], idx

def voxelize(x, y, resolution=1., vertex_assign_cut = 0.) :
    if len(x.shape) != 2:
        raise Exception('x should have 2 dims')
    
    # all voxel indices needs to be non-negative
    for i in range(x.shape[1]) :
        x[:,i] = x[:,i] - np.min(x[:,i])
    
    # in unit of resolution
    x = x/resolution
    
    # digitize
    x = x.astype(int)
    
    # filling histogram
    d = dict()
    w = dict()
    for idx in range(x.shape[0]) :
        key = tuple(x[idx,])
        if key in d :
            d[key][0] = d[key][0] + y[idx,][0]
            w[key][0] = w[key][0] + 1
            d[key][1:] = np.maximum(d[key][1:],y[idx,][1:])
            w[key][1:] = np.ones_like(y[idx,][1:])
        else :
            d[key] = np.copy(y[idx,])
            w[key] = np.ones_like(y[idx,])
    
    keys = []
    vals = []
    for key in d :
        keys.append(list(key))
        vals.append(list(d[key]/w[key]))
    
    coords = np.array(keys)
    ft = np.array(vals)

    if vertex_assign_cut <= 0 :
        return coords, ft
    
    # assigne the vertex flag to a closest reco charge
    vindex = np.argmax(ft[:,-1])
    if ft[vindex,0] == 0 :
        vcoords = coords[vindex]
        qcoords = coords[ft[:,0]>0]
        d, i = closest(qcoords, vcoords)
        if d*resolution < vertex_assign_cut :
            ft[i,-1] = 1
    
    return coords, ft

def batch_load(list) :
    coords = []
    fts = []
    for i, meta in enumerate(list) :
        coord, ft = load(meta, vis=False, vox=True)
        batch_id = np.ones((coord.shape[0],1))*i
        coord = np.concatenate((coord, batch_id), axis=1)
        coords.append(coord)
        fts.append(ft)
        # print('coord: ', coord[0:5,])
        # print('ft: ', ft[0:5,])
    
    coords = np.concatenate(coords, axis=0)
    fts = np.concatenate(fts, axis=0)
    # print('coords: ', coords[0:5,])
    # print('fts: ', fts[0:5,])

    return coords, fts

def dist2prob(x, t, s = 1.0) :
    d = []
    for i in range(x.shape[0]) :
        d.append(math.exp(-(np.linalg.norm(x[i]-t)/s)**2/2))
    return np.array(d)

def load(meta, vis=False, vox = True, resolution = 1., vertex_assign_cut = 0., mode='dist', sigma=1.0) :

    # root_file = uproot.open(meta[0])
    try:
        root_file = uproot.open(meta[0])
    except :
        print('Failed to load ', meta[0])
        return None
    tblob = root_file['T_rec_charge_blob']
    tvtx = root_file['T_vtx']

    tvx = np.array([float(meta[2])])
    tvy = np.array([float(meta[3])])
    tvz = np.array([float(meta[4])])
    tvtx_coords = np.stack((tvx,tvy,tvz), axis=1)
    tvtx_ft = np.stack((np.zeros_like(tvx),np.zeros_like(tvx),np.zeros_like(tvx),np.ones_like(tvx)), axis=1)

    x = tblob.array('x')
    y = tblob.array('y')
    z = tblob.array('z')
    q = tblob.array('q')
    blob_coords = np.stack((x,y,z), axis=1)

    blob_truth = np.zeros_like(q)
    if mode == 'dist' :
        blob_truth = dist2prob(blob_coords,tvtx_coords[0],sigma)
    blob_ft = np.stack((q,np.zeros_like(q),np.zeros_like(q),blob_truth), axis=1)

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

    if mode == 'dist' :
        coords = np.concatenate((vtx_coords, blob_coords), axis=0)
        ft = np.concatenate((vtx_ft, blob_ft), axis=0)
    else :
        coords = np.concatenate((vtx_coords, blob_coords, tvtx_coords), axis=0)
        ft = np.concatenate((vtx_ft, blob_ft, tvtx_ft), axis=0)
    # np.savez('/home/yuhw/wc/nue-cc/tmp.npz',coords=coords, ft=ft)
    if vox :
        vox_coords, vox_ft = voxelize(coords, ft, resolution, vertex_assign_cut)
    else :
        return coords, ft

    # input visualize
    if vis:
        fig = plt.figure()
        
        # 2D
        ax = fig.add_subplot(121)
        if mode == 'dist' :
            charge_filter = ft[:,0] > 0
            img = ax.scatter(coords[charge_filter,2], coords[charge_filter,1], c=ft[charge_filter,-1], cmap="jet", alpha=0.5)
            plt.colorbar(img)
            cand_filter = ft[:,1] > 0
            img = ax.scatter(coords[cand_filter,2], coords[cand_filter,1], marker='*', facecolors='none', edgecolors='y')
        else :
            charge_filter = ft[:,0] > 0
            img = ax.scatter(coords[charge_filter,2], coords[charge_filter,1], c=ft[charge_filter,0], cmap="jet", alpha=0.5)
            cand_filter = ft[:,1] > 0
            img = ax.scatter(coords[cand_filter,2], coords[cand_filter,1], marker='*', facecolors='none', edgecolors='y')
            truth_fiter = ft[:,3] > 0
            img = ax.scatter(coords[truth_fiter,2], coords[truth_fiter,1], marker='s', facecolors='none', edgecolors='r')
        plt.xlabel('Z [cm]')
        plt.ylabel('Y [cm]')
        plt.grid()
        
        # after voxelize
        ax = fig.add_subplot(122)
        if mode == 'dist' :
            charge_filter = vox_ft[:,0] > 0
            img = ax.scatter(vox_coords[charge_filter,2], vox_coords[charge_filter,1], c=vox_ft[charge_filter,-1], cmap="jet", alpha=0.5)
            plt.colorbar(img)
            cand_filter = vox_ft[:,1] > 0
            img = ax.scatter(vox_coords[cand_filter,2], vox_coords[cand_filter,1], marker='*', facecolors='none', edgecolors='y')
        else :
            charge_filter = vox_ft[:,0] > 0
            img = ax.scatter(vox_coords[charge_filter,2], vox_coords[charge_filter,1], c=vox_ft[charge_filter,0], cmap="jet", alpha=0.5)
            cand_filter = vox_ft[:,1] > 0
            img = ax.scatter(vox_coords[cand_filter,2], vox_coords[cand_filter,1], marker='*', facecolors='none', edgecolors='y')
            truth_fiter = vox_ft[:,3] > 0
            img = ax.scatter(vox_coords[truth_fiter,2], vox_coords[truth_fiter,1], marker='s', facecolors='none', edgecolors='r')
        plt.xlabel('Z [{}cm]'.format(resolution))
        plt.ylabel('Y [{}cm]'.format(resolution))
        plt.grid()
        
        plt.show()

    return vox_coords, vox_ft

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