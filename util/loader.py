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

def voxelize(x, y, res=1.) :
    if len(x.shape) != 2:
        raise Exception('x should have 2 dims')
    
    # all voxel indices needs to be non-negative
    for i in range(x.shape[1]) :
        x[:,i] = x[:,i] - np.min(x[:,i])
    
    # in unit of resolution
    x = x/res
    
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
            d[key] = y[idx,]
            w[key] = np.ones_like(y[idx,])
    
    keys = []
    vals = []
    for key in d :
        keys.append(list(key))
        vals.append(list(d[key]/w[key]))
    
    return np.array(keys), np.array(vals)

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



def load(meta, vis=False, vox = True, res = 1.) :

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

    coords = np.concatenate((vtx_coords, blob_coords, tvtx_coords), axis=0)
    ft = np.concatenate((vtx_ft, blob_ft, tvtx_ft), axis=0)
    
    if vox :
        vox_coords, vox_ft = voxelize(coords, ft, res)
    else :
        return coords, ft

    # input visualize
    if vis:
        fig = plt.figure()
        
        # 2D
        ax = fig.add_subplot(121)
        charge_filter = ft[:,0] > 0
        img = ax.scatter(coords[charge_filter,2], coords[charge_filter,1], c=ft[charge_filter,0], cmap="jet", alpha=0.1)
        cand_filter = ft[:,1] > 0
        img = ax.scatter(coords[cand_filter,2], coords[cand_filter,1], marker='*', facecolors='none', edgecolors='y')
        truth_fiter = ft[:,3] > 0
        img = ax.scatter(coords[truth_fiter,2], coords[truth_fiter,1], marker='s', facecolors='none', edgecolors='r')
        plt.xlabel('Z [cm]')
        plt.ylabel('Y [cm]')
        plt.grid()
        
        # after voxelize
        ax = fig.add_subplot(122)
        charge_filter = vox_ft[:,0] > 0
        img = ax.scatter(vox_coords[charge_filter,2], vox_coords[charge_filter,1], c=vox_ft[charge_filter,0], cmap="jet", alpha=0.1)
        cand_filter = vox_ft[:,1] > 0
        img = ax.scatter(vox_coords[cand_filter,2], vox_coords[cand_filter,1], marker='*', facecolors='none', edgecolors='y')
        truth_fiter = vox_ft[:,3] > 0
        img = ax.scatter(vox_coords[truth_fiter,2], vox_coords[truth_fiter,1], marker='s', facecolors='none', edgecolors='r')
        plt.xlabel('Z [cm]')
        plt.ylabel('Y [cm]')
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

def type_to_color_func(type):
    if type == 1:
        return 'b'
    elif type == 2:
        return 'g'
    elif type == 3:
        return 'y'
    return 'k'
type_to_color = np.vectorize(type_to_color_func)

def vis_prediction(coords, ft, prediction, truth, ref=None, threshold=0, vis=True):
    # print('{} points pass {} threshold'.format(np.count_nonzero(prediction>threshold), threshold))

    if ft[np.argmax(ft[:,-1]), 0] <= 0 :
        print('no charge for vtx, skip')
        return 'Skip'

    truth_idx = np.argmax(truth)
    pred_idx = np.argmax(prediction)
    pred_idx = prediction >= prediction[np.argmax(prediction)]
    
    match  = 'Miss'
    for pred_coords in coords[pred_idx] :
        d = np.linalg.norm(pred_coords - coords[truth_idx])
        if d <= 2 :
            match = 'Loose'
        print(pred_coords, coords[truth_idx], d)
    if pred_idx[truth_idx] == True :
        match = 'Exact'
    print('{} points pass prob {} match: {}'.format(np.count_nonzero(pred_idx), prediction[np.argmax(prediction)], match))
    
    if not vis :
        return match
        if match != 'Miss' :
            plt.close()
            return match
    
    fig = plt.figure(1)
    
    ax = fig.add_subplot(121)
    img = ax.hist(prediction,100)
    plt.xlabel('prediction')
    
    ax = fig.add_subplot(122)
    # ax.scatter(coords[:,2], coords[:,1], cmap="Greys", alpha=0.05)
    charge_filter = ft[:,0] > 0
    ax.scatter(coords[charge_filter,2], coords[charge_filter,1], c=ft[charge_filter,0], cmap="jet", alpha=0.1)
    
    ax.scatter(coords[truth_idx,2], coords[truth_idx,1], marker='s', facecolors='none', edgecolors='r')
    
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

    img = ax.scatter(
        coords[pred_idx,2],
        coords[pred_idx,1],
        marker='^',
        facecolors='none',
        edgecolors='r')
    
    if ref is not None :
        ref_filter = ref>0
    if ref is not None :
        ax.scatter(
            coords[:,2][ref_filter],
            coords[:,1][ref_filter],
            facecolors='r',
            edgecolors='none',
            # edgecolors=type_to_color(ref[ref_filter]),
            marker='+')

    plt.xlabel('Z [cm]')
    plt.ylabel('Y [cm]')
    
    plt.show()
    plt.close()