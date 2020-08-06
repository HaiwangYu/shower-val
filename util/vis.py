import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def type_to_color_func(type):
    if type == 1:
        return 'b'
    elif type == 2:
        return 'g'
    elif type == 3:
        return 'y'
    return 'k'
type_to_color = np.vectorize(type_to_color_func)

def vis_prediction_seg(coords, ft, prediction, truth, ref1=None, ref2=None, resolution=1.0, loose_cut=2.0, threshold=0, vis=True):
    # print('{} points pass {} threshold'.format(np.count_nonzero(prediction>threshold), threshold))

    if ft[np.argmax(ft[:,-1]), 0] <= 0 :
        print('no charge for vtx, skip')
        return 'Skip'

    # if ref1 is not None :
    #     if np.count_nonzero(truth[ref1>0]) > 0 :
    #         return 'Skip'

    truth_idx = np.argmax(truth)
    pred_idx = np.argmax(prediction)
    pred_idx = prediction >= prediction[np.argmax(prediction)]
    
    match  = 'Miss'
    for pred_coords in coords[pred_idx] :
        d = np.linalg.norm(pred_coords - coords[truth_idx])*resolution
        if d <= loose_cut :
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
    
    # hist
    # ax = fig.add_subplot(121)
    # img = ax.hist(prediction,100)
    # plt.xlabel('prediction')

    # 3D
    ax = fig.add_subplot(121, projection='3d')
    charge_filter = ft[:,0] > 0
    ax.scatter(coords[charge_filter,0], coords[charge_filter,1], coords[charge_filter,2], c=ft[charge_filter,0], cmap="jet", alpha=0.05)
    ax.scatter(coords[pred_idx,0], coords[pred_idx,1], coords[pred_idx,2], c='r', marker='*')
    plt.xlabel('X [cm]')
    plt.ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    
    ax = fig.add_subplot(122)
    # ax.scatter(coords[:,2], coords[:,1], cmap="Greys", alpha=0.05)
    charge_filter = ft[:,0] > 0
    ax.scatter(coords[charge_filter,2], coords[charge_filter,1], c=ft[charge_filter,0], cmap="jet", alpha=0.1)
    
    ax.scatter(coords[truth_idx,2], coords[truth_idx,1], marker='s', facecolors='none', edgecolors='r', label='Truth')
    
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
        coords[pred_idx,2]+0.1,
        coords[pred_idx,1],
        marker='^',
        facecolors='none',
        edgecolors='r',
        label='DNN')
    
    if ref1 is not None :
        ref_filter = ref1>0
        ax.scatter(
            coords[:,2][ref_filter]-0.1,
            coords[:,1][ref_filter],
            facecolors='g',
            edgecolors='none',
            # edgecolors=type_to_color(ref1[ref_filter]),
            marker='+',
            label='Candidate')
        print('ref1: ', np.count_nonzero(truth[ref_filter]))
    
    if ref2 is not None :
        ref_filter = ref2>0
        ax.scatter(
            coords[:,2][ref_filter]-0.1,
            coords[:,1][ref_filter],
            facecolors='none',
            edgecolors='g',
            marker='s',
            label='Traditional')
        print('ref2: ', np.count_nonzero(truth[ref_filter]))

    plt.legend(loc='best', fontsize=24)
    plt.xlabel('Z [cm]')
    plt.ylabel('Y [cm]')
    
    plt.show()
    plt.close()



def vis_prediction_regseg(pred, truth, trad=None, cand=None, x=2, y=1, resolution=0.5, loose_cut=1.0, vis=True):
    fontsize = 24

    fig = plt.figure(0)
    
    ax = fig.add_subplot(111)
    
    img = ax.scatter(pred[:,x], pred[:,y], c=pred[:,-1], cmap='jet', alpha=0.2, label='Prediction')
    plt.colorbar(img)

    pred_idx = np.argmax(pred[:,-1])
    ax.scatter(pred[pred_idx,x], pred[pred_idx,y], marker='^', facecolors='none', edgecolors='r', label='Prediction Max')

    truth_idx = np.argmax(truth[:,-1])
    img = ax.scatter(truth[truth_idx,x], truth[truth_idx,y], marker='s', facecolors='none', edgecolors='r', label='Truth')

    dist_dnn_truth = np.linalg.norm(pred[pred_idx,0:3]-truth[truth_idx,0:3]) * resolution
    
    if cand is not None :
        flt = cand[:,-1]>0
        img = ax.scatter(cand[flt,x], cand[flt,y], marker='+', facecolors='g', edgecolors='none', label='Candidate')

    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel('Z [{}cm]'.format(resolution), fontsize=fontsize)
    plt.ylabel('Y [{}cm]'.format(resolution), fontsize=fontsize)
    
    if vis :
        print('{:.2f} {:.2f} {}'.format(pred[pred_idx,-1], dist_dnn_truth, dist_dnn_truth<loose_cut))
        plt.show()
    
    plt.close()
    return pred[pred_idx,-1], dist_dnn_truth