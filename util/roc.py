import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def roc_one_sample(coords, ft, prediction, truth, resolution=1.0, verbose=False ):

    if ft[np.argmax(ft[:,-1]), 0] <= 0 :
        if verbose :
            print('no charge for vtx, skip')
        return None

    truth_idx = np.argmax(truth)
    pred_prob = prediction[np.argmax(prediction)]
    pred_filter = prediction >= pred_prob
    
    best  = 9999
    worst = 0
    match = 0
    
    for pred_coords in coords[pred_filter] :
        d = np.linalg.norm(pred_coords - coords[truth_idx])*resolution
        if verbose :
            print(pred_coords, coords[truth_idx], d)
        if d < best :
            best = d
        if d > worst :
            worst = d
    
    if pred_filter[truth_idx] == True :
        match = 1
    
    if verbose :
        print('{} points pass prob {} match:{} best: {} worst: {}'.format(
            np.count_nonzero(pred_filter),
            pred_prob,
            match,
            best,
            worst
            ))

    return [pred_prob, match, best, worst]

def roc(samples, match_criteria = [2.0], nbin = 100) :
    rocs = []
    for match_criterion in match_criteria :
        roc = []
        nsample = samples.shape[0]
        max_bin = np.max(samples[:,0])
        min_bin = np.min(samples[:,0])
        interval = (max_bin- min_bin) / nbin
        for ibin in range(nbin) :
            threashold = min_bin + ibin*interval
            prob_filter = samples[:,0] > threashold
            match_filter = samples[prob_filter, 2] < match_criterion
            eff = np.count_nonzero(prob_filter) / nsample
            pur = np.count_nonzero(match_filter) / np.count_nonzero(prob_filter)
            roc.append([threashold, eff, pur])
        rocs.append(np.array(roc))

    fontsize = 24
    fig = plt.figure(0)
    
    ax = fig.add_subplot(121)
    plt.hist(samples[:,0])
    plt.xlabel('Best event prob', fontsize=fontsize)
    plt.ylabel('', fontsize=fontsize)
    
    ax = fig.add_subplot(122)
    for i, roc in enumerate(rocs) :
        img = ax.scatter(roc[:,1], roc[:,2],
        # c=roc[:,0], cmap='jet',
        alpha=0.5,
        label='{} cm'.format(match_criteria[i]))
    # plt.colorbar(img)
    plt.legend(loc='best', fontsize=fontsize)
    plt.grid()
    # plt.ylim(0,1)
    plt.xlabel('Eff.', fontsize=fontsize)
    plt.ylabel('Pur.', fontsize=fontsize)
    
    plt.show()
    plt.close()

