from Network5 import *
from sklearn import cluster, preprocessing
import random
import numpy as np
from mgwr import gwr
from mgwr.sel_bw import Sel_BW


def greedy_merge(Xarr, Yarr, w, label, lamda):
    units = np.arange(w.n).astype(int)
    reg_label = list(set(label))
    reg_label.sort()
    regions = [units[label == r].tolist() for r in reg_label]
    regtree = dict()

    for rid in range(len(regions)):
        curnode = Node(reg_label[rid],regions[rid])
        curnode.err = region_error(Xarr,Yarr,curnode.units)
        regtree[reg_label[rid]] = curnode
    for cid in regtree.keys():
        regtree[cid].links = set([label[u] for u in region_neighbors(regtree[cid].units, w)])

    iter = 0
    while True:
        maxdrop = 0
        maxp,maxq = -1,-1
        for p in regtree.keys():
            for q in regtree[p].links:
                if q <= p:
                    continue
                drop = lamda - delta_err(regtree,p,q,Xarr,Yarr)
                if drop > maxdrop:
                    maxdrop = drop
                    maxp = p
                    maxq = q
        if maxp == -1:
            break
        else:
            merge_node(regtree,maxp,maxq,Xarr,Yarr)
            iter += 1

    regions = [regtree[k].units for k in regtree.keys()]
    coeffs = fit_equations(Xarr, Yarr, regions)
    return regions, coeffs


def kmodels(Xarr, Yarr, n_regions, w, init_stoc_step = True):
    # w: pysal.weights.W object
    k = n_regions
    units = np.arange(w.n).astype(int)
    label = initial_zones(w, k, init_stoc_step)

    # iteration
    iters = 1
    regions = [units[label == r].tolist() for r in range(k)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    closest = np.array(closest_equation(Xarr, Yarr, coeffs))
    moves = units[closest != label]
    while len(moves) > 0:
        # make move and update assignments, coeffs, closest, candidates
        for u in moves:
            label[u] = closest[u]
        regions = [units[label == r].tolist() for r in range(k)]
        coeffs = fit_equations(Xarr, Yarr, regions)
        closest = np.array(closest_equation(Xarr, Yarr, coeffs))
        moves = units[closest != label]
        iters += 1
    return label, iters


def gwr_cluster(Xarr, Yarr,  coord, n_regions):
    # Initialization
    units = Xarr.shape[0]
    nvar = Xarr.shape[1] - 1
    k = n_regions

    # gwr
    coeff = [None for u in range(units)]
    Xprm = np.asarray([Xarr[u,1:] for u in range(units)])
    Xprm = Xprm.reshape((units, nvar))
    Yprm = np.asarray([Yarr[u] for u in range(units)])
    Yprm = Yprm.reshape((units,1))
    bw = Sel_BW(coord, Yprm, Xprm, fixed=False, kernel='bisquare').search(criterion='AICc')
    coord = np.array(coord)
    model = gwr.GWR(coord, Yprm, Xprm, bw=bw, fixed=False, kernel='bisquare')
    results = model.fit()
    for u in range(units):
         coeff[u] = results.params[u]

    # Data Pre-processing
    X = np.asarray([coeff[u] for u in range(units)])
    X1 = preprocessing.StandardScaler().fit_transform(X)

    classifier = cluster.KMeans(n_clusters=k).fit(X1)
    label = classifier.labels_
    return label


def split_components(w, clabel):
    g = weights_to_graph(w)
    units = np.arange(w.n).astype(int)
    clusters = [units[clabel == r].tolist() for r in set(clabel)]
    rid = 0
    rlabel = np.array([-1] * w.n).astype(int)

    for c in clusters:
        clus = g.subgraph(c)
        regs = networkx.connected_components(clus)
        for r in regs:
            for u in r:
                rlabel[u] = rid
            rid += 1
    return rlabel


def split_merge(Xarr, Yarr, w, clabel, lamda):
    rlabel = split_components(w,clabel)
    regions, coeffs = greedy_merge(Xarr, Yarr, w, rlabel, lamda)
    return regions, coeffs


def azp(Xarr, Yarr, n_regions, w, init_stoc_step = True):
    # w: pysal.weights.W object
    k = n_regions
    units = np.arange(w.n).astype(int)
    label = initial_zones(w, k, init_stoc_step)

    # iteration
    g = weights_to_graph(w)
    iters = 1
    regions = [units[label == r].tolist() for r in range(k)]
    while True:
        no_move = True
        valid_regions = [r for r in range(k) if len(regions[r]) > 0]
        while len(valid_regions) > 0:
            r = random.choice(valid_regions)
            valid_regions.remove(r)
            if len(regions[r]) == 0:
                continue
            moves = region_neighbors(regions[r], w)
            valid_moves = batch_check_AZP(moves, units, label, r, g, w, Xarr, Yarr)
            if len(valid_moves) == 0:
                continue

            no_move = False
            u = random.choice(valid_moves)
            label[u] = r
            regions = [units[label == r].tolist() for r in range(k)]
            iters += 1
            # print(iters,evaluation_func(regions,Xarr,Yarr,0))
        if no_move:
            break
    regions = [units[label == r].tolist() for r in set(label)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    return regions, coeffs, iters


def region_k_models(Xarr, Yarr, n_regions, w, init_stoc_step = True):
    # w: pysal.weights.W object
    k = n_regions
    units = np.arange(w.n).astype(int)
    label = initial_zones(w, k, init_stoc_step)

    # iteration
    g = weights_to_graph(w)
    iters = 1
    regions = [units[label == r].tolist() for r in range(k)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    closest = np.array(closest_equation(Xarr, Yarr, coeffs))
    moves = units[closest != label]
    valid_moves = batch_check_RKM(moves, units, label, closest, g, w)
    while valid_moves:
        # make move and update assignments, coeffs, closest, candidates
        u = np.random.choice(valid_moves)
        label[u] = closest[u]
        regions = [units[label == r].tolist() for r in range(k)]
        coeffs = fit_equations(Xarr, Yarr, regions)
        closest = np.array(closest_equation(Xarr, Yarr, coeffs))
        moves = units[closest != label]
        valid_moves = batch_check_RKM(moves, units, label, closest, g, w)
        iters += 1
    regions = [units[label == r].tolist() for r in set(label)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    return regions, coeffs, iters