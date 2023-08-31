from Network9 import *
from sklearn import preprocessing, metrics
import numpy as np
import geopandas
from mgwr import gwr
from mgwr.sel_bw import Sel_BW
from spopt.region import Skater
import spreg
from spreg.skater_reg import Skater_reg


def kmodels(Xarr, Yarr, K, w,  max_iter=10000, min_size=None, init_stoc_step=True, verbose=False):
    # w: pysal.weights.W object
    if min_size is None:
        min_size = Xarr.shape[1]
    units = np.arange(w.n).astype(int)
    label = init_zones(w, K, min_size, stoc_step=init_stoc_step)

    # iteration
    iters = 0
    regions = [units[label == r].tolist() for r in range(K)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    closest = np.array(closest_equation(Xarr, Yarr, coeffs))
    moves = units[closest != label]
    while len(moves) > 0 and iters < max_iter:
        # make move and update assignments, coeffs, closest, candidates
        for u in moves:
            donor_region = units[label == label[u]].tolist()
            if len(donor_region) <= min_size:
                continue
            label[u] = closest[u]
        regions = [units[label == r].tolist() for r in range(K)]
        coeffs = fit_equations(Xarr, Yarr, regions)
        closest = np.array(closest_equation(Xarr, Yarr, coeffs))
        candidate_moves = units[closest != label]
        moves = []
        for u in candidate_moves:
            donor_region = units[label == label[u]].tolist()
            if len(donor_region) > min_size:
                moves.append(u)
        iters += 1
        if verbose and iters%10 == 0:
            print(iters, len(moves), regression_error(regions,Xarr,Yarr), moves)
    return label, iters


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


def greedy_merge(Xarr, Yarr, n_regions, w, label, min_size=None, verbose=False):
    if min_size is None:
        min_size = Xarr.shape[1]
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

    merges = 0
    # merge segment regions first, to enforce minimum size constraint
    fragments = [p for p in regtree.keys() if len(regtree[p].units) < min_size]
    while len(fragments) > 0:
        p = np.random.choice(fragments)
        min_derr = inf  # min increase of error
        minq = -1
        for q in regtree[p].links:
            derr = delta_err(regtree, p, q, Xarr, Yarr)
            if derr < min_derr:
                min_derr = derr
                minq = q
        merge_node(regtree,p,minq,Xarr,Yarr)
        merges += 1
        fragments = [p for p in regtree.keys() if len(regtree[p].units) < min_size]
        if verbose and len(regtree.keys()) % 10 == 0:
            print(len(regtree.keys()), minq, min_derr)
    if len(regtree.keys()) < n_regions:
        raise RuntimeError("Failed to achieve required number of regions.")

    while len(regtree.keys()) > n_regions:
        min_derr = inf
        minp,minq = -1,-1
        for p in regtree.keys():
            for q in regtree[p].links:
                if q <= p:
                    continue
                derr = delta_err(regtree, p, q, Xarr, Yarr)
                if derr < min_derr:
                    min_derr = derr
                    minp = p
                    minq = q
        merge_node(regtree,minp,minq,Xarr,Yarr)
        merges += 1
        if verbose and len(regtree.keys()) % 10 == 0:
            print(len(regtree.keys()), minp, minq, min_derr)

    regions = [regtree[k].units for k in regtree.keys()]
    rlabel = region_to_label(w.n,regions)
    coeffs = fit_equations(Xarr, Yarr, regions)
    return rlabel, coeffs, merges


def azp(Xarr, Yarr, n_regions, w, max_iter=10000, min_size=None, init_stoc_step=True):
    # w: pysal.weights.W object
    # min_size: lower bound for region size, default: #params
    if min_size is None:
        min_size = Xarr.shape[1]
    k = n_regions
    units = np.arange(w.n).astype(int)
    label = init_zones(w, k, min_size, init_stoc_step)
    regions = [units[label == r].tolist() for r in range(k)]
    xtxinvs = init_xtxinv(Xarr, regions)

    # iteration
    g = weights_to_graph(w)
    iters = 0

    while True:
        stable = True
        region_list = list(range(k))
        while len(region_list) > 0 and iters < max_iter:
            r = np.random.choice(region_list)
            region_list.remove(r)
            moves = region_neighbors(regions[r], w)
            valid_moves = batch_check_AZP(moves, units, label, r, g, w, Xarr, Yarr, xtxinvs, min_size)
            if len(valid_moves) == 0:
                continue
            stable = False
            u = np.random.choice(valid_moves)
            xtxinvs[label[u]] = sherman_morrison(xtxinvs[label[u]], Xarr[u].reshape(-1), add=False)
            xtxinvs[r] = sherman_morrison(xtxinvs[r], Xarr[u].reshape(-1), add=True)
            label[u] = r
            regions = [units[label == r].tolist() for r in range(k)]
            # print(iters,evaluation_func(regions,Xarr,Yarr,0))
        iters += 1
        if stable or iters >= max_iter:
            break
    coeffs = fit_equations(Xarr, Yarr, regions)
    return label, coeffs, iters


def region_k_models(Xarr, Yarr, n_regions, w, max_iter=10000, min_size=None, init_stoc_step=True):
    # w: pysal.weights.W object
    if min_size is None:
        min_size = Xarr.shape[1]
    k = n_regions
    units = np.arange(w.n).astype(int)
    label = init_zones(w, k, min_size, init_stoc_step)

    # iteration
    g = weights_to_graph(w)
    iters = 0
    regions = [units[label == r].tolist() for r in range(k)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    closest = np.array(closest_equation(Xarr, Yarr, coeffs))
    moves = units[closest != label]
    valid_moves = batch_check_RKM(moves, units, label, closest, g, w, min_size)
    while valid_moves and iters < max_iter:
        # make move and update assignments, coeffs, closest, candidates
        u = np.random.choice(valid_moves)
        label[u] = closest[u]
        regions = [units[label == r].tolist() for r in range(k)]
        coeffs = fit_equations(Xarr, Yarr, regions)
        closest = np.array(closest_equation(Xarr, Yarr, coeffs))
        moves = units[closest != label]
        valid_moves = batch_check_RKM(moves, units, label, closest, g, w, min_size)
        iters += 1
    coeffs = fit_equations(Xarr, Yarr, regions)
    return label, coeffs, iters


def gwr_skater(Xarr, Yarr, n_regions, w, coord, min_size=None):
    # Initialization
    nobs = Xarr.shape[0]
    nvar = Xarr.shape[1] - 1

    # gwr
    coeff = [None for u in range(nobs)]
    Xprm = np.asarray([Xarr[u,1:] for u in range(nobs)])
    Xprm = Xprm.reshape((nobs, nvar))
    Yprm = np.asarray([Yarr[u] for u in range(nobs)])
    Yprm = Yprm.reshape((nobs,1))
    bw = Sel_BW(coord, Yprm, Xprm, fixed=False, kernel='bisquare').search(criterion='AICc')
    # for debug
    # print(f"bw:{bw}")

    coord = np.array(coord)
    model = gwr.GWR(coord, Yprm, Xprm, bw=bw, fixed=False, kernel='bisquare')
    results = model.fit()
    for u in range(nobs):
         coeff[u] = results.params[u]
    # for debug
    # print("GWR finished")

    # pre-processing
    X = np.asarray([coeff[u] for u in range(nobs)])
    X1 = preprocessing.StandardScaler().fit_transform(X)

    # SKATER regionalization
    fields = ['intercept']+[f'slope{v}' for v in range(nvar)]
    pd = geopandas.GeoDataFrame(X1, columns=fields, dtype=float)
    # Use default configurations
    spconfig = dict(dissimilarity=metrics.pairwise.manhattan_distances, affinity=None, reduction=np.sum, center=np.mean)
    model = Skater(pd, w, attrs_name=fields, n_clusters=n_regions, floor=min_size, trace=False, spanning_forest_kwds=spconfig)
    model.solve()

    label = model.labels_
    units = np.arange(w.n).astype(int)
    regions = [units[label == r].tolist() for r in range(n_regions)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    return label, coeffs


def skater_reg(Xarr, Yarr, n_regions, w, min_size=None):
    nobs = Xarr.shape[0]
    nvar = Xarr.shape[1] - 1

    Xreg = np.asarray([Xarr[u, 1:] for u in range(nobs)])
    Xreg = Xreg.reshape((nobs, nvar))
    Yreg = np.asarray([Yarr[u] for u in range(nobs)])
    Yreg = Yreg.reshape((nobs, 1))
    results = Skater_reg().fit(n_clusters=n_regions, W=w, data=Xreg,
              data_reg={'reg': spreg.OLS, 'y': Yreg, 'x': Xreg}, quorum=min_size)

    label = results.current_labels_
    units = np.arange(w.n).astype(int)
    regions = [units[label == r].tolist() for r in range(n_regions)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    return label, coeffs

