import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
import networkx
import copy

inf = 1e10


# General
def region_error(Xarr,Yarr,region):
    variables = Xarr.shape[1] - 1
    reg = linear_model.LinearRegression()
    nsize = len(region)
    if nsize == 0:
        return 0
    Xreg = np.asarray([Xarr[u, 1:] for u in region])
    Xreg = Xreg.reshape((nsize, variables))
    Yreg = np.asarray([Yarr[u] for u in region])
    Yreg = Yreg.reshape(nsize)

    reg.fit(Xreg, Yreg)
    coeff = [reg.intercept_] + list(reg.coef_)

    err = 0
    for u in region:
        err += (np.inner(coeff,Xarr[u,:])-Yarr[u])**2
    return err


def initial_zones(w, n_regions, stoc_step = True):
    # w: libpysal.weights.W
    units = np.arange(w.n).astype(int)
    k = n_regions
    seeds = np.random.choice(units, size=k, replace=False)

    label = np.array([-1] * w.n).astype(int)
    for i, seed in enumerate(seeds):
        label[seed] = i
    to_assign = units[label == -1]

    while to_assign.size > 0:
        for rid in range(k):
            region = units[label == rid]
            neighbors = region_neighbors(region, w)
            neighbors = [j for j in neighbors if j in to_assign]
            if len(neighbors) > 0:
                if stoc_step:
                    u = np.random.choice(neighbors)
                    label[u] = rid
                else:
                    for u in neighbors:
                        label[u] = rid
        to_assign = units[label == -1]

    return label


def evaluation_func(regions, Xarr, Yarr, lamda):
    rgl = len(regions)
    acc = 0
    for r in regions:
        acc += region_error(Xarr,Yarr,r)
    tc = lamda * rgl + acc
    return tc, acc, rgl


def is_neighbor(unit, region, w):
    if unit in region:
        return False
    for member in region:
        if unit in w[member]:
            return True
    return False


def region_neighbors(region, w):
    # Get neighboring units for members of a region.
    n_list = []
    for member in region:
        n_list.extend(w[member])
    n_set = list(set(n_list))
    for u in n_set:
        if u in region:
            n_set.remove(u)
    return n_set


def weights_to_graph(w):
    # transform a PySAL W to a networkx graph
    g = networkx.Graph()
    for ego, alters in w.neighbors.items():
        for alter in alters:
            g.add_edge(ego, alter)
    return g


def fit_equations(Xarr,Yarr,regions):
    # Xarr: (n_samples, n_units, n_variables)
    variables = Xarr.shape[1] - 1
    reg = linear_model.LinearRegression()
    coeffs = []
    for region in regions:
        if len(region) == 0:
            coeffs.append([inf]+[0 for i in range(variables)])
            continue
        nsize = len(region)

        Xreg = np.asarray([Xarr[u, 1:] for u in region])
        Xreg = Xreg.reshape((nsize, variables))
        Yreg = np.asarray([Yarr[u] for u in region])
        Yreg = Yreg.reshape(nsize)

        reg.fit(Xreg, Yreg)
        coeffs.append([reg.intercept_] + list(reg.coef_))
    return np.array(coeffs)


def closest_equation(Xarr, Yarr, coeffs):
    units = Xarr.shape[0]
    closest = []
    for u in range(units):
        reg_err = [np.abs(Yarr[u] - np.inner(coeffs[z], Xarr[u, :])) for z in range(len(coeffs))]
        closest.append(reg_err.index(min(reg_err)))
    return closest


# Explicit Methods
def contiguity_check(unit, from_region, to_region, g, w):
    # first check if area has a neighbor in destination
    if not is_neighbor(unit, to_region, w):
        return False
    # check if moving area would break source connectivity
    new_source = [j for j in from_region if j != unit]
    if len(new_source) == 0 or networkx.is_connected(g.subgraph(new_source)):
        return True
    else:
        return False


def accuracy_check(u, from_region, to_region, Xarr, Yarr):
    ps = from_region
    pd = to_region
    ns = copy.copy(from_region)
    ns.remove(u)
    nd = copy.copy(to_region)
    nd.append(u)
    err_prev = region_error(Xarr,Yarr,ps) + region_error(Xarr,Yarr,pd)
    err_new = region_error(Xarr,Yarr,ns) + region_error(Xarr,Yarr,nd)
    return err_new <= err_prev


def batch_check_RKM(moves, units, label, closest, g, w):
    valid_moves = []
    for u in moves:
        from_region = units[label == label[u]]
        to_region = units[label == closest[u]]
        if contiguity_check(u, from_region, to_region, g, w):
            valid_moves.append(u)
    return valid_moves


def batch_check_AZP(moves, units, label, target, g, w, Xarr, Yarr):
    valid_moves = []
    for u in moves:
        from_region = units[label == label[u]].tolist()
        to_region = units[label == target].tolist()
        # connectivity O(n^2); solve linear equations O(n^3)
        if contiguity_check(u, from_region, to_region, g, w):
            if accuracy_check(u,from_region,to_region, Xarr, Yarr):
                valid_moves.append(u)
    return valid_moves


# Implicit Methods
class Node:
    def __init__ (self, id_: int , units_:list):
        self.id = id_
        self.units = units_  # index in Xarr,Yarr,coord,Neighbors
        self.links = set()  # id in graph
        self.err = 0

    def __str__(self):
        return str(self.id)+" "+str(self.units)+" "+str(self.links)+" "+str(self.err)


def merge_node(g:dict, k1, k2, Xarr, Yarr):
    assert k2 in g[k1].links and k1 in g[k2].links
    g[k1].units += g[k2].units
    g[k1].err = region_error(Xarr,Yarr,g[k1].units)
    g[k1].links = g[k1].links.union(g[k2].links).difference({k1,k2})
    g.pop(k2)
    for k in g[k1].links:
        if k2 in g[k].links:
            g[k].links.remove(k2)
        g[k].links.add(k1)
    return g


def delta_err(g:dict, k1, k2, Xarr, Yarr):
    virnode = Node(-1,g[k1].units + g[k2].units)
    virnode.err = region_error(Xarr, Yarr, virnode.units)
    return virnode.err - g[k1].err - g[k2].err


def Test_Equations(regions,Xarr,Yarr,log):
    zone_id = 0
    for zone in regions:
        zone_id += 1
        variables = Xarr.shape[1] - 1
        zsize = len(zone)
        if zsize <= 4:
            log.write(str(zone_id) + " " + str(zsize) + '\n')
            continue

        X = np.asarray([Xarr[u, 1:] for u in zone])
        X = X.reshape((zsize, variables))
        X = sm.add_constant(X)
        Y = np.asarray([Yarr[u] for u in zone])
        Y = Y.reshape(zsize)

        results = sm.OLS(Y, X).fit()
        print(results.summary())
        f_test = results.f_test(np.identity(len(results.params))[1:, :])
        log.write(str(zone_id) + " " + str(zsize) + " " + str(results.params)
                  + " " + str(f_test.fvalue[0][0]) + " " + str(f_test.pvalue)+'\n')
    return

