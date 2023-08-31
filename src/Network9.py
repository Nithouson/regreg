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
        err += (np.inner(coeff, Xarr[u, :])-Yarr[u])**2
    return err


def regression_error(regions, Xarr, Yarr):
    ssr = 0
    for r in regions:
        ssr += region_error(Xarr,Yarr,r)
    return ssr


def init_zones(w, n_regions, min_size, stoc_step=True, max_attempt=50):
    units = np.arange(w.n).astype(int)
    trial = 0
    while trial < max_attempt:
        label = init_zones_generation(w, n_regions, stoc_step)
        regions = [units[label == r].tolist() for r in range(n_regions)]
        if min([len(region) for region in regions]) >= min_size:
            break
        trial += 1
    if trial == max_attempt:
        raise RuntimeError("Initial zoning failed. Please check minimum size constraint.")
    return label


def init_zones_generation(w, n_regions, stoc_step):
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
    n_set = set(n_list).difference(set(region))
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


def region_to_label(N, regions):
    label = np.array([-1] * N).astype(int)
    for r in range(len(regions)):
        for u in regions[r]:
            label[u] = r
    return label


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


def accuracy_check(u, from_region, to_region, inv_source, inv_target, Xarr, Yarr):
    ps, pd = from_region, to_region
    ns = copy.copy(from_region)
    ns.remove(u)
    nd = copy.copy(to_region)
    nd.append(u)
    psinv, pdinv = inv_source, inv_target
    v = Xarr[u].reshape((-1))
    nsinv = sherman_morrison(psinv, v, add=False)
    ndinv = sherman_morrison(pdinv, v, add=True)
    err_prev = linreg_err(Xarr, Yarr, ps, psinv) + linreg_err(Xarr, Yarr, pd, pdinv)
    err_new = linreg_err(Xarr, Yarr, ns, nsinv) + linreg_err(Xarr, Yarr, nd, ndinv)
    return err_new <= err_prev


def batch_check_RKM(moves, units, label, closest, g, w, min_size):
    valid_moves = []
    for u in moves:
        from_region = units[label == label[u]]
        if len(from_region) <= min_size:
            continue
        to_region = units[label == closest[u]]
        if contiguity_check(u, from_region, to_region, g, w):
            valid_moves.append(u)
    return valid_moves


def batch_check_AZP(moves, units, label, target, g, w, Xarr, Yarr, xtxinvs, min_size):
    valid_moves = []
    for u in moves:
        from_region = units[label == label[u]].tolist()
        if len(from_region) <= min_size:
            continue
        to_region = units[label == target].tolist()
        # connectivity O(kn); sherman-morrison update O(kn)
        if accuracy_check(u, from_region, to_region, xtxinvs[label[u]],
                             xtxinvs[target], Xarr, Yarr):
            if contiguity_check(u, from_region, to_region, g, w):
                valid_moves.append(u)
    return valid_moves


def linreg_err(Xarr,Yarr,region,xtxinv=None):
    variables = Xarr.shape[1]
    nsize = len(region)
    if nsize == 0:
        return 0
    Xreg = np.asarray([Xarr[u] for u in region])
    Xreg = Xreg.reshape((nsize, variables))
    Xreg_t = np.transpose(Xreg)
    Yreg = np.asarray([Yarr[u] for u in region])
    Yreg = Yreg.reshape(nsize)
    if xtxinv is None:
        xtxinv = np.linalg.inv(np.matmul(Xreg_t, Xreg))
    lin_coeff = np.matmul(xtxinv,np.matmul(Xreg_t, Yreg))
    err = 0
    for u in region:
        err += (np.inner(lin_coeff, Xarr[u, :]) - Yarr[u]) ** 2
    return err


def sherman_morrison(xtxinv, v, add=True):
    m = v.shape[0]
    w = np.matmul(xtxinv, v)
    k = np.matmul(v, w)
    w = w.reshape((m,1))
    if add:
        res = xtxinv - np.matmul(w, np.transpose(w))/(1 + k)
    else:
        res = xtxinv + np.matmul(w, np.transpose(w))/(1 - k)
    return res


def init_xtxinv(Xarr, regions):
    xtxinvs = []
    for region in regions:
        Xreg = np.asarray([Xarr[u] for u in region])
        Xreg_t = np.transpose(Xreg)
        xtxinv = np.linalg.inv(np.matmul(Xreg_t, Xreg))
        xtxinvs.append(xtxinv)
    return xtxinvs



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
        # print(results.summary())
        f_test = results.f_test(np.identity(len(results.params))[1:, :])
        log.write(str(zone_id) + " " + str(zsize) + " " + str(results.params)
                  + " " + str(f_test.fvalue) + " " + str(f_test.pvalue)+'\n')
    log.write('\n')
    return


