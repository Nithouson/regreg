import numpy as np
import math
from Network9 import weights_to_graph, regression_error, region_neighbors
from libpysal import weights
from sklearn.metrics import mean_absolute_error
import networkx

# General

def Pos_Encode(r,c,side):
    return r*side+c


def Pos_Decode(code,side):
    return code//side, code%side


def GetNeighbors(pos,side):
    nlist = []
    if pos+side < side*side:
        nlist.append(pos + side)
    if pos-side >= 0:
        nlist.append(pos - side)
    if pos % side != side - 1:
        nlist.append(pos + 1)
    if pos % side != 0:
        nlist.append(pos - 1)
    return nlist


def GetAllNeighbors(side):
    neighbors =[]
    for pos in range(side*side):
        neighbors.append(GetNeighbors(pos,side))
    return neighbors


def GetCoord(side):
    return [(pos // side, pos % side) for pos in range(side * side)]


# Grid Regionlization Evaluation
# Xarr (Side*Side, Variables(including constant)); Yarr (Side*Side)
# Label (Side*Side); Coeff (Side*Side, Variables(including constant))
class Grid_Region_Metrics:
    def __init__(self, Xarr, Yarr, true_label, pred_label, true_coeff, pred_coeff):
        self.X = Xarr
        self.Y = Yarr
        self.N = len(self.X)
        self.label = true_label
        self.rlabel = np.asarray(pred_label, dtype=int)
        units = np.arange(self.N).astype(int)
        self.reg = [units[self.label == r].tolist() for r in set(self.label)]
        self.rreg = [units[self.rlabel == r].tolist() for r in set(self.rlabel)]
        self.coeff = true_coeff
        self.rcoeff = pred_coeff

        self.SSR()
        self.Rand()
        self.Mutual_Info()
        self.Coeff_MAE()
        return

    def SSR(self):
        self.ssr = regression_error(self.rreg, self.X, self.Y)
        return

    def Rand(self):
        tp = fn = fp = tn = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.label[i] == self.label[j]:
                    if self.rlabel[i] == self.rlabel[j]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if self.rlabel[i] == self.rlabel[j]:
                        fp += 1
                    else:
                        tn += 1
        npair = self.N * (self.N - 1) // 2
        assert npair == tp + fn + fp + tn
        self.tp, self.fn, self.fp, self.tn = tp, fn, fp, tn
        self.randi = (tp + tn) / npair
        return

    def Mutual_Info(self):
        n = sum([len(r) for r in self.reg])
        assert n == sum([len(r) for r in self.rreg])
        h = -sum([(len(r) / n) * math.log2(len(r) / n) for r in self.reg])
        hr = -sum([(len(r) / n) * math.log2(len(r) / n) for r in self.rreg])
        reg_set = [set(r) for r in self.reg]
        rreg_set = [set(r) for r in self.rreg]
        mi = 0
        for r in reg_set:
            for rr in rreg_set:
                n_int = len(r.intersection(rr))
                if n_int == 0:
                    continue
                mi += (n_int / n) * math.log2(n * n_int / (len(r) * len(rr)))
        nmi = 2 * mi / (h + hr)
        self.h, self.hr, self.mi, self.nmi = h, hr, mi, nmi
        return

    def Coeff_MAE(self):
        self.coeff_mae=[]
        for v in range(self.X.shape[1]):
            coeffv = [self.coeff[u][v] for u in range(self.N)]
            rcoeffv = [self.rcoeff[self.rlabel[u]][v] for u in range(self.N)]
            self.coeff_mae.append(mean_absolute_error(coeffv,rcoeffv))
        return

    def result_full_str(self):
        str = f'{self.ssr:.4f} '
        str += f'{self.tp:d} {self.fn:d} {self.fp:d} {self.tn:d} {self.randi:.4f} '
        str += f'{self.h:.4f} {self.hr:.4f} {self.mi:.4f} {self.nmi:.4f} '
        str += ' '.join([f'{mae:.4f}' for mae in self.coeff_mae])
        return str

    def result_str(self):
        str = f'{self.ssr:.4f} {self.randi:.4f} {self.nmi:.4f} '
        str += ' '.join([f'{mae:.4f}' for mae in self.coeff_mae])
        return str

# Grid Data Simulation

# data[0] Xarr (Side*Side)*Variables
# data[1] Yarr (Side*Side)
# data[2] Coeff (Side*Side)*Variables


def dist(r1,c1,r2,c2):
    return math.sqrt((r1-r2)*(r1-r2)+(c1-c2)*(c1-c2))


def simulate_zone(side, zonemap, valarr, bias):
    # zonemap: side*side; valarr: d*zones
    Xarr = np.asarray([[1]+[np.random.rand() for d in range(len(valarr)-1)]
                         for u in range(side*side)])
    coeff = np.asarray([[valarr[d][zonemap[u//side][u % side]]
                         for d in range(len(valarr))] for u in range(side*side)])
    Yarr = np.asarray([np.inner(coeff[u], Xarr[u])+bias*np.random.randn()
                         for u in range(side*side)])
    return Xarr, Yarr, coeff


def generate_regular_zones(side,zones):
    width = int(round(side/zones))
    zonemap = [[r//width for c in range(side)] for r in range(side)]
    return zonemap


def generate_voronoi_zones(side,zones,min_size):
    zonemap = [[-1 for c in range(side)] for r in range(side)]
    w = weights.lat2W(side,side)
    g = weights_to_graph(w)
    while True:
        seeds = np.random.choice(range(side*side),zones,replace=False)
        for r in range(side):
            for c in range(side):
                dists = [dist(r, c, seeds[i]//side, seeds[i]%side) for i in range(zones)]
                zonemap[r][c] = dists.index(min(dists))
        valid = True
        for i in range(zones):
            zone = [p for p in range(side*side) if zonemap[p//side][p%side]==i]
            if not networkx.is_connected(g.subgraph(zone)):
                valid = False
            if len(zone) < min_size:
                valid = False
        if valid:
            break
    return zonemap


def generate_random_zones(side, zones, min_size):
    w = weights.lat2W(side, side)
    units = np.arange(w.n).astype(int)
    while True:
        seeds = np.random.choice(units, size=zones, replace=False)
        label = np.array([-1] * w.n).astype(int)
        for i, seed in enumerate(seeds):
            label[seed] = i
        to_assign = units[label == -1]

        while to_assign.size > 0:
            for rid in range(zones):
                region = units[label == rid]
                neighbors = region_neighbors(region, w)
                neighbors = [j for j in neighbors if j in to_assign]
                if len(neighbors) > 0:
                    u = np.random.choice(neighbors)
                    label[u] = rid
            to_assign = units[label == -1]

        regions = [units[label == r].tolist() for r in range(zones)]
        if min([len(region) for region in regions]) >= min_size:
            break

    zonemap = [[label[Pos_Encode(r, c, side)] for c in range(side)] for r in range(side)]
    return zonemap

# Input/Output

def reg_pic(side, label, coeffs=None,dim=-1):
    if dim == -1:
        arr = np.asarray([[-1 for c in range(side)] for r in range(side)])
        # Show shape of zones
        for u in range(len(label)):
            r,c = Pos_Decode(u,side)
            arr[r][c] = label[u]
    else:
        # Show coefficients
        arr = np.asarray([[-1.0 for c in range(side)] for r in range(side)])
        for u in range(len(label)):
            r,c = Pos_Decode(u,side)
            arr[r][c] = coeffs[label[u]][dim]
    return arr


def input_data(i, side, nvar):
    Xarr = np.asarray([[1.0]+[0.0]*nvar for u in range(side*side)])
    Yarr = np.asarray([-1.0 for u in range(side*side)])
    label = np.asarray([-1 for u in range(side*side)])
    coeff = np.asarray([[0.0]*(nvar+1) for u in range(side * side)])
    # Read Xarr
    for v in range(nvar):
        for r in range(side):
            line = i.readline().split()
            for c in range(side):
                Xarr[Pos_Encode(r,c,side)][v+1] = float(line[c])
        line = i.readline()
    # Read Yarr
    for r in range(side):
        line = i.readline().split()
        for c in range(side):
            Yarr[Pos_Encode(r, c, side)] = float(line[c])
    line = i.readline()
    # Read label
    for r in range(side):
        line = i.readline().split()
        for c in range(side):
            label[Pos_Encode(r, c, side)] = int(line[c])
    line = i.readline()
    # Read Coeff
    for v in range(nvar):
        for r in range(side):
            line = i.readline().split()
            for c in range(side):
                coeff[Pos_Encode(r,c,side)][v+1] = float(line[c])
        line = i.readline()
    return Xarr, Yarr, label, coeff


def output_data(o, side, Xarr, Yarr, coeff=None, regmap=None):
    variables = Xarr.shape[1]

    # X
    for v in range(1,variables):
        for r in range(side):
            for c in range(side):
                u = Pos_Encode(r,c,side)
                o.write('%.6f' % Xarr[u][v]+" ")
            o.write("\n")
        o.write("\n")
    # Y
    for r in range(side):
        for c in range(side):
            u = Pos_Encode(r, c, side)
            o.write('%.6f' % Yarr[u]+" ")
        o.write("\n")
    o.write("\n")
    # regmap
    if regmap is not None:
        for r in range(side):
            for c in range(side):
                o.write(str(regmap[r][c]) + " ")
            o.write("\n")
        o.write("\n")
    # coeff
    if coeff is not None:
        for v in range(1, variables):
            for r in range(side):
                for c in range(side):
                    u = Pos_Encode(r, c, side)
                    o.write('%.1f' % coeff[u][v] + " ")
                o.write("\n")
            o.write("\n")
    return


def output_result(o, side, rlabel, coeffs, method):
    o.write(method + "\n")
    regmap = reg_pic(side, rlabel)
    for r in range(side):
        for c in range(side):
            o.write(str(regmap[r][c])+" ")
        o.write("\n")
    o.write("\n")
    for z in range(len(coeffs)):
        for entry in range(len(coeffs[z])):
            o.write('%.4f' % coeffs[z][entry]+" ")
        o.write("\n")
    o.write("\n")
    return


def output_coeff(o, coeffs):
    for z in range(len(coeffs)):
        for entry in range(len(coeffs[z])):
            o.write('%.4f' % coeffs[z][entry]+" ")
        o.write("\n")
    o.write("\n")
    return
