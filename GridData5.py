import numpy as np
import math


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


def reg_pic(regions,side,coeffs=None,dim=-1):

    if dim == -1:
        arr = np.asarray([[-1 for c in range(side)] for r in range(side)])
        # Show shape of zones
        for z in range(len(regions)):
            for p in regions[z]:
                r,c = Pos_Decode(p,side)
                arr[r][c] = z
    else:
        # Show coefficients
        arr = np.asarray([[-1.0 for c in range(side)] for r in range(side)])
        for z in range(len(regions)):
            for p in regions[z]:
                r,c = Pos_Decode(p,side)
                arr[r][c] = coeffs[z][dim]
    return arr


# Grid Data Simulation

# data[0] Xarr (Side*Side)*Variables
# data[1] Yarr (Side*Side)
# data[2] Coeff (Side*Side)*Variables

def Const(con):
    return lambda r,c: con


def Slope(rslope,cslope,bias):
    return lambda r,c:  r * rslope + c * cslope + bias


def CosAdd(rperiod,cperiod):
    return lambda r, c: np.cos(r * np.pi / rperiod + c * np.pi / cperiod)


def CosMul(rperiod,cperiod):
    return lambda r,c:  np.cos(r*np.pi/rperiod)*np.cos(c*np.pi/cperiod)


def CosExpMul(u,v):
    return lambda r, c:  np.cos(np.exp(r/u)*np.pi) * np.cos(np.exp(c/v)*np.pi)


def Dist(r1,c1,r2,c2):
    return math.sqrt((r1-r2)*(r1-r2)+(c1-c2)*(c1-c2))

def Simulate(side, func, bias):
    Xarr = np.asarray([[1]+[np.random.rand() for d in range(len(func)-1)]
                for u in range(side*side)])
    coeff = np.asarray([[func[d](u//side,u%side) for d in range(len(func))] for u in range(side*side)])
    Yarr = np.asarray([np.inner(coeff[u],Xarr[u])+bias*np.random.randn()
                         for u in range(side*side)])
    return Xarr, Yarr, coeff


def GetCoeff(side, func):
    coeff = np.asarray([[func[d](u // side, u % side) for d in range(len(func))] for u in range(side * side)])
    return coeff

def Simulate_zone(side, zonemap, valarr, bias):
    # zonemap: side*side; valarr: d*z
    Xarr = np.asarray([[1]+[np.random.rand() for d in range(len(valarr)-1)]
                         for u in range(side*side)] )
    coeff = np.asarray([[valarr[d][zonemap[u//side][u%side]]
                         for d in range(len(valarr))] for u in range(side*side)])
    Yarr = np.asarray([np.inner(coeff[u],Xarr[u])+bias*np.random.randn()
                         for u in range(side*side)])
    return Xarr, Yarr, coeff


def Generate_zone(side,zones):
    zonemap = [[-1 for c in range(side)] for r in range(side)]
    seeds = np.random.choice(range(side*side),zones,replace=False)
    for r in range(side):
        for c in range(side):
            dists = [Dist(r,c,seeds[i]//side,seeds[i]%side) for i in range(zones)]
            zonemap[r][c] = dists.index(min(dists))
    return zonemap


def input_data_con(i, side, variables):
    Xarr = np.asarray([[1.0]+[-1.0 for d in range(variables-1)] for u in range(side*side)])
    Yarr = np.asarray([-1.0 for u in range(side*side)])
    for v in range(variables-1):
        for r in range(side):
            line = i.readline().split()
            for c in range(side):
                Xarr[Pos_Encode(r,c,side)][v+1] = float(line[c])
        line = i.readline()
    for r in range(side):
        line = i.readline().split()
        for c in range(side):
            Yarr[Pos_Encode(r, c, side)] = float(line[c])
    return Xarr, Yarr

def input_data_dis(i, side):
    coeff = np.asarray([[0.0, -1.0] for u in range(side*side)])
    Xarr = np.asarray([[1.0, -1.0] for u in range(side*side)])
    Yarr = np.asarray([-1.0 for u in range(side*side)])
    for r in range(side):
        line = i.readline().split()
        for c in range(side):
            coeff[Pos_Encode(r,c,side)][1] = float(line[c])
    line = i.readline()
    for r in range(side):
        line = i.readline().split()
        for c in range(side):
            Xarr[Pos_Encode(r,c,side)][1] = float(line[c])
    line = i.readline()
    for r in range(side):
        line = i.readline().split()
        for c in range(side):
            Yarr[Pos_Encode(r, c, side)] = float(line[c])
    return Xarr, Yarr, coeff

def output_data(o, side, Xarr, Yarr, coeff = None):
    variables = Xarr.shape[1]

    # coeff
    if coeff is not None:
        for v in range(1,variables):
            for r in range(side):
                for c in range(side):
                    u = Pos_Encode(r, c, side)
                    o.write('%.1f' % coeff[u][v] + " ")
                o.write("\n")
            o.write("\n")

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
    return


def output_result(o, side, reg, method):
    o.write(method + "\n")
    regions, coeff = reg
    regmap = reg_pic(regions,side)
    for r in range(side):
        for c in range(side):
            o.write(str(regmap[r][c])+" ")
        o.write("\n")
    o.write("\n")
    for z in range(len(coeff)):
        for entry in range(len(coeff[z])):
            o.write('%.4f' % coeff[z][entry]+" ")
        o.write("\n")
    o.write("\n")
    return
