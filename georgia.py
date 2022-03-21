from Algorithm5 import *
import copy
import datetime
import xlrd
import xlwt
import libpysal

lamda = 3
cmp = "bwr"
Kmax = [20, 20, 20, 60]
Kmin = 2
Kstep = 1
runs = 10
numid = 4

log = open("Georgia_"+str(numid)+".txt", 'w')
log.write(str(datetime.datetime.now().ctime())+'\n')
log.write("Lamda: "+str(lamda)+'\n')
log.write("Method:\n KModels " + str(Kmin) + " " + str(Kmax[0]) + " "+str(Kstep)+"\n"
          + " AZP " + str(Kmin) + " " + str(Kmax[1])+" "+str(Kstep)+"\n"
          + " Reg_KModels " + str(Kmin)+" "+str(Kmax[2])+" "+str(Kstep)+"\n"
          + " GWR_cluster " + str(Kmin)+" "+str(Kmax[3])+" "+str(Kstep)+"\n")

data = xlrd.open_workbook("./Georgia/GData_utm.xls")
table = data.sheets()[1]
NCounty = table.nrows - 1
NVar = 3
print(NCounty)
AreaIndex = {}
IndexArea = []
X = []
Y = []
coord = []

for r in range(1, NCounty+1):
    Areaid = table.cell_value(r, 0)
    PerBach = table.cell_value(r, 1)
    PerRural = table.cell_value(r, 2)
    PerFB = table.cell_value(r, 3)
    PerBlack = table.cell_value(r, 4)
    coordX = table.cell_value(r, 5)
    coordY = table.cell_value(r, 6)
    AreaIndex[Areaid] = r-1
    IndexArea.append(Areaid)
    X.append([PerRural,PerFB,PerBlack])
    Y.append([PerBach])
    coord.append((coordX,coordY))

Xarr = preprocessing.StandardScaler().fit_transform(np.array(X))
Yarr = preprocessing.StandardScaler().fit_transform(np.array(Y))
Xarr = np.array([[1]+list(datapoint) for datapoint in Xarr])
Xarr = Xarr.reshape((NCounty, NVar+1))
Yarr = Yarr.reshape(NCounty)

w = libpysal.weights.Rook.from_shapefile("./Georgia/Aggre.shp",idVariable="AreaId")

'''
linkdata = xlrd.open_workbook("./Georgia/Links.xls")
links = linkdata.sheets()[0]
NLinks = links.nrows-1
print(NLinks)

linkarea = [set() for i in range(NCounty)]
for r in range(1,NLinks+1):
    AreaA = links.cell_value(r, 0)
    AreaB = links.cell_value(r, 1)
    ida = AreaIndex[AreaA]
    idb = AreaIndex[AreaB]
    linkarea[ida].add(idb)
    linkarea[idb].add(ida)
neighbors = {id: list(linkarea[id]) for id in range(NCounty)}
w = libpysal.weights.W(neighbors)
'''

outxls = xlwt.Workbook(encoding='utf-8')
params = outxls.add_sheet('params')
params.write(0, 0, label='Area_Key')
params.write(0, 1, label='Zone_KM')
params.write(0, 2, label='Const_KM')
params.write(0, 3, label='Rural_KM')
params.write(0, 4, label='FB_KM')
params.write(0, 5, label='Black_KM')
params.write(0, 6, label='Zone_AZP')
params.write(0, 7, label='Const_AZP')
params.write(0, 8, label='Rural_AZP')
params.write(0, 9, label='FB_AZP')
params.write(0, 10, label='Black_AZP')
params.write(0, 11, label='Zone_RKM')
params.write(0, 12, label='Const_RKM')
params.write(0, 13, label='Rural_RKM')
params.write(0, 14, label='FB_RKM')
params.write(0, 15, label='Black_RKM')
params.write(0, 16, label='Const_GWR')
params.write(0, 17, label='Rural_GWR')
params.write(0, 18, label='FB_GWR')
params.write(0, 19, label='Black_GWR')
params.write(0, 20, label='Zone_GWC')
params.write(0, 21, label='Const_GWC')
params.write(0, 22, label='Rural_GWC')
params.write(0, 23, label='FB_GWC')
params.write(0, 24, label='Black_GWC')

for u in range(NCounty):
    params.write(u+1, 0, label=int(IndexArea[u]))

# KModels
bestobj = 1e10
bestreg = None
for run in range(runs):
    st = datetime.datetime.now()
    mincost = 1e10
    mininfo = None
    minreg = None
    for K in range(Kmin, Kmax[0]+Kstep, Kstep):
        clabel, iters = kmodels(Xarr, Yarr, K, w)
        regions, coeffs = split_merge(Xarr, Yarr, w, clabel, lamda)
        cost = evaluation_func(regions, Xarr, Yarr, lamda)
        if cost[0] < mincost:
            mincost = cost[0]
            mininfo = (cost[0], cost[1], cost[2], K, iters)
            minreg = copy.copy((regions, coeffs))
    ed = datetime.datetime.now()
    print(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " " + str(ed - st)
          + " " + str(mininfo[3]) + " " + str(mininfo[4]))
    log.write(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " " + str(ed - st)
              + " " + str(mininfo[3]) + " "+str(mininfo[4]) + '\n')
    if mincost < bestobj:
        bestobj = mincost
        bestreg = copy.copy(minreg)

for zoneid in range(len(bestreg[0])):
    z = bestreg[0][zoneid]
    for u in z:
        params.write(u + 1, 1, label=zoneid)
        for v in range(NVar + 1):
            params.write(u + 1, 2 + v, label=bestreg[1][zoneid][v])
Test_Equations(bestreg[0], Xarr, Yarr, log)
log.write("\n")

# AZP
bestobj = 1e10
bestreg = None
for run in range(runs):
    st = datetime.datetime.now()
    mincost = 1e10
    mininfo = None
    minreg = None
    for K in range(Kmin, Kmax[1] + Kstep, Kstep):
        regions, coeffs, iters = azp(Xarr, Yarr, K, w)
        cost = evaluation_func(regions, Xarr, Yarr, lamda)
        if cost[0] < mincost:
            mincost = cost[0]
            mininfo = (cost[0], cost[1], cost[2], K, iters)
            minreg = copy.copy((regions, coeffs))
    ed = datetime.datetime.now()
    print(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
          + str(ed - st) + " " + str(mininfo[3]) + " " + str(mininfo[4]))
    log.write(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
              + str(ed - st) + " " + str(mininfo[3]) + " " + str(mininfo[4]) + '\n')
    if mincost < bestobj:
        bestobj = mincost
        bestreg = copy.copy(minreg)

for zoneid in range(len(bestreg[0])):
    z = bestreg[0][zoneid]
    for u in z:
        params.write(u + 1, 6, label=zoneid)
        for v in range(NVar + 1):
            params.write(u + 1, 7 + v, label=bestreg[1][zoneid][v])
Test_Equations(bestreg[0], Xarr, Yarr, log)
log.write("\n")

# Region-K-Models
bestobj = 1e10
bestreg = None
for run in range(runs):
    st = datetime.datetime.now()
    mincost = 1e10
    mininfo = None
    minreg = None
    for K in range(Kmin, Kmax[2] + Kstep, Kstep):
        regions, coeffs, iters = region_k_models(Xarr, Yarr, K, w)
        cost = evaluation_func(regions, Xarr, Yarr, lamda)
        if cost[0] < mincost:
            mincost = cost[0]
            mininfo = (cost[0], cost[1], cost[2], K, iters)
            minreg = copy.copy((regions, coeffs))
    ed = datetime.datetime.now()
    print(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
          + str(ed - st) + " " + str(mininfo[3]) + " " + str(mininfo[4]))
    log.write(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
              + str(ed - st) + " " + str(mininfo[3]) + " " + str(mininfo[4]) + '\n')
    if mincost < bestobj:
        bestobj = mincost
        bestreg = copy.copy(minreg)

for zoneid in range(len(bestreg[0])):
    z = bestreg[0][zoneid]
    for u in z:
        params.write(u + 1, 11, label=zoneid)
        for v in range(NVar + 1):
            params.write(u + 1, 12 + v, label=bestreg[1][zoneid][v])

Test_Equations(bestreg[0], Xarr, Yarr, log)
log.write("\n")

# GWR_coeff
bestobj = 1e10
bestreg = None
Xprm = np.asarray([Xarr[u,1:] for u in range(NCounty)])
Yprm = np.asarray([Yarr[u] for u in range(NCounty)])
Xprm = Xprm.reshape((NCounty,NVar))
Yprm = Yprm.reshape((NCounty,1))
bw = Sel_BW(coord, Yprm, Xprm, fixed=False, kernel='bisquare').search(criterion='AICc')
coord = np.array(coord)
model = gwr.GWR(coord, Yprm, Xprm, bw=bw, fixed=False, kernel='bisquare')
results = model.fit()

for u in range(NCounty):
    for v in range(NVar + 1):
        params.write(u + 1, 16 + v, label=results.params[u][v])


# GWR_Cluster
for run in range(runs):
    st = datetime.datetime.now()
    mincost = 1e10
    mininfo = None
    minreg = None
    for K in range(Kmin, Kmax[3] + Kstep, Kstep):
        clabel = gwr_cluster(Xarr, Yarr, coord, K)
        regions, coeffs = split_merge(Xarr, Yarr, w, clabel, lamda)
        cost = evaluation_func(regions, Xarr, Yarr, lamda)
        if cost[0] < mincost:
            mincost = cost[0]
            mininfo = (cost[0], cost[1], cost[2], K)
            minreg = copy.copy((regions, coeffs))
    ed = datetime.datetime.now()
    print(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " " + str(ed - st) + " " + str(mininfo[3]))
    log.write(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " " + str(
        ed - st) + " " + str(mininfo[3]) + '\n')
    if mincost < bestobj:
        bestobj = mincost
        bestreg = copy.copy(minreg)

for zoneid in range(len(bestreg[0])):
    z = bestreg[0][zoneid]
    for u in z:
        params.write(u + 1, 20, label=zoneid)
        for v in range(NVar + 1):
            params.write(u + 1, 21 + v, label=bestreg[1][zoneid][v])
Test_Equations(bestreg[0], Xarr, Yarr, log)
log.write("\n")

log.close()
outxls.save('Georgia_'+str(numid)+'.xls')