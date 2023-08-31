from Algorithm9 import *
import copy
import datetime
import xlrd
import xlwt
import libpysal

cmp = "bwr"
pmin = 2
pmax = 10
Kfac = 2
runs = 10
numid = 7
min_region = 5

log = open("Georgia_"+str(numid)+".txt", 'w')
log.write(str(datetime.datetime.now().ctime())+'\n')
log.write("Method: KModels AZP RegKModels GWRSkater SkaterReg \n")
log.write(f"pmin: {pmin} pmax: {pmax} Kfac: {Kfac} repeat: {runs}\n")

data = xlrd.open_workbook("../georgia/GData_utm.xls")
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

w = libpysal.weights.Rook.from_shapefile("../Georgia/Aggre.shp",idVariable="AreaId")
units = np.arange(w.n).astype(int)

outxls = xlwt.Workbook(encoding='utf-8')

for n_regions in range(pmin, pmax+1):
    micro_clusters = Kfac*n_regions
    print(f"{n_regions} Regions")

    params = outxls.add_sheet(f'R{n_regions}')
    params.write(0, 0, label='Area_Key')
    params.write(0, 1, label='Const_GWR')
    params.write(0, 2, label='Rural_GWR')
    params.write(0, 3, label='FB_GWR')
    params.write(0, 4, label='Black_GWR')
    params.write(0, 5, label='Zone_KM')
    params.write(0, 6, label='Const_KM')
    params.write(0, 7, label='Rural_KM')
    params.write(0, 8, label='FB_KM')
    params.write(0, 9, label='Black_KM')
    params.write(0, 10, label='Zone_AZP')
    params.write(0, 11, label='Const_AZP')
    params.write(0, 12, label='Rural_AZP')
    params.write(0, 13, label='FB_AZP')
    params.write(0, 14, label='Black_AZP')
    params.write(0, 15, label='Zone_RKM')
    params.write(0, 16, label='Const_RKM')
    params.write(0, 17, label='Rural_RKM')
    params.write(0, 18, label='FB_RKM')
    params.write(0, 19, label='Black_RKM')
    params.write(0, 20, label='Zone_GSK')
    params.write(0, 21, label='Const_GSK')
    params.write(0, 22, label='Rural_GSK')
    params.write(0, 23, label='FB_GSK')
    params.write(0, 24, label='Black_GSK')
    params.write(0, 25, label='Zone_SKR')
    params.write(0, 26, label='Const_SKR')
    params.write(0, 27, label='Rural_SKR')
    params.write(0, 28, label='FB_SKR')
    params.write(0, 29, label='Black_SKR')

    for u in range(NCounty):
        params.write(u + 1, 0, label=int(IndexArea[u]))

    # GWR_coeff
    Xprm = np.asarray([Xarr[u, 1:] for u in range(NCounty)])
    Yprm = np.asarray([Yarr[u] for u in range(NCounty)])
    Xprm = Xprm.reshape((NCounty, NVar))
    Yprm = Yprm.reshape((NCounty, 1))
    bw = Sel_BW(coord, Yprm, Xprm, fixed=False, kernel='bisquare').search(criterion='AICc')
    coord = np.array(coord)
    model = gwr.GWR(coord, Yprm, Xprm, bw=bw, fixed=False, kernel='bisquare')
    results = model.fit()
    for u in range(NCounty):
        for v in range(NVar + 1):
            params.write(u + 1, 1 + v, label=results.params[u][v])

    # KModels
    bestres = None
    bestssr = 1e10
    for run in range(runs):
        st = datetime.datetime.now()
        clabel, iters = kmodels(Xarr, Yarr, micro_clusters, w)
        slabel = split_components(w, clabel)
        rlabel, rcoeff, merges = greedy_merge(Xarr, Yarr, n_regions, w, slabel, min_size=min_region)
        ed = datetime.datetime.now()
        regions = [units[rlabel == r].tolist() for r in set(rlabel)]
        ssr = regression_error(regions, Xarr, Yarr)
        log.write(f'{ssr} {ed - st} {iters} {merges}\n')
        if ssr < bestssr:
            bestres = (copy.copy(regions), copy.copy(rcoeff))
            bestssr = ssr
    print(f'{bestssr}')
    log.write('\n')
    breg, bcoeff = bestres
    for zoneid in range(len(breg)):
        z = breg[zoneid]
        for u in z:
            params.write(u + 1, 5, label=zoneid)
            for v in range(NVar + 1):
                params.write(u + 1, 6 + v, label=bcoeff[zoneid][v])
    Test_Equations(breg, Xarr, Yarr, log)

    # AZP
    bestres = None
    bestssr = 1e10
    for run in range(runs):
        st = datetime.datetime.now()
        rlabel, rcoeff, iters = azp(Xarr, Yarr, n_regions, w, min_size=min_region)
        ed = datetime.datetime.now()
        regions = [units[rlabel == r].tolist() for r in set(rlabel)]
        ssr = regression_error(regions, Xarr, Yarr)
        log.write(f'{ssr} {ed - st} {iters}\n')
        if ssr < bestssr:
            bestres = (copy.copy(regions), copy.copy(rcoeff))
            bestssr = ssr
    print(f'{bestssr}')
    log.write('\n')
    breg, bcoeff = bestres
    for zoneid in range(len(breg)):
        z = breg[zoneid]
        for u in z:
            params.write(u + 1, 10, label=zoneid)
            for v in range(NVar + 1):
                params.write(u + 1, 11 + v, label=bcoeff[zoneid][v])
    Test_Equations(breg, Xarr, Yarr, log)

    # Region-K-Models
    bestres = None
    bestssr = 1e10
    for run in range(runs):
        st = datetime.datetime.now()
        rlabel, rcoeff, iters = region_k_models(Xarr, Yarr, n_regions, w, min_size=min_region)
        ed = datetime.datetime.now()
        regions = [units[rlabel == r].tolist() for r in set(rlabel)]
        ssr = regression_error(regions, Xarr, Yarr)
        log.write(f'{ssr} {ed - st} {iters}\n')
        if ssr < bestssr:
            bestres = (copy.copy(regions), copy.copy(rcoeff))
            bestssr = ssr

    print(f'{bestssr}')
    log.write('\n')
    breg, bcoeff = bestres
    for zoneid in range(len(breg)):
        z = breg[zoneid]
        for u in z:
            params.write(u + 1, 15, label=zoneid)
            for v in range(NVar + 1):
                params.write(u + 1, 16 + v, label=bcoeff[zoneid][v])
    Test_Equations(breg, Xarr, Yarr, log)

    # GWR_Skater
    st = datetime.datetime.now()
    rlabel, rcoeff = gwr_skater(Xarr, Yarr, n_regions, w, coord, min_size=min_region)
    ed = datetime.datetime.now()
    regions = [units[rlabel == r].tolist() for r in set(rlabel)]
    ssr = regression_error(regions, Xarr, Yarr)
    print(f'{ssr}')
    log.write(f'{ssr} {ed - st}\n\n')
    for zoneid in range(len(regions)):
        z = regions[zoneid]
        for u in z:
            params.write(u + 1, 20, label=zoneid)
            for v in range(NVar + 1):
                params.write(u + 1, 21 + v, label=rcoeff[zoneid][v])
    Test_Equations(regions, Xarr, Yarr, log)

    # SkaterReg
    st = datetime.datetime.now()
    rlabel, rcoeff = skater_reg(Xarr, Yarr, n_regions, w, min_size=min_region)
    ed = datetime.datetime.now()
    regions = [units[rlabel == r].tolist() for r in set(rlabel)]
    ssr = regression_error(regions, Xarr, Yarr)
    print(f'{ssr}')
    log.write(f'{ssr} {ed - st}\n\n')
    for zoneid in range(len(regions)):
        z = regions[zoneid]
        for u in z:
            params.write(u + 1, 25, label=zoneid)
            for v in range(NVar + 1):
                params.write(u + 1, 26 + v, label=rcoeff[zoneid][v])
    Test_Equations(regions, Xarr, Yarr, log)

log.close()
outxls.save(f'Georgia_{numid}.xls')
