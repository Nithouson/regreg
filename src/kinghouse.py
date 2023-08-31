from Algorithm9 import *
from GridData9 import output_coeff
import networkx
import datetime
import xlrd
import xlwt
import libpysal

cmp = "bwr"
pmin = 5
pmax = 5
Kfac = 2
numid = 1
min_region = 20

log = open("Kinghouse_"+str(numid)+".txt", 'w')
log.write(str(datetime.datetime.now().ctime())+'\n')
log.write("Method: KModels SkaterReg \n")
log.write(f"pmin: {pmin} pmax: {pmax} Kfac: {Kfac}\n")

data = xlrd.open_workbook("../hedonic/kc_house_r5.xls")
table = data.sheets()[0]
nobs = table.nrows - 1
nvar = 16
print(nobs, nvar)
AreaIndex = {}
IndexArea = []
X = []
Y = []
coord = []

for r in range(1, nobs+1):
    Areaid = table.cell_value(r, 0)
    logprice = table.cell_value(r, 2)
    bedroom = table.cell_value(r, 3)
    bathroom = table.cell_value(r, 4)
    sqft_liv = table.cell_value(r, 5)
    sqft_lot = table.cell_value(r, 6)
    floors = table.cell_value(r, 7)
    renovated = table.cell_value(r, 8)
    age = table.cell_value(r, 9)
    age2 = table.cell_value(r, 10)
    sqft_liv15 = table.cell_value(r, 11)
    sqft_lot15 = table.cell_value(r, 12)
    viewd = table.cell_value(r, 13)
    condn = table.cell_value(r, 14)
    avggrade = table.cell_value(r, 15)
    abvavgrd = table.cell_value(r, 16)
    greatgrd = table.cell_value(r, 17)
    distnn = table.cell_value(r, 18)
    coordX = table.cell_value(r, 19)
    coordY = table.cell_value(r, 20)
    AreaIndex[Areaid] = r-1
    IndexArea.append(Areaid)
    X.append([bedroom, bathroom, sqft_liv, sqft_lot, floors, renovated, age, age2,
              sqft_liv15, sqft_lot15, viewd, condn, avggrade, abvavgrd, greatgrd, distnn])
    Y.append([logprice])
    coord.append((coordX,coordY))

Xarr = preprocessing.StandardScaler().fit_transform(np.array(X))
Yarr = preprocessing.StandardScaler().fit_transform(np.array(Y))
Xarr = np.array([[1]+list(datapoint) for datapoint in Xarr])
Xarr = Xarr.reshape((nobs, nvar+1))
Yarr = Yarr.reshape(nobs)

knn = libpysal.weights.KNN.from_shapefile("../hedonic/kc_house_utm_r5.shp", k=18)
w = knn.symmetrize()
print(networkx.is_connected(weights_to_graph(w)))
units = np.arange(w.n).astype(int)

outxls = xlwt.Workbook(encoding='utf-8')

for n_regions in range(pmin, pmax+1):
    micro_clusters = Kfac*n_regions
    print(f"{n_regions} Regions")

    params = outxls.add_sheet(f'R{n_regions}')
    params.write(0, 0, label='Area_Key')
    params.write(0, 1, label='Zone_KM')
    params.write(0, 2, label='Zone_SKR')
    for u in range(nobs):
        params.write(u + 1, 0, label=int(IndexArea[u]))

    # KModels
    st = datetime.datetime.now()
    clabel, iters = kmodels(Xarr, Yarr, micro_clusters, w, init_stoc_step=False, verbose=True)
    print(f"km_finish {datetime.datetime.now()-st}")
    slabel = split_components(w, clabel)
    print(f"sl_finish {datetime.datetime.now()-st}")
    rlabel, rcoeff, merges = greedy_merge(Xarr, Yarr, n_regions, w, slabel, min_size=min_region, verbose=True)
    ed = datetime.datetime.now()
    regions = [units[rlabel == r].tolist() for r in set(rlabel)]
    ssr = regression_error(regions, Xarr, Yarr)
    log.write(f'{ssr} {ed - st} {iters} {merges}\n')
    print(f'{ssr} {ed - st} {iters} {merges}')
    for zoneid in range(len(regions)):
        z = regions[zoneid]
        for u in z:
            params.write(u + 1, 1, label=zoneid)
    output_coeff(log, rcoeff)
    Test_Equations(regions, Xarr, Yarr, log)

    '''
    # GWR_Skater
    st = datetime.datetime.now()
    rlabel, rcoeff = gwr_skater(Xarr, Yarr, n_regions, w, coord, min_size=min_region)
    ed = datetime.datetime.now()
    regions = [units[rlabel == r].tolist() for r in set(rlabel)]
    ssr = regression_error(regions, Xarr, Yarr)
    print(f'{ssr}')
    log.write(f'{ssr} {ed - st}\n')
    for zoneid in range(len(regions)):
        z = regions[zoneid]
        for u in z:
            params.write(u + 1, 20, label=zoneid)
            for v in range(nvar + 1):
                params.write(u + 1, 21 + v, label=rcoeff[zoneid][v])
    output_coeff(log, rcoeff)
    Test_Equations(regions, Xarr, Yarr, log)
    
    DNF in 30 min
    '''

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
            params.write(u + 1, 2, label=zoneid)
    output_coeff(log, rcoeff)
    Test_Equations(regions, Xarr, Yarr, log)

log.close()
outxls.save(f'kc_house_{numid}.xls')