from GridData5 import *
from Algorithm5 import *
import copy
import datetime
from matplotlib import pyplot as plt
import libpysal

# Test Mode
# Side = 10
# dataid = 0
# func = [Const(0), Slope(1/9, 1/9, -1), CosExpMul(9, 9)]
# Kmax = 5
# Lamdamax = 5
# prefix = "econtest_"

# Run Mode
Side = 25
dataid = 0
func = [Const(0), Slope(1/24, 1/24, -1), CosExpMul(22, 22)]
Kmax = 20
Lamdamax = 20
prefix = "econ_"

recordnum = 53
cmp = "RdBu_r"

# Zones params
Kmin = 2
Kstep = 1
Lamdamin = 1
Lamdastep = 1
vmin, vmax = -1, 1

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # inches; 3 row: 10,10 5 row: 10,14
shrink_ratio = 0.57  # 2 row: 0.57 3 row:0.81 5 row: 1
title = ['$β_0$', '$β_1$', '$β_2$']

log = open("RG"+str(recordnum)+".txt", 'w')
log.write(str(datetime.datetime.now().ctime())+'\n')
log.write("Lambda\n")
log.write("Side: "+str(Side)+" Data: "+str(dataid)+"\n")
log.write("LamdaMin: "+str(Lamdamin)+" LamdaMax: "+str(Lamdamax)+" LamdaStep: "+str(Lamdastep)+"\n")
log.write("Method:\n KModels " + str(Kmin) + " " + str(Kmax) + " "+str(Kstep)+"\n")

simdata = open("./synthetic/"+prefix + str(dataid)+".txt")
Xarr, Yarr = input_data_con(simdata, Side, len(func))
coeff = GetCoeff(Side, func)
coord = GetCoord(Side)
w = libpysal.weights.lat2W(Side, Side)

for lamda in range(Lamdamin,Lamdamax+1,Lamdastep):
    print("Lamda " + str(lamda))
    ofile = open("result_" + str(recordnum) + "_" + str(lamda) + ".txt", "w")
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))

    for d in range(coeff.shape[1]):
        axes[0, d].set_xticks([])
        axes[0, d].set_yticks([])
        axes[0, d].set_title(title[d])
        coeff_img = coeff[:, d].reshape((Side, Side))
        im = axes[0, d].imshow(coeff_img, cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[0, d], shrink=shrink_ratio)

    # KModels
    st = datetime.datetime.now()
    mincost = 1e10
    mininfo = None
    minreg = None
    for K in range(Kmin, Kmax+Kstep, Kstep):
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
    for d in range(coeff.shape[1]):
        axes[1, d].set_title("KModels "+title[d])
        axes[1, d].set_xticks([])  # 去掉x轴
        axes[1, d].set_yticks([])  # 去掉y轴
        im = axes[1, d].imshow(reg_pic(minreg[0], Side, minreg[1], dim=d), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[1, d], shrink=shrink_ratio)
    output_result(ofile, Side, minreg, "KModels")

    plt.tight_layout()
    fig.savefig('RG'+str(recordnum) + '-ECon' + str(lamda) + '-'
                + str(datetime.datetime.now().strftime('%y%m%d%H%M%S')) + '.png')
    plt.clf()
    ofile.close()
    log.write("\n")

log.close()
