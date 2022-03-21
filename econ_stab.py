from GridData5 import *
from Algorithm5 import *
import copy
import datetime
from matplotlib import pyplot as plt
import libpysal

# Test Mode
# Side = 10
# dataid = 0
# repeat = 5
# Kmax = [5, 5, 5, 5]
# func = [Const(0), Slope(1/9, 1/9, -1), CosExpMul(9, 9)]

# Run Mode
Side = 25
dataid = 0
repeat = 50
Kmax = [20, 20, 20, 60]
func = [Const(0), Slope(1/24, 1/24, -1), CosExpMul(22, 22)]

lamda = 3
recordnum = 47
cmp = "bwr"

# Zones params
Kmin = 2
Kstep = 1
vmin, vmax = -1, 1

plt.rcParams['figure.figsize'] = (10.0, 14.0)  # inches; 3 row: 10,10 5 row: 10,14
shrink_ratio = 1  # 3 row:0.81 5 row: 0.98
title = ['$β_0$', '$β_1$', '$β_2$']

log = open("RG"+str(recordnum)+".txt", 'w')
log.write(str(datetime.datetime.now().ctime())+'\n')
log.write("Stability\n")
log.write("Side: "+str(Side)+" lamda: "+str(lamda)+" Data: "+str(dataid)+"\n")
log.write("Method:\n KModels " + str(Kmin) + " " + str(Kmax[0]) + " "+str(Kstep)+"\n"
          + " AZP " + str(Kmin) + " " + str(Kmax[1])+" "+str(Kstep)+"\n"
          + " Reg_KModels " + str(Kmin)+" "+str(Kmax[2])+" "+str(Kstep)+"\n"
          + " GWR_cluster " + str(Kmin)+" "+str(Kmax[3])+" "+str(Kstep)+"\n")

simdata = open("./synthetic/econ_"+str(dataid)+".txt")
Xarr, Yarr = input_data_con(simdata, Side, len(func))
coeff = GetCoeff(Side, func)
coord = GetCoord(Side)
w = libpysal.weights.lat2W(Side, Side)

for run in range(repeat):
    print("Run " + str(run))
    ofile = open("result_" + str(recordnum) + "_" + str(run) + ".txt", "w")
    fig, axes = plt.subplots(5, 3, figsize=(10, 14))

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
    for d in range(coeff.shape[1]):
        axes[1, d].set_title("KModels "+title[d])
        axes[1, d].set_xticks([])  # 去掉x轴
        axes[1, d].set_yticks([])  # 去掉y轴
        im = axes[1, d].imshow(reg_pic(minreg[0], Side, minreg[1], dim=d), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[1, d], shrink=shrink_ratio)
    output_result(ofile, Side, minreg, "KModels")

    # AZP
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
    for d in range(coeff.shape[1]):
        axes[2, d].set_title("AZP " + title[d])
        axes[2, d].set_xticks([])  # 去掉x轴
        axes[2, d].set_yticks([])  # 去掉y轴
        im = axes[2, d].imshow(reg_pic(minreg[0], Side, minreg[1], dim=d), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[2, d], shrink=shrink_ratio)
    output_result(ofile, Side, minreg, "AZP")

    # Region-K-Models
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
    for d in range(coeff.shape[1]):
        axes[3, d].set_title("RegKModels "+title[d])
        axes[3, d].set_xticks([])  # 去掉x轴
        axes[3, d].set_yticks([])  # 去掉y轴
        im = axes[3, d].imshow(reg_pic(minreg[0], Side, minreg[1], dim=d), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[3, d], shrink=shrink_ratio)
    output_result(ofile, Side, minreg, "RegionKModels")

    # GWR_Cluster
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
    for d in range(coeff.shape[1]):
        axes[4, d].set_title("GWR "+title[d])
        axes[4, d].set_xticks([])  # 去掉x轴
        axes[4, d].set_yticks([])  # 去掉y轴
        im = axes[4, d].imshow(reg_pic(minreg[0], Side, minreg[1], dim=d), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[4, d], shrink=shrink_ratio)
    output_result(ofile, Side, minreg, "GWR_cluster")

    plt.tight_layout()
    fig.savefig('RG'+str(recordnum) + '-ECon' + str(run) + '-'
                + str(datetime.datetime.now().strftime('%y%m%d%H%M%S')) + '.png')
    plt.clf()
    ofile.close()
    log.write("\n")

log.close()
