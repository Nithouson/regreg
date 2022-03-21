from GridData5 import *
from Algorithm5 import *
import copy
import datetime
from matplotlib import pyplot as plt
import libpysal

# Test Mode
# Side, runs = 10, 5
# ids, idt = 0, 3
# Kmax = [5, 5, 5, 5]

# Run Mode
Side = 25
ids, idt = 0, 20
Kmax = [20, 20, 20, 60]

lamda = 5
recordnum = 46
cmp = "bwr"

# Zones params
Kmin = 2
Kstep = 1
vmin, vmax = -3, 3

plt.rcParams['figure.figsize'] = (10.0, 14.0)  # inches; 3 row: 10,10 5 row: 10,14
shrink_ratio = 1  # 3 row:0.81 5 row: 0.98
title = ['low noise', 'medium noise', 'high noise']
noiselevel = ['l', 'm', 'h']

log = open("RG"+str(recordnum)+".txt", 'w')
log.write(str(datetime.datetime.now().ctime())+'\n')
log.write("Rebuild\n")
log.write("Side: "+str(Side)+" lamda: "+str(lamda)+" Data: "+str(ids)+"-"+str(idt-1)+"\n")
log.write("Method:\n KModels "+str(Kmin)+" "+str(Kmax[0])+" "+str(Kstep)+"\n"
          + " AZP " + str(Kmin)+" "+str(Kmax[1])+" "+str(Kstep)+"\n"
          + " Reg_KModels " + str(Kmin)+" "+str(Kmax[2])+" "+str(Kstep)+"\n"
          + " GWR_cluster " + str(Kmin)+" "+str(Kmax[3])+" "+str(Kstep)+"\n")

#for dataid in range(ids, idt):
for dataid in [1,5,16]:
    ofile = open("result_" + str(recordnum) + "_" + str(dataid) + ".txt", "w")
    kfile = open("kcurve_" + str(recordnum) + "_" + str(dataid) + ".txt", "w")
    fig, axes = plt.subplots(5, 3, figsize=(10, 14))
    for noi in range(len(noiselevel)):
        print("Data " + str(dataid) + str(noiselevel[noi]))
        simdata = open("./synthetic/edis_"+str(dataid)+noiselevel[noi]+".txt")
        Xarr, Yarr, coeff = input_data_dis(simdata, Side)
        coord = GetCoord(Side)
        w = libpysal.weights.lat2W(Side, Side)

        axes[0, noi].set_xticks([])
        axes[0, noi].set_yticks([])
        axes[0, noi].set_title(title[noi])
        coeff_img = coeff[:, 1].reshape((Side, Side))
        im = axes[0, noi].imshow(coeff_img, cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[0, noi], shrink=shrink_ratio)

        # KModels
        st = datetime.datetime.now()
        mincost = 1e10
        mininfo = None
        minreg = None
        for K in range(Kmin, Kmax[0]+Kstep, Kstep):
            clabel, iters = kmodels(Xarr, Yarr, K, w)
            regions, coeffs = split_merge(Xarr, Yarr, w, clabel, lamda)
            cost = evaluation_func(regions, Xarr, Yarr, lamda)
            kfile.write(str(K) + " " + str(cost[0]) + " " + str(cost[1]) + " "
                        + str(cost[2]) + " " + str(iters) + '\n')
            if cost[0] < mincost:
                mincost = cost[0]
                mininfo = (cost[0], cost[1], cost[2], K, iters)
                minreg = copy.copy((regions, coeffs))
        ed = datetime.datetime.now()
        print(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " " +
              str(ed - st)+" " + str(mininfo[3]) + " "+str(mininfo[4]))
        log.write(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " " +
                  str(ed - st) + " " + str(mininfo[3]) + " "+str(mininfo[4]) + '\n')
        output_result(ofile, Side, minreg, "KModels")
        kfile.write('\n')
        axes[1, noi].set_title("KModels "+title[noi])
        axes[1, noi].set_xticks([])  # 去掉x轴
        axes[1, noi].set_yticks([])  # 去掉y轴
        im = axes[1, noi].imshow(reg_pic(minreg[0], Side, minreg[1], dim=1), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[1, noi], shrink=shrink_ratio)

        # AZP
        st = datetime.datetime.now()
        mincost = 1e10
        mininfo = None
        minreg = None
        for K in range(Kmin, Kmax[1] + Kstep, Kstep):
            regions, coeffs, iters = azp(Xarr, Yarr, K, w)
            cost = evaluation_func(regions, Xarr, Yarr, lamda)
            kfile.write(str(K) + " " + str(cost[0]) + " " + str(cost[1]) + " "
                        + str(cost[2]) + " " + str(iters) + '\n')
            if cost[0] < mincost:
                mincost = cost[0]
                mininfo = (cost[0], cost[1], cost[2], K ,iters)
                minreg = copy.copy((regions, coeffs))
        ed = datetime.datetime.now()
        print(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
              + str(ed - st) + " " + str(mininfo[3]) + " " + str(mininfo[4]))
        log.write(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
                  + str(ed - st) + " " + str(mininfo[3]) + " " + str(mininfo[4]) + '\n')
        output_result(ofile, Side, minreg, "AZP")
        kfile.write('\n')
        axes[2, noi].set_title("AZP " + title[noi])
        axes[2, noi].set_xticks([])  # 去掉x轴
        axes[2, noi].set_yticks([])  # 去掉y轴
        im = axes[2, noi].imshow(reg_pic(minreg[0], Side, minreg[1], dim=1), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[2, noi], shrink=shrink_ratio)

        # Region-K-Models
        st = datetime.datetime.now()
        mincost = 1e10
        mininfo = None
        minreg = None
        for K in range(Kmin, Kmax[2] + Kstep, Kstep):
            regions, coeffs, iters = region_k_models(Xarr, Yarr, K, w)
            cost = evaluation_func(regions, Xarr, Yarr, lamda)
            kfile.write(str(K) + " " + str(cost[0]) + " " + str(cost[1]) + " "
                        + str(cost[2]) + " " + str(iters) + '\n')
            if cost[0] < mincost:
                mincost = cost[0]
                mininfo = (cost[0], cost[1], cost[2], K, iters)
                minreg = copy.copy((regions, coeffs))
        ed = datetime.datetime.now()
        print(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
              + str(ed - st) + " " + str(mininfo[3]) + " " + str(mininfo[4]))
        log.write(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
                  + str(ed - st) + " " + str(mininfo[3]) + " " + str(mininfo[4]) + '\n')
        output_result(ofile, Side, minreg, "RegionKModels")
        kfile.write('\n')
        axes[3, noi].set_title("RegKModels "+title[noi])
        axes[3, noi].set_xticks([])  # 去掉x轴
        axes[3, noi].set_yticks([])  # 去掉y轴
        im = axes[3, noi].imshow(reg_pic(minreg[0], Side, minreg[1], dim=1), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[3, noi], shrink=shrink_ratio)

        # GWR_Cluster
        st = datetime.datetime.now()
        mincost = 1e10
        mininfo = None
        minreg = None
        for K in range(Kmin, Kmax[3] + Kstep, Kstep):
            clabel = gwr_cluster(Xarr, Yarr, coord, K)
            regions, coeffs = split_merge(Xarr, Yarr, w, clabel, lamda)
            cost = evaluation_func(regions, Xarr, Yarr, lamda)
            kfile.write(str(K) + " " + str(cost[0]) + " " + str(cost[1]) + " "
                        + str(cost[2]) + '\n')
            if cost[0] < mincost:
                mincost = cost[0]
                mininfo = (cost[0], cost[1], cost[2], K)
                minreg = copy.copy((regions, coeffs))
        ed = datetime.datetime.now()
        print(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
              + str(ed - st) + " " + str(mininfo[3]))
        log.write(str(mininfo[0]) + " " + str(mininfo[1]) + " " + str(mininfo[2]) + " "
                  + str(ed - st) + " " + str(mininfo[3]) + '\n')
        output_result(ofile, Side, minreg, "GWR_cluster")
        kfile.write('\n')
        axes[4, noi].set_title("GWR "+title[noi])
        axes[4, noi].set_xticks([])  # 去掉x轴
        axes[4, noi].set_yticks([])  # 去掉y轴
        im = axes[4, noi].imshow(reg_pic(minreg[0], Side, minreg[1], dim=1), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[4, noi], shrink=shrink_ratio)

        log.write("\n")

    plt.tight_layout()
    fig.savefig('RG'+str(recordnum) + '-EDis' + str(dataid) + '-' +
                str(datetime.datetime.now().strftime('%y%m%d%H%M%S')) + '.png')
    plt.clf()
    ofile.close()
    kfile.close()

log.close()
