from GridData9 import *
from Algorithm9 import *
import datetime
from matplotlib import pyplot as plt
import libpysal

# Test Mode
# Side = 10
# dataid = 0
# n_regions = 5
# prefix = 'gridtest_'
# micro_clusters = 10
# min_region = 5
# repeat = 3

# Run Mode
Side = 25
dataid = 4
n_regions = 5
prefix = 'grid_'
micro_clusters = 20
min_region = 10
repeat = 50

recordnum = 67

# Zones params
nvar = 2
plt.rcParams['figure.figsize'] = (10.0, 10.0)  # inches; 3 row: 10,10 5 row: 10,14
shrink_ratio = 1  # 3 row:0.81 5 row: 0.98
title = ['low noise', 'medium noise', 'high noise']
noiselevel = ['l', 'm', 'h']

log = open("RG"+str(recordnum)+".txt", 'w')
log.write(str(datetime.datetime.now().ctime())+'\n')
log.write("Stability\n")
log.write(f"Side: {Side} Data: {dataid} Repeat: {repeat}\n")
log.write("Regions:"+str(n_regions)+" Micro_clusters:"+str(micro_clusters)+"\n")
log.write("Method: KModels AZP Reg_KModels\n")

for rep in range(repeat):
    ofile = open(f"result_{recordnum}_{dataid}_R{rep}.txt", "w")
    fig, axes = plt.subplots(4, len(noiselevel), figsize=(10, 10))
    for noi in range(len(noiselevel)):
        print(f"Data {dataid}{noiselevel[noi]} R{rep}")
        simdata = open("../synthetic/" + prefix + str(dataid) + noiselevel[noi]+".txt")
        Xarr, Yarr, label, coeff = input_data(simdata, Side, nvar)
        coord = GetCoord(Side)
        w = libpysal.weights.lat2W(Side, Side)

        axes[0, noi].set_xticks([])
        axes[0, noi].set_yticks([])
        axes[0, noi].set_title(title[noi])
        regmap = label.reshape((Side, Side))
        axes[0, noi].imshow(regmap)

        # KModels
        st = datetime.datetime.now()
        clabel, iters = kmodels(Xarr, Yarr, micro_clusters, w)
        slabel = split_components(w, clabel)
        rlabel, rcoeff, merges = greedy_merge(Xarr, Yarr, n_regions, w, slabel, min_size=min_region)
        ed = datetime.datetime.now()
        metrics = Grid_Region_Metrics(Xarr, Yarr, label, rlabel, coeff, rcoeff)
        print(metrics.result_str() + " " + str(ed - st) + " " + str(iters) + " " + str(merges))
        log.write(metrics.result_full_str() + " " + str(ed - st) + " " + str(iters) + " " + str(merges) + '\n')
        output_result(ofile, Side, rlabel, rcoeff, "KModels")
        axes[1, noi].set_title("KModels "+title[noi])
        axes[1, noi].set_xticks([])  # 去掉x轴
        axes[1, noi].set_yticks([])  # 去掉y轴
        axes[1, noi].imshow(reg_pic(Side, rlabel, dim=-1))

        # AZP
        st = datetime.datetime.now()
        rlabel, rcoeff, iters = azp(Xarr, Yarr, n_regions, w, min_size=min_region)
        ed = datetime.datetime.now()
        metrics = Grid_Region_Metrics(Xarr, Yarr, label, rlabel, coeff, rcoeff)
        print(metrics.result_str() + " " + str(ed - st) + " " + str(iters))
        log.write(metrics.result_full_str() + " " + str(ed - st) + " " + str(iters) + '\n')
        output_result(ofile, Side, rlabel, rcoeff, "AZP")
        axes[2, noi].set_title("AZP " + title[noi])
        axes[2, noi].set_xticks([])  # 去掉x轴
        axes[2, noi].set_yticks([])  # 去掉y轴
        axes[2, noi].imshow(reg_pic(Side, rlabel, dim=-1))

        # Region-K-Models
        st = datetime.datetime.now()
        rlabel, rcoeff, iters = region_k_models(Xarr, Yarr, n_regions, w, min_size=min_region)
        ed = datetime.datetime.now()
        metrics = Grid_Region_Metrics(Xarr, Yarr, label, rlabel, coeff, rcoeff)
        print(metrics.result_str() + " " + str(ed - st) + " " + str(iters))
        log.write(metrics.result_full_str() + " " + str(ed - st) + " " + str(iters) + '\n')
        output_result(ofile, Side, rlabel, rcoeff, "RegionKModels")
        axes[3, noi].set_title("RegKModels "+title[noi])
        axes[3, noi].set_xticks([])  # 去掉x轴
        axes[3, noi].set_yticks([])  # 去掉y轴
        axes[3, noi].imshow(reg_pic(Side, rlabel, dim=-1))

        log.write("\n")

    plt.tight_layout()
    fig.savefig(f'RG{recordnum}-{dataid}-R{rep}-'+
                str(datetime.datetime.now().strftime('%y%m%d%H%M%S')) + '.png')
    plt.clf()
    ofile.close()

log.close()