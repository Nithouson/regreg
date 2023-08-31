from GridData9 import *
from Algorithm9 import *
import datetime
from matplotlib import pyplot as plt
import libpysal

# Test Mode
#Side = 10
#idlist = [0,3,6]
#n_regions = 5
#prefix = 'gridtest_'
#mclist = range(n_regions, 15)
#min_region = 5

# Run Mode
Side = 25
idlist = range(150)
n_regions = 5
prefix = 'grid_'
min_region = 10
mclist = range(10, 31)

recordnum = 66
# Zones params
nvar = 2
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # inches; 3 row: 10,10 5 row: 10,14
shrink_ratio = 1  # 3 row:0.81 5 row: 0.98
title = ['low noise', 'medium noise', 'high noise']
noiselevel = ['l', 'm', 'h']

log = open("RG"+str(recordnum)+".txt", 'w')
log.write(str(datetime.datetime.now().ctime())+'\n')
log.write("Hyper\n")
log.write("Side: "+str(Side)+" Data: "+str(idlist)+"\n")
log.write("Regions:"+str(n_regions)+" Micro_clusters:"+str(mclist)+"\n")
log.write("Method: KModels\n")

for dataid in idlist:
    for micro_clusters in mclist:
        ofile = open(f"result_{recordnum}_{dataid}_{micro_clusters}.txt", "w")
        fig, axes = plt.subplots(2, len(noiselevel), figsize=(10, 8))
        for noi in range(len(noiselevel)):
            print(f"Data {dataid}{noiselevel[noi]} {micro_clusters}")
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

        plt.tight_layout()
        fig.savefig('RG'+str(recordnum) + '-' + str(dataid) + '-' + str(micro_clusters) +'-'+
                    str(datetime.datetime.now().strftime('%y%m%d%H%M%S')) + '.png')
        plt.close()
        ofile.close()
        log.write("\n")

log.close()
