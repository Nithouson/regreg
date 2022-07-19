from GridData5 import *
import datetime
from matplotlib import pyplot as plt

Side = 25
dataid = 30

recordnum = 49
cmp = "plasma"

vmin, vmax = -2.25, 2.25
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10.0, 14.0)  # inches; 3 row: 10,10 5 row: 10,14
shrink_ratio = 1  # 3 row:0.81 5 row: 0.98
title = ['low noise', 'medium noise', 'high noise']
method = ['KModels', 'AZP', 'RegKModels', 'GWR']
noiselevel = ['l', 'm', 'h']

fig, axes = plt.subplots(5, 3, figsize=(10, 14))

resfile = open("../log/result_"+str(recordnum)+"_"+str(dataid)+".txt")

for noi in range(len(noiselevel)):
    print("Data " + str(dataid) + str(noiselevel[noi]))
    simdata = open("../synthetic/edis_"+str(dataid)+noiselevel[noi]+".txt")
    Xarr, Yarr, coeff = input_data_dis(simdata, Side)

    axes[0, noi].set_xticks([])
    axes[0, noi].set_yticks([])
    axes[0, noi].set_title(title[noi])
    coeff_img = coeff[:, 1].reshape((Side, Side))
    im = axes[0, noi].imshow(coeff_img, cmap=cmp, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axes[0, noi], shrink=shrink_ratio)

    for met in range(4):
        zonemap = [[-1 for c in range(Side)] for r in range(Side)]
        for r in range(Side):
            zoneline = resfile.readline().split(" ")
            while len(zoneline) <= 1:
                zoneline = resfile.readline().split(" ")
            zoneline.remove('\n')
            assert len(zoneline) == Side
            for c in range(Side):
                zonemap[r][c] = int(zoneline[c])
        n_regions = max([max(zonemap[r]) for r in range(Side)]) + 1
        regions = [[] for z in range(n_regions)]
        for r in range(Side):
            for c in range(Side):
                regions[zonemap[r][c]].append(Pos_Encode(r,c,Side))

        rcoeff = [[-9999,-9999] for z in range(n_regions)]
        for z in range(n_regions):
            zoneline = resfile.readline().split(" ")
            while len(zoneline) <= 1:
                zoneline = resfile.readline().split(" ")
            zoneline.remove('\n')
            assert len(zoneline) == 2
            for var in range(2):
                rcoeff[z][var] = float(zoneline[var])

        print(max([rcoeff[z][1] for z in range(n_regions)]),min([rcoeff[z][1] for z in range(n_regions)]))
        axes[met+1, noi].set_title(method[met]+" "+title[noi])
        axes[met+1, noi].set_xticks([])  # 去掉x轴
        axes[met+1, noi].set_yticks([])  # 去掉y轴
        im = axes[met+1, noi].imshow(reg_pic(regions, Side, rcoeff, dim=1), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[met+1, noi], shrink=shrink_ratio)

plt.tight_layout()
fig.savefig('RG'+str(recordnum) + '-EDis' + str(dataid) + '-' +
        str(datetime.datetime.now().strftime('%y%m%d%H%M%S')) + '.png')
