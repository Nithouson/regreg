from GridData5 import *
import datetime
from matplotlib import pyplot as plt

Side = 25
dataid = 12
func = [Const(0), Slope(1/24, 1/24, -1), CosExpMul(22, 22)]

recordnum = 42
cmp = "plasma"

vmin, vmax = -1, 1
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10.0, 14.0)  # inches; 3 row: 10,10 5 row: 10,14
shrink_ratio = 1  # 3 row:0.81 5 row: 0.98
title = ['$β_0$', '$β_1$', '$β_2$']
method = ['KModels', 'AZP', 'RegKModels', 'GWR']

print("Data " + str(dataid))
simdata = open("../synthetic/econ_"+str(dataid)+".txt")
Xarr, Yarr = input_data_con(simdata, Side, len(func))
coeff = GetCoeff(Side, func)
resfile = open("../log/result_"+str(recordnum)+"_"+str(dataid)+".txt")

fig, axes = plt.subplots(5, 3, figsize=(10, 14))

for d in range(coeff.shape[1]):
    axes[0, d].set_xticks([])
    axes[0, d].set_yticks([])
    axes[0, d].set_title(title[d])
    coeff_img = coeff[:, d].reshape((Side, Side))
    im = axes[0, d].imshow(coeff_img, cmap=cmp, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axes[0, d], shrink=shrink_ratio)

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
            regions[zonemap[r][c]].append(Pos_Encode(r, c, Side))

    rcoeff = [[-9999, -9999, -9999] for z in range(n_regions)]
    for z in range(n_regions):
        zoneline = resfile.readline().split(" ")
        while len(zoneline) <= 1:
            zoneline = resfile.readline().split(" ")
        zoneline.remove('\n')
        assert len(zoneline) == coeff.shape[1]
        for var in range(coeff.shape[1]):
            rcoeff[z][var] = float(zoneline[var])
    print(max([rcoeff[z][0] for z in range(n_regions)]), min([rcoeff[z][0] for z in range(n_regions)]))
    print(max([rcoeff[z][1] for z in range(n_regions)]), min([rcoeff[z][1] for z in range(n_regions)]))
    print(max([rcoeff[z][2] for z in range(n_regions)]), min([rcoeff[z][2] for z in range(n_regions)]))
    for d in range(coeff.shape[1]):
        axes[met + 1, d].set_title(method[met]+" "+title[d])
        axes[met + 1, d].set_xticks([])  # 去掉x轴
        axes[met + 1, d].set_yticks([])  # 去掉y轴
        im = axes[met + 1, d].imshow(reg_pic(regions, Side, rcoeff, dim=d), cmap=cmp, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[met + 1, d], shrink=shrink_ratio)

plt.tight_layout()
fig.savefig('RG' + str(recordnum) + '-ECon' + str(dataid) + '-' +
        str(datetime.datetime.now().strftime('%y%m%d%H%M%S')) + '.png')
