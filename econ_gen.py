from GridData5 import *
from matplotlib import pyplot as plt

Side = 10
runs = 5
bias_y = 0.1

cmp = "bwr"
vmin, vmax = -1, 1
title = ['$β_0$','$β_1$','$β_2$']

for run in range(runs):
    print("Run " + str(run))
    # data = Simulate(Side, [Const(0), Slope(1/24, 1/24, -1) , CosExpMul(22, 22)], bias_y)
    data = Simulate(Side, [Const(0), Slope(1/9, 1/9, -1), CosExpMul(9, 9)], bias_y)
    Xarr, Yarr, coeff = data[0], data[1], data[2]
    #ofile = open("econ_"+str(run)+".txt","w")
    ofile = open("econtest_" + str(run) + ".txt", "w")
    output_data(ofile,Side,Xarr,Yarr)

    if run == runs-1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        for d in range(coeff.shape[1]):
            axes[d].set_xticks([])
            axes[d].set_yticks([])
            axes[d].set_title(title[d])
            coeff_img = coeff[:, d].reshape((Side, Side))
            im = axes[d].imshow(coeff_img, cmap=cmp, vmin=vmin, vmax=vmax)

        plt.subplots_adjust(right=0.8)
        cax = plt.axes([0.84, 0.285, 0.01, 0.425])
        fig.colorbar(im, cax=cax)
        plt.show()

    ofile.close()
