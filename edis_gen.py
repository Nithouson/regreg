from GridData5 import *
from matplotlib import pyplot as plt

Side = 25
runs = 3
bias_y = [0.1, 0.2, 0.3]
noise_label = ["l","m","h"]
zonenum = 5
valarr = [[0,0,0,0,0],[-2,-1,0,1,2]]
cmp = 'bwr'
vmin, vmax = -3, 3

fig, axes = plt.subplots(1, 3, figsize=(16, 8))

for r in range(runs):
    print("Run " + str(r))
    zonemap = Generate_zone(Side, zonenum)

    for noi in range(len(bias_y)):
        data = Simulate_zone(Side, zonemap, valarr, bias_y[noi])
        Xarr, Yarr, coeff = data[0], data[1], data[2]
        ofile = open("edis_" + str(50+r) + noise_label[noi]+".txt", "w")
        #ofile = open("edistest_" + str(r) + noise_label[noi]+".txt", "w")
        output_data(ofile,Side,Xarr,Yarr,coeff)
        ofile.close()
        if r < 3 and noi == 0:
            axes[r].set_xticks([])
            axes[r].set_yticks([])
            coeff_img = coeff[:, 1].reshape((Side, Side))
            im = axes[r].imshow(coeff_img, cmap=cmp, vmin=vmin, vmax=vmax)
    if r == 2:
        plt.subplots_adjust(right=0.8)
        cax = plt.axes([0.84, 0.285, 0.01, 0.425])
        fig.colorbar(im, cax=cax,ticks=[-2,-1,0,1,2])
        plt.show()


