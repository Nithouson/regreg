import numpy as np
import libpysal as ps
from mgwr.utils import shift_colormap, truncate_colormap
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.size'] = 36

georgia_shp = gp.read_file('../georgia/Join4.shp')
fig, ax = plt.subplots(figsize=(20,16))

# georgia_shp.plot(ax=ax, **{'edgecolor':'black', 'facecolor':'white'})
# georgia_shp.centroid.plot(ax=ax, c='black')
# plt.show()

cmp = mpl.cm.coolwarm

# Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmp, norm=plt.Normalize(vmin=-1,vmax=1))
inv = list(georgia_shp['Zone_KM'] == 3)

# Plot GWR parameters
georgia_shp.plot('Const_KM', cmap=sm.cmap, ax=ax, vmin=-1, vmax=1, **{'edgecolor': 'black'})
georgia_shp[inv].plot(color='grey', ax=ax, **{'edgecolor': 'black'})
#cax = fig.add_axes([0.7, 0.63, 0.015, 0.2])
#fig.colorbar(sm, cax=cax)
ax.axis('off')
fig.savefig('KM_const.png')
plt.clf()

fig, ax = plt.subplots(figsize=(20,16))
georgia_shp.plot('FB_KM', cmap=sm.cmap, ax=ax, vmin=-1, vmax=1, **{'edgecolor': 'black'})
georgia_shp[inv].plot(color='grey', ax=ax, **{'edgecolor': 'black'})
#cax = fig.add_axes([0.7, 0.63, 0.015, 0.2])
#fig.colorbar(sm, cax=cax)
ax.axis('off')
fig.savefig('KM_fb.png')
plt.clf()

fig, ax = plt.subplots(figsize=(20,16))
georgia_shp.plot('Black_KM', cmap=sm.cmap, ax=ax, vmin=-1, vmax=1, **{'edgecolor': 'black'})
georgia_shp[inv].plot(color='grey', ax=ax, **{'edgecolor': 'black'})
#cax = fig.add_axes([0.8, 0.18, 0.015, 0.64])
#fig.colorbar(sm, cax=cax)
ax.axis('off')
fig.savefig('KM_black.png')
plt.clf()

fig, ax = plt.subplots(figsize=(22,16))
georgia_shp.plot('Rural_KM', cmap=sm.cmap, ax=ax, vmin=-1, vmax=1, **{'edgecolor': 'black'})
georgia_shp[inv].plot(color='grey', ax=ax, **{'edgecolor': 'black'})
cax = fig.add_axes([0.8, 0.18, 0.015, 0.64])
fig.colorbar(sm, cax=cax)
ax.axis('off')
fig.savefig('KM_rural.png')
plt.clf()
