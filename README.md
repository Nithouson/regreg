# Overview

This software aims to delineate geographically connected regions in the context of linear regression models with varying coefficients across regions. 

# Environment & Dependencies

The software is built on Python 3.8 (or higher versions) as well as several Python packages.
- Necessary
  - numpy, scikit-learn, networkx, libpysal, mgwr
  - matplotlib, xlrd, xlwt
- Optional
  - statsmodels (significance testing of linear equations) 
  - geopandas (visualization results on Georgia dataset)

# File Structure

## Core Module

- ``Algorithm5.py`` : implementation of the regionalization algorithms, including AZP, K-Models, Regional-K-Models, and GWR+K-Means.
- ``Network5.py`` : tool functions dealing with the adjacency network, region contiguity, etc.
- ``GridData5.py`` : generating and handling simulated grid data.

## Simulated Data

- ``synthetic/edis_*.txt`` : Dataset 1 (distinct latent regions), 25*25 grid, 50 simulations (number 0-49), each with three noise levels ('h' for high, 'm' for medium, 'l' for low).

- ``synthetic/econ_*.txt`` : Dataset 2 (continuous coefficient surfaces), 25*25 grid, 50 simulations (number 0-49).

We also include 3 simulations on 10\*10 grid for debugging codes (``synthetic/edistest_*.txt``, ``synthetic/econtest_*.txt``), on which regionalization takes only a few seconds. 

## Scripts for Regionalization Experiments

### Simulated Data: Dataset 1

- ``edis_gen.py`` : data generation
- ``edis_reg.py`` : regionalization with four considered algorithms
- ``evaluation/Rand.py`` : calculation of Rand Index (RI)
- ``evaluation/Mutual_Info.py`` : calculation of Normalized Mutual Information (NMI)
- ``visualization/Plot_Regdis.py`` : visualization of produced regions

### Simulated Data: Dataset 2

- ``econ_gen.py`` : data generation
- ``econ_reg.py`` : regionalization with four considered algorithms
- ``econ_stab.py`` : repeated regionalization on one simulation, with four considered algorithms
- ``econ_lambda.py`` : regionalization with different $\lambda$ values, using K-Models algorithm
- ``visualization/Plot_Regcon.py`` : visualization of produced regions

### Real Data: Georgia Dataset

- ``georgia.py`` : regionalization on the Georgia Dataset, with four considered algorithms
- ``visualization/Plot_Georgia.py`` : visualization of produced regions
  

# Function Interfaces

## Algorithm5

 ```Python
 kmodels(Xarr, Yarr, n_regions, w, init_stoc_step = True)
 ```

 K-Models zone design algorithm for linear regression.
 - **Parameters:**
    - **Xarr** (``numpy.ndarray``): 2D array of explanatory variables with size $N*M$. $N$ is the number of spatial units, $M$ is the number of features (If the model includes an intercept, there should be a feature where all units has value 1).
    - **Yarr** (``numpy.ndarray``): 1D array of the response variable with size $N$. 
    - **n_regions** (``int``): initial number of regions $K$.
    - **w** (``pysal.weights.W``): spatial contiguity matrix of the $N$ areal units. The order of units must be consistent with **Xarr** and **Yarr**.
    - **init_stoc_step** (``bool, default=True``): the way of region growing in initialization. If ``False``, all neighboring units of current region are merged into it. Otherwise, a randomly chosen neighboring unit is assigned into it. All our experiment use the default ``True`` value, as it increases the diversity of initial solutions.
 - **Returns:** 
    - **label** (``list``): the region label for each unit, with length $N$. The $u$th unit is assigned to the region with index ``label[u]``. The region indices are integers from 0 to $K-1$.
    - **iters** (``int``): the number of iterations.

  ```Python
 azp(Xarr, Yarr, n_regions, w, init_stoc_step = True)
 ```
AZP zone design algorithm for linear regression.
 - **Parameters**:  
    The same with ***kmodels***.
 - **Returns**: 
    - **regions** (``list``): produced regions. Each element ``regions[r]`` is a list, containing indices of all units belong to it. It can be converted from ***label*** as follows: 
    ```Python
    units = np.arange(w.n).astype(int)
    regions = [units[label == r].tolist() for r in set(label)]
    ```
    - **coeffs** (``list``): estimated coefficients for each region. ``coeff[r]`` is a list of length $M$, containing $M$ coefficients corresponding to the features in ***Xarr***, estimated with units in the $r$th region.
    - **iters** (``int``): the number of iterations.


```Python
region_k_models(Xarr, Yarr, n_regions, w, init_stoc_step = True)
 ```
Regional-K-Models zone design algorithm for linear regression.
- **Parameters**:   
  The same with ***azp***.
- **Returns**:   
  The same with ***azp***.

```Python
gwr_cluster(Xarr, Yarr, coord, n_regions)
```
GWR+K-Means zone design algorithm for linear regression. 
- **Parameters**:   
   - **Xarr**, **Yarr**, **n_regions**: see ***kmodels***.
   - **coord** (```list```): coordinates of the spatial units. Each of the $N$ elements is a coordinate pair (tuple of length 2) indicating the position of a unit.
- **Returns**:   
   - **label**: see ***kmodels***.

```Python
split_merge(Xarr, Yarr, w, clabel, lamda)
```
Post-processing procedure to improve the solution and assure contiguity. Used after K-Models or GWR+K-Means.

- **Parameters**:   
   - **Xarr**, **Yarr**, **w**: see ***kmodels***.
   - **clabel** (```list```): the same as ***label*** in ***kmodels***.
   - **lamda** (```float```): the penalty factor $\lambda$ in the objective function.
- **Returns**:   
   - **regions**, **coeffs**: see ***azp***.

## Network5



# Reproduction & Replication Guide

