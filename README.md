# Overview

This software aims to delineate spatial regimes (geographically connected regions with varying coefficients across regions) in the context of linear regression models. Please refer to our paper for more details:    
Hao Guo, Andre Python & Yu Liu (2023) Extending regionalization algorithms to explore spatial process heterogeneity, International Journal of Geographical Information Science, 37:11, 2319-2344, DOI: 10.1080/13658816.2023.2266493

# Supplementary Notes on the Paper

- p.2336 "For multiple sales at the same location, we only retain the latest record." This involves ambiguity as several pairs of sales have the same location and date. One possible solution is to retain the sale with higher price. We have reproduced the King County experiment in this setting, verifying that this issue does not affect the main finding (lower SSR for K-Models than Skater-reg).   

# Environment & Dependencies

The software is built on Python 3.9 (or higher versions) as well as several Python packages.

- Necessary
  - numpy, matplotlib, networkx
  - scikit-learn
  - geopandas
  - libpysal, mgwr, spopt, spreg
  - xlrd, xlwt

- Optional
  - statsmodels (significance testing of linear equations)  

# File Structure

## Core Modules

- ``src/Algorithm9.py`` : Implementation of the regime delineation algorithms, including the two-stage K-Models (KM),  AZP, Regional-K-Models (RKM). We also incorporate the interfaces of GWR-Skater (GSK, from mgwr/spopt) and Skater regression (SKR, from spreg) for comparison.  
- ``src/Network9.py`` : tool functions dealing with the adjacency network, region contiguity, etc.
- ``src/GridData9.py`` : generating and handling simulated grid data.

## Simulated Data

- ``synthetic/grid_*.txt`` : three synthetic datasets on 25\*25 grid, each with 50 simulations (```dataid```  0-49: *Rectangular*; 50-99: *Voronoi*; 100-149: *Arbitrary*). Each simulation contains three samples under different noise levels ('h' for high, 'm' for medium, 'l' for low).

We also include several simulations on 10\*10 grid for debugging codes (``synthetic/gridtest_*.txt``).  

## Scripts for Regime Delineation Experiments

### Synthetic Data

- ``src/grid_gen.py`` : data generation
- ``src/grid_reg.py`` : spatial regime delineation with five considered algorithms
- ``src/grid_K.py`` : experiment on the choice of parameter K (number of micro-clusters) in two-stage K-Models  
- ``src/grid_stab.py`` : experiment on the algorithm stability of two-stage K-Models, AZP and Regional-K-Models

### Real Data

- ``src/georgia.py`` : spatial regime delineation on the Georgia Dataset, using five considered algorithms
- ``src/kinghouse.py``: spatial regime delineation on the King County house price Dataset, using two-stage K-Models and Skater-reg

# Function Interfaces

## Algorithm9

```Python
kmodels(Xarr, Yarr, K, w, max_iter=10000, min_size=None, init_stoc_step=True, verbose=False)
```

The first stage (partition stage) of the two-stage K-Models algorithm for spatial regime delineation.

- **Parameters:**
  - **Xarr** (``numpy.ndarray``): 2D array of explanatory variables with size $n\times m$. $n$ is the number of spatial units, $m$ is the number of features (If the model includes an intercept, there should be a feature where all units has value 1).
  - **Yarr** (``numpy.ndarray``): 1D array of the response variable with size $n$.  
  - **K** (``int``): number of micro-clusters in the partition stage.
  - **w** (``pysal.weights.W``): spatial contiguity matrix of the $n$ areal units. The order of units must be consistent with ```Xarr``` and ```Yarr```.
  - **max_iter** (``int``): maximum number of iterations.  
  - **min_size** (``int``): minimum number of spatial units in each micro-cluster. If ``None``, the default value of $m$ is used.
  - **init_stoc_step** (``bool``): the way of region growing in initialization. If ``False``, all neighboring units of current region are merged into it. Otherwise, a randomly chosen neighboring unit is assigned into it in each step. This increases the diversity of initial solutions, yet may be time-consuming for large datasets.
  - **verbose** (``bool``): If ``True``, there will be additional output in the algorithm process.  
- **Returns:**  
  - **label** (``list``): the region label for each unit, with length $n$. The $u$ th unit is assigned to the region with index ``label[u]``. The region indices are integers from 0 to $K-1$.
  - **iters** (``int``): the number of iterations.

The second stage (merge stage) of the two-stage K-Models algorithm is implemented in two functions. ``split_components`` breaks disconnected regions into branches, and ``greedy_merge`` merges the branches to achieve the required number of regions. They should be called sequentially.

```Python
split_components(w, clabel)
```

- **Parameters**:
  - **w**: see ```kmodels```.
  - **clabel** (```list```): ```label``` from ```kmodels```.
- **Returns**:
  - **rlabel** (```list```): the region label after split components, with length $n$.

```Python
greedy_merge(Xarr, Yarr, n_regions, w, label, min_size=None, verbose=False)
```

- **Parameters**:
  - **Xarr**, **Yarr**, **w**, **verbose**: see ```kmodels```.
  - **n_regions** (```int```): the number of output regions.
  - **label** (```list```): ```rlabel``` from ```split_components```.
  - **min_size** (```int```): minimum number of units in each output region. If ``None``, the default value of $m$ is used. It may be different from ``min_size`` in ```kmodels```.

- **Returns**:
  - **rlabel** (``list``): the final region label for each unit, with length $n$. The region indices are integers from 0 to ```n_regions-1```.
  - **coeffs** (``list``): estimated coefficients for each region. ``coeff[r]`` is a list of length $m$, containing $m$ coefficients corresponding to the features in ```Xarr```.  
  - **merges** (``int``): the number of merge operations.

```Python
azp(Xarr, Yarr, n_regions, w, max_iter=10000, min_size=None, init_stoc_step=True)
```

AZP algorithm for spatial regime delineation.

- **Parameters**:  
  - **Xarr**, **Yarr**, **w**, **max_iter**, **init_stoc_step**: see ```kmodels```.
  - **n_regions**, **min_size**: see ```greedy_merge```.
- **Returns**:  
  - **label**: same as ```rlabel``` in ```greedy_merge```.  
  - **coeffs**: see ```greedy_merge```.  
  - **iters** (``int``): the number of iterations.

```Python
region_k_models(Xarr, Yarr, n_regions, w, max_iter=10000, min_size=None, init_stoc_step=True):
```

Regional-K-Models algorithm for spatial regime delineation.

- **Parameters**:  
  The same with ```azp```.
- **Returns**:  
  The same with ```azp```.

```Python
gwr_skater(Xarr, Yarr, n_regions, w, coord, min_size=None)
```

GWR-Skater for spatial regime delineation.

- **Parameters**:  
  - **Xarr**, **Yarr**, **n_regions**, **w**, **min_size**: see ```azp```.
  - **coord** (```list```): coordinates of the spatial units. Each of the $n$ elements is a coordinate pair (tuple of length 2) indicating the position of a unit.
- **Returns**:
  - **label**, **coeffs**: see ```azp```.

```Python
skater_reg(Xarr, Yarr, n_regions, w, min_size=None, verbose=0)
```

Skater regression for spatial regime delineation. 

- **Parameters**:  
  - **Xarr**, **Yarr**, **n_regions**, **w**, **min_size**: see ```azp```.
  - **verbose** (```int```): see document of [spreg.skater_reg](https://pysal.org/spreg/notebooks/skater_reg.html).
- **Returns**:
  - **label**, **coeffs**: see ```azp```.

# Reproduction & Replication Guide

## Experiments on Synthetic Data

### 1. Data Generation

Simulated data used in our experiments are provided in ```synthetic/grid_*.txt```. The file structure for each sample is: the array of $x_1$, $x_2$, $y$, region index, $\beta_1$, and $\beta_2$. Each array contains 25 lines, and each line has 25 numbers, corresponding to 25*25 grid cells. These files can be read by ```grid_reg.py``` automatically for spatial regime delineation.

If you use provided data, go straightly to step 2. Otherwise, use the script ```grid_gen.py``` to generate new data samples. Open ```grid_gen.py``` and change the following parameters if needed:

- **Side** : number of cells in each row and column of the grid.
- **repeat** : number of simulations in each dataset (*Rectangular*, *Voronoi*, *Arbitrary*).  
- **min_region** : minimum number of units in each region.  

Then run the script. Files named ```grid_*.txt``` will be produced. Note that three samples with different noise levels are generated in each simulation. Move these files to the ```synthetic``` directory for future use. The script will also show the latent regions of one simulation for each dataset, which produced **Figure 2**.  

### 2. Spatial Regime Delineation

Use the script ```grid_reg.py``` to perform spatial regime delineation with the five considered algorithms. Open ```grid_reg.py``` and change the following parameters if needed:

- **Side** : number of cells in each side of the grid. Should be consistent with input data.
- **ids, idt** : index range of input data. The code will process data with id from ```ids``` to ```idt```-1.  
- **prefix** : 'edis_' or 'edistest_'. Should be consistent with input data.  
- **micro_clusters** : the parameter K in the two-stage K-Models algorithm.  
- **min_region** : minimum number of units in each region.  
- **recordnum** : the number for this experiment, used to label output files.  

The script will generate a log file named ```RG(recordnum).txt```.  It records five lines for each sample, which is the summary for KM, AZP, RKM, GSK, SKR, respectively. Each line includes: total sum of squared residuals (SSR); number of true positives (TP), false negatives (FN), false positives (FP), true negatives (TN), Rand index; entropy of true regions, entropy of reconstructed regions, mutual information (MI), normalized mutual information (NMI); mean absolute error (MAE) of $\beta_0$, $\beta_1$, $\beta_2$; execution time; number of iterations and merge operations (if applicable). Please refer to our paper for detailed explanation of these metrics. **Table 1**, **Table 2**, and **Table A1** are summarized from these results.

For each ```dataid```, the program outputs:

- A TXT file named ```result_(recordnum)_(dataid).txt```. The result of low noise data comes first, then medium and high noise, each contains results from five algorithms. For each algorithm, the file first records map of the delineated regions (a ```Side*Side``` matrix containing region index of each unit), then the estimated coefficients (the $r$-th line is the intercept and linear coefficients for the $r$-th region). **Figure 4** is a visualization of the delineated regions recorded in these files.

- A PNG file named with ```recordnum```, ```dataid``` and time, visualizing the region delineation at each noise level.  

### 3. Algorithm Stability

To reproduce **Figure A1a**, run the two-stage K-Models, AZP, and Regional-K-Models algorithms repeatedly on the same sample of data, using the script ```grid_stab.py```. The usage is the same with ```grid_reg.py``` except that ```dataid``` and ```repeat``` (number of repeated runs) need to be specified rather than ```ids``` and ```idt```.

### 4. The Effect of $K$

To reproduce **Figure A1b**, run the two-stage K-Models algorithm with different $K$ values, using the script ```grid_K.py```. The usage is the same with ```grid_reg.py``` except that  ```idlist``` (a list of ```dataid```) needs to be specified rather than ```ids``` and ```idt```. You may also change ```mclist```, which specify the range of $K$ values.

## Real Data: Georgia Dataset

The Georgia dataset is available as sample data of the [MGWR software](https://sgsup.asu.edu/sparc/multiscale-gwr). We aggregated polygons belong to the same county into multi-polygons, making it easy to generate PySAL spatial weights.

The script ```src/georgia.py``` run the five considered algorithms on the Georgia dataset. You may change the following parameters:

- **pmin, pmax** : the range of number of regions. The code runs the algorithms for each $p$ in [```pmin```,```pmax```].  
- **Kfac** : the $K$ parameter in the two-stage K-Models is calculated as ```Kfac```$\times p$.  
- **runs** : number of repetitions. Increasing this parameter may find better solution, with the cost of more computation time.  
- **numid** : number of the experiment, used to label output files.  
- **min_region** : minimum number of units for each region. Should be at least 5 in this analysis for valid significance test.  

After running the script, results are recorded in two output files:

- ```Georgia_(numid).txt``` : the log file. It records the results at each number of regions $p$ (from ```pmin``` to ```pmax```). For each $p$, in the order of KM, AZP, RKM, GSK, and SKR, it first lists total SSR, execution time, number of iterations and merge operations (if applicable) for each run. Then for the best solution (with the lowest SSR), it records the coefficients and F-test results. Each line contains: region index, number of units, estimated coefficients, the F-value and p-value of the overall F-test. **Figure 5** visualizes the total SSR from the five considered algorithm at different $p$ values.  
- ```Georgia_(numid).xls``` : an Excel workbook. For each county, it records the region it belongs to, as well as the estimated coefficients of that region. For each of the five algorithms, only the best solution is recorded.

To visualize the results, join the output table ```Georgia_(numid).xls``` with the polygon Shapefile via the ```Area_key/AreaKey``` field with any standard GIS software (e.g. QGIS). After that, **Figure 6** and **Figure 7** can be rendered in a GIS environment using the joined Shapefile.  

## Real Data: King County Dataset  

The King County house price dataset is available from [GeoDa center](https://geodacenter.github.io/data-and-lab//KingCounty-HouseSales2015/). The data processing procedure is described in our paper.   

The script ```src/kinghouse.py``` run the two-stage K-Models and Skater regression  algorithms on the King County dataset. You may change the parameters including **pmin, pmax**, **Kfac**, **runs**, **numid**, and **min_region**. The meanings of these parameters are the same with those in ```src/georgia.py```. Note that ```min_region``` should be at least 18 in this analysis for valid significance test.  

After running the script, results are recorded in two output files:

- ```Kinghouse_(numid).txt``` : the log file. The file structure is the same with the log file for the Georgia dataset.  
- ```Kinghouse_(numid).xls``` : an Excel workbook. For each house location, it records the region it belongs to.  

To visualize the results, join the output table ```Kinghouse_(numid).xls``` with the point Shapefile via the ```id``` field with any standard GIS software (e.g. QGIS). After that, **Figure 8** can be rendered in a GIS environment using the joined Shapefile.  
