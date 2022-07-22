# Overview

This software aims to delineate geographically connected regions in the context of linear regression models with varying coefficients across regions. 

# Environment & Dependencies

The software is built on Python 3.8 (or higher versions) as well as several Python packages.
- Necessary
  - numpy, matplotlib, networkx
  - scikit-learn (0.24.2, higher versions may encounter compatibility error)
  - libpysal, mgwr
  - xlrd, xlwt
- Optional
  - statsmodels (significance testing of linear equations) 

# File Structure

## Core Modules

- ``Algorithm5.py`` : implementation of the regionalization algorithms, including K-Models (KM),  AZP, Regional-K-Models (RKM), and GWR+K-Means (GWR).
- ``Network5.py`` : tool functions dealing with the adjacency network, region contiguity, etc.
- ``GridData5.py`` : generating and handling simulated grid data.

## Simulated Data

- ``synthetic/edis_*.txt`` : Dataset 1 (distinct latent regions), 25*25 grid, 50 simulations (```dataid```  0-49), each with three samples under different noise levels ('h' for high, 'm' for medium, 'l' for low), yet with the same latent regions of regression coefficients.

- ``synthetic/econ_*.txt`` : Dataset 2 (continuous coefficient surfaces), 25*25 grid, 50 simulations (```dataid``` 0-49).

We also include several simulations on 10\*10 grid for debugging codes (``synthetic/edistest_*.txt``, ``synthetic/econtest_*.txt``), on which regionalization takes only a few seconds. 

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

- ``georgia.py`` : regionalization on the Georgia Dataset, using four considered algorithms

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
    - **w** (``pysal.weights.W``): spatial contiguity matrix of the $N$ areal units. The order of units must be consistent with ```Xarr``` and ```Yarr```.
    - **init_stoc_step** (``bool, default=True``): the way of region growing in initialization. If ``False``, all neighboring units of current region are merged into it. Otherwise, a randomly chosen neighboring unit is assigned into it. All our experiment use the default ``True`` value, as it increases the diversity of initial solutions.
 - **Returns:** 
    - **label** (``list``): the region label for each unit, with length $N$. The $u$ th unit is assigned to the region with index ``label[u]``. The region indices are integers from 0 to $K-1$.
    - **iters** (``int``): the number of iterations.

  ```Python
 azp(Xarr, Yarr, n_regions, w, init_stoc_step = True)
 ```
AZP zone design algorithm for linear regression.
 - **Parameters**:  
    The same with ```kmodels```.
 - **Returns**: 
    - **regions** (``list``): produced regions. Each element ``regions[r]`` is a list, containing indices of all units belong to the $r$ th region. It can be converted from ```label``` as follows: 
    ```Python
    units = np.arange(w.n).astype(int)
    regions = [units[label == r].tolist() for r in set(label)]
    ```
    - **coeffs** (``list``): estimated coefficients for each region. ``coeff[r]`` is a list of length $M$, containing $M$ coefficients corresponding to the features in ```Xarr```. 
    - **iters** (``int``): the number of iterations.


```Python
region_k_models(Xarr, Yarr, n_regions, w, init_stoc_step = True)
 ```
Regional-K-Models zone design algorithm for linear regression.
- **Parameters**:   
  The same with ```azp```.
- **Returns**:   
  The same with ```azp```.

```Python
gwr_cluster(Xarr, Yarr, coord, n_regions)
```
GWR+K-Means zone design algorithm for linear regression. 
- **Parameters**:   
   - **Xarr**, **Yarr**, **n_regions**: see ```kmodels```.
   - **coord** (```list```): coordinates of the spatial units. Each of the $N$ elements is a coordinate pair (tuple of length 2) indicating the position of a unit.
- **Returns**:   
   - **label**: see ```kmodels```.

```Python
split_merge(Xarr, Yarr, w, clabel, lamda)
```
Post-processing procedure to improve the solution and impose contiguity. Used after K-Models and GWR+K-Means.

- **Parameters**:   
   - **Xarr**, **Yarr**, **w**: see ```kmodels```.
   - **clabel** (```list```): the same as ```label``` in ```kmodels```.
   - **lamda** (```float```): the penalty factor $\lambda$ in the objective function.
- **Returns**:   
   - **regions**, **coeffs**: see ```azp```.

## Network5

```Python
evaluation_func(regions, Xarr, Yarr, lamda)
```
The proposed objective function.
- **Parameters**:   
   - **Xarr**, **Yarr**: see ```kmodels```.
   - **regions**: see ```azp```.
   - **lamda**: see ```split_merge```.
- **Returns**:   
   - **tc** (```float```): objective function value, equals ```acc``` + $\lambda$ ```rgl```.
   - **acc** (```float```): total sum of squared errors  (SSE) of the regional regression models.
   - **rgl** (```int```): number of produced regions.

```Python
Test_Equations(regions,Xarr,Yarr,log)
```
Overall F-test of the derived regression model for each region.
- **Parameters**:   
   - **Xarr**, **Yarr**: see ```kmodels```.
   - **regions**: see ```azp```.
   - **log** (```_io.TextIOWrapper```): output file. Results of the F-test is recorded there.
- **Returns**: None. 

# Reproduction & Replication Guide

## Simulated Data: Dataset 1

### 1. Data Generation
Simulated data used in our experiments are provided in ```synthetic/edis_*.txt```. The file structure for each sample is: the array of $\beta_1$, the array of $x$, and the array of $y$. Each array contains 25 lines, and each line has 25 float numbers, corresponding to 25*25 grid cells. These files can be read by ```edis_reg.py``` automatically for regionalization. 

If you use provided data, go straightly to step 2. Otherwise, use the script ```edis_gen.py``` to generate new data samples. Open ```edis_gen.py``` and change the following parameters if needed:
- **Side** : number of cells in each row and column of the grid. 
- **runs**: number of simulations.  

Then run the script. Files named ```edis_*.txt``` will be produced, and ```dataid```  numbered from 50. Note that three samples with different noise levels are generated in each simulation. Move these files to the ```synthetic``` directory for future use. The script will also show the latent region schemes generated with Voronoi polygons in the first three simulations , which produced **Figure 3**. This may cause error if ```runs```<3.

### 2. Regionalization

Use the script ```edis_reg.py``` to perform regionalization with the four considered algorithms. Open ```edis_reg.py``` and change the following parameters:
- **Side** : number of cells in each side of the grid. Should be consistent with input data.
- **ids, idt** : index range of input data. The code will process data with id from ```ids``` to ```idt```-1. 
- **lamda** : the penalty factor $\lambda$ in the objective function.
- **recordnum** : the number for this experiment, used to label output files. 
- **prefix** : 'edis_' or 'edistest_'. Should be consistent with input data. 

The script will generate a log file named ```RG(recordnum).txt```.  It records four lines for each sample, which is the summary for KM, AZP, RKM, GWR, respectively. Each line includes: objective function value ```tc```, modelling error ```acc```, number of produced regions ```rgl```, executing time, selected $K$, and number of iterations ```iters``` (not applicable for GWR). **Table 2** is summarized from ```tc``` of 50 simulations. **Figure A1b** is summarized from execution times of 50 simulations.

For each ```dataid```, the program outputs:
- A TXT file named ```result_(recordnum)_(dataid).txt```. The result of low noise data comes first, then medium and high noise, each contains results from four algorithms. For each algorithm, the file first records map of the delineated regions (a ```Side*Side``` matrix containing region index of each unit), then the estimated coefficients (the $r$ th line is the intercept and linear coefficients for the $r$ th region). Move these files to the ```log``` directory for future use.

- A PNG file named with ```recordnum```, ```dataid``` and time, visualizing the estimated coefficients for each region (**Figure 6**). Futhermore, the script ```visualization/Plot_Regdis.py``` can reproduce such figures based on saved regionalization results in the ```log``` directory (need to specify ```recordnum``` and ```dataid```). 

- A TXT file named ```kcurve_(recordnum)_(dataid).txt```. The result of low noise data comes first, then medium and high noise, each contains results from four algorithms (in the order KM, AZP, RKM, GWR). This file records ```tc```, ```acc```, ```rgl```, ```iters``` for each $K$. **Figure B2** shows ```tc``` values at a range of $K$.

### 3. Evaluation with RI and NMI

To reproduce **Table 3**, use the scripts ```evaluation/Rand.py``` and ```evaluation/Mutual_Info.py``` to calculate RI and NMI for a set of regionalization results. For both scripts, first open and specify ```Side```, ```ids``` and ```idt```, as well as ```recordnum```. If all results use the same ```recordnum```, the ```recordnum``` function body should be ```return (recordnum)```. Else, use conditional statements on ```dataid``` to specify ```recordnum```. Note that the related result files should be in the ```log``` directory. Then run the script. The RI and NMI results for each sample of data can be found in generated Excel workbooks. 

## Simulated Data: Dataset 2

### 1. Data Generation

Simulated data used in our experiments are provided in ```synthetic/econ_*.txt```. The file structure for each sample is: the array of $x_1$, the array of $x_2$, and the array of $y$. Each array contains 25 lines, and each line has 25 float numbers, corresponding to 25*25 grid cells. These files can be read by ```econ_reg.py``` automatically for regionalization. 

If you use provided data, go straightly to step 2. Otherwise, use the script ```econ_gen.py``` to generate new data samples. Open ```econ_gen.py``` and change ```runs``` (number of simulations) if needed. We do not recommend change ```Side``` arbitrarily since the parameter surfaces are designed according to it (a parameter surfaces setting for 10*10 grid is provided in the comment, used to generate ```econtest_*.txt``` files). 

Then run the script. Files named ```econ_*.txt``` will be produced, and ```dataid```  numbered from 50. Move these files to the ```synthetic``` directory for future use. The script will also show the latent parameter surfaces (**Figure 4**).

### 2. Regionalization

Use the script ```econ_reg.py``` to perform regionalization with the four considered algorithms. Open ```econ_reg.py``` and change the following parameters:
- **Side**, **ids**, **idt**, **lamda**, **recordnum** : The same with ```edis_reg.py```
- **prefix** : 'econ_' or 'econtest_'. Should be consistent with input data. 

The script will generate a log file named ```RG(recordnum).txt```. The file structure is the same with log file of ```edis_reg.py```. Results of ```tc``` and execution times are also summarized in **Table 2** and **Figure A1b**, respectively. 

For each ```dataid```, the program outputs:
- A TXT file named ```result_(recordnum)_(dataid).txt```. It contains results from four algorithms. For each algorithm, the file first records the delineated regions (a ```Side*Side``` matrix containing region index of each unit), then the estimated coefficients (the $r$ th line is the intercept and linear coefficients for the $r$ th region). Move these files to the ```log``` directory for future use.

- A PNG file named with ```recordnum```, ```dataid``` and time, visualizing the estimated coefficients for each region (**Figure 7**). Futhermore, the script ```visualization/Plot_Regcon.py``` can reproduce such figures based on saved regionalization results in the ```log``` directory (need to specify ```recordnum```, ```dataid```). 

### 3. Stability Test

To reproduce **Figure A1a**, run regionalization algorithms repeatedly on the same sample of data, using the script ```econ_stab.py```. The usage is the same with ```econ_reg.py``` except that ```dataid``` and ```repeat``` (number of repeats) need to be specified rather than ```ids``` and ```idt```.

### 4. The Effect of $\lambda$

To reproduce **Figure B1**, run K-Models algorithm with different $\lambda$ values, using the script ```econ_lambda.py```. The usage is the same with ```econ_reg.py``` except that  ```dataid``` needs to be specified rather than ```ids``` and ```idt```. You may also change ```Lamdamin```, ```Lamdastep```, and ```Lamdamax```, which specify the range of $\lambda$ values.

## Real Data: Georgia Dataset

The Georgia dataset is available from https://sgsup.asu.edu/sparc/multiscale-gwr. We provide simplified data to ease usage in our experiment:
- ```Aggre.shp``` : a polygon Shapefile data of 159 Georgia counties. We aggregated polygons belong to the same county into multi-polygons, making it easy to generate PySAL spatial weights.
- ```GData_utm.xls``` : a table containing explanatory and dependent variables for each county. Our code only uses data in the 'use' sheet, where we remove irrelevant fields.

The script ```georgia.py``` run the four considered regionalization algorithms on the Georgia dataset. You may change the following parameters:
- **lambda** : the penalty factor for number of regions. $\lambda=3$ is used in our experiment.
- **runs** : number of repetitions. Increasing this parameter may find better solution, with the cost of more computation time. 
- **numid** : number of the experiment, used to label output files.

After running the script, results are recorded in two output files:
- ```Georgia_(numid).txt``` : the log file. In the order of KM, AZP, RKM and GWR, it first lists ```tc```,  ```acc```, ```rgl```, executing time, selected $K$, and ```iters``` (not applicable for GWR) for each run. Then for the best solution (with the lowest ```tc```), it records the coefficients and F-test results. Each line contains: region index, number of units, estimated intercept and coefficients, the F-value and p-value of the overall F-test. Note that regions with less than 5 units are considered outliers, so the coefficients and F-statistics are not calculated.
- ```Georgia_(numid).xls``` : an Excel workbook. For each county, it records the region it belongs to, as well as the estimated intercept and coefficients of that region. For each of the four algorithms, only the best solution is recorded .

To visualize the results, join the output table ```Georgia_(numid).xls``` with ```Aggre.shp``` via the ```Area_key/AreaKey``` field. This can be performed using a standard GIS software (e.g. QGIS). Our results are in the Shapefile ```Join4.shp```. After that, **Figure 8** and **Figure 9** can be rendered in a GIS environment using the joined Shapefile. 
