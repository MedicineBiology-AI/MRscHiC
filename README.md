# MRscHiC
A multi-scale representation learning framework

## Data

* The Flyamer dataset is downloaded from GSE80006.
* The Ramani dataset is downloaded from GSE84920.
* The 4DN dataset is downloaded from https://noble.gs.washington.edu/proj/schic-topicmodel.


## Usage
### The input file format
As an example, at 1mb resolution, if there are 5 reads in Cell_1 supporting the 
interaction between chr1:2000000-3000000 and chr1:7000000-8000000, this should be represented as


* 2 7   5

This file can be named Cell_1_chr1.txt.

### Clustering
1.Run MRscHiC_run.py to convert input files to feature matrix.

2.Run ftrain.py(Flyamer dataset) or rtrain.py(Ramani dataset 4DN dataset) to 
convert feature matrix to the cell embedding.

3.Use cell embedding for downstream analysis.

## Citation
Zhou, J., Ma, J., Chen, Y., Cheng, C., Bao, B., Peng, J., ... & Ecker, J. R. (2019). 
Robust single-cell Hi-C clustering by convolution-and random-walk–based imputation. 
Proceedings of the National Academy of Sciences, 116(28), 14011-14018.
