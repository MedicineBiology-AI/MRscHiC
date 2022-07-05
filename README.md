# MRscHiC

**This repo is moved to [23AIBox](https://github.com/23AIBox/23AIBox-MRscHiC)**

The three-dimensional genome structure plays a key role in cellular function and gene regulation. Single-cell Hi-C technology can capture genome structure information at the cell level, which provides the opportunity to study how genome structure varies among different cell types. However, few methods are well designed for single-cell Hi-C clustering, because of high sparsity, noise and heterogeneity of single-cell Hi-C data. In this manuscript, we propose a multi-scale representation learning framework, named MRscHiC, for single-cell Hi-C data representation and clustering. MRscHiC mainly contains two parts: imputation at bin level, feature extraction at chromosome and cell level. The evaluation results show that the proposed method outperforms existing state-of-the-art approaches on both human and mouse datasets.


<!-- TOC depthFrom:1 depthTo:8 withLinks:1 updateOnSave:1 orderedList:0 -->

- [MRscHiC](#mrschic)  
	- [The environment of MRscHiC](#the-environment-of-mrschic)  
    - [Data](#data)
         - [Contact matrix](#contact-matrix)
         - [Contact matrix preprocessing](#contact-matrix-preprocessing)
    - [Input files](#input-files)
         - [Contact matrix file](#contact-matrix-file)
         - [Cell list file](#cell-list-file)
    - [Usage](#usage)
    - [Acknowledgments](#acknowledgments)

<!-- /TOC -->



## The environment of MRscHiC
    Linux OS
    python 3.8.5 
    PyTorch 1.4.0

## Data

* The Flyamer dataset is downloaded from [GSE80006](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80006).
* The Ramani dataset is downloaded from [GSE84920](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84920).
* The 4DN dataset is downloaded from https://noble.gs.washington.edu/proj/schic-topicmodel.

 ### Contact matrix
The current single-cell Hi-C experimental data is generally stored as a two-dimensional contact matrix. The data obtained by high-throughput sequencing technologies such as Hi-C can be used to construct a contact matrix. The data information obtained by the Hi-C experiment represents the interaction information between any two sites in the whole genome (the information is usually reflected by the number of read-pairs or interaction pairs), and the information is stored in the contact matrix, which is a symmetric matrix. Each element in the contact matrix of the chromosome is said to support the number of interactions between the corresponding two chromosome fragments. For a chromosome contact matrix *M*, each element *Mij* in the matrix represents the number of read-pairs that support the interaction between two chromosome fragments *i* and *j*. As shown in Table 1, the table is a part of the contact matrix of chromosome 1 of an Oocyte cell. The number 62 in bold in the table indicates that the number of read-pairs that support the interaction between the two chromosome fragments chr1: 3000000-4000000 and chr1:3000000-4000000 is 62, and 12 in bold represents the number of read-pairs that support the interaction between the two chromosome fragments chr1: 3000000-4000000 and chr1: 4000000 -5000000 is 12. 

Table 1 Part of the contact matrix. 
bin| 3000000-4000000 |4000000-5000000	| 5000000-6000000 | 6000000-7000000 |	7000000-8000000 |	8000000-9000000
:-----:|:-----:|:-----:|:----------:|:----:|:-----:|:--------:|
3000000-4000000 |	**62.00** |	**12.00**  |	2.00      |	1.00   |	0.00  |	2.00 
4000000-5000000 |	12.00 |	132.00 |	8.00      |	10.00  |	0.00  |	6.00 
5000000-6000000	| 2.00  |	8.00   |	86.00     |	16.00  |	21.00 |	16.00 
6000000-7000000 |	1.00 	| 10.00  |	16.00     |	172.00 |	24.00 |	10.00 
7000000-8000000	|0.00 	|  0.00  |	21.00 	  | 24.00  |  104.00|8.00 
8000000-9000000	|2.00 	| 6.00 	 |16.00       | 10.00  |   8.00 |	58.00 

 ### Contact matrix preprocessing 
The raw data downloaded in [GSE80006](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80006) is shown in Table 2. The bolded data rows in the table indicate: the number of read-pairs that support the interaction between the two chromosome fragments chr1:3000000-3200000 and chr1:3200000-3400000 is 1.

Table 2 Raw data of GSE80006.
chrom1 |	chrom2 |	start1 |	end1 |	start2 |	end2 |	count
:-----:|:-----:|:-----:|:----------:|:----:|:-----:|:--------:|
1	|1	|3000000	|3200000	|3000000	|3200000	|4
**1**	|**1**	|**3000000**	|**3200000**	|**3200000**	|**3400000**	|**1**
1	|1	|3000000	|3200000	|4000000	|4200000	|1
1	|1	|3000000	|3200000	|4400000	|4600000	|1
1	|1	|3000000	|3200000	|4600000	|4800000	|1
1	|1	|3000000	|3200000	|5200000	|5400000	|1

chrom1:  chromosome name for fragment 1.   
chrom2:  chromosome name for fragment 2.   
start1:  start location for fragment 1.   
end1:    end location for fragment 1.   
start2:  start location for fragment 2.   
end2:    end location for fragment 2.   
count:   count number or normalized weight for the interaction.   
(Note that only intra-chromosomal reads are used in MRscHiC.)  

## Input files
### Contact matrix file
In this work, we process the raw data in Table 2 into the data in Table 3 to enter the script. The input data is divided into three columns, and each column is separated by a tab, respectively indicating bins and the number of read-pairs that support interaction. As shown in Table 3, the bolded data row indicates that at the 200-kb resolution, the number of read-pairs that support the interaction between the two chromosome fragments chr1: 3000000-3200000 and chr1: 3200000-3400000 is 78. The first column indicates the starting position of the fragment of the first chromosome divided by the resolution, the second column indicates the starting position of the fragment of the second chromosome divided by the resolution, and the third column indicates the number of read-pairs that support the interaction in these two fragments. The end position of the chromosomes fragments can be calculated using the start position plus the resolution.

Table 3 Part of contact matrix file.
chrom1_start|	chrom2_start|	count
:-----:|:-----:|:-----:|
15|	15|	237
**15**|	**16**|	**78**
15|	17|	22
15|	18|	24
15|	19|	11
15|	20|	14
15|	21|	8
15|	22|	10
15|	23|	9

chrom1_start:  start location for fragment 1.   
chrom2_start:  start location for fragment 2.      
count:   count number or normalized weight for the interaction.   
(Note that the chrom1_start and chrom2_start are calculated in the following way: chrom1_start=(start location for fragment 1)/resolution.)  

### Cell list file
Information in this file: the location information of the cells that are input to the script.   


    1mbres/Oocyte/NSN/NSN_1  
    1mbres/Oocyte/NSN/NSN_4  
    1mbres/Oocyte/NSN/NSN_5  
    1mbres/Oocyte/NSN/NSN_6
     
    



## Usage
### 1.Run MRscHiC_run.py to convert input files（contact matrix file & cell list file） to feature matrix.

    $ python MRscHiC_run.py

### 2.Run ftrain.py (Flyamer dataset) or rtrain.py (Ramani dataset 4DN dataset) to convert feature matrix to the cell embedding.
    $ python ftrain.py
    or
    $ python rtrain.py

### 3.Use cell embedding for downstream analysis.

## Acknowledgments
We really thank the Zhou et al. open the source code of scHiCluster at this [link](https://github.com/zhoujt1994/scHiCluster). 
