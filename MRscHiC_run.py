from imputation import *
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import time
import numpy as np
import os


mm9dim = [197195432,181748087,159599783,155630120,152537259,149517037,152524553,131738871,124076172,129993255,121843856,121257530,120284312,125194864,103494974,98319150,95272651,90772031,61342430]
hg19dim = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]
mm10dim=[195471971,182113224,160039680,156508116,151834684,149736546,145441459,129401213,124595110,130694993,122082543,120129022,120421639,124902244,104043685,98207768,94987271,90702639,61431566,171031299]

#flyamer
#ctlist=['Oocyte','ZygM','ZygP']
#network = [np.loadtxt('/mnt/d/jjpeng/cwzhen/data/pythonProject/Flyamer/1mbres/'+ct+'/samplelist_filter_linux.txt',dtype=np.str) for ct in ctlist]
#label = np.array([ctlist[i] for i in range(len(ctlist)) for j in range(len(network[i]))]).astype('U8')
#label=np.concatenate((label,label),axis=0)
#network = np.concatenate(network)
#chrom = [str(i+1) for i in range(19)]
#chromsize = {chrom[i]:mm9dim[i] for i in range(len(chrom))}
#nc = 3



#ramani dataset start
#ctlist = ['HeLa', 'HAP1', 'GM12878', 'K562']
#network = [np.loadtxt('/mnt/d/jjpeng/cwzhen/data/pythonProject/Ramani/'+ct+'/9_12_samplelist_filter_nonnum_linux.txt',dtype=np.str) for ct in ctlist]
#label = np.array([ctlist[i] for i in range(len(ctlist)) for j in range(len(network[i]))]).astype('U8')
#network = np.concatenate(network)
#chrom = [str(i+1) for i in range(22)] + ['X']
#chromsize = {chrom[i]:hg19dim[i] for i in range(len(chrom))}
#nc=4

#4DN dataset start
ctlist=['GM12878','H1Esc','HAP1','HFF','IMR90']
network = [np.loadtxt('1mbfaster/'+ct+'/kim_samplelistfilter2.txt',dtype=np.str) for ct in ctlist]
label = np.array([ctlist[i] for i in range(len(ctlist)) for j in range(len(network[i]))]).astype('U8')
network = np.concatenate(network)
chrom = [str(i+1) for i in range(22)] + ['X']
chromsize = {chrom[i]:hg19dim[i] for i in range(len(chrom))}
nc = 5


if __name__ == "__main__":
    start_time = time.time()
    cluster, embedding = dataprocess(network, chromsize, nc=nc,ncpus=5)
    print(time.time() - start_time)
    print(embedding.shape)

    label_num=KMeans(n_clusters=nc, n_init=200).fit(embedding[:,:]).labels_
    ari =ARI(label, label_num)
    print(ari)



