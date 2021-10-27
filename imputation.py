from __future__ import division
from __future__ import print_function


import time
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score as ARI
#from rtrain import *
from loadmodel import *
from data_preprocess import *
#from DeepAE import *
#from flyamer_train import *
#import vae
#from multimodel import *
#from multitrain import *

def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # 所有节点的�?
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)

def neighbor_ave_cpu(A, pad):
    if pad == 0:
        return A
    ngene, _ = A.shape
    ll = pad * 2 + 1
    B, C, D, E = [np.zeros((ngene + ll, ngene + ll)) for i in range(4)]
    B[(pad + 1):(pad + ngene + 1), (pad + 1):(pad + ngene + 1)] = A[:]
    F = B.cumsum(axis=0).cumsum(axis=1)
    C[ll:, ll:] = F[:-ll, :-ll]
    D[ll:, :] = F[:-ll, :]
    E[:, ll:] = F[:, :-ll]
    return (np.around(F + C - D - E, decimals=8)[ll:, ll:] / float(ll * ll))


def random_walk_cpu(A, rp):
    ngene, _ = A.shape
    A = A - np.diag(np.diag(A))
    A = A + np.diag(np.sum(A, axis=0) == 0)
    P = np.divide(A, np.sum(A, axis=0))
    Q = np.eye(ngene)
    I = np.eye(ngene)
    for i in range(30):
        Q_new = (1 - rp) * I + rp * np.dot(Q, P)
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        if delta < 1e-6:
            break
    return Q



def impute_cpu(args):
    cell, c, ngene, pad, rp= args
    D = np.loadtxt(cell + '_chr' + c + '.txt')
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
    A = np.log2(A + A.T + 1)
    #A=minmax_scale(A)
    #A=standrad_scale(A)
    #A = neighbor_ave_cpu(A, pad)

    if rp == -1:
        Q = A[:]
    else:
        #B = random_walk_cpu(A, rp)
#        A = neighbor_ave_cpu(A, pad)
#        Q = A[:]
        Q = random_walk_cpu(A, rp)
        adj_norm = normalize(torch.FloatTensor(Q), True)
        features = torch.FloatTensor(Q)
        ax1 = adj_norm.mm(features)
        ax1 = ax1.numpy()
#        ax2=adj_norm.mm(ax1)
#        ax2 = ax2.numpy()


    return [cell,ax1.reshape(ngene*ngene)]

def zscore(matrix):
    num1,num2=matrix.shape
    newmat=np.zeros((num1,num2))
    std=np.std(matrix)
    ave=np.mean(matrix)
    for i in range(num1):
        for j in range(num2):
            newmat[i][j]=(matrix[i][j]-ave)/std
    return newmat


def dataprocess(network, chromsize, nc,  res=1000000, pad=1, rp=0.5, prct=95, ndim=20, ncpus=10):
    matrix = []
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res) + 1
        start_time = time.time()
        result=[]
        item=[]
        for cell in network:
            args=cell,c,ngene,pad,rp
            item=impute_cpu(args)
            result.append(item)
        index = {x[0]: j for j, x in enumerate(result)}
        Q_concat = np.array([result[index[x]][1] for x in network])
        print('adj closed!')
        
#        n1, n2 = Q_concat.shape
#        b = []
#        for i in range(n2):
#            chr_count = 0
#            for j in range(n1):
#                flag = 0
#                if (Q_concat[j][i] != 0):
#                    chr_count += 1
#                if (chr_count >= n1*0.9):
#                    b.append(Q_concat[0:n1, i])
#                    flag = 1
#                if (flag == 1):
#                    break
#            #print('feature ' + str(i) + 'count(0)=' + str(chr_count0))
#
#        b = np.array(b)
#        Q_concat = b.T
#        print('feature selection closed!')
#        print(Q_concat.shape)
        
        #Q_concat=zscore(Q_concat)
        #print("z_score closed!")
#        Q_concat=standrad_scale(Q_concat)
#        print('standrad_scale closed!')
#        Q_concat=minmax_scale(Q_concat)
#        print('minmax_scale closed!')

        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:,None ])

#        np.savetxt('final_mat/topmat01.csv',Q_concat,delimiter=',')
        ndim = int(min(Q_concat.shape)*0.95) - 1
        pca = PCA(n_components=ndim)
        R_reduce = pca.fit_transform(Q_concat)
        print('pca->chromosome closed!')
        print(R_reduce.shape)
#        R_reduce = train(Q_concat)
#        print('AE->chromosome closed!')
#        print(R_reduce.shape)
        
#        R_reduce = vae.Dhaka(Q_concat)
#        print('VAE->chromosome closed!')
#        print(R_reduce.shape)


#        ndim = int(min(Q_concat.shape) * 0.15) - 1
#        pca = PCA(n_components=ndim)
#        R_reduce = pca.fit_transform(Q_concat)
#        print('pca->chromosome closed!')
#        print(R_reduce.shape)

        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        matrix.append(R_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    np.savetxt('copy4_pca/flyamer_copy4_top95_mat.txt',matrix)
    print(matrix.shape)
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix[:, :ndim])
    label_true=np.loadtxt('outputdata/flyamer_copy4_label_true.txt')
    ari_20=ARI(label_true,kmeans.labels_)
    print('ari_20: ',ari_20)
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix[:, :])
    ari_all=ARI(label_true,kmeans.labels_)
    print('ari_all: ',ari_all)
    matrix_reduce=train(matrix)

    
#    matrix=np.concatenate(matrix,axis=1)
#    matrix_reduce = vae.Dhaka(matrix)

#    pca = PCA(n_components=min(matrix.shape) - 1)
#    matrix_reduce = pca.fit_transform(matrix)

    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, :ndim])
    # kmeans.labels_,
    return kmeans.labels_, matrix_reduce







