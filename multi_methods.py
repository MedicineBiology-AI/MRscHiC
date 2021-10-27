from sklearn.decomposition import PCA, FastICA, NMF
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import sklearn.preprocessing as prep
from sklearn.decomposition import TruncatedSVD
#from rtrain import *
from ftrain import *

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

def minmax_scale(x_train):
    '''
    do min-max normalization
    '''
    preprocessor = prep.MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    return x_train


def standrad_scale(x_train):
    '''
    do standard normalization
    '''
    preprocessor = prep.StandardScaler()
    x_train = preprocessor.fit_transform(x_train)
    return x_train
    


def neighbor_ave_cpu(A, pad):
    if pad==0:
        return A
    ngene, _ = A.shape
    ll = pad * 2 + 1
    B, C, D, E = [np.zeros((ngene + ll, ngene + ll)) for i in range(4)]
    B[(pad + 1):(pad + ngene + 1), (pad + 1):(pad + ngene + 1)] = A[:]
    F = B.cumsum(axis = 0).cumsum(axis = 1)
    C[ll :, ll:] = F[:-ll, :-ll]
    D[ll:, :] = F[:-ll, :]
    E[:, ll:] = F[:, :-ll]
    return (np.around(F + C - D - E, decimals=8)[ll:, ll:] / float(ll * ll))

def random_walk_cpu(A, rp):
    ngene, _ = A.shape
    A = A - np.diag(np.diag(A))
    A = A + np.diag(np.sum(A, axis=0) == 0)
    P = np.divide(A, np.sum(A, axis = 0))
    Q = np.eye(ngene)
    I = np.eye(ngene)
    for i in range(30):
        Q_new = (1 - rp) * I + rp * np.dot(Q, P)
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        if delta < 1e-6:
            break
    return Q
def impute_gc(args):
    cell, c, ngene, pad, rp= args
    D = np.loadtxt(cell + '_chr' + c + '.txt')
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
    A = np.log2(A + A.T + 1)

    if rp == -1:
        Q = A[:]
    else:
        A = neighbor_ave_cpu(A, pad)
        Q = random_walk_cpu(A, rp)
        adj_norm = normalize(torch.FloatTensor(Q), True)
        features = torch.FloatTensor(Q)
        ax1 = adj_norm.mm(features)
        ax1 = ax1.numpy()

    return [cell,ax1.reshape(ngene*ngene)]

def impute_cpu(args):
    cell, c, ngene, pad, rp = args
    D = np.loadtxt(cell + '_chr' + c + '.txt')
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape = (ngene, ngene)).toarray()
    A = np.log2(A + A.T + 1)
    A = neighbor_ave_cpu(A, pad)
    if rp==-1:
        Q = A[:]
    else:
        Q = random_walk_cpu(A, rp)
    return [cell, Q.reshape(ngene*ngene)]

def hicluster_cpu(network, chromsize, nc, res=1000000, pad=1, rp=0.5, prct=20, ndim=20, ncpus=10):
    matrix=[]
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res) + 1
        start_time = time.time()
        result = []
        item = []
        for cell in network:
            args = cell, c, ngene, pad, rp
            item = impute_cpu(args)
            result.append(item)
        index = {x[0]: j for j, x in enumerate(result)}
        Q_concat = np.array([result[index[x]][1] for x in network])
        print('adj closed!')
        if prct>-1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:, None])
        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        ndim = int(min(Q_concat.shape) * 0.2) - 1
        pca = PCA(n_components = ndim)
        R_reduce = pca.fit_transform(Q_concat)
        matrix.append(R_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis = 1)
    pca = PCA(n_components = min(matrix.shape) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    kmeans = KMeans(n_clusters = nc, n_init = 200).fit(matrix_reduce[:, :ndim])
    return kmeans.labels_, matrix_reduce



def ica(network, chromsize, nc, label,res=1000000, ndim=20):
    matrix = []
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res) + 1
        start_time = time.time()
        uptri = np.triu_indices(ngene, 1)
        A_concat = np.zeros((len(label), len(uptri[0]))).astype(float)
        j = 0
        for cell in network:
            D = np.loadtxt(cell + '_chr' + c + '.txt')
            A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
            A = np.log2(A + A.T + 1)
#            A = neighbor_ave_cpu(A, 1)
            A_concat[j, :] = A[uptri]
            j += 1
        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        #A_concat=minmax_scale(A_concat)
        ica = FastICA(n_components=ndim)
        A_reduce = ica.fit_transform(A_concat)
        matrix.append(A_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    ica = FastICA(n_components=min(matrix.shape) - 1)
    matrix_reduce = ica.fit_transform(matrix)
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, 1:(ndim + 1)])
    return kmeans.labels_, matrix_reduce


def nmf(network, chromsize, nc, label,res=1000000, ndim=20):
    matrix = []
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res) + 1
        start_time = time.time()
        uptri = np.triu_indices(ngene, 1)
        A_concat = np.zeros((len(label), len(uptri[0]))).astype(float)
        j = 0
        for cell in network:
            D = np.loadtxt(cell + '_chr' + c + '.txt')
            A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
            A = np.log2(A + A.T + 1)
#            A = neighbor_ave_cpu(A, 1)
            A_concat[j, :] = A[uptri]
            j += 1
        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        A_concat=minmax_scale(A_concat)
        nmf = NMF(n_components=ndim,max_iter=10000)
        A_reduce = nmf.fit_transform(A_concat)
        matrix.append(A_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    nmf = NMF(n_components=min(matrix.shape) - 1,max_iter=10000)
    matrix_reduce = nmf.fit_transform(matrix)
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, 1:(ndim + 1)])
    return kmeans.labels_, matrix_reduce

def svd(network, chromsize, nc,label ,res=1000000, ndim=20):
    matrix = []
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res) + 1
        start_time = time.time()
        uptri = np.triu_indices(ngene, 1)
        A_concat = np.zeros((len(label), len(uptri[0]))).astype(float)
        j = 0
        for cell in network:
            D = np.loadtxt(cell + '_chr' + c + '.txt')
            A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
            A = np.log2(A + A.T + 1)
#            A = neighbor_ave_cpu(A, 1)
            A_concat[j, :] = A[uptri]
            j += 1
        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        svd = TruncatedSVD(n_components=ndim)
        A_reduce = svd.fit_transform(A_concat)
        matrix.append(A_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    svd = TruncatedSVD(n_components=min(matrix.shape) - 1)
    matrix_reduce = svd.fit_transform(matrix)
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, 1:(ndim + 1)])
    return kmeans.labels_, matrix_reduce

def decay(network, chromsize, nc,label ,res=1000000, ndim=20):
    matrix = []
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res)+1
        start_time = time.time()
        dec = np.zeros((len(label), ngene-1)).astype(float)
        j = 0
        for j, cell in enumerate(network):
            D = np.loadtxt(cell+'_chr'+c+'.txt')
            A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
            tmp = np.array([np.sum(np.diag(A, k)) for k in range(1,ngene)])
            dec[j] = tmp / np.sum(tmp)
        end_time = time.time()
        print('Load and random walk take', end_time - start_time, 'seconds')
        matrix.append(dec)
        print(c)
    matrix = np.concatenate(matrix, axis = 1)
    pca = PCA(n_components = min(matrix.shape) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    kmeans = KMeans(n_clusters = nc, n_init = 200).fit(matrix_reduce[:, 1:3])
    return kmeans.labels_, matrix_reduce


def raw_pca(network, chromsize, nc, label,res=1000000, ndim=20):
    matrix=[]
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res)+1
        start_time = time.time()
        uptri = np.triu_indices(ngene, 1)
        A_concat = np.zeros((len(label), len(uptri[0]))).astype(float)
        j = 0
        for cell in network:
            D = np.loadtxt(cell + '_chr' + c + '.txt')
            A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape = (ngene, ngene)).toarray()
            A = np.log2(A + A.T + 1)
            A_concat[j, :] = A[uptri]
            j += 1
        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        pca = PCA(n_components = min(A_concat.shape)-1)
        A_reduce = pca.fit_transform(A_concat)
        matrix.append(A_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis = 1)
    pca = PCA(n_components = min(matrix.shape) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    kmeans = KMeans(n_clusters = nc, n_init = 200).fit(matrix_reduce[:, 1:(3)])
    return kmeans.labels_, matrix_reduce


def gcpca(network, chromsize, nc, res=1000000, pad=1, rp=0.5, prct=15, ndim=20, ncpus=10):
    matrix = []
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res) + 1
        start_time = time.time()
        result = []
        item = []
        for cell in network:
            args = cell, c, ngene, pad, rp
            item = impute_gc(args)
            result.append(item)
        index = {x[0]: j for j, x in enumerate(result)}
        Q_concat = np.array([result[index[x]][1] for x in network])
        print('adj closed!')
        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:, None])
        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        ndim = int(min(Q_concat.shape) * 0.15) - 1
        pca = PCA(n_components=ndim)
        R_reduce = pca.fit_transform(Q_concat)
        matrix.append(R_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    pca = PCA(n_components=min(matrix.shape) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, :ndim])
    return kmeans.labels_, matrix_reduce


def crpcaae(network, chromsize, nc,  res=1000000, pad=1, rp=0.5, prct=15, ndim=20, ncpus=10):
    matrix = []
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res) + 1
        start_time = time.time()
        result = []
        item = []
        for cell in network:
            args = cell, c, ngene, pad, rp
            item = impute_cpu(args)
            result.append(item)
        index = {x[0]: j for j, x in enumerate(result)}
        Q_concat = np.array([result[index[x]][1] for x in network])
        print('adj closed!')

        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:, None])


        ndim = int(min(Q_concat.shape) * 0.15) - 1
        pca = PCA(n_components=ndim)
        R_reduce = pca.fit_transform(Q_concat)
        print('pca->chromosome closed!')
        print(R_reduce.shape)

        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        matrix.append(R_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    matrix_reduce = train(matrix)
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, :ndim])
    # kmeans.labels_,
    return kmeans.labels_, matrix_reduce