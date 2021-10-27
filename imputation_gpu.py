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
from train import *

def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)

def neighbor_ave_gpu(A, pad):
    if pad==0:
        return torch.from_numpy(A).float().cuda()
    ll = pad * 2 + 1
    conv_filter = torch.ones(1, 1, ll, ll).cuda()
    B = F.conv2d(torch.from_numpy(A[None, None,: ,:]).float().cuda(), conv_filter, padding = pad * 2)
    return (B[0, 0 ,pad:-pad, pad:-pad] / float(ll * ll))

def random_walk_gpu(A, rp):
    ngene, _ = A.shape
    A = A - torch.diag(torch.diag(A))
#    A = A + torch.diag(torch.sum(A, 0) == 0).float()
    P = torch.div(A, torch.sum(A, 0))
    Q = torch.eye(ngene).cuda()
    I = torch.eye(ngene).cuda()
    for i in range(30):
        Q_new = (1 - rp) * I + rp * torch.mm(Q, P)
        delta = torch.norm(Q - Q_new, 2)
        Q = Q_new
        if delta < 1e-6:
            break
    return Q

def impute_gpu(args):
    cell, c, ngene, pad, rp = args
    D = np.loadtxt(cell + '_chr' + c + '.txt')
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape = (ngene, ngene)).toarray()
    A = np.log2(A + A.T + 1)
    #A = neighbor_ave_gpu(A, pad)
    if rp==-1:
        Q = A[:]
    else:
        A = neighbor_ave_gpu(A, pad)
        Q = random_walk_gpu(A, rp)
        adj_norm = normalize(torch.FloatTensor(Q), True)
        features = torch.FloatTensor(Q)
        ax1 = adj_norm.mm(features)
        ax1 = ax1.numpy()
    return ax1.reshape(ngene*ngene)

def dataprocess_gpu(network, chromsize, nc, res=1000000, pad=1, rp=0.5, prct=20, ndim=20):
    matrix=[]
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res)+1
        start_time = time.time()
        Q_concat = torch.zeros(len(network), ngene * ngene).float().cuda()
        for j, cell in enumerate(network):
            Q_concat[j] = impute_gpu([cell, c, ngene, pad, rp])
        Q_concat = Q_concat.cpu().numpy()
        print('adj closed!')
        if prct>-1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
        Q_concat = (Q_concat > thres[:, None])
        R_reduce = train(Q_concat,c)
        print('AE->chromosome closed!')
        print(R_reduce.shape)
        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        # ndim = int(min(Q_concat.shape) * 0.2) - 1
        # # U, S, V = torch.svd(Q_concat, some=True)
        # # R_reduce = torch.mm(U[:, :ndim], torch.diag(S[:ndim])).cuda().numpy()
        # pca = PCA(n_components = ndim)
        # R_reduce = pca.fit_transform(Q_concat)
        matrix.append(R_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis = 1)
    matrix = np.concatenate(matrix, axis=1)
    matrix_reduce=train(matrix,c)
    # pca = PCA(n_components = min(matrix.shape) - 1)
    # matrix_reduce = pca.fit_transform(matrix)
    kmeans = KMeans(n_clusters = nc, n_init = 200).fit(matrix_reduce[:, :ndim])
    return kmeans.labels_, matrix_reduce