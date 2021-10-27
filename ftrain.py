import numpy as np
import torch
from flyamer_model import *
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import matplotlib.pyplot as plt
import time
import os
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def plot_loss(epoch_list, train_loss_list, path):
    plt.figure(figsize=(7, 5))
    plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train')
    plt.legend()
    plt.xlabel(u'Epoch')
    plt.ylabel(u'Loss')
    plt.title('Loss of Train')
    plt.savefig(path + 'loss_2.png', dpi=300)


def similarity_mat(label_pred):

    label_pred = label_pred
    n1, = label_pred.shape

    label_pred = label_pred.reshape((n1, 1))
    sim_mat_pred = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n1):
            if (label_pred[i][0] == label_pred[j][0]):
                sim_mat_pred[i][j] = 1

    return sim_mat_pred


def train(matrix):
    device = torch.device("cuda:0")
    dimm, dimension = matrix.shape
    ndim = dimension

    hid1 = 220
    hid2 = 50
    outdim = 16
    EPOCH = 500
    LR = 0.0001
    decay = 0



    autoencoder = AutoEncoder(ndim, outdim, hid1, hid2)
    autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=decay)

    loss_func = nn.MSELoss()


    for epoch in range(EPOCH):
        b_x = torch.from_numpy(matrix).unsqueeze(0).float().to(device)
        b_y = torch.from_numpy(matrix).unsqueeze(0).float().to(device)
        _, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())


    encoded_data, _ = autoencoder(torch.from_numpy(matrix).unsqueeze(0).float().to(device))

    time.sleep(0.1)
    Q = encoded_data.squeeze(0)
    Q = Q.detach().cpu().numpy()
    outdim1, outdim2 = Q.shape
    torch.cuda.empty_cache()
    return Q


if __name__ == "__main__":
    label = np.loadtxt('outputdata/flyamer_label_true.txt')
    matrix = np.loadtxt('top_pca/flyamer_top15_mat.txt')
    label_pred = KMeans(n_clusters=3, n_init=200).fit(matrix[:, :]).labels_
    ari0 = ARI(label, label_pred)
    print(matrix.shape)

    ari_list = []
    for i in range(10):
        mat_ae = train(matrix)


        label_pred = KMeans(n_clusters=3, n_init=200).fit(mat_ae[:, :]).labels_

        ari1 = ARI(label, label_pred)
        print('ari0: ', ari0)
        print('ari1: ', ari1)
        ari_list.append(ari1)

    ari_list = np.array(ari_list)

    ari_ave = ari_list.sum() / 10
    print('are_list:',ari_list)
    print('ari_ave: ', ari_ave)
    


