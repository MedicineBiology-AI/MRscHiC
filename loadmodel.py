import numpy as np
import torch
from flyamer_model import *
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import matplotlib.pyplot as plt
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def plot_loss(epoch_list, train_loss_list, path):
    plt.figure(figsize=(7, 5))
    plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train')
    plt.legend()
    plt.xlabel(u'Epoch')
    plt.ylabel(u'Loss')
    plt.title('Loss of Train')
    plt.savefig(path + 'loss_2.png', dpi=300)


def similarity_mat(label_pred):
    #    label_true=np.loadtxt('outputdata/flyamer_label_true.txt')
    label_pred = label_pred
    n1, = label_pred.shape
    #    label_true=label_true.reshape((n1,1))
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

#    hid1 = 900
#    hid2 = 500
#    outdim = 100
#    EPOCH = 800
#    LR = 4e-05
#    decay = 1e-05

    hid1 = 140
    hid2 = 100
    outdim = 30
    EPOCH = 300
    LR = 3e-05
    decay = 0.001
    

    autoencoder = AutoEncoder(ndim, outdim, hid1, hid2)
    autoencoder.load_state_dict(torch.load('save_model/flyamer_model3.pkl'))
    autoencoder.to(device)
    loss_func = nn.MSELoss()
    loss_list = []
    b_x = torch.from_numpy(matrix).unsqueeze(0).float().to(device)
    b_y = torch.from_numpy(matrix).unsqueeze(0).float().to(device)
    _, decoded = autoencoder(b_x)
    loss = loss_func(decoded, b_y)
#    print(loss.data.cpu().numpy())
    encoded_data, _ = autoencoder(torch.from_numpy(matrix).unsqueeze(0).float().to(device))
    time.sleep(0.1)
    Q = encoded_data.squeeze(0)
    Q = Q.detach().cpu().numpy()
    outdim1, outdim2 = Q.shape
    torch.cuda.empty_cache()
    return Q


if __name__ == "__main__":
    label = np.loadtxt('outputdata/ramani_label_true.txt')
    matrix = np.loadtxt('/mnt/d/jjpeng/cwzhen/data/AE/top_pca/ramani_top15_mat.txt')
    label_pred = KMeans(n_clusters=4, n_init=200).fit(matrix[:, :]).labels_
    ari0 = ARI(label, label_pred)
    print(matrix.shape)

    ari_list = []
    for i in range(1):
        mat_ae = train(matrix)
        #    np.savetxt('outputdata/ramain_pca2chr0ae2cell_mat.txt',mat_ae)

        label_pred = KMeans(n_clusters=4, n_init=200).fit(mat_ae[:, :]).labels_
        ari1 = ARI(label, label_pred)
        print('ari0: ', ari0)
        print('ari1: ', ari1)
        ari_list.append(ari1)

    ari_list = np.array(ari_list)
#    np.savetxt('ari/5ramani_lx_top15_pca_ae_ari_yellow.csv', ari_list, delimiter=",")
    ari_ave = ari_list.sum() / 1
    print('ari_ave: ', ari_ave)
#    np.savetxt('outputdata/ramain_pca2chr0ae2cell_label.txt', label_pred)

