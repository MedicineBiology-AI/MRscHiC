import numpy as np
import torch
from model import *
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


def train(matrix,EPOCH,LR,decay,hid1,hid2,outdim):
    device = torch.device("cuda:0")
    dimm, dimension = matrix.shape
    ndim = dimension
    # outdim = int(ndim * 0.2)
    # hid1 = 1000
    # hid2 = 500
    # outdim = 90
    # EPOCH = 700
    # if (dimension == 23 * 32):
    #     hid1 = 1000
    #     hid2 = 300
    #     outdim = 32
    #     EPOCH = 500
    # LR = 0.0007

    autoencoder = AutoEncoder(ndim, outdim, hid1, hid2)
    autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=decay)
    # optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)
    loss_func = nn.MSELoss()
    #    loss_func=loss_func
    loss_list = []
    # matrix=matrix.cuda()

    for epoch in range(EPOCH):
        b_x = torch.from_numpy(matrix).unsqueeze(0).float().to(device)
        b_y = torch.from_numpy(matrix).unsqueeze(0).float().to(device)
        _, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())

    #    chr=str(c)
    #    epoch_list=np.arange(1,EPOCH+1)
    #    path='/Users/zhencaiwei/PycharmProjects/pythonProject/Ramani/picture/'+chr
    # plot_loss(epoch_list,loss_list,path)
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
    ari_max=0
    ari_max_string=''
    ari_max_list=[]
    for lr in [0.0001,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009]:
        for epoch in [500,550,600,650,700,750,800]:
            for decay in [0.00001,0.0001,0.001,0]:
                for hid1 in [1000,900,800]:
                    for hid2 in [600,500,400,300]:
                        for outdim in[100,80,60,40,30,20]:
                            ari_list = []
                            for i in range(5):
                                mat_ae = train(matrix, epoch, lr, decay, hid1, hid2, outdim)
                                label_pred = KMeans(n_clusters=4, n_init=200).fit(mat_ae[:, :]).labels_
                                ari1 = ARI(label, label_pred)
                                print('ari0: ', ari0)
                                print('ari1: ', ari1)
                                ari_list.append(ari1)
                            ari_list = np.array(ari_list)
                            ari_ave = ari_list.sum() / 5
                            if (ari_max < ari_ave):
                                ari_max = ari_ave
                                ari_max_string = 'lr: ' + str(lr) + 'epoch: ' + str(epoch) + 'decay: ' + str(decay)+\
                                                 'hid1: '+str(hid1)+'hid2: '+str(hid2)+'outdim: '+str(outdim)
                                ari_max_list = ari_list


    ari_max_list = np.array(ari_max_list)
    np.savetxt('ari/5ramani_lx_top15_pca_ae_ari_run3.csv', ari_max_list, delimiter=",")
    print('ari_max: ', ari_max)
    print('ari_max_string: ',ari_max_string)
    # for i in range(20):
    #     mat_ae = train(matrix)
    #     #    np.savetxt('outputdata/ramain_pca2chr0ae2cell_mat.txt',mat_ae)
    #
    #     label_pred = KMeans(n_clusters=4, n_init=200).fit(mat_ae[:, :]).labels_
    #     ari1 = ARI(label, label_pred)
    #     print('ari0: ', ari0)
    #     print('ari1: ', ari1)
    #     ari_list.append(ari1)
    #
    # ari_list = np.array(ari_list)
    # np.savetxt('ari/20ramani_lx_top15_pca_ae_ari_test32.csv', ari_list, delimiter=",")
    # ari_ave = ari_list.sum() / 20
    # print('ari_ave: ', ari_ave)
#    np.savetxt('outputdata/ramain_pca2chr0ae2cell_label.txt', label_pred)

