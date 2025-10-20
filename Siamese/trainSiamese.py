import os
import pickle

import numpy as np
import torchinfo
from sklearn.metrics import confusion_matrix
from torch import optim
import torch
from torchinfo import summary
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from SiameseNetwork import SiameseNetwork, ContrastiveLoss
from SiameseDataset import SiameseDataset
import time

# matplotlib.use('TkAgg')

def save_checkpoint(state, is_best, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth.tar')
        torch.save(state, best_filepath)


def save_list(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def early_stopping(val_accs, patience=5):
    if len(val_accs) >= patience + 1:
        recent_accs = val_accs[-patience - 1:]
        if all(earlier >= later for earlier, later in zip(recent_accs, recent_accs[1:])):
            return True
    return False


if __name__ == '__main__':
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_batch_size = 256  # 训练时batch_size
    train_number_epochs = 10  # 训练的epoch

    feature_dir = '../all_dataset'
    train_negative_sample_ratio = 0.015
    val_negative_sample_ratio = 0.01
    train_file_idx_else = ['10', '09']
    val_file_idx_else = ['09']

    # train_dataset = SiameseDataset(feature_dir, train_negative_sample_ratio, train_file_idx_else, 'train')
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    # val_dataset = SiameseDataset(feature_dir, val_negative_sample_ratio, val_file_idx_else, 'test')
    # val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=train_batch_size)

    print(f"Loading Datasets: {time.time() - start_time:.4f} seconds")

    net = SiameseNetwork().cuda()
    print(net)
    x1 = torch.randn(1, 1, 1050)
    x2 = torch.randn(1, 1, 1050)
    torchinfo.summary(net, input_data=(x1, x2), device="cuda:0")

    '''
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    print(device)
    net.to(device)

    counter = []
    loss_history = []
    train_accs = []
    val_accs = []
    iteration_number = 0
    base_loss = 1
    patience = 2
    checkpoint_dir = '../10_folders/'

    for epoch in range(train_number_epochs):
        total_loss = 0
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            feature1, feature2, label = data
            feature1, feature2, label = feature1.to(device), feature2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = net(feature1, feature2)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            total_loss += loss_contrastive.item()

            # if i % 250 == 0 and i != 0:
            #     print(f"!!! Save plot: Epoch number {epoch}, Batch {i} Current loss {loss_contrastive.item()}")
            #     iteration_number += 250
            #     counter.append(iteration_number)
            #     loss_history.append(loss_contrastive.detach().cpu().numpy())
            #     save_list(counter, os.path.join(checkpoint_dir, 'counter.pkl'))
            #     save_list(loss_history, os.path.join(checkpoint_dir, 'loss_history.pkl'))
            # if loss_contrastive.item() < base_loss:
            #     base_loss = loss_contrastive.item()
            #     save_checkpoint({
            #         'epoch': epoch,
            #         'state_dict': net.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best=True, checkpoint_dir=checkpoint_dir, filename=f'epoch_{epoch}_loss_{base_loss}.pth.tar')
            #     print(
            #         f"!!! Save model: Epoch number {epoch}, Batch {i} Current loss {base_loss}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"\n-----------------------------------------\n"
              f"Epoch {epoch} average loss: {avg_loss}"
              f"\n-----------------------------------------\n")

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loss = 0.0
            val_labels = []
            val_score = []
            for i, data in enumerate(val_dataloader, 0):
                feature1, feature2, label = data
                feature1, feature2, label = feature1.to(device), feature2.to(device), label.to(device)
                val_labels.extend(label.cpu().numpy())

                output1, output2 = net(feature1, feature2)
                loss_contrastive = criterion(output1, output2, label)
                val_loss += loss_contrastive.item()

                euclidean_distance = F.pairwise_distance(output1, output2)
                val_score.extend(euclidean_distance.cpu().numpy())

        val_scores_temp = np.array(val_score)
        val_predictions = (val_scores_temp >= 1).astype(int)
        cm = confusion_matrix(val_labels, val_predictions)
        TP, FN, FP, TN = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        val_accs.append(accuracy)

        # save_checkpoint({
        #     'epoch': epoch,
        #     'state_dict': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, is_best=False, checkpoint_dir=checkpoint_dir, filename=f'{train_file_idx_else[0]}_neuron_08.pth.tar')
        #
        print(f"Epoch {epoch} average validation accuracy: {accuracy}"
              f"\n-----------------------------------------\n")

        if early_stopping(val_accs, patience=patience):
            print(f"Validation accuracy has not improved for {patience} epochs. Stopping training early.")
            break
        elif val_accs[-1] >= max(val_accs):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=True, checkpoint_dir=checkpoint_dir, filename=f'epoch_{epoch}_acc_{accuracy}.pth.tar')
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch} average validation loss: {avg_val_loss}"
              f"\n-----------------------------------------\n")

        print(f"Epoch {epoch}: {time.time() - start_time:.4f} seconds")

    # fig = plt.plot(counter, loss_history)
    # # plt.show()
    # plt.savefig(os.path.join(checkpoint_dir, 'loss_neuron_16.png'))
    # plt.show()
    '''