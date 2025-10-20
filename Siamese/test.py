import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, \
    precision_recall_curve, auc, roc_curve
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from SiameseNetwork import SiameseNetwork, ContrastiveLoss
from SiameseDataset import SiameseDataset
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def save_list(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    feature_dir = '../all_dataset'
    negative_sample_ratio = 0.01
    file_idx_else = ['01']

    all_labels = []
    all_predictions = []
    all_scores = []
    total_loss = 0.0

    # net.load_state_dict(torch.load('../checkpoints/try/best_model.pth.tar'))
    checkpoint_path = '../10_folders/01_neuron_08.pth.tar'
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    net.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    siamese_dataset = SiameseDataset(feature_dir, negative_sample_ratio, file_idx_else, 'test')
    test_dataloader = DataLoader(siamese_dataset, shuffle=True)

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            feature1, feature2, label = data
            feature1, feature2, label = feature1.cuda(), feature2.cuda(), label.cuda()
            all_labels.extend(label.cpu().numpy())

            output1, output2 = net(feature1, feature2)
            loss_contrastive = criterion(output1, output2, label)
            total_loss += loss_contrastive

            # output1 = feature1
            # output2 = feature2

            euclidean_distance = F.pairwise_distance(output1, output2)
            all_scores.extend(euclidean_distance.cpu().numpy())
            predictions = (euclidean_distance >= 0.5).float().cpu().numpy()
            all_predictions.extend(predictions)

            # if label.cpu().numpy() == 1.0:
            #     print(f'{label.cpu().numpy()}:{euclidean_distance.float().cpu().numpy()}:{predictions}')

    print(f'数据集长度{len(siamese_dataset.pairs)}')
    cm = confusion_matrix(all_labels, all_predictions)
    TP, FN, FP, TN = cm.ravel()
    print(f'TN: {TN}\tFP: {FP}\tFN: {FN}\tTP: {TP}')

    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # f1_score = 2 * (precision * recall) / (precision + recall)
    # print(f"{accuracy}, {precision}, {recall}, {f1_score}")

    print(
        f'{len(siamese_dataset.pairs)}, {siamese_dataset.positive}, {siamese_dataset.negative},'
        f' {TP}, {TN}, {FP}, {FN}')

    # cm = confusion_matrix(all_labels, all_predictions)
    # TP, FN, FP, TN = cm.ravel()
    # print(f'TN: {TN}\tFP: {FP}\tFN: {FN}\tTP: {TP}')
    #
    # average_loss = total_loss / len(test_dataloader)
    # accuracy = accuracy_score(all_labels, all_predictions)
    # precision = precision_score(all_labels, all_predictions)
    # recall = recall_score(all_labels, all_predictions)
    # f1 = f1_score(all_labels, all_predictions)
    #
    # print(f"Test Loss: {average_loss}")
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")

    # TP = 0
    # FP = 0
    # TN = 0
    # FN = 0
    # # 遍历标签和预测值
    # for i in range(len(all_labels)):
    #     if all_labels[i] == 0 and all_predictions[i] == 0:
    #         TP += 1  # 真正例
    #     elif all_labels[i] == 0 and all_predictions[i] == 1:
    #         FN += 1  # 假负例
    #     elif all_labels[i] == 1 and all_predictions[i] == 0:
    #         FP += 1  # 假正例
    #     elif all_labels[i] == 1 and all_predictions[i] == 1:
    #         TN += 1  # 真负例
    # print(f'TN: {TN}\nFP: {FP}\nFN: {FN}\nTP: {TP}')

    # # 计算准确率
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # print(f"Accuracy: {accuracy}")
    # # 计算精确率
    # if TP + FP > 0:
    #     precision = TP / (TP + FP)
    # else:
    #     precision = 0
    # print(f"Precision: {precision}")
    # # 计算召回率
    # if TP + FN > 0:
    #     recall = TP / (TP + FN)
    # else:
    #     recall = 0
    # print(f"Recall: {recall}")
    # # 计算F1分数
    # if precision + recall > 0:
    #     f1_score = 2 * (precision * recall) / (precision + recall)
    # else:
    #     f1_score = 0
    # print(f"F1 Score: {f1_score}")


    # 手动设置阈值范围
    # min_th = min(all_scores)
    # max_th = max(all_scores)
    # print(f'min:{min_th}, max:{max_th}')
    # thresholds = np.linspace(min_th, max_th, num=100)
    # precisions = []
    # recalls = []
    # for threshold in thresholds:
    #     predictions = (all_scores > threshold).astype(int)
    #     cm = confusion_matrix(all_labels, predictions)
    #     TP, FN, FP, TN = cm.ravel()
    #     precision = TP / (TP + FP)
    #     recall = TP / (TP + FN)
    #     # precision = precision_score(all_labels, predictions)
    #     # recall = recall_score(all_labels, predictions)
    #     precisions.append(precision)
    #     recalls.append(recall)
    #     print(f'{threshold}, P: {precision}, R: {recall}')
    #     print(f'TN: {TN}\tFP: {FP}\tFN: {FN}\tTP: {TP}')
    #
    # # save_list(recalls, os.path.join('../exp/exp_00/neuron_16', 'recalls_neuron_16.pkl'))
    # # save_list(precisions, os.path.join('../exp/exp_00/neuron_16', 'precisions_neuron_16.pkl'))
    # pr_auc = auc(recalls, precisions)
    # # 绘制 Precision-Recall 曲线
    # plt.figure()
    # # plt.plot(recalls, precisions, marker='.')
    # plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.savefig(os.path.join('../10_folders', "01_neuron_08.png"))

    #
    # # 计算 Precision-Recall 曲线
    # p, r, thresholds = precision_recall_curve(all_labels, all_scores)
    # pr_auc = auc(r, p)
    #
    # # 绘制 PR 曲线
    # plt.figure()
    # plt.plot(r, p, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall curve')
    # plt.legend(loc="lower left")
    # plt.savefig("PR_all.png")

    # # 计算ROC曲线
    # fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    # # 绘制ROC曲线
    # plt.figure()
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve')
    # plt.text(0.5, 0.5, f'AUC = {auc}', horizontalalignment='center', verticalalignment='center', fontsize=15)
    # # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.fill_between(fpr, tpr, color='gray', alpha=0.3)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.savefig("AUC_all.png")

