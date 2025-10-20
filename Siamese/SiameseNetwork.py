# -*- coding: utf-8 -*-

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn

import torch.nn.functional as F
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # vgg16 = models.vgg16(pretrained=True)

        # 01 02 03 04
        # 修改第一层卷积层以适应1个输入通道
        # self.cnn1 = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道从3改为1
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 21x50 -> 10x25
        #
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 10x25 -> 5x12
        #
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # 5x12 -> 3x6
        #
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # 3x6 -> 2x3
        #
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)   # 2x3 -> 1x2
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512 * 1 * 2, 4096),  # 输入尺寸根据前面卷积层的输出尺寸调整
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 128)
        # )

        # 05
        # self.cnn1 = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道从3改为1
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 21x50 -> 10x25
        #
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 10x25 -> 5x12
        #
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)  # 5x12 -> 2x6
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512 * 2 * 6, 4096),  # 输入尺寸根据前面卷积层的输出尺寸调整
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 128)
        # )


        # self.cnn1 = nn.Sequential(
        #     nn.Conv2d(1, 3, kernel_size=3, padding=1),  # 将单通道变为三通道以适配VGG16
        #     vgg16.features,
        #     nn.Flatten(),
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 128)
        # )

        # 06 一维输入
        # self.cnn1 = nn.Sequential(
        #     nn.Conv1d(1, 64, kernel_size=3, padding=1),  # 输入通道从1改为64
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2),  # 1050 -> 525
        #
        #     nn.Conv1d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2),  # 525 -> 262
        #
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2)  # 262 -> 131
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512 * 131, 4096),  # 根据前面的卷积层输出形状调整输入尺寸
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 128)
        # )

        # # 07 一维输入
        # self.cnn1 = nn.Sequential(
        #     nn.Conv1d(1, 64, kernel_size=3, padding=1),  # 输入通道从1改为64
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2),  # 1050 -> 525
        #
        #     nn.Conv1d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2)  # 525 -> 262
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128 * 262, 512),  # 根据前面的卷积层输出形状调整输入尺寸
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 128)
        # )

        # 08 一维输入
        # self.cnn1 = nn.Sequential(
        #     nn.Conv1d(1, 64, kernel_size=3, padding=1),  # (1, 1050)   (21, 50)
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2),  # 1050 -> 525
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(64 * 525, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 128)
        # )


        # 09 一维输入
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),  # (1, 1050)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 1050 -> 525
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 525, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def eudience_distance(selfself, output1, output2):
        eudience_dis = torch.sum((output1 - output2) ** 2, dim=1) ** 0.5
        return eudience_dis

    def forward_once(self, x):
        output = self.cnn1(x)
        output = self.fc1(output)
        # output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # output = self.eudience_distance(output1, output2)
        # return output
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
