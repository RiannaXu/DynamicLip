import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import numpy as np
import os

from itertools import combinations


class SiameseDataset(Dataset):
    def __init__(self, feature_dir, negative_sample_ratio, file_idx_else, split='train'):
        self.feature_dir = feature_dir
        self.negative_sample_ratio = negative_sample_ratio

        print(f'rate: {self.negative_sample_ratio}')

        if len(file_idx_else) == 2:
            self.file_idx_else1 = file_idx_else[0]
            self.file_idx_else2 = file_idx_else[1]
        elif len(file_idx_else) == 1:
            self.file_idx_else1 = file_idx_else[0]
        self.split = split
        # if self.split == 'test':
        #     self.negative_sample_ratio = 0.2

        self.data_folders = self._get_dirs()

        self.pairs = self._generate_pairs()

    def _get_dirs(self):
        file_dirs = os.listdir(self.feature_dir)
        data_dirs = []
        for dir in file_dirs:
            temp = os.path.join(self.feature_dir, dir)
            for size in ['large', 'middle', 'small']:
                data_dirs.append(os.path.join(temp, size))
        return data_dirs

    def _load_features(self, folder):
        idx = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        datas = []

        if self.split == 'test':
            split_folder = os.path.join(folder, self.file_idx_else1)
            # print(split_folder)
            for filename in os.listdir(split_folder):
                if filename.endswith('.npy'):
                    filepath = os.path.join(split_folder, filename)
                    feature = np.load(filepath).flatten()
                    if len(feature) == 1050:
                        datas.append(feature)
            # datas = [np.load(os.path.join(split_folder, f)).flatten() for f in os.listdir(split_folder) if
            #          f.endswith('.npy')]
        elif self.split == 'train':
            for id in idx:
                if id != self.file_idx_else1 and id != self.file_idx_else2:
                    split_folder = os.path.join(folder, id)
                    for filename in os.listdir(split_folder):
                        if filename.endswith('.npy'):
                            filepath = os.path.join(split_folder, filename)
                            feature = np.load(filepath).flatten()
                            if len(feature) == 1050:
                                datas.append(feature)
                    # print(split_folder)
                    # data.append([np.load(os.path.join(split_folder, f)).flatten() for f in os.listdir(split_folder) if
                    #              f.endswith('.npy')])
            # datas = [ndarray for sublist in data for ndarray in sublist]

        # 归一化
        normalized_datas = []
        for row in datas:
            # 找到两个最大值的索引
            max_indices = np.argpartition(row, -2)[-2:]
            # 将这两个索引对应的值置为0
            row[max_indices] = 0

            part1 = row[:502]
            part2 = row[502:562]
            part3 = row[562:942]
            part4 = row[942:]

            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            scaler3 = MinMaxScaler()
            scaler4 = MinMaxScaler()

            # 逐段归一化
            part1 = scaler1.fit_transform(part1.reshape(-1, 1)).flatten()
            part2 = scaler2.fit_transform(part2.reshape(-1, 1)).flatten()
            part3 = scaler3.fit_transform(part3.reshape(-1, 1)).flatten()
            part4 = scaler4.fit_transform(part4.reshape(-1, 1)).flatten()

            # 合并归一化后的各部分
            normalized_row = np.concatenate((part1, part2, part3, part4))
            normalized_datas.append(normalized_row)

        # 不要形状特征
        # for i in range(len(datas)):
        #     datas[i] = datas[i][:562]
        return normalized_datas

    def _generate_pairs(self):
        pairs = []
        labels = []
        all_data = {}

        for folder in self.data_folders:
            all_data[folder] = self._load_features(folder)

        folder_pairs = list(combinations(self.data_folders, 2))

        # 正样本 1: 同一个文件中的特征对
        for folder, data in all_data.items():
            print(f'正：{folder}')
            for pair in combinations(data, 2):
                pairs.append((pair[0], pair[1], 0))
        self.positive = len(pairs)
        print(f'正样本数：{ self.positive}\n\n')

        # 负样本 -1：跨文件夹的特征对
        neg_filtered_pairs = []
        for temp in folder_pairs:
            # print(f'temp:{temp}')
            # temp1 = temp[0].split('\\')
            temp1 = temp[0].split('/')  # linux temp:('../all_dataset/data_00/large', '../all_dataset/data_00/middle')
            # temp2 = temp[1].split('\\')
            temp2 = temp[1].split('/')
            if (temp1[2] == temp2[2] and temp1[3] != temp2[3]) or (temp1[2] != temp2[2] and temp1[3] == temp2[3]):
            # if (temp1[1] == temp2[1] and temp1[2] != temp2[2]) or (temp1[1] != temp2[1] and temp1[2] == temp2[2]):
                neg_filtered_pairs.append(temp)
                neg_filtered_pairs.append((temp[1], temp[0]))
        print(f'文件夹数量: {len(neg_filtered_pairs)}')
        step = 1
        sample_size = 0
        for p in neg_filtered_pairs:
            print(f'\t{sample_size} 负样本{step}: {p}')
            data_1 = all_data[p[0]]
            data_2 = all_data[p[1]]
            neg_pairs = self._generate_negative_pairs(data_1, data_2)
            if len(neg_pairs) == 0:
                continue
            # print(f"Length of neg_pairs: {len(neg_pairs)}")
            sample_size = int(len(neg_pairs) * self.negative_sample_ratio)
            sampled_indices = np.random.choice(len(neg_pairs), sample_size, replace=False)
            sampled_pairs = [neg_pairs[i] for i in sampled_indices]
            for pair in sampled_pairs:
                pairs.append((pair[0], pair[1], 1))
            step += 1

        self.negative = len(pairs) - self.positive
        # negative = index - positive
        print(f'正样本数：{ self.positive}   负样本数：{ self.negative}')
        # pairs = pairs[:index]
        # labels = labels[:index]
        print(f'总样本数：pairs {len(pairs)} ')

        # for pair in snt(pair[0], pair[1], pair[2])

        return pairs

    def _generate_negative_pairs(self, data_1, data_2):
        pairs = []
        for a in data_1:
            for b in data_2:
                pairs.append((a, b))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        feature1 = torch.tensor(pair[0], dtype=torch.float32).view(1, -1)
        feature2 = torch.tensor(pair[1], dtype=torch.float32).view(1, -1)
        label = torch.tensor(pair[2], dtype=torch.float32).view(-1)

        # pair = self.pairs[idx]
        # feature1 = torch.tensor(pair[0], dtype=torch.float32).view(1, 21, 50)
        # feature2 = torch.tensor(pair[1], dtype=torch.float32).view(1, 21, 50)
        # label = torch.tensor(pair[2], dtype=torch.float32).view(1)
        return feature1, feature2, label


if __name__ == '__main__':
    file_idx_else = '01'
    spilt = 'train'
    test_dataset = SiameseDataset('../try_dataset', 0.1, file_idx_else=file_idx_else, split=spilt)
