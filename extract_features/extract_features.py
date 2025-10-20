import os
import time
import warnings
import cv2
import numpy as np

from dyn_correlation import extract_correlation
from sta_glcm import extract_GLCM
from sta_shape import extract_shape
from dyn_texture import extract_texture

# 忽略所有的警告
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="iCCP: known incorrect sRGB profile")
warnings.filterwarnings('ignore',
                        message="RankWarning: Polyfit may be poorly conditioned poly = np.polyfit(x, y, degree)")


def get_all_features(video_path, start_frame, lip_width=250, lip_height=110, video_length=50, interval=5,
                     components_num=2):
    features = []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret is not None:
        feature_shape = extract_shape(frame, lip_width, lip_height)# width * 2 + 2
        shape_count = 0
        while feature_shape is None:
            ret, frame = cap.read()
            if ret is not None:
                feature_shape = extract_shape(frame, lip_width, lip_height)
            shape_count += 1
            if shape_count >= 10:
                return None
        features.extend(feature_shape)
        print(f'\t\tshape')
        feature_glcm = extract_GLCM(frame, components_num)  # 5 * num * 6
        features.extend(feature_glcm)
        print(f'\t\tglcm')
        cap.release()

        feature_correlation = extract_correlation(video_path, start_frame, video_length, interval)
        if feature_correlation is None:
            print('feature is None')  # 190 * 2
            return None
        features.extend(feature_correlation)
        print(f'\t\tcorrelation')

        features_texture = extract_texture(video_path, start_frame, video_length, interval)
        if features_texture is None:
            return None
        features.extend(features_texture)
        print(f'\t\ttexture')
    else:
        cap.release()

    return np.array(features)


def save_as_npy(arr, output_dir, output_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_name, arr)
    print(f'\t保存{output_name}成功')


def get_videos_features(file_path, output_dir, lip_width=250, lip_height=110, video_length=50, interval=5,
                        components_num=2):
    videos_name = sorted(os.listdir(file_path))
    # print(videos_name)

    for v_name in videos_name:

        video_path = os.path.join(file_path, v_name)  # xxxxxx/xxx.mp4
        print(f'\n\n正在处理视频{video_path}......')

        video_name = v_name.split('.')[0]  # xxx
        video_spilt = video_name.split('_')
        temp_dir = os.path.join(output_dir, f'{video_spilt[0]}_{video_spilt[1]}')

        if video_spilt[3] == 's':
            result_dir = os.path.join(temp_dir, 'small')
        elif video_spilt[3] == 'm':
            result_dir = os.path.join(temp_dir, 'middle')
        elif video_spilt[3] == 'l':
            result_dir = os.path.join(temp_dir, 'large')

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(frame_count, video_length * 30)  # 提取30个单词
        # end_frame = min(frame_count, video_length * 4)  # 只提取四个单词
        frame_indices = np.arange(0, end_frame, 50)  # 每个时间间隔的起点
        # print(frame_indices)

        for i in frame_indices:
            print(f'\t第{i}——{i + 49}帧.....')
            word_idx = i / 50 + 1
            # print(f'\t\t{int(word_idx):02d}')

            # def get_all_features(video_path, start_frame, lip_width=150, lip_height=75, video_length=100,
            # interval=4, components_num=2):
            start_time = time.time()
            feature = get_all_features(video_path, i, lip_width, lip_height, video_length, interval, components_num)
            if feature is None:
                continue

            # ../dataset\data_00\data_00_0_l_word01.npy
            output_name = os.path.join(result_dir, f'{video_name}_word{int(word_idx):02d}.npy')
            print(f'\t\t{output_name}')
            save_as_npy(feature, result_dir, output_name)
            end_time = time.time()
            run_time = end_time - start_time
            print("\t\t代码执行时间为：%s秒" % run_time)


if __name__ == '__main__':
    file_path = r'J:\LipPrint\14_distance2'
    output_dir = r'E:\研究生\研究所\deepfakes\lip print\FeatureExtraction\dataset\14_distance'

    lip_width = 250
    lip_height = 110
    video_legth = 50
    components_num = 2
    interval = 5

    get_videos_features(file_path, output_dir, lip_width, lip_height, video_legth, interval, components_num)

    # print(feature)
    # print(len(feature))


    # video_path = r'H:\10_device_rear\data_10_0_s.mp4'
    # video_name = 'data_10_0_m'
    # result_dir = '../dataset/10_device_rear/data_10/small'
    # start = 0
    # end = 1500
    # frame_indices = np.arange(start, end, 50)
    # for i in frame_indices:
    #     print(f'\t第{i}——{i + 49}帧.....')
    #     word_idx = i / 50 + 1
    #     # print(f'\t\t{int(word_idx):02d}')
    #
    #     # def get_all_features(video_path, start_frame, lip_width=150, lip_height=75, video_length=100,
    #     # interval=4, components_num=2):
    #     start_time = time.time()
    #     feature = get_all_features(video_path, i, 250, 110, 50, 5, 2)
    #     if feature is None:
    #         continue
    #
    #     # ../dataset\data_00\data_00_0_l_word01.npy
    #     output_name = os.path.join(result_dir, f'{video_name}_word{int(word_idx):02d}.npy')
    #     print(f'\t\t{output_name}')
    #     save_as_npy(feature, result_dir, output_name)
    #     end_time = time.time()
    #     run_time = end_time - start_time
    #     print("\t\t代码执行时间为：%s秒" % run_time)

