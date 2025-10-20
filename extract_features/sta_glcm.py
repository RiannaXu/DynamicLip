# 输入是单张图片
# 5 * num * 6

import numpy as np
import cv2
import dlib
import pandas as pd
from skimage.transform import radon
from skimage.feature import graycomatrix, graycoprops

from sklearn.decomposition import PCA

import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

# 读取图像并进行颜色转换
def read_and_convert_image(image):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr_image[:, :, 0]
    # print("YCR颜色转换...")
    return y_channel


# 图像对比度拉伸
def contrast_stretch(image, gamma=1.0):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = 255 * ((image - min_val) / (max_val - min_val)) ** gamma
    # print("对比度拉伸...")
    return stretched.astype(np.uint8)


def Regional_Division_TO_6(image):

    # 读取输入图像并转换为灰度图

    # image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    # 检测面部区域
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        # 获取唇部特征点
        points = []
        for i in range(68):  # 68点模型中，48-67是唇部特征点
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append((x, y))

        # 定义唇部的6个区域（UL, UR, LL, LR, UM, LM）
        regions = {
            'UL': [points[i] for i in [48, 49, 50, 61, 60]],
            'UM': [points[i] for i in [50, 51, 52, 63, 62, 61]],
            'UR': [points[i] for i in [52, 53, 54, 64, 63]],
            'LL': [points[i] for i in [48, 60, 67, 58, 59]],
            'LM': [points[i] for i in [56, 57, 58, 67, 66, 65]],
            'LR': [points[i] for i in [54, 55, 56, 65, 64]],
        }

        # for point in points:
        #     cv2.circle(image, point, 3, (0, 0, 255), -1)
        # cv2.imwrite("all_point.png", image)
        rois = []
        # 为每个区域创建蒙版并保存图像
        for region_name, region_points in regions.items():
            region_points = np.array(region_points, np.int32)
            mask = np.zeros_like(gray)

            cv2.fillPoly(mask, [region_points], 255)

            # 提取ROI并将其他部分设为背景
            region_image = cv2.bitwise_and(image, image, mask=mask)
            background = np.full_like(image, 255)  # 白色背景
            mask_inv = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(background, background, mask=mask_inv)

            final_image = cv2.add(region_image, background)

            # 提取实际唇部区域并保存
            x, y, w, h = cv2.boundingRect(region_points)
            roi = final_image[y:y + h, x:x + w]
            rois.append(roi)
            # cv2.imwrite(f'{region_name}.png', roi)

    # 显示原始图像和区域图像
    # cv2.imshow("Original Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("划分为6个区域...")
    return rois


# 可转向滤波器
def steerable_filters(image, orientations=8, region_name=''):
    # 定义高斯函数及其二阶导数
    sigma = 1.0
    size = int(2 * np.ceil(3 * sigma) + 1)

    def gaussian(x, y, sigma):
        return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    def d2_gaussian_dx2(x, y, sigma):
        return (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * gaussian(x, y, sigma)

    def d2_gaussian_dxy(x, y, sigma):
        return (x * y / sigma ** 4) * gaussian(x, y, sigma)

    def d2_gaussian_dy2(x, y, sigma):
        return (y ** 2 / sigma ** 4 - 1 / sigma ** 2) * gaussian(x, y, sigma)

    x = np.arange(-size // 2 + 1, size // 2 + 1)
    y = np.arange(-size // 2 + 1, size // 2 + 1)
    X, Y = np.meshgrid(x, y)

    gxx = d2_gaussian_dx2(X, Y, sigma)
    gxy = d2_gaussian_dxy(X, Y, sigma)
    gyy = d2_gaussian_dy2(X, Y, sigma)
    steerable_images = []

    for t in range(orientations):
        theta = np.pi * t / orientations
        g_theta = (gxx * np.cos(theta) ** 2 - 2 * gxy * np.sin(theta) * np.cos(theta) + gyy * np.sin(theta) ** 2)
        filtered_image = cv2.filter2D(image, -1, g_theta)
        # theta_str = f"{theta:.2f}"
        # pic_name = f"{region_name}_{theta_str}_image.png"
        # full_path = os.path.join("H:\STUDY\PG\sensing\lip-print\code\pic\\two", pic_name)
        # cv2.imwrite(full_path, filtered_image)
        steerable_images.append(filtered_image)

    steerable_images = np.array(steerable_images)
    # combined_image = np.sum(steerable_images ** 2, axis=0) ** 0.5
    # cv2.imwrite('combined_image.png', combined_image)
    return steerable_images


# 计算共生矩阵特征
def co_occurrence_features(images):
    distances = [5]
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    sam_roi = []
    contrast_roi = []
    correlation_roi = []
    homogeneity_roi = []
    entropy_roi = []
    for image in images:
        glcm = graycomatrix(image, distances, angles, 256, symmetric=True, normed=True)

        # 计算二阶矩（Second Angular Moment）
        sam = graycoprops(glcm, 'ASM').mean()
        sam_roi.append(sam)
        # 计算对比度（Contrast）
        contrast = graycoprops(glcm, 'contrast').mean()
        contrast_roi.append(contrast)
        # 计算相关性（Correlation）
        correlation = graycoprops(glcm, 'correlation').mean()
        correlation_roi.append(correlation)
        # 计算逆差距矩（Inverse Differential Moment）
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        homogeneity_roi.append(homogeneity)
        # 计算熵（Entropy）
        entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
        entropy_roi.append(entropy)
    return {
        'Second Angular Moment': sam_roi,
        'Contrast': contrast_roi,
        'Correlation': correlation_roi,
        'Inverse Differential Moment': homogeneity_roi,
        'Entropy': entropy_roi
    }


def Feature_Steerable_and_Radon(image):
    y_channel = read_and_convert_image(image)
    stretched_image = contrast_stretch(y_channel)
    # cv2.imwrite('y_channel.png', y_channel)
    # cv2.imwrite('stretched_image.png', stretched_image)
    rois = Regional_Division_TO_6(stretched_image)
    co_occurrences = []
    radons_features = []
    n = 1
    for region_name, roi in zip(['UL', 'UM', 'UR', 'LL', 'LM', 'LR'], rois):
        steerable_response = steerable_filters(roi, region_name=region_name)
        # 从单精度浮点数（即float32）转换为uint8
        steerable_response = steerable_response * 255
        output1 = steerable_response.astype(np.uint8)
        co_occurrence = co_occurrence_features(output1)
        co_occurrences.append(co_occurrence)
        # print(f"Region {region_name}:")
        # for feature_name, value in co_occurrence.items():
        #     print(f"{feature_name}: {value}")
        # print("Radon Transform Features:", radon_features)
        n += 1
    # roi = rois[0]
    # steerable_response = steerable_filters(roi)
    # steerable_response = steerable_response*255
    # output1 = steerable_response.astype(np.uint8)
    # co_occurrence = co_occurrence_features(output1)
    # for feature_name, value in co_occurrence.items():
    #     print(f"{feature_name}: {value}")
    # radon_features = radon_transform_features(output1)
    # print("Radon Transform Features:", radon_features)
    return co_occurrences


def extract_GLCM(image, components_num=2):
    co_occurrences = Feature_Steerable_and_Radon(image)
    reductions = []

    for i in range(len(co_occurrences)):
        # 对每个特征进行归一化
        normalized_data = {}
        for key, values in co_occurrences[i].items():
            min_val = min(values)
            max_val = max(values)
            normalized_data[key] = [(x - min_val) / (max_val - min_val) for x in values]
        df_normalized = pd.DataFrame(normalized_data)
        try_matrix = df_normalized.T.values

        # 用PCA进行主成份分析
        clf = PCA(n_components=components_num)  # 初始化PCA对象
        clf.fit(try_matrix)  # 对X进行主成份分析

        reduction = clf.transform(try_matrix)
        reduction = reduction.flatten()

        # print('降维后的结果', clf.transform(try_matrix))  # 降维后的结果
        # print('各主成分的方差值', clf.explained_variance_)  # 降维后的各主成分的方差值
        # print('各主成分的方差值占总方差值的比例', clf.explained_variance_ratio_)
        # sum111 = sum(clf.explained_variance_ratio_)
        # print('sum', sum111)

        reductions.extend(reduction)

    # print(len(np.array(reductions)))

    return np.array(reductions)


if __name__ == '__main__':

    # image_path = "../image/TRY.png"
    # image = cv2.imread(image_path)

    video_path = '../videos/try1.MP4'
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()

    re = extract_GLCM(image)
    print(re)
