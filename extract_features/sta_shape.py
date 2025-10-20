# 输入是单张图片
# width*2 + 2

import cv2
import dlib
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')


def extract_lip_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        lip = []
        for i in range(48, 68):
            lip.append((landmarks.part(i).x, landmarks.part(i).y))  # (x,y)是坐标
        return np.array(lip)
    return None


def fit_curve_segment(points, degree=3, num_points=50):
    x = points[:, 0]
    y = points[:, 1]
    poly = np.polyfit(x, y, degree)
    x_vals = np.linspace(min(x), max(x), num=num_points)
    y_vals = np.polyval(poly, x_vals)
    return np.column_stack((x_vals, y_vals))


def fit_lip_contour(lip_landmarks):
    # upper_lip_indices = [0, 1, 2, 3, 4, 5, 6, 16, 15, 14, 13, 12, 0]
    # lower_lip_indices = [6, 7, 8, 9, 10, 11, 0, 12, 19, 18, 17, 16, 6]
    upper_lip_indices = [0, 1, 2, 3, 4, 5, 6, 15, 14, 13, 0]
    lower_lip_indices = [6, 7, 8, 9, 10, 11, 0, 19, 18, 17, 6]

    upper_lip = lip_landmarks[upper_lip_indices]
    lower_lip = lip_landmarks[lower_lip_indices]

    upper_curve = []
    lower_curve = []
    upper_curve.append(fit_curve_segment(upper_lip[0:3], degree=3, num_points=20))
    upper_curve.append(fit_curve_segment(upper_lip[2:5], degree=3, num_points=10))
    upper_curve.append(fit_curve_segment(upper_lip[4:7], degree=3, num_points=20))
    temp = fit_curve_segment(upper_lip[6:], degree=5, num_points=50)  # 翻转
    temp1 = np.flipud(temp)
    upper_curve.append(temp1)

    lower_curve.append(fit_curve_segment(lower_lip[:7], degree=5, num_points=50))
    temp = fit_curve_segment(lower_lip[7:], degree=5, num_points=50) # 翻转
    temp1 = np.flipud(temp)
    lower_curve.append(temp1)


    upper_curve = np.vstack(upper_curve)
    lower_curve = np.vstack(lower_curve)

    return upper_curve, lower_curve


def draw_lip_contour(image, upper_curve, lower_curve):
    for (x, y) in np.vstack(lower_curve):
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    for (x, y) in np.vstack(upper_curve):
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
    cv2.namedWindow("Lip Contour", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow('Lip Contour', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


def create_lip_mask(image, upper_curve, lower_curve):
    mask = np.zeros_like(image)
    upper_curve = np.array(upper_curve, dtype=np.int32)
    lower_curve = np.array(lower_curve, dtype=np.int32)

    cv2.fillPoly(mask, [upper_curve], (255, 255, 255))
    cv2.fillPoly(mask, [lower_curve], (255, 255, 255))
    return mask


def mask_lips(image, upper_curve, lower_curve):
    mask = create_lip_mask(image, upper_curve, lower_curve)
    lips = cv2.bitwise_and(image, mask)
    white_background = np.ones_like(image) * 255
    white_background[mask == 255] = lips[mask == 255]
    return white_background


def draw_points(image, keypoints):
    if keypoints is not None:
        for (x, y) in keypoints:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            # 显示结果图像
        cv2.namedWindow("Lip Keypoints", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('Lip Keypoints', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def count_contour(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0
    for i in range(len(contours)):
        perimeter += cv2.arcLength(contours[i], True)
    return perimeter


def count_area(image):
    white_area = np.sum(image == 255)
    return white_area


def count_thickness(image):
    upper = []
    lower = []
    for col in range(image.shape[1]):
        col_data = image[:, col]
        white_pixels = np.where(col_data == 255)[0]
        if white_pixels.size > 0:
            data = white_pixels
            # 计算不连续部分的长度
            discontinuities = []
            start_idx = 0
            for i in range(1, len(data)):
                if data[i] != data[i - 1] + 1:
                    length = data[i - 1] - data[start_idx] + 1
                    discontinuities.append(length)
                    start_idx = i
            if start_idx < len(data):
                length = data[-1] - data[start_idx] + 1
                discontinuities.append(length)
            # 录入上下唇
            if len(discontinuities) == 2:
                upper.append(discontinuities[0])
                lower.append(discontinuities[1])
            else:
                upper.append(discontinuities[0])
        else:
            continue
    return upper, lower


def padding_zero(image, arr):
    lower_len = len(arr)
    padding = image.shape[1] - lower_len
    padding_left = round(padding / 2)
    padding_right = padding - padding_left
    for i in range(padding_left):
        arr.insert(0, 0)
    for i in range(padding_right):
        arr.append(0)
    return arr


def extract_shape(image, width, heigh):
    # new_size = (heigh, width)
    # 提取唇部特征点
    lip_landmarks = extract_lip_landmarks(image)
    if lip_landmarks is None:
        return None
    # 求上下左右边界
    x_min = np.min(lip_landmarks[:, 0])
    x_max = np.max(lip_landmarks[:, 0])
    y_min = np.min(lip_landmarks[:, 1])
    y_max = np.max(lip_landmarks[:, 1])
    # draw_points(image.copy(), lip_landmarks)

    if lip_landmarks is not None:
        # 二项式拟合
        upper_curve, lower_curve = fit_lip_contour(lip_landmarks)
        # image_with_contour = draw_lip_contour(image.copy(), upper_curve, lower_curve)
        # 提取、裁剪出只有嘴唇的部分
        lips_extracted = mask_lips(image, upper_curve, lower_curve)
        cropped_lips = lips_extracted[y_min:y_max, x_min:x_max]
        # cv2.imshow('Cropped', cropped_lips)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        scale_face = cv2.resize(cropped_lips, (width, heigh))  # (width, heigh)
        # cv2.imshow('Scaled', scale_face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('../image/scaled_roi.png', scale_face)

        # 灰度转换
        gray = cv2.cvtColor(scale_face, cv2.COLOR_BGR2GRAY)

        # 图像二值化，127为界，大于254置为白色255，小于254置黑色0
        _, thresh = cv2.threshold(gray, 254, 255, 0)
        # 黑白反转
        inverted_image = cv2.bitwise_not(thresh)  # 黑白反转
        # cv2.imshow('Inverted', inverted_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('../image/binarization.png', inverted_image)

        # 面积
        area = count_area(inverted_image)
        # print('面积: ', area)
        # 周长
        perimeter = count_contour(inverted_image)
        # print('周长: ', perimeter)
        # 上下唇厚度
        upper, lower = count_thickness(inverted_image)
        lower = padding_zero(inverted_image, lower)
        lips = np.hstack((upper, lower))
        # print('上唇厚度', upper)
        # print('下唇厚度', lower)

        features = [area, perimeter]
        features = np.hstack((features, lips))
        # print(len(features))
        return features  # 面积、周长、上唇厚度、下唇厚度
    return None


if __name__ == '__main__':
    # image_path = "../image/TRY.png"
    # image = cv2.imread(image_path)

    video_path = 'G:/00_raw/data_00_0_l.MP4'
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()

    re = extract_shape(image, 250, 110)
    print(re)
    print(len(re))
