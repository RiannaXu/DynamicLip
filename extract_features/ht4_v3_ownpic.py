import numpy as np
import cv2
from matplotlib import pyplot as plt
import dlib
from crop_lip import extract_lip_from_face


class Segment:
    def __init__(self, points):
        self.points = points
        self.center = self.calculate_center()
        self.length = self.calculate_length()
        self.angle = self.calculate_angle()

    def calculate_center(self):
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        return (x_center, y_center)

    def calculate_length(self):
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return np.sqrt((x_coords[-1] - x_coords[0]) ** 2 + (y_coords[-1] - y_coords[0]) ** 2)

    def calculate_angle(self):
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return np.degrees(np.arctan2(y_coords[-1] - y_coords[0], x_coords[-1] - x_coords[0]))


def calculate_global_threshold(image):
    h, w = image.shape
    avg_brightness = np.sum(image) / (w * h)
    gamma = 255 - 0.75 * (255 - avg_brightness)
    return gamma


def calculate_local_brightness(image, x, y):
    h, w = image.shape
    if x < 3 or x >= w - 3 or y < 1 or y >= h - 1:
        return 255  # 边缘处默认设为255白（背景黑255）
    local_area = image[y - 1:y + 2, x - 3:x + 4]
    local_brightness = np.sum(local_area) / 21
    return local_brightness


def background_detection(image, gamma):
    h, w = image.shape
    bg_mask = np.zeros_like(image)
    for y in range(0, h):
        for x in range(0, w):
            local_brightness = calculate_local_brightness(image, x, y)
            if local_brightness > gamma:
                bg_mask[y, x] = 0  # 背景 黑色
            else:
                bg_mask[y, x] = image[y, x]  # 唇纹区域
    return bg_mask


def calculate_b(image, x, y):
    local_area = image[y - 4:y + 5, x - 4:x + 5]
    b_value = 1.1 * np.sum(local_area) / 81
    return b_value


def binarization(image, bg_mask):
    h, w = image.shape
    bin_image = np.zeros_like(image)
    for y in range(0, h):
        for x in range(0, w):
            b_value = calculate_b(bg_mask, x, y)
            if bg_mask[y, x] == 0:
                bin_image[y, x] = 255  # 白色
            else:
                if bg_mask[y, x] < b_value:
                    bin_image[y, x] = 0  # 黑色
                else:
                    bin_image[y, x] = 255  # 白色
    return bin_image


def get_line_points(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points


def valid_segment(points, mask):
    line_detection_array = []
    for (x, y) in points:
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if mask[y, x] == 0:  # 唇纹区域,用边缘检测的，唇纹区域白色，其他黑（0）
                flag = 0
            else:
                flag = 1
        else:
            flag = 0
        line_detection_array.append(flag)
    return line_detection_array


def extract_segments_from_detection_array_2(points, detection_array, n):
    segments = []
    current_segment = []
    gap_count = 0

    for i, flag in enumerate(detection_array):
        if flag == 1:
            if gap_count <= n and current_segment:
                current_segment.extend(points[i - gap_count:i + 1])
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = [points[i]]
            gap_count = 0
        else:
            if current_segment:
                gap_count += 1

    if current_segment:
        segments.append(current_segment)

    return segments


def extract_segments_from_detection_array(points, detection_array):
    segments = []
    current_segment = []

    for i, flag in enumerate(detection_array):
        if flag == 1:
            current_segment.append(points[i])
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []

    if current_segment:
        segments.append(current_segment)

    return segments


def filter_segments_by_length(segments, min_length=16, max_length=100):
    filtered_segments = []
    for segment in segments:
        length = cv2.arcLength(np.array(segment, dtype=np.int32).reshape((-1, 1, 2)), closed=False)
        if min_length < length < max_length:
            filtered_segments.append(segment)
    return filtered_segments


def calculate_angle(segment):
    # 计算线段的角度（弧度）
    x1, y1, x2, y2 = segment[0][0], segment[0][1], segment[1][0], segment[1][1]
    angle = np.arctan2(y2 - y1, x2 - x1)
    # 将角度转换为度数
    angle_degrees = np.degrees(angle)
    if angle_degrees < 0:
        angle_degrees = - angle_degrees
    return angle_degrees


def filter_segments_by_angle(segments, min_angle, max_angle):
    filtered_segments = []
    for segment in segments:
        angle = calculate_angle(segment)
        if min_angle < angle < max_angle:
            filtered_segments.append(segment)
    return filtered_segments


def keep_only_lip(edges, bg_mask):
    # 遍历每个像素
    rows, cols = edges.shape
    for i in range(rows):
        for j in range(cols):
            # 如果 bg_mask 中的像素为黑色，则将 edges 中的相应像素设置为黑色
            if bg_mask[i, j] == 0:
                edges[i, j] = 0
    return edges


def Feature_HT(image, lip_region):  # image只有嘴唇，lip_region区域
    # 读取图片
    # image = cv2.imread("H:\\STUDY\\PG\\sensing\\lip-print\\database\\yjxxv9xbds-2\\LipPrintDatabase\\U002L03.png")
    # image = cv2.imread("test.jpg")

    image_copy_0 = image.copy()
    image_copy_1 = image.copy()
    # 转换成灰度图
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯滤波降噪
    # image_gaussian = cv2.GaussianBlur(image_gray, (5, 5), 0)
    # image_gray = image
    # 计算全局阈值
    gamma = calculate_global_threshold(image_gray)
    # 背景检测
    bg_mask = background_detection(image_gray, gamma)
    # # 二值化处理
    bin_image = binarization(image_gray, bg_mask)
    # 二值化处理，生成掩码
    # _, bin_image = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)#HRESH_BINARY<127置0（黑色），大于置255（默认，白），黑色部分唇纹
    # cv2.imshow('THRESH_BINARY', bin_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    lip_region_gray = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # 1,4,4
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(lip_region_gray)

    # # 高斯模糊，平滑图像
    # blurred = cv2.GaussianBlur(lip_region_gray, (1, 1), 0)
    # 边缘检测, Sobel算子大小为3
    edges_1 = cv2.Canny(dst, 50, 150, apertureSize=3)  # 低阈值和高阈值 (threshold1 和 threshold2),越小越细致,算子大小
    # Use morphological operations to thin the edges
    # kernel = np.ones((2, 2), np.uint8)
    # edges = cv2.morphologyEx(edges_1, cv2.MORPH_CLOSE, kernel)
    # #尝试使用膨胀和腐蚀的组合
    # dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    # eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    # 滤波
    edges_2 = cv2.bilateralFilter(edges_1, 150, 80, 80)

    # 使用形态学操作增强竖直边缘
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    # edges_3 = cv2.morphologyEx(edges_2, cv2.MORPH_CLOSE, kernel)

    # 去除非嘴唇区域
    edges = keep_only_lip(edges_2, bg_mask)

    # cv2.imwrite('H:\STUDY\PG\sensing\lip-print\code\ht4_v2\data_new\canny50-150\\canny.png',edges_1)
    # cv2.imwrite('H:\STUDY\PG\sensing\lip-print\code\ht4_v2\data_new\canny50-150\\keep_only_lip.png', edges)
    # # cv2.imshow('edge', edges)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # 霍夫曼直线检测
    rho = 5  # 距离分辨率,越小检测的直线越少，5
    theta = np.pi / 180  # 角度分辨率180
    threshold = 25  # 累加平面的阈值，越小越细致  25
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
    # # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #

    segment_objects = []  # 存储每个 segment 的属性
    lines_detection_array = []
    # # 遍历

    if lines is None:
        return None
    for line in lines:
        # 获取rho和theta
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_copy_0, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)

        # # 计算直线上所有点的坐标
        line_points = get_line_points(x1, y1, x2, y2)

        # 检查这些点是否在唇纹区域内
        line_detection_array = valid_segment(line_points, edges)
        lines_detection_array.append(line_detection_array)

        # 提取连续在唇纹区域内的线段
        n = 2  # 允许出现间断的0的个数:2
        segments = extract_segments_from_detection_array_2(line_points, line_detection_array, n)  # 可能会有只包含一个点的segment
        # segments = extract_segments_from_detection_array(line_points, line_detection_array)

        # 筛选线段长度10-100,30-150
        segments = filter_segments_by_length(segments, min_length=10, max_length=100)  # 几乎都很小，max_length无太大用处35-40
        segments = filter_segments_by_angle(segments, min_angle=40, max_angle=140)
        # 绘制这些线段
        # blank_image = np.ones_like(image)*255
        for segment in segments:
            segment_obj = Segment(segment)
            segment_objects.append(segment_obj)
            # if len(segment) > 1:
            # cv2.line(image_copy, segment[0], segment[len(segment)-1], (0, 0, 255), 2)
            segment_array = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image_copy_1, [segment_array], isClosed=False, color=(0, 0, 255), thickness=1)
            # cv2.polylines(blank_image, [segment_array], isClosed=False, color=(0, 0, 0), thickness=3)

    # 保存结果
    # cv2.imshow('hough_lines', image_copy_1)
    # # cv2.imwrite('H:\STUDY\PG\sensing\lip-print\code\ht4_v2\data_new\canny50-150\\filter-angle.png', image_copy_1)
    # # # cv2.imwrite('H:\STUDY\PG\sensing\lip-print\code\ht4_v2\data_new\canny50-150\len15-100.png', image_copy_1)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # # 显示结果
    # plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    # plt.title('Hough Segments with Attributes')
    # plt.xticks([]), plt.yticks([])

    # # # 图片展示
    # f, ax = plt.subplots(2, 3, figsize=(12, 12))
    #
    # # 检查并转换 bg_mask
    #
    # bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)
    # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    #
    # # 子图
    # # ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # ax[0, 0].imshow(lip_region_gray, "gray")
    # ax[0, 1].imshow(cv2.cvtColor(bg_mask, cv2.COLOR_BGR2RGB))
    # ax[0, 2].imshow(bin_image, "gray")
    # ax[1, 0].imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    # ax[1, 1].imshow(cv2.cvtColor(image_copy_0, cv2.COLOR_BGR2RGB))
    # ax[1, 2].imshow(cv2.cvtColor(image_copy_1, cv2.COLOR_BGR2RGB))
    # # ax[1, 2].imshow(cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB))
    #
    # # 标题
    # # ax[0, 0].set_title("image")
    # ax[0, 0].set_title("image gray")
    # ax[0, 1].set_title("image back_det")
    # ax[0, 2].set_title("image binarization")
    # ax[1, 0].set_title("image edge")
    # ax[1, 1].set_title("image line_all")
    # ax[1, 2].set_title("image line_in_pic")
    # #
    # plt.show(block=True)

    return segment_objects


# def Static_HT(image_path):
#     image = cv2.imread(image_path)
def Static_HT(lip_only, lip_region, position, lip_landmarks):
    # lip_only, lip_region, [x_min, y_min], lip_landmarks = extract_lip_from_face(image)

    # 只有嘴唇lip_only,嘴唇区域lip_region
    segment_objects = Feature_HT(lip_only, lip_region)  # 匹配的直线坐标是相对于裁剪的图片的
    if segment_objects is None:
        segment_objects, lip_only, position, lip_landmarks = None, None, None, None
    return segment_objects, lip_only, position, lip_landmarks


if __name__ == '__main__':
    image_path = "H:\STUDY\PG\sensing\lip-print\database\databese\output_frames\\frame_0004.png"
    image = cv2.imread(image_path)
    lip_only, lip_region, _, _ = extract_lip_from_face(image)  # 只有嘴唇lip_only,嘴唇区域lip_region
    segment_objects = Feature_HT(lip_only, lip_region)
    # len = 200
    # segment_objects_alignment  = Len_Aligmnet(segment_objects,len,MODE = )
    # print(len(segment_objects))
