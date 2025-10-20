# 输入是视频的路径 len = 380 恒定
# 20*20的关键点矩阵只取下三角矩阵
import pickle

import cv2
import dlib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, cm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('..\shape_predictor_68_face_landmarks.dat')

colormap = plt.get_cmap('tab20')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def extract_lip_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        lip = []
        for i in range(48, 68):
            lip.append((landmarks.part(i).x, landmarks.part(i).y))  # (x,y)是坐标

        # draw_points(image, lip)
        return np.array(lip)
    return None


def draw_points(image, keypoints):
    if keypoints is not None:
        for (x, y) in keypoints:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            # 显示结果图像
            print(f'({x}, {y})')
        # cv2.namedWindow("Lip Keypoints", cv2.WINDOW_GUI_NORMAL)
        # cv2.imshow('Lip Keypoints', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def generate_sequence(video_path, start_frame, length, interval):
    landmarks_sequence = []

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(frame_count, start_frame + length)
    frame_indices = np.arange(start_frame, end_frame, interval)
    # print(frame_indices)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    temp_landmarks = extract_lip_landmarks(frame)
    for i in frame_indices:
        # print(f'正在处理第{i}帧')
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the video.")
            break
        lip_landmarks = extract_lip_landmarks(frame)
        if lip_landmarks is None and temp_landmarks is None:
            return None
        elif lip_landmarks is None:
            lip_landmarks = temp_landmarks

        else:
            temp_landmarks = lip_landmarks
        landmarks_sequence.append(lip_landmarks)
    cap.release()


    # i = 1
    # cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Failed to read the video.")
    #         break
    #     # print(f'正在处理第{i}帧')
    #     lip_landmarks = extract_lip_landmarks(frame)
    #     # draw_motion_points(frame, lip_landmarks)
    #     landmarks_sequence.append(lip_landmarks)
    #     i = i + 1
    #     if i > length:
    #         break
    # cap.release()
    return landmarks_sequence


def compute_motion_tracks(landmarks_sequence):
    motion_tracks = {i: [] for i in range(48, 68)}

    try:
        for landmarks in landmarks_sequence:
            for i, (x, y) in enumerate(landmarks):
                motion_tracks[48 + i].append((x, y))
    except Exception:
        return None
    finally:
        return motion_tracks

    return motion_tracks


def build_motion_matrices(motion_tracks):
    motion_matrices_x = []  # 矩阵Cx
    motion_matrices_y = []  # 矩阵Cy
    num_points = len(motion_tracks)
    num_frames = len(next(iter(motion_tracks.values())))


    for i in range(num_points):
        track_x = [motion_tracks[48 + i][j][0] for j in range(num_frames)]
        track_y = [motion_tracks[48 + i][j][1] for j in range(num_frames)]
        motion_matrices_x.append(track_x)
        motion_matrices_y.append(track_y)

    motion_matrices_x = np.array(motion_matrices_x)
    motion_matrices_y = np.array(motion_matrices_y)

    return motion_matrices_x, motion_matrices_y


def calculate_correlation(matrix):
    num_points = matrix.shape[0]  # 20个关键点
    correlation_matrix = np.zeros((num_points, num_points))
    correlation = []

    for i in range(num_points):
        for j in range(i, num_points):
            if np.std(matrix[i]) == 0 or np.std(matrix[j]) == 0:
                correlation_matrix[i, j] = 0
            else:
                correlation_matrix[i, j] = np.corrcoef(matrix[i], matrix[j])[0, 1]
            correlation_matrix[j, i] = correlation_matrix[i, j]  # 对称矩阵

    for i in range(1, num_points):
        for j in range(0, i):
            correlation.append(correlation_matrix[i, j])  # 取除去对角的单个元素

    return correlation_matrix, correlation


def calculate_correlation_coefficients(motion_matrices_x, motion_matrices_y):
    _, correlation_x = calculate_correlation(motion_matrices_x)
    _, correlation_y = calculate_correlation(motion_matrices_y)

    return correlation_x, correlation_y


def plot_landmarks_over_time(landmarks_sequence):
    num_frames = len(landmarks_sequence)
    frame_range = range(0, 50)

    plt.figure(figsize=(12, 8))

    height = []
    for frame in range(len(landmarks_sequence)):
        high_max = landmarks_sequence[frame][18][1]
        high_min = landmarks_sequence[frame][14][1]
        h = high_max - high_min
        height.append(h)

    with open('read.pkl', 'wb') as f:
        pickle.dump(height, f)

    plt.xlabel('Frame')
    plt.ylabel('Mouth opening height (pixels)')
    plt.plot(frame_range, height, label=f'stop', color='gray', linestyle='--')
    plt.savefig('../image/read.png')



    # # want_to_draw = [2, 3, 4]
    # want_to_draw = [13, 14, 15]
    # for i in want_to_draw:
    #     x_coords = [landmarks_sequence[frame][i][0] for frame in frame_range]
    #     y_coords = [landmarks_sequence[frame][i][1] for frame in frame_range]
    #
    #     # plt.subplot(4, 1, 1)
    #     # plt.plot(range(num_frames), x_coords, label=f'Point {48 + i} X')
    #     # plt.ylabel('X Coordinate')
    #     # plt.title('Lip Landmark Coordinates Over Time')
    #     # plt.legend()
    #     # plt.gca().invert_yaxis()
    #
    #     plt.subplot(2, 1, 1)
    #     plt.plot(frame_range, y_coords, label=f'Point {48 + i}', color=colormap(i))
    #     plt.ylabel('Y Coordinate')
    #     # plt.legend()
    #     plt.gca().invert_yaxis()
    #
    # # want_to_draw = [8, 9, 10]
    # want_to_draw = [17, 18, 19]
    # for i in want_to_draw:
    #     x_coords = [landmarks_sequence[frame][i][0] for frame in frame_range]
    #     y_coords = [landmarks_sequence[frame][i][1] for frame in frame_range]
    #
    #     # plt.subplot(4, 1, 3)
    #     # plt.plot(range(num_frames), x_coords, label=f'Point {48 + i} X')
    #     # plt.ylabel('X Coordinate')
    #     # plt.legend()
    #     # plt.gca().invert_yaxis()
    #
    #     plt.subplot(2, 1, 2)
    #     plt.plot(frame_range, y_coords, label=f'Point {48 + i}', color=colormap(i))
    #     plt.xlabel('Frame')
    #     plt.ylabel('Y Coordinate')
    #
    #     # plt.legend()
    #     plt.gca().invert_yaxis()
    #
    # # plt.tight_layout()
    # # plt.gca().invert_yaxis()
    # plt.savefig('../image/middle_hey_move_y.png')
    # plt.show()


def draw_motion_points(image, keypoints):
    if keypoints is not None:
        for (x, y) in keypoints:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        # 显示结果图像
        cv2.imshow('Lip Keypoints', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_motion_tracks(motion_tracks):
    plt.figure(figsize=(10, 8))

    x_coords = []
    y_coords = []
    for i, track in motion_tracks.items():

        x_coords.append([point[0] for point in track])
        y_coords.append([point[1] for point in track])
        # x_coords = [point[0] for point in track]
        # y_coords = [point[1] for point in track]
        # plt.plot(x_coords, y_coords, lw=3, color=colormap(i - 48))

    y_min = 99999
    y_max = 0
    for i in range(20):
        max_ = max(y_coords[i])
        if max_ > y_max:
            y_max = max_
        min_ = min(y_coords[i])
        if min_ < y_min:
            y_min = min_
    x_min = 99999
    x_max = 0
    for i in range(20):
        max_ = max(x_coords[i])
        if max_ > x_max:
            x_max = max_
        min_ = min(x_coords[i])
        if min_ < x_min:
            x_min = min_

    for i in range(20):
        y = [t - 1500 - 70 for t in y_coords[i]]  # l:0 m:120
        x = x_coords[i] - x_min
        plt.plot(x, y, lw=3, color=colormap(i))

        # y_min = min(y_coords)
        # y_max = max(y_coords)
        # print(y_min, y_max, y_max-y_min)
        # y = []
        # for idx in range(len(y_coords)):
        #     y.append(y_coords[idx]-y_min)
        # plt.plot(x_coords, y, label=f'Point {i}', color=colormap(i - 48))


    plt.xlabel('X Coordinate', fontsize=30)
    plt.ylabel('Y Coordinate', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    # plt.title('Lip Landmark Motion Tracks')
    # plt.legend(loc='upper left')
    # plt.legend()
    plt.ylim(0, 301)
    plt.xlim(0, 451)
    plt.xticks(range(0, 451, 150))
    plt.yticks(range(0, 301, 150))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../image/read_track.png')
    # plt.show()


def extract_correlation(video_path, start_frame, length, interval):
    # print('Start')
    landmarks_sequence = generate_sequence(video_path, start_frame, length, interval)
    if landmarks_sequence is None:
        return None
    # print(f'读取{video_path}的唇部关键点完毕。。。')
    # plot_landmarks_over_time(landmarks_sequence)

    motion_tracks = compute_motion_tracks(landmarks_sequence)  # 20个关键点在length帧的移动
    plot_motion_tracks(motion_tracks)
    motion_matrices_x, motion_matrices_y = build_motion_matrices(motion_tracks)  # 20个关键点在length帧的移动的x坐标、y坐标
    correlation_x, correlation_y = calculate_correlation_coefficients(motion_matrices_x, motion_matrices_y)

    # print(len(np.hstack((correlation_x, correlation_y))))

    return np.hstack((correlation_x, correlation_y))


if __name__ == '__main__':

    video_path = 'H:/00_raw/data_00_0_s.MP4'
    length = 22  # l:23  m:26
    start_frame = 271  # l:20  m:205
    interval = 1

    re = extract_correlation(video_path, start_frame, length, interval)

    print(re)
    print(len(re))





