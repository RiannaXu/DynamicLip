import time

import numpy as np
import cv2
import pandas as pd

from crop_lip import extract_lip_from_face
from dynamic_HT import get_matches_between_two, segment_objects_Coordinate_Conversions
# import pickle
import warnings

# 忽略所有的警告
from ht4_v3_ownpic import Static_HT

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="iCCP: known incorrect sRGB profile")
warnings.filterwarnings('ignore',
                        message="RankWarning: Polyfit may be poorly conditioned poly = np.polyfit(x, y, degree)")


def calculate_motion_features(motions_list):
    avg_vector_allframes = []
    std_vector_allframes = []
    avg_distance_allframes = []
    std_distance_allframes = []
    max_distance_allframes = []
    min_distance_allframes = []
    avg_angle_allframes = []
    std_angle_allframes = []
    trajectory_length_allframes = []
    curvature_length_allframes = []
    for motions in motions_list:
        vectors = np.array([motion[2] for motion in motions])
        distances = np.array([motion[3] for motion in motions])
        angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))

        avg_vector = np.mean(vectors, axis=0)
        avg_vector_allframes.append(avg_vector)

        std_vector = np.std(vectors, axis=0)
        std_vector_allframes.append(std_vector)

        avg_distance = np.mean(distances)
        avg_distance_allframes.append(avg_distance)

        std_distance = np.std(distances)
        std_distance_allframes.append(std_distance)

        max_distance = np.max(distances)
        max_distance_allframes.append(max_distance)
        min_distance = np.min(distances)
        min_distance_allframes.append(min_distance)

        avg_angle = np.mean(angles)
        avg_angle_allframes.append(avg_angle)
        std_angle = np.std(angles)
        std_angle_allframes.append(std_angle)

        # angle_hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
        # angle_hist = angle_hist / len(angles)

        trajectory_length = np.sum(distances)
        trajectory_length_allframes.append(trajectory_length)
        curvature = np.std([np.linalg.norm(vectors[i] - vectors[i - 1]) for i in range(1, len(vectors))])
        curvature_length_allframes.append(curvature)

    # autocorr_coeffs = correlate(distances, distances)
    # fft_coeffs = fft(distances)
    avg_vector_allframes = np.array(avg_vector_allframes)
    std_vector_allframes = np.array(std_vector_allframes)

    features = {  # 12种
        'avg_vector_x': [item[0] for item in avg_vector_allframes],
        'avg_vector_y': [item[1] for item in avg_vector_allframes],
        'std_vector_x': [item[0] for item in std_vector_allframes],
        'std_vector_y': [item[1] for item in std_vector_allframes],
        'avg_distance': avg_distance_allframes,
        'std_distance': std_distance_allframes,
        'max_distance': max_distance_allframes,
        'min_distance': min_distance_allframes,
        'avg_angle': avg_angle_allframes,
        'std_angle': std_angle_allframes,
        # 'angle_histogram': angle_hist,
        'trajectory_length': trajectory_length_allframes,
        'curvature': curvature_length_allframes,
        # 'autocorr_coeffs': autocorr_coeffs,
        # 'fft_coeffs': fft_coeffs
    }
    return features


def extract_frames_from_video(video_path, start_frame, num_frames=50, interval=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(frame_count, start_frame + num_frames)
    frame_indices = np.arange(start_frame, end_frame, interval)
    frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    temp1, temp2, temp3, temp4 = extract_lip_from_face(frame)

    for i in frame_indices:
        print(f'\t\t第{i}帧')
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            lip_only, lip_region, [x_min, y_min], lip_landmarks = extract_lip_from_face(frame)
            if lip_landmarks is None:
                lip_only, lip_region, [x_min, y_min], lip_landmarks =  temp1, temp2, temp3, temp4
            else:
                temp1, temp2, temp3, temp4 = lip_only, lip_region, [x_min, y_min], lip_landmarks
            segment_objects_1, lip_image_1, offset_1, lip_landmarks_1 = Static_HT(lip_only, lip_region, [x_min, y_min], lip_landmarks)
            if lip_landmarks_1 is None:
                return None
            segment_objects_1_Absolute = segment_objects_Coordinate_Conversions(segment_objects_1, offset_1)
            frames.append((segment_objects_1_Absolute, lip_landmarks_1))
            # frames.append(frame)
    cap.release()
    return frames


def print_features(features):
    print("运动向量的统计特征")
    print(f"平均运动向量 x: {features['avg_vector_x']}")
    print(f"平均运动向量 y: {features['avg_vector_y']}")
    print(f"运动向量标准差 x: {features['std_vector_x']}")
    print(f"运动向量标准差 y: {features['std_vector_y']}")
    # print(f"最大运动向量: {features['max_distance']}")
    # print(f"最小运动向量: {features['min_distance']}")

    print("\n运动幅度的统计特征")
    print(f"平均运动幅度: {features['avg_distance']}")
    print(f"运动幅度标准差: {features['std_distance']}")
    print(f"最大运动幅度: {features['max_distance']}")
    print(f"最小运动幅度: {features['min_distance']}")

    print("\n运动方向的统计特征")
    print(f"运动方向的平均值: {features['avg_angle']}")
    print(f"运动方向的标准差: {features['std_angle']}")

    print("\n运动轨迹特征")
    print(f"运动轨迹长度: {features['trajectory_length']}")
    print(f"运动轨迹的弯曲度: {features['curvature']}")


def extract_texture(video_path, start_frame, num_frames=50, interval=5):
    frames = extract_frames_from_video(video_path, start_frame, num_frames, interval)
    if frames is None:
        return None

    matches_list = []

    # for i in range(len(frames) - 1):
    for (item1, item2) in zip(frames, frames[1:]):
        _, matches = get_matches_between_two(item1, item2)
        matches_list.append(matches)
        # print(f"\t\t提取第{i + 1}组运动向量")

    # combined_motions = []
    # for matches in matches_list:
    #     combined_motions.extend(matches)
    #
    motion_features = calculate_motion_features(matches_list)

    df = pd.DataFrame(motion_features)
    motion_matrix = df.T.values
    return motion_matrix.flatten()
    # return motion_features


if __name__ == '__main__':
    start_time = time.time()
    video_path = "../videos/try1.mp4"
    motion_features = extract_texture(video_path, start_frame=0, num_frames=50, interval=5)
    print("完成")
    print(f"Motion Features: {motion_features}")
    # print_features(motion_features)
    end_time = time.time()
    run_time = end_time - start_time
    print("代码执行时间为：%s秒" % run_time)
