import numpy as np
import cv2
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon
import copy

# 导入其他必要的模块和函数
from ht4_v3_ownpic import Static_HT, Segment


def get_regions(lip_landmarks):
    regions = {
        'UL': [0, 1, 2, 13, 12],
        'UM': [2, 3, 4, 15, 14, 13],
        'UR': [4, 5, 6, 16, 15],
        'LL': [0, 12, 19, 10, 11],
        'LM': [8, 9, 10, 19, 18, 17],
        'LR': [6, 7, 8, 17, 16]
    }
    return {name: [lip_landmarks[i] for i in indices] for name, indices in regions.items()}


def point_in_polygon(point, polygon):
    return Polygon(polygon).contains(Point(point))


def calculate_max_movement_distances(region_polygons1, region_polygons2):
    max_movements = {}
    for name in region_polygons1.keys():
        max_x_movement = 0
        max_y_movement = 0
        for p1, p2 in zip(region_polygons1[name], region_polygons2[name]):
            max_x_movement = max(max_x_movement, abs(p1[0] - p2[0]))
            max_y_movement = max(max_y_movement, abs(p1[1] - p2[1]))
        max_movements[name] = (max_x_movement, max_y_movement)
    return max_movements


def match_and_filter_lines(segment_objects1, segment_objects2, lip_landmarks1, lip_landmarks2, length_threshold=10,
                           angle_threshold=10, region_counts=[1, 2, 1, 1, 2, 1]):
    motions = []
    filtered_vectors = []

    region_polygons1 = get_regions(lip_landmarks1)
    region_polygons2 = get_regions(lip_landmarks2)
    region_vectors = {name: [] for name in region_polygons1}

    max_movements = calculate_max_movement_distances(region_polygons1, region_polygons2)

    for region_name, polygon1 in region_polygons1.items():
        region_segment_objects1 = [seg for seg in segment_objects1 if point_in_polygon(seg.center, polygon1)]
        polygon2 = region_polygons2[region_name]
        region_segment_objects2 = [seg for seg in segment_objects2 if point_in_polygon(seg.center, polygon2)]

        max_x_movement = max_movements[region_name][0]
        max_y_movement = max_movements[region_name][1]

        for seg1 in region_segment_objects1:
            best_match = None
            best_score = float('inf')
            for seg2 in region_segment_objects2:
                dist_x = abs(seg1.center[0] - seg2.center[0])
                dist_y = abs(seg1.center[1] - seg2.center[1])
                length_diff = abs(seg1.length - seg2.length)
                angle_diff = abs(seg1.angle - seg2.angle)

                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if dist_x < max_x_movement and dist_y < max_y_movement and length_diff < length_threshold and angle_diff < angle_threshold:
                    score = length_diff + angle_diff
                    if score < best_score:
                        best_score = score
                        best_match = seg2

            if best_match is not None:
                motion_vector = np.array(best_match.center) - np.array(seg1.center)
                distance = np.linalg.norm(motion_vector)
                motions.append((seg1, best_match, motion_vector, distance))
                region_vectors[region_name].append((seg1, best_match, motion_vector, distance))

    def filter_by_y_direction(vectors, overall_direction):
        if not vectors:
            return vectors
        if overall_direction > 0:
            return [v for v in vectors if v[2][1] > 0]
        else:
            return [v for v in vectors if v[2][1] < 0]

    upper_lip_vectors = region_vectors['UL'] + region_vectors['UM'] + region_vectors['UR']
    lower_lip_vectors = region_vectors['LL'] + region_vectors['LM'] + region_vectors['LR']

    upper_lip_y_directions = [v[2][1] for v in upper_lip_vectors]
    lower_lip_y_directions = [v[2][1] for v in lower_lip_vectors]

    upper_lip_overall_direction = np.mean(upper_lip_y_directions) if upper_lip_y_directions else 0
    lower_lip_overall_direction = np.mean(lower_lip_y_directions) if lower_lip_y_directions else 0

    for i, (name, vectors) in enumerate(region_vectors.items()):
        if name in ['UL', 'UM', 'UR']:
            vectors = filter_by_y_direction(vectors, upper_lip_overall_direction)
        else:
            vectors = filter_by_y_direction(vectors, lower_lip_overall_direction)

        vectors = sorted(vectors, key=lambda x: x[3], reverse=True)[:region_counts[i]]
        while len(vectors) < region_counts[i]:
            zero_segment = Segment(points=[(0, 0), (0, 0)])
            vectors.append((zero_segment, zero_segment, np.array([0, 0]), 0))
        filtered_vectors.extend(vectors)

    return motions, filtered_vectors


def draw_motion(image, matches, offset):
    for (seg1, seg2, _, _) in matches:
        seg1_new = line_Coordinate_Conversions(copy.deepcopy(seg1), offset, "Relative")
        seg2_new = line_Coordinate_Conversions(copy.deepcopy(seg2), offset, "Relative")
        center1 = tuple(map(int, seg1_new.center))
        center2 = tuple(map(int, seg2_new.center))
        cv2.arrowedLine(image, center1, center2, (0, 255, 0), 1, tipLength=0.5)
    return image


def draw_matches_segments(image, matches, offset, frame_index):
    for (seg1, seg2, _, _) in matches:
        if frame_index == 1:
            seg1_new = line_Coordinate_Conversions(copy.deepcopy(seg1), offset, "Relative")
            pts = np.array(seg1_new.points, dtype=np.int32).reshape((-1, 1, 2))
        else:
            seg2_new = line_Coordinate_Conversions(copy.deepcopy(seg2), offset, "Relative")
            pts = np.array(seg2_new.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(0, 255, 0), thickness=1)
    return image


def line_Coordinate_Conversions(segment, offset, MODE="Absolute"):
    x_min, y_min = offset
    for i in range(len(segment.points)):
        if MODE == "Absolute":
            segment.points[i] = (segment.points[i][0] + x_min, segment.points[i][1] + y_min)
        elif MODE == "Relative":
            segment.points[i] = (segment.points[i][0] - x_min, segment.points[i][1] - y_min)
    segment.center = segment.calculate_center()
    segment.length = segment.calculate_length()
    segment.angle = segment.calculate_angle()
    return segment


def segment_objects_Coordinate_Conversions(segment_objects, offset):
    for segment_old in segment_objects:
        segment = line_Coordinate_Conversions(segment_old, offset)
    return segment_objects


def get_matches_between_two(frame1, frame2):
    list1, array1 = frame1
    list2, array2 = frame2
    all_matches, filtered_matches = match_and_filter_lines(list1, list2,
                                                           array1, array2, length_threshold=10,
                                                           angle_threshold=10, region_counts=[1, 2, 1, 1, 2, 1])
    return all_matches, filtered_matches


if __name__ == '__main__':
    image_path_1 = "H:/STUDY/PG/sensing/lip-print/database/databese/output_frames/frame_0000.png"
    image_path_2 = "H:/STUDY/PG/sensing/lip-print/database/databese/output_frames/frame_0003.png"
    image_1 = cv2.imread(image_path_1)
    image_2 = cv2.imread(image_path_2)

    all_matches, filtered_matches, lip_image_1, offset_1, lip_landmarks_1, lip_image_2, offset_2, lip_landmarks_2 = get_matches_between_two(
        image_1, image_2)

    image1_segments = draw_matches_segments(lip_image_1.copy(), copy.deepcopy(filtered_matches), offset_1,
                                            frame_index=1)
    image2_segments = draw_matches_segments(lip_image_2.copy(), copy.deepcopy(filtered_matches), offset_2,
                                            frame_index=2)
    image1_with_motion = draw_motion(lip_image_1.copy(), copy.deepcopy(filtered_matches), offset_1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image1_segments, cv2.COLOR_BGR2RGB))
    plt.title('Frame 1 Segments')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image2_segments, cv2.COLOR_BGR2RGB))
    plt.title('Frame 2 Segments')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image1_with_motion, cv2.COLOR_BGR2RGB))
    plt.title('Motion Frame 1-2')

    plt.show()
