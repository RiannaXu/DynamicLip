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
    upper_lip_indices = [0, 1, 2, 3, 4, 5, 6, 16, 15, 14, 13, 12, 0]
    lower_lip_indices = [6, 7, 8, 9, 10, 11, 0, 12, 19, 18, 17, 16, 6]
    # upper_lip_indices = [0, 1, 2, 3, 4, 5, 6, 15, 14, 13, 0]
    # lower_lip_indices = [6, 7, 8, 9, 10, 11, 0, 19, 18, 17, 6]

    upper_lip = lip_landmarks[upper_lip_indices]
    lower_lip = lip_landmarks[lower_lip_indices]

    upper_curve = []
    lower_curve = []
    upper_curve.append(fit_curve_segment(upper_lip[0:3], degree=5, num_points=20))
    upper_curve.append(fit_curve_segment(upper_lip[2:5], degree=5, num_points=10))
    upper_curve.append(fit_curve_segment(upper_lip[4:7], degree=5, num_points=20))
    temp = fit_curve_segment(upper_lip[6:], degree=5, num_points=50)
    # print(temp)
    temp1 = np.flipud(temp)
    upper_curve.append(temp1)
    # upper_curve.append(fit_curve_segment(upper_lip[6:], degree=5, num_points=50))
    lower_curve.append(fit_curve_segment(lower_lip[:7], degree=5, num_points=50))
    temp = fit_curve_segment(fit_curve_segment(lower_lip[7:], degree=5, num_points=50))
    temp1 = np.flipud(temp)
    lower_curve.append(temp1)
    # lower_curve.append(fit_curve_segment(lower_lip[7:], degree=5, num_points=50))

    upper_curve = np.vstack(upper_curve)
    lower_curve = np.vstack(lower_curve)

    return upper_curve, lower_curve


def create_lip_mask(image, upper_curve, lower_curve):
    mask = np.zeros_like(image)
    upper_curve = np.array(upper_curve, dtype=np.int32)
    lower_curve = np.array(lower_curve, dtype=np.int32)

    cv2.fillPoly(mask, [upper_curve], (255, 255, 255))
    cv2.fillPoly(mask, [lower_curve], (255, 255, 255))
    return mask


def extract_lips(image, upper_curve, lower_curve):
    mask = create_lip_mask(image, upper_curve, lower_curve)
    lips = cv2.bitwise_and(image, mask)
    white_background = np.ones_like(image) * 255
    white_background[mask == 255] = lips[mask == 255]
    return white_background


# def extract_lip_from_face(image):
#     lip_landmarks = extract_lip_landmarks(image)
#     x_min = np.min(lip_landmarks[:, 0])
#     x_max = np.max(lip_landmarks[:, 0])
#     y_min = np.min(lip_landmarks[:, 1])
#     y_max = np.max(lip_landmarks[:, 1])
#
#     # 裁剪唇部区域
#     lips_region = image[y_min:y_max, x_min:x_max]
#     if lip_landmarks is not None:
#         upper_curve, lower_curve = fit_lip_contour(lip_landmarks)
#         lips_extracted = extract_lips(image, upper_curve, lower_curve)
#         cropped_lips = lips_extracted[y_min:y_max, x_min:x_max]
#
#     return cropped_lips, lips_region, [x_min, y_min]


def extract_lip_from_face(image):
    lip_landmarks = extract_lip_landmarks(image)
    if lip_landmarks is None:
        cropped_lips = None
        lips_region = None
        [x_min, y_min] = [0,0]
        lip_landmarks = None
    else:
        x_min = np.min(lip_landmarks[:, 0])
        x_max = np.max(lip_landmarks[:, 0])
        y_min = np.min(lip_landmarks[:, 1])
        y_max = np.max(lip_landmarks[:, 1])

        # draw_points(image, lip_landmarks)
        # 裁剪唇部区域
        lips_region = image[y_min:y_max, x_min:x_max]
        if lip_landmarks is not None:
            upper_curve, lower_curve = fit_lip_contour(lip_landmarks)
            image_with_contour = draw_lip_contour(image.copy(), upper_curve, lower_curve)
            lips_extracted = extract_lips(image, upper_curve, lower_curve)

            cropped_lips = lips_extracted[y_min:y_max, x_min:x_max]

            # cv2.imshow('Cropped Lips', cropped_lips)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            #gray = cv2.cvtColor(cropped_lips, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('Cropped Gray Lips', gray)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    return cropped_lips, lips_region, [x_min,y_min], lip_landmarks


def draw_lip_contour(image, upper_curve, lower_curve):
    for (x, y) in np.vstack(lower_curve):
        cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
    for (x, y) in np.vstack(upper_curve):
        cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), -1)

    # cv2.imshow('Lip Contour', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image


if __name__ == '__main__':

    # video_path = 'image/xu.mp4'
    # cap = cv2.VideoCapture(video_path)
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     image = frame
    #
    #     lip_landmarks = extract_lip_landmarks(image)
    #     x_min = np.min(lip_landmarks[:, 0])
    #     x_max = np.max(lip_landmarks[:, 0])
    #     y_min = np.min(lip_landmarks[:, 1])
    #     y_max = np.max(lip_landmarks[:, 1])
    #
    #     # draw_points(image, lip_landmarks)
    #
    #     if lip_landmarks is not None:
    #         upper_curve, lower_curve = fit_lip_contour(lip_landmarks)
    #         image_with_contour = draw_lip_contour(image.copy(), upper_curve, lower_curve)
    #         lips_extracted = extract_lips(image, upper_curve, lower_curve)
    #
    #         cropped_lips = lips_extracted[y_min:y_max, x_min:x_max]
    #
    #         # cv2.imshow('Cropped Lips', cropped_lips)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    #
    #         gray = cv2.cvtColor(cropped_lips, cv2.COLOR_BGR2GRAY)
    #
    #         cv2.imshow('Cropped Gray Lips', gray)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #
    #     #     output_path = 'save/111.png'
    #     #     cv2.imwrite(output_path, lips_extracted)
    #     #     print(f'Extracted lips saved to {output_path}')
    #
    # cap.release()

    image_path = 'CFD-AF-200-228-N.jpg'
    image = cv2.imread(image_path)

    lip_landmarks = extract_lip_landmarks(image)
    x_min = np.min(lip_landmarks[:, 0])
    x_max = np.max(lip_landmarks[:, 0])
    y_min = np.min(lip_landmarks[:, 1])
    y_max = np.max(lip_landmarks[:, 1])

    # draw_points(image, lip_landmarks)

    if lip_landmarks is not None:
        upper_curve, lower_curve = fit_lip_contour(lip_landmarks)
        # image_with_contour = draw_lip_contour(image.copy(), upper_curve, lower_curve)
        lips_extracted = extract_lips(image, upper_curve, lower_curve)

        cropped_lips = lips_extracted[y_min:y_max, x_min:x_max]

        # cv2.imshow('Cropped Lips', cropped_lips)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        gray = cv2.cvtColor(cropped_lips, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Cropped Gray Lips', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #     output_path = 'save/111.png'
    #     cv2.imwrite(output_path, lips_extracted)
    #     print(f'Extracted lips saved to {output_path}')
