import cv2
import numpy as np
import matplotlib.pyplot as plt
import unittest

######################## TASK 1 FUNCTIONS  ########################


def object_points(width, spacing):
    objpts = []
    for j in range(2):
        for i in range(3):
            objpts.append(
                np.array(
                    [
                        [i * (spacing + width), j * (spacing + width), 0],
                        [i * (spacing + width), j * (spacing + width) + width, 0],
                        [
                            i * (spacing + width) + width,
                            j * (spacing + width) + width,
                            0,
                        ],
                        [i * (spacing + width) + width, j * (spacing + width), 0],
                    ],
                    dtype=np.float32,
                )
            )
    objpts = np.array(objpts)
    return [objpts.reshape(1, 24, 3)]


def simpler_object_points(width, spacing):
    objpts = [
        np.array(
            [[0, 0, 0], [0, width, 0], [width, width, 0], [width, 0, 0]],
            dtype=np.float32,
        )
    ]
    objpts = np.array(objpts)
    return [objpts.reshape(1, 4, 3)]


def sort_corners(corners, ids):
    order = np.argsort(ids.flatten()).astype(int)
    sorted_corners = [
        corners[order[5]],
        corners[order[3]],
        corners[order[1]],
        corners[order[4]],
        corners[order[2]],
        corners[order[0]],
    ]
    sorted_corners = np.array(sorted_corners)
    sorted_corners = sorted_corners.reshape(1, 24, 2)
    return sorted_corners


def undistort_image(image, cameraMatrix, distCoeffs):
    h, w = image.shape[:2]

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, (w, h), alpha=0
    )

    R = np.eye(3)
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix, distCoeffs, R, new_camera_matrix, (w, h), m1type=cv2.CV_32FC1
    )

    undistorted_image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    return undistorted_image


def calibrating(calibration_images, tag_size, spacing):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    imgptsf = []
    objptsf = []
    objpts = object_points(tag_size, spacing)
    shape = (calibration_images[0].shape[1], calibration_images[0].shape[0])
    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        corners = sort_corners(corners, ids)

        objptsf.append(objpts[0])
        imgptsf.append(corners)
    ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
        objptsf, imgptsf, shape, None, None
    )
    return cameraMatrix, distCoeffs, ret


def simpler_calibrating(calibration_images, tag_size, spacing):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    undistorted_img_2 = []
    for img in calibration_images:
        imgptsf = []
        objptsf = []
        shape = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        corners = sort_corners(corners, ids)
        objpts = object_points(tag_size, spacing)

        objptsf.append(objpts[0])
        imgptsf.append(corners)
        _, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
            objptsf, imgptsf, shape, None, None
        )
        undistorted_img = undistort_image(img, cameraMatrix, distCoeffs)
        undistorted_img_2.append(undistorted_img)

    return undistorted_img_2


def calibrating_6_different(calibration_images, tag_size, spacing):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    objptsf = []
    imgptsf = []
    objpts = simpler_object_points(tag_size, spacing)
    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            corners = sort_corners(corners, ids)
            shape = (img.shape[1], img.shape[0])
            for i in range(6):
                objptsf.append(objpts[0])
                imgptsf.append(corners[:, 4 * i : 4 * (i + 1), :])

    ret, cameraMatrix_, distCoeffs_, _, _ = cv2.calibrateCamera(
        objptsf, imgptsf, shape, None, None
    )
    return cameraMatrix_, distCoeffs_, ret


######################## TASK 2 FUNCTIONS  ########################


def apply_projective_transform(
    source_image, destination_image, projective_transformation_matrix
):
    h, w = destination_image.shape[:2]
    inv_matrix = np.linalg.inv(projective_transformation_matrix)
    transformed_image = np.zeros_like(destination_image)

    for y in range(h):
        for x in range(w):
            src_coords = inv_matrix @ np.array([x, y, 1])
            src_x, src_y = src_coords[:2] / src_coords[2]
            src_x, src_y = int(round(src_x)), int(round(src_y))

            if (
                0 <= src_x < source_image.shape[1]
                and 0 <= src_y < source_image.shape[0]
            ):
                transformed_image[y, x] = source_image[src_y, src_x]

    final_image = np.zeros_like(destination_image)
    for y in range(h):
        for x in range(w):
            dst_coords = projective_transformation_matrix @ np.array([x, y, 1])
            dst_x, dst_y = dst_coords[:2] / dst_coords[2]
            dst_x, dst_y = int(round(dst_x)), int(round(dst_y))

            if (
                0 <= dst_x < destination_image.shape[1]
                and 0 <= dst_y < destination_image.shape[0]
            ):
                final_image[dst_y, dst_x] = transformed_image[y, x]
            else:
                final_image[y, x] = [0, 0, 0]  # Make the square black if out of image

    # Display the original and transformed images
    cv2.imshow("Original Image", source_image)
    cv2.imshow("Transformed Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_image


######################## TASK 3 FUNCTIONS  ########################


def find_homography_matrix(source, destination):
    A = np.zeros((2 * source.shape[1], 9), dtype=np.float64)
    for i in range(source.shape[1]):
        part = np.hstack([source[:, i].T, np.array([1])])

        A[2 * i, 0:3] = part
        A[2 * i, 6:9] = -part * destination[0, i]
        A[2 * i + 1, 3:6] = part
        A[2 * i + 1, 6:9] = -part * destination[1, i]
    _, _, V = np.linalg.svd(A)
    matrix = V[-1, :].reshape(3, 3)
    matrix = matrix / matrix[2, 2]
    return matrix


def get_random_homography():
    random_matrix = np.zeros((3, 3))
    while np.linalg.det(random_matrix) == 0:
        random_matrix = np.random.rand(3, 3)
        random_matrix = random_matrix / random_matrix[2, 2]
    return random_matrix


def generate_random_points(num_points=100, range_min=0, range_max=100):
    points = np.random.uniform(range_min, range_max, (2, num_points))
    return points


def test_homography():
    np.set_printoptions(precision=18, suppress=True)
    source = generate_random_points()
    random_matrix = get_random_homography()
    source_homogeneous = np.vstack([source, np.ones((1, source.shape[1]))])
    destination_homogeneous = random_matrix @ source_homogeneous
    destination_homogeneous /= destination_homogeneous[2, :]
    destination = destination_homogeneous[:2, :]
    test_matrix = find_homography_matrix(source, destination)
    print(
        f"The differences between homography matrix and its calculation: {np.subtract(random_matrix, test_matrix)}"
    )
    print()


######################## TASK 4 FUNCTIONS  ########################
######################## TASK 5 FUNCTIONS  ########################


######################## MAIN FUNCTION  ########################


def main():

    calibration_images = [cv2.imread(f"calibration\img{i}.png") for i in range(1, 29)]
    stitching_images = [cv2.imread(f"stitching\img{i}.png") for i in range(1, 10)]
    tag_size = 1.68
    spacing = 0.70

    ######################## TASK 1 ########################

    cameraMatrix, distCoeffs, ret = calibrating(
        calibration_images=calibration_images, tag_size=tag_size, spacing=spacing
    )

    print(cameraMatrix)
    print(distCoeffs)

    ###### testing errors for different methods ######
    test_img = [
        calibration_images[5],
        calibration_images[6],
        calibration_images[7],
        calibration_images[8],
        calibration_images[9],
        calibration_images[10],
        calibration_images[11],
    ]
    cameraMatrix1, distCoeffs1, ret1 = calibrating(
        calibration_images=test_img, tag_size=tag_size, spacing=spacing
    )
    cameraMatrix2, distCoeffs2, ret2 = calibrating_6_different(
        calibration_images=test_img, tag_size=tag_size, spacing=spacing
    )

    print(cameraMatrix1)
    print(distCoeffs1)
    print(f"normal method with info: {ret1}")
    print(cameraMatrix2)
    print(distCoeffs2)
    print(f"one image 6 times: {ret2}")

    # undistorted_images2 = simpler_calibrating(
    #     calibration_images=calibration_images, tag_size=tag_size, spacing=spacing
    # )
    # undistorted_images = []
    # for img in calibration_images:
    #     undistorted_image = undistort_image(img, cameraMatrix, distCoeffs)
    #     undistorted_images.append(undistorted_image)
    # #     # cv2.imshow("original image", img)   #WE CAN SHOW THE IMAGES TO SEE THE DIFFERENCE
    # #     # cv2.imshow("undistorted image", undistorted_image)
    # #     # cv2.waitKey(0)
    # cv2.imshow(
    #     "original image", calibration_images[0]
    # )  # WE CAN SHOW THE IMAGES TO SEE THE DIFFERENCE
    # cv2.imshow("undistorted image", undistorted_images[0])
    # cv2.imshow("undistorted image 2", undistorted_images2[0])
    # # FOR UNDISTORTION IT IS BETTER TO USE CALIBRATECAMERA FOR SINGLE IMAGE AS IT BETTER FITS TO THIS SPECIFIC IMAGE
    # cv2.waitKey(0)

    ######################## TASK 2 ########################
    ######################## TASK 3 ########################
    # for i in range(10):
    #     test_homography()

    ######################## TASK 4 ########################
    # left = stitching_images[0]
    # right = stitching_images[1]
    # # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # # axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    # # axes[0].set_title("Left Image")
    # # axes[0].axis("on")
    # # axes[1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    # # axes[1].set_title("Right Image")
    # # axes[1].axis("on")
    # # plt.show()

    # left_points = np.array(
    #     [[431, 1096], [424, 458], [328, 339], [362, 934], [183, 756], [584, 855]]
    # )

    # right_points = np.array(
    #     [[440, 1228], [570, 429], [337, 459], [368, 1052], [187, 864], [596, 968]]
    # )

    # homography_matrix = find_homography_matrix(left_points.T, right_points.T)
    # print(homography_matrix)
    # final_img = apply_projective_transform(right, left, homography_matrix)
    #################### TASK 5 ####################


if __name__ == "__main__":
    main()
