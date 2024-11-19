import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    shape = (calibration_images[0].shape[1], calibration_images[0].shape[0])
    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        corners = sort_corners(corners, ids)
        objpts = object_points(tag_size, spacing)

        objptsf.append(objpts)
        imgptsf.append(corners)
    _, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
        objpts, corners, shape, None, None
    )
    return cameraMatrix, distCoeffs


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

        objptsf.append(objpts)
        imgptsf.append(corners)
        _, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
            objpts, corners, shape, None, None
        )
        undistorted_img = undistort_image(img, cameraMatrix, distCoeffs)
        undistorted_img_2.append(undistorted_img)
    return undistorted_img_2


def calibrating_6_different(calibration_images, tag_size, spacing):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    objptsf = [[], [], [], [], [], []]
    imgptsf = [[], [], [], [], [], []]
    objpts = simpler_object_points(tag_size, spacing)
    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            corners = sort_corners(corners, ids)
            shape = (img.shape[1], img.shape[0])
            for i in range(6):
                objptsf[i].append(objpts[0])
                imgptsf[i].append(corners[:, 4 * i : 4 * (i + 1), :])

    cameraMatrix = []
    distCoeffs = []
    for j in range(6):
        _, cameraMatrix_, distCoeffs_, _, _ = cv2.calibrateCamera(
            objptsf[j], imgptsf[j], shape, None, None
        )
        cameraMatrix.append(cameraMatrix_)
        distCoeffs.append(distCoeffs_)

    cameraMatrix = np.mean(cameraMatrix, axis=0)
    distCoeffs = np.mean(distCoeffs, axis=0)
    return cameraMatrix, distCoeffs


def main():
    ######################## TASK 1 ########################
    calibration_images = [cv2.imread(f"calibration\img{i}.png") for i in range(1, 29)]
    tag_size = 1.68
    spacing = 0.70

    cameraMatrix, distCoeffs = calibrating(
        calibration_images=calibration_images, tag_size=tag_size, spacing=spacing
    )

    print(cameraMatrix)
    print(distCoeffs)

    # cameraMatrix2, distCoeffs2 = calibrating_6_different(
    #     calibration_images=calibration_images, tag_size=tag_size, spacing=spacing
    # )

    # print(cameraMatrix2)
    # print(distCoeffs2)
    # print(
    #     "The results are quite different, the first method(considering all markers and distances between them at once) is more accurate, because it uses all the information given. We we will be using intrinisic camera matrix from the first method further on."
    # )

    # undistorted_images2 = simpler_calibrating(
    #     calibration_images=calibration_images, tag_size=tag_size, spacing=spacing
    # )
    # undistorted_images = []
    # for img in calibration_images:
    #     undistorted_image = undistort_image(img, cameraMatrix, distCoeffs)
    #     undistorted_images.append(undistorted_image)
    #     # cv2.imshow("original image", img)   #WE CAN SHOW THE IMAGES TO SEE THE DIFFERENCE
    #     # cv2.imshow("undistorted image", undistorted_image)
    #     # cv2.waitKey(0)
    # cv2.imshow(
    #     "original image", calibration_images[0]
    # )  # WE CAN SHOW THE IMAGES TO SEE THE DIFFERENCE
    # cv2.imshow("undistorted image", undistorted_images[0])
    # cv2.imshow("undistorted image 2", undistorted_images2[0])
    # # FOR UNDISTORTION IT IS BETTER TO USE CALIBRATECAMERA FOR SINGLE IMAGE AS IT BETTER FITS TO THIS SPECIFIC IMAGE
    # cv2.waitKey(0)

    ######################## TASK 2 ########################


if __name__ == "__main__":
    main()
