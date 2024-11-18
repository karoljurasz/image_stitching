import cv2
import numpy as np
import matplotlib.pyplot as plt


def object_points(width, spacing):
    objpts = []
    for j in range(3):
        for i in range(2):
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
    return objpts


def sort_corners(corners, ids):
    cos = np.argsort(ids.flatten()).astype(int)
    sorted_corners = [
        corners[cos[0]],
        corners[cos[2]],
        corners[cos[4]],
        corners[cos[1]],
        corners[cos[3]],
        corners[cos[5]],
    ]
    return sorted_corners


def undistort_image(image, cameraMatrix, distCoeffs):
    h, w = image.shape[:2]

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, (w, h), alpha=1
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
    for img in calibration_images:
        # Find ArUcO tags
        img = cv2.imread("calibration\img1.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        corners = sort_corners(corners, ids)
        objpts = object_points(tag_size, spacing)
        shape = (img.shape[1], img.shape[0])
        objptsf.append(objpts)
        imgptsf.append(corners)
    _, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
        objpts, corners, shape, None, None
    )
    return cameraMatrix, distCoeffs


def main():
    # Load your images here
    calibration_images = [cv2.imread(f"calibration\img{i}.png") for i in range(1, 29)]
    tag_size = 1.68
    spacing = 0.70

    cameraMatrix, distCoeffs = calibrating(
        calibration_images=calibration_images, tag_size=tag_size, spacing=spacing
    )

    print(cameraMatrix)
    print(distCoeffs)

    # undistorted_image = undistort_image(img, cameraMatrix, distCoeffs)
    # cv2.imshow("original image", img)
    # cv2.imshow("undistorted image", undistorted_image)
    # cv2.waitKey(0)
    # for img in images:


if __name__ == "__main__":
    main()
