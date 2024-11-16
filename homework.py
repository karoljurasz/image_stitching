import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_aruco_tags(images):
    # Initialize the dictionary and parameters for ArUcO detection
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    all_corners = []
    all_ids = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            all_corners.append(corners)
            all_ids.append(ids)

    return all_corners, all_ids


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
    print(f"cos: {cos}")
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
    # Get image dimensions
    h, w = image.shape[:2]

    # Optimize the camera matrix based on a balance between field of view and no pixel loss
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, (w, h), alpha=1
    )

    # Compute the undistortion and rectification transformation map
    R = np.eye(3)  # No additional rotation #+0.1 rotate
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix, distCoeffs, R, new_camera_matrix, (w, h), m1type=cv2.CV_32FC1
    )

    # Apply the remap to get the undistorted image
    undistorted_image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    # Crop the image using the ROI if desired (optional)
    # x, y, w, h = roi
    # undistorted_image = undistorted_image[y:y+h, x:x+w]

    return undistorted_image


def main():
    # Load your images here
    calibration_images = [cv2.imread(f"calibration\img{i}.png") for i in range(1, 29)]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    # Find ArUcO tags
    img = cv2.imread("calibration\img1.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    print(ids)
    print()
    tag_size = 1.68
    spacing = 0.70
    corners = sort_corners(corners, ids)
    objpts = object_points(tag_size, spacing)
    shape = (img.shape[1], img.shape[0])

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpts, corners, shape, None, None
    )

    undistorted_image = undistort_image(img, cameraMatrix, distCoeffs)
    cv2.imshow("original image", img)
    cv2.imshow("undistorted image", undistorted_image)
    cv2.waitKey(0)
    # for img in images:


if __name__ == "__main__":
    main()
