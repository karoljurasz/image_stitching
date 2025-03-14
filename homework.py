import cv2
import numpy as np
import matplotlib.pyplot as plt


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
def get_1nn_pixel(approximate_pixel_coordinates):
    # the np.round() rounds to a nearest pixel_coordinates in L1 norm
    return np.round(approximate_pixel_coordinates).astype(int)


def projective_transform(source, H, display=True):
    inverse_H = np.linalg.inv(H)
    source_pixels = (
        np.dstack(np.meshgrid(np.arange(source.shape[1]), np.arange(source.shape[0])))
        .reshape(-1, 2)
        .T
    )
    # source pixels in homogeneous coordinates
    source_pixels = np.vstack((source_pixels, np.ones((1, source_pixels.shape[1]))))

    part_of_destination_points = H @ source_pixels
    part_of_destination_points = (
        part_of_destination_points / part_of_destination_points[2, :]
    )

    destination_begining = np.floor(np.min(part_of_destination_points, axis=1))[
        :2
    ].astype(int)
    destination_begining[0], destination_begining[1] = (
        destination_begining[1],
        destination_begining[0],
    )
    destination_end = np.ceil(np.max(part_of_destination_points, axis=1))[:2].astype(
        int
    )
    destination_end[0], destination_end[1] = destination_end[1], destination_end[0]

    destination_img_shape = np.array(
        list(destination_end - destination_begining + 1) + [3]
    )
    destination_pixels = (
        np.dstack(
            np.meshgrid(
                np.arange(
                    destination_begining[1], destination_end[1] + 1, dtype=np.int32
                ),
                np.arange(
                    destination_begining[0], destination_end[0] + 1, dtype=np.int32
                ),
            )
        )
        .reshape(-1, 2)
        .T
    )
    destination_pixels = np.vstack(
        [destination_pixels, np.ones((1, destination_pixels.shape[1]))]
    ).astype(np.int32)

    destination_img = np.zeros(destination_img_shape, dtype=np.uint8)

    source_pixels = source_pixels[:2].astype(int)
    part_of_destination_points = part_of_destination_points[:2]

    mask = np.ones(destination_img_shape[:2], dtype=np.bool_)

    # more robust version with using inverse transformation and 1nn
    for dp in destination_pixels.T:
        sp = inverse_H @ dp
        sp = sp / sp[2]
        sp = sp[:2]
        sp = get_1nn_pixel(sp)

        dimg_index = (dp[0] - destination_begining[1], dp[1] - destination_begining[0])
        if (
            sp[0] < 0
            or sp[0] >= source.shape[1]
            or sp[1] < 0
            or sp[1] >= source.shape[0]
        ):
            mask[dimg_index[1], dimg_index[0]] = 0
        else:
            destination_img[dimg_index[1], dimg_index[0]] = source[sp[1], sp[0]]
    if display == True:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("on")
        axes[1].imshow(cv2.cvtColor(destination_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Transformed Image")
        axes[1].axis("on")
        plt.show()

    return destination_img, mask, destination_begining, destination_end


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
    matrix = np.array(matrix)
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
    ## uncheck next line to change results visualisation
    # np.set_printoptions(precision=18, suppress=True)
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


def find_overlapping_area_and_shift(destination_begining, destination_end, base):
    # _, _, destination_begining, destination_end = projective_transform(
    #     image, H, display=False
    # )
    x_min, y_min = max(0, destination_begining[1]), max(0, destination_begining[0])
    x_max, y_max = min(base.shape[1], destination_end[1]), min(
        base.shape[0], destination_end[0]
    )
    shift = [-destination_begining[0], -destination_begining[1]]
    return x_min, y_min, x_max, y_max, shift


def some_dynamic_seam(image, base, H):
    img, _, destination_begining, destination_end = projective_transform(
        image, H, display=False
    )
    image = img

    x_min, y_min, x_max, y_max, shift = find_overlapping_area_and_shift(
        destination_begining, destination_end, base
    )

    image = image[
        y_min + shift[0] : y_max + shift[0], x_min + shift[1] : x_max + shift[1]
    ]
    base = base[y_min:y_max, x_min:x_max]

    h, w = image.shape[:2]

    cost_table = np.zeros((h, w))

    # Calculate the initial cost for the first row
    for j in range(w):
        diff = np.abs(image[0, j] - base[0, j])
        grayscale_diff = 0.3 * diff[0] + 0.59 * diff[1] + 0.11 * diff[1]
        cost_table[0, j] = grayscale_diff**2

    # Fill the cost table using dynamic programming
    for i in range(1, h):
        for j in range(w):
            diff = np.abs(image[i, j] - base[i, j])
            grayscale_diff = 0.11 * diff[0] + 0.59 * diff[1] + 0.3 * diff[2]
            cost = np.sqrt(grayscale_diff)

            if j == 0:
                cost_table[i, j] = cost + min(
                    cost_table[i - 1, j], cost_table[i - 1, j + 1]
                )
            elif j == w - 1:
                cost_table[i, j] = cost + min(
                    cost_table[i - 1, j - 1], cost_table[i - 1, j]
                )
            else:
                cost_table[i, j] = cost + min(
                    cost_table[i - 1, j - 1],
                    cost_table[i - 1, j],
                    cost_table[i - 1, j + 1],
                )

    # Backtrack to find the optimal seam
    seam = []
    min_cost_index = int(np.argmin(cost_table[-1, :]))
    seam.append((h - 1, min_cost_index))

    for i in range(h - 2, -1, -1):
        j = seam[-1][1]
        if j == 0:
            min_cost_index = int(np.argmin(cost_table[i, j : j + 2]))
            seam.append((i, j + min_cost_index))
        elif j == w - 1:
            min_cost_index = int(np.argmin(cost_table[i, j - 1 : j + 1]))
            seam.append((i, j - 1 + min_cost_index))
        else:
            min_cost_index = int(np.argmin(cost_table[i, j - 1 : j + 2]))
            seam.append((i, j - 1 + min_cost_index))

    seam.reverse()
    return seam


def get_final_image_shape_and_coordinates_of_images(
    destination_begining, destination_end, base
):
    x_min, y_min, x_max, y_max, shift = find_overlapping_area_and_shift(
        destination_begining, destination_end, base
    )
    h, w = base.shape[:2]
    len_a = max(destination_end[1], w) - min(destination_begining[1], 0) + 1
    len_b = max(destination_end[0], h) - min(destination_begining[0], 0) + 1
    image_beg_coords = (
        max(0, destination_begining[0]),
        max(0, destination_begining[1]),
    )
    image_end_coords = (
        image_beg_coords[0] + destination_end[0] - destination_begining[0] + 1,
        image_beg_coords[1] + destination_end[1] - destination_begining[1] + 1,
    )
    if destination_begining[0] > 0:
        base_beg_coords1 = 0
        base_end_coords1 = h
    else:
        base_beg_coords1 = -destination_begining[0]
        base_end_coords1 = -destination_begining[0] + h

    if destination_begining[1] > 0:
        base_beg_coords2 = 0
        base_end_coords2 = w
    else:
        base_beg_coords2 = -destination_begining[1]
        base_end_coords2 = -destination_begining[1] + w
    base_beg_coords = (base_beg_coords1, base_beg_coords2)
    base_end_coords = (base_end_coords1, base_end_coords2)
    final_img_shape = (len_b, len_a, 3)
    return (
        final_img_shape,
        image_beg_coords,
        image_end_coords,
        base_beg_coords,
        base_end_coords,
    )


def stich_images(image, base, seam, destination_begining, destination_end):
    x_min, y_min, x_max, y_max, shift = find_overlapping_area_and_shift(
        destination_begining, destination_end, base
    )

    (
        final_img_shape,
        image_beg_coords,
        image_end_coords,
        base_beg_coords,
        base_end_coords,
    ) = get_final_image_shape_and_coordinates_of_images(
        destination_begining, destination_end, base
    )

    final_img = np.zeros(shape=final_img_shape)

    final_img[
        image_beg_coords[0] : image_end_coords[0],
        image_beg_coords[1] : image_end_coords[1],
        :,
    ] = image
    final_img[
        base_beg_coords[0] : base_end_coords[0],
        base_beg_coords[1] : base_end_coords[1],
    ] = base

    overlapping_img = image[
        y_min + shift[0] : y_max + shift[0], x_min + shift[1] : x_max + shift[1]
    ]

    # Check which image is on the left

    img_on_the_left = x_min == 0
    if img_on_the_left:
        for i in range(len(seam)):
            final_img[
                max(image_beg_coords[0], base_beg_coords[0]) + i,
                base_beg_coords[1] : base_beg_coords[1] + seam[i][1],
            ] = overlapping_img[i, : seam[i][1]]
            # # checking the sewing line
            # final_img[
            #     max(image_beg_coords[0], base_beg_coords[0]) + i,
            #     base_beg_coords[1] + seam[i][1],
            # ] = [
            #     0,
            #     240,
            #     0,
            # ]
            # # end of checking the sewing line
    else:
        for i in range(len(seam)):
            final_img[
                max(image_beg_coords[0], base_beg_coords[0]) + i,
                image_beg_coords[1] + seam[i][1] : base_end_coords[1],
            ] = overlapping_img[
                i, seam[i][1] :  # base_end_coords[1] - image_beg_coords[1]
            ]

            # #  checking the sewing line
            # final_img[
            #     max(image_beg_coords[0], base_beg_coords[0]) + i,
            #     image_beg_coords[1] + seam[i][1] - 1,
            # ] = [
            #     0,
            #     240,
            #     0,
            # ]
            # # end of checking the sewing line
    return final_img.astype(np.uint8)


########### TASK 6 FUNCTIONS ############
########### TASK 7 FUNCTIONS ############
def quantify_error(H, source, destination):
    source = np.vstack([source, np.ones((1, source.shape[1]))])

    destination = np.vstack([destination, np.ones((1, destination.shape[1]))])

    destination_hat = H @ source

    destination_hat = destination_hat / destination_hat[2, :]
    return np.linalg.norm(destination_hat[:2] - destination[:2], ord=2, axis=0)


def RANSAC_find_homography_matrix(
    source, destination, n_interations, n_drop, epsilon=1e-3
):
    N = len(source[0])

    indexes = np.arange(N)
    best_number_of_inliers = 0
    best_H = None
    best_indexes = None

    for _ in range(n_interations):
        sample_indexes = np.random.choice(indexes, N - n_drop, replace=False)

        source_sample = source[:, sample_indexes].reshape(2, -1)
        destination_sample = destination[:, sample_indexes].reshape(2, -1)
        H = find_homography_matrix(source_sample, destination_sample)
        error = quantify_error(H, source.reshape(2, -1), destination.reshape(2, -1))
        number_of_inliers = np.sum(error < epsilon)
        if number_of_inliers > best_number_of_inliers:
            best_number_of_inliers = number_of_inliers
            best_H = H
            best_indexes = sample_indexes

    return best_H, best_indexes


######################## MAIN FUNCTION  ########################


def main():

    ## don't comment this part
    calibration_images = [cv2.imread(f"calibration\img{i}.png") for i in range(1, 29)]
    stitching_images = [cv2.imread(f"stitching\img{i}.png") for i in range(1, 10)]
    tag_size = 1.68
    spacing = 0.70

    ######################## TASK 1 ########################

    # cameraMatrix, distCoeffs, ret = calibrating(
    #     calibration_images=calibration_images, tag_size=tag_size, spacing=spacing
    # )
    # ####checking the results
    # print(cameraMatrix)
    # print(distCoeffs)

    # ###### testing errors for different methods ######
    # test_img = [
    #     calibration_images[5],
    #     calibration_images[6],
    #     calibration_images[7],
    #     calibration_images[8],
    #     calibration_images[9],
    #     calibration_images[10],
    #     calibration_images[11],
    # ]
    # cameraMatrix1, distCoeffs1, ret1 = calibrating(
    #     calibration_images=test_img, tag_size=tag_size, spacing=spacing
    # )
    # cameraMatrix2, distCoeffs2, ret2 = calibrating_6_different(
    #     calibration_images=test_img, tag_size=tag_size, spacing=spacing
    # )

    # print(f"normal method with spacing info error: {ret1}")

    # print(f"one image 6 times error: {ret2}")

    # undistorted_images2 = simpler_calibrating(
    #     calibration_images=calibration_images, tag_size=tag_size, spacing=spacing
    # )
    # undistorted_images = []
    # for img in calibration_images:
    #     undistorted_image = undistort_image(img, cameraMatrix, distCoeffs)
    #     undistorted_images.append(undistorted_image)

    # ## WE CAN SHOW THE IMAGES TO SEE THE DIFFERENCE
    # cv2.imshow("original image", calibration_images[0])
    # cv2.imshow("undistorted image", undistorted_images[0])
    # cv2.imshow("undistorted image 2", undistorted_images2[0])
    # cv2.waitKey(0)

    ##folders with undistorted images: undistorted_images, undistorted_images2

    ######################## TASK 2 ########################
    ######################## TASK 3 ########################
    # for i in range(10):
    #     test_homography()

    ######################## TASK 4 ########################

    ## images I compare

    left = stitching_images[0]
    right = stitching_images[1]

    ## Plots section 1 - display images to find corresponding points

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    # axes[0].set_title("Left Image")
    # axes[0].axis("on")
    # axes[1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    # axes[1].set_title("Right Image")
    # axes[1].axis("on")
    # plt.show()

    ## found these corresponding points:

    left_points = np.array(
        [[1096, 431], [458, 424], [339, 328], [934, 362], [756, 183], [855, 584]]
    )

    right_points = np.array(
        [[1228, 440], [570, 429], [459, 337], [1052, 368], [864, 187], [968, 596]]
    )

    ## Plots section 2 - just to see if the points coordinates aren't reversed

    # for point in left_points:
    #     cv2.circle(left, tuple(point), 5, (0, 0, 255), -1)
    # for point in right_points:
    #     cv2.circle(right, tuple(point), 5, (0, 0, 255), -1)

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    # axes[0].set_title("Left Image")
    # axes[0].axis("on")
    # axes[1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    # axes[1].set_title("Right Image")
    # axes[1].axis("on")
    # plt.show()

    ## End of checking

    #### Finding projective transformation

    # homography_matrix = find_homography_matrix(left_points.T, right_points.T)
    # print(homography_matrix)
    # final_img, _, _, _ = projective_transform(left, homography_matrix, display=True)

    ######################## TASK 5 ########################
    # left = stitching_images[0]
    # right = stitching_images[1]
    # left_points = np.array(
    #     [[1096, 431], [458, 424], [339, 328], [934, 362], [756, 183], [855, 584]]
    # )

    # right_points = np.array(
    #     [[1228, 440], [570, 429], [459, 337], [1052, 368], [864, 187], [968, 596]]
    # )
    # homography_matrix = find_homography_matrix(left_points.T, right_points.T)
    # final_img, _, beg, end = projective_transform(
    #     left, homography_matrix, display=False
    # )
    # seam = some_dynamic_seam(left, right, homography_matrix)
    ## If you want to sea the seam line, uncomment this or two segments in stitch_images function
    # # print(seam)

    # final_img = stich_images(final_img, right, seam, beg, end)

    # plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # cv2.imwrite("task_5_stitched.jpg", final_img)

    ################### TASK 6 ####################

    # # First run this:
    # text = "img1.png img2.png"
    # with open("stitching_pairs_task_6.txt", "w") as file:
    #     file.write(text)
    # # After preapring proper folders(stitching_pairs_task_6.txt, stitching_task_6 folder) i want to use this command in cmd:

    # python superglue\SuperGluePretrainedNetwork\match_pairs.py --resize -1 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir stitching_task_6/ --input_pairs stitching_pairs_task_6.txt --output_dir matching_pairs --viz
    # in my folder I want to have img2(as base) and img1 and then do some standard things as I read file with matches, find homography and stitch images

    # npz = np.load("matching_pairs/img1_img2_matches.npz")
    # left_points = npz["keypoints0"]
    # new_left_points = []
    # right_points = []
    # for i in range(len(left_points)):
    #     if npz["matches"][i] != -1:
    #         right_points.append(npz["keypoints1"][npz["matches"][i]])
    #         new_left_points.append(left_points[i])

    # left_points = np.array(new_left_points)
    # right_points = np.array(right_points)

    # homography_matrix = find_homography_matrix(left_points.T, right_points.T)
    # final_img, _, beg, end = projective_transform(
    #     left, homography_matrix, display=False
    # )
    # seam = some_dynamic_seam(left, right, homography_matrix)
    # final_img = stich_images(final_img, right, seam, beg, end)
    # plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # cv2.imwrite("task_6_stitched.jpg", final_img)

    ######################## TASK 7 ########################

    # # Run this first to have proper files:
    # current_image = cv2.imread("task_6_stitched.jpg")
    # for i in range(4):
    #     text = f"current_image.png img{i+3}.png"
    #     with open(f"add_img_{i+3}.txt", "w") as file:
    #         file.write(text)

    ########## THE PROCEDURE (don't uncomment this, it just shows the way I do things later on) #########
    ##These 3 lines first:
    # for i in range(4):
    # base = cv2.imread("stitching_task_7_folder\current_image.png")
    # left = cv2.imread(f"img{i+3},png")
    # Run this in cmd:
    # #python superglue\SuperGluePretrainedNetwork\match_pairs.py --resize -1 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir stitching_task_7_folder/ --input_pairs add_img_{i+3}.txt --output_dir matching_pairs_{i} --viz
    # npz = np.load("matching_pairs/img1_img2_matches.npz")
    # base_points = npz["keypoints0"]
    # new_base_points = []
    # left_points = []
    # for i in range(len(base_points)):
    #     if npz["matches"][i] != -1:
    #         left_points.append(npz["keypoints1"][npz["matches"][i]])
    #         new_base_points.append(base_points[i])

    # left_points = np.array(left_points)
    # base_points = np.array(base_points)

    # homography_matrix = find_homography_matrix(left_points.T, base_points.T)
    # final_img, _, beg, end = projective_transform(
    #     left, homography_matrix, display=False
    # )
    # seam = some_dynamic_seam(left, base, homography_matrix)
    # final_img = stich_images(final_img, base, seam, beg, end)
    # cv2.imwrite("stitching_task_7_folder/current_image.png", final_img)
    ########## END OF THE PROCEDURE #########

    ####### ADD IMAGE 3 #############
    # Run this first to fill the folder with current_image.png
    # base = cv2.imread("task_6_stitched.jpg")
    # left = cv2.imread(f"stitching_task_7_folder\img3.png")
    # cv2.imwrite("stitching_task_7_folder/current_image.png", base)

    ## cmd time:
    # # python superglue\SuperGluePretrainedNetwork\match_pairs.py --resize -1 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir stitching_task_7_folder/ --input_pairs add_img_3.txt --output_dir matching_pairs_3 --viz
    # base = cv2.imread("task_6_stitched.jpg")
    # left = cv2.imread(f"stitching_task_7_folder\img3.png")
    # npz = np.load("matching_pairs_3\current_image_img3_matches.npz")
    # base_points = npz["keypoints0"]
    # new_base_points = []
    # left_points = []
    # for i in range(len(base_points)):
    #     if npz["matches"][i] != -1:
    #         left_points.append(npz["keypoints1"][npz["matches"][i]])
    #         new_base_points.append(base_points[i])

    # left_points = np.array(left_points)
    # base_points = np.array(new_base_points)

    # homography_matrix, indexes = RANSAC_find_homography_matrix(
    #     left_points.T, base_points.T, n_interations=70, n_drop=10, epsilon=1
    # )
    # print("RANSAC finished")
    # base_points = base_points[indexes]
    # left_points = left_points[indexes]
    # final_img, _, beg, end = projective_transform(
    #     left, homography_matrix, display=False
    # )
    # seam = some_dynamic_seam(left, base, homography_matrix)
    # final_img = stich_images(final_img, base, seam, beg, end)
    # cv2.imwrite("stitching_task_7_folder/current_image.png", final_img)
    # cv2.imwrite("stitching_task_7_folder/image_after_3.png", final_img)

    # ############ ADD IMAGE 4 ########

    ## cmd time:
    # # python superglue\SuperGluePretrainedNetwork\match_pairs.py --resize -1 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir stitching_task_7_folder/ --input_pairs add_img_4.txt --output_dir matching_pairs_4 --viz
    # base = cv2.imread("stitching_task_7_folder\current_image.png")
    # left = cv2.imread(f"stitching_task_7_folder\img4.png")
    # npz = np.load("matching_pairs_4\current_image_img4_matches.npz")
    # base_points = npz["keypoints0"]
    # new_base_points = []
    # left_points = []
    # for i in range(len(base_points)):
    #     if npz["matches"][i] != -1:
    #         left_points.append(npz["keypoints1"][npz["matches"][i]])
    #         new_base_points.append(base_points[i])

    # left_points = np.array(left_points)
    # base_points = np.array(new_base_points)

    # homography_matrix, indexes = RANSAC_find_homography_matrix(
    #     left_points.T, base_points.T, n_interations=300, n_drop=10, epsilon=1
    # )
    # print("RANSAC finished")
    # base_points = base_points[indexes]
    # left_points = left_points[indexes]

    # final_img, _, beg, end = projective_transform(
    #     left, homography_matrix, display=False
    # )
    # seam = some_dynamic_seam(left, base, homography_matrix)
    # final_img = stich_images(final_img, base, seam, beg, end)
    # cv2.imwrite("stitching_task_7_folder/current_image.png", final_img)
    # cv2.imwrite("stitching_task_7_folder/image_after_4.png", final_img)

    ############ ADD IMAGE 6 ########

    ## cmd time:
    ## python superglue\SuperGluePretrainedNetwork\match_pairs.py --resize -1 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir stitching_task_7_folder/ --input_pairs add_img_6.txt --output_dir matching_pairs_6 --viz
    base = cv2.imread("stitching_task_7_folder\current_image.png")
    left = cv2.imread(f"stitching_task_7_folder\img6.png")

    npz = np.load("matching_pairs_6\current_image_img6_matches.npz")
    base_points = npz["keypoints0"]
    new_base_points = []
    left_points = []
    for i in range(len(base_points)):
        if npz["matches"][i] != -1:
            left_points.append(npz["keypoints1"][npz["matches"][i]])
            new_base_points.append(base_points[i])

    left_points = np.array(left_points)
    base_points = np.array(new_base_points)

    homography_matrix, indexes = RANSAC_find_homography_matrix(
        left_points.T, base_points.T, n_interations=300, n_drop=10, epsilon=1
    )
    print("RANSAC finished")
    base_points = base_points[indexes]
    left_points = left_points[indexes]

    final_img, _, beg, end = projective_transform(
        left, homography_matrix, display=False
    )
    seam = some_dynamic_seam(left, base, homography_matrix)
    final_img = stich_images(final_img, base, seam, beg, end)
    cv2.imwrite("task_7_stitched.jpg", final_img)
    cv2.imwrite("stitching_task_7_folder/image_after_6.png", final_img)


if __name__ == "__main__":
    main()
