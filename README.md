# homework1

########## TASK 1 ################

I created functions that sort corners so that we use order of markers from the board on images, so that I can connect it with corresponding objectpoints and use corners and object points to calibrate camera. Unexpectedly, the error with cutting object 6 times is smaller - this could occur due to the bigger ease to undistort 4 points precisely than to do it with 24 points, and moreover, model has more images to fit to(6 compared to 1)

When undistorting images, I found out that it is usually more effective to calculate camera matrix and distortion coefficients for one single image - the undistortion works better on this specific image, even though it probably generalizes worse. However, I prepared lists with undistorted images created by using both individual calibration and general calibration for all images.

I compared calibrating camera with all markers at once and by averaging calibrations for each marker individually. The difference was significant and resulted probably from not including information about spacing between markers in calculating intrinisic camera matrix.

########### TASK 2 #################

I created the needed function with both backward and forward homography and using a nearest neighbor to find a pixel in the source image.

########### TASK 3 #################

I created function find_homography_matrix(), that finds the homography between two images based on a sequence of matching point coordinates. I prepared functions testing it, ensuring our random homography is invertible(otherwise it would transform an image into one of dimension 1 or 0, which is not what homography does). I've shown that in all/almost all of test examples the calculated homography matrix doesn't differ too much from original homography

############ TASK 4 #################

I took two images from the "stitching" folder (img1 and img2) and identified 6 correspoinding points coordinates. Using them as ground truth, I have found the homography matrix. I also highlighted the points and checked whether the transormation doesn't do something crazy to some example image.

############ TASK 5 #################
