# homework1

########## TASK 1 ################

I created functions that sort corners so that we use order of markers from the board on images, so that I can connect it with corresponding objectpoints and use corners and object points to calibrate camera.
When undistorting images, I found out that it is usually more effective to calculate camera matrix and distrortion coefficients for one single image - the undistortion works better on this specific image, even though it probably generalizes worse. However, I prepared lists with undistorted images created by using both individual calibration and general calibration for all images.

I compared calibrating camera with all markers at once and by averaging calibrations for each marker individually. The difference was significant and resulted probably from not including information about spacing between markers in calculating intrinisic camera matrix.


########### TASK 2 #################
