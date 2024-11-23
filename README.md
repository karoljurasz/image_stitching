

########## TASK 1 ################

I created functions that sort corners so that we use order of markers from the board on images, so that I can connect it with corresponding objectpoints and use corners and object points to calibrate camera. Unexpectedly, the error with cutting object 6 times is smaller - this could occur due to the bigger ease to undistort 4 points precisely than to do it with 24 points, and moreover, model has more images to fit to(6 compared to 1)

When undistorting images, I found out that it is usually more effective to calculate camera matrix and distortion coefficients for one single image - the undistortion works better on this specific image, even though it probably generalizes worse. However, I prepared lists with undistorted images created by using both individual calibration and general calibration for all images.

I compared calibrating camera with all markers at once and by averaging calibrations for each marker individually. The difference was significant and resulted probably from not including information about spacing between markers in calculating intrinisic camera matrix.

Uncomment the whole section to run the Task files, you can check some plots in the end. The variables with undistorted images are given at the bottom of the task main section.

########### TASK 2 #################

I created the needed function with both backward and forward homography and using a nearest neighbor to find a pixel in the source image.

########### TASK 3 #################

I created function find_homography_matrix(), that finds the homography between two images based on a sequence of matching point coordinates. I prepared functions testing it, ensuring our random homography is invertible(otherwise it would transform an image into one of dimension 1 or 0, which is not what homography does). I've shown that in all/almost all of test examples the calculated homography matrix doesn't differ too much from original homography. In test_homography you can uncomment one line to change the way the results are presented

############ TASK 4 #################

I took two images from the "stitching" folder (img1 and img2) and identified 6 correspoinding points coordinates. Using them as ground truth, I have found the homography matrix. I also highlighted the points and checked whether the transformation doesn't do something crazy to some example image. I recommend uncommenting plots section one by one, not uncommenting everything at once.

############ TASK 5 #################

I created some functions for stitching, in order to be able to create an empty image of proper size, be able to identify where component images shoukd lie on this, to create sime line in overlapping area and to always know where correspodnding points lie. The results looks pretty good. You could uncomment part of stitching function to see the sewing line.

########### TASK 6 ##################

After preparing proper folders I used this command in cmd[as I am using Windows]:
    python superglue\SuperGluePretrainedNetwork\match_pairs.py --resize -1 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir stitching_task_6/ --input_pairs stitching_pairs_task_6.txt --output_dir matching_pairs --viz
In my folder I have img2(as base) and img1 and then do some standard things as I read file with matches, find homography, project img1 on img2 and stitch images. The obstacles were finding matches and reading the file, rest was pretty standard.
No RANSAC was needed.

########### TASK 7 ###############

Having already 2 images stitched, I want to stitch 3 more images to it. So i repetively compare current stitched images with new image to add on the image plane, find correspodning points, find homography and stitch. I prepared the folder with images  1-2(stitched together as current_img) 3,4,6 that I want to add. In the code there is shown place where I use terminal to run the superglue algorithm and the code I use. I prepared the first folder and pairs to stitch manually, then I started overwriting the current_image file that I use as a projection plane for next images. 

There was a need to use RANSAC, I manually changed the accepted error so the image looks good.

As there is a need to use terminal, at first I showed the concept and then I iterated 4 times, showing where the terminal was used.

Note that after adding image 4 the image seemed pretty clean and only after adding image 6 it started to look somehow strange - which porbably is because the camera in image 6 seems to be in a bit different place - which contradicts the assumptions and may result in worse final stitched image.

You can check images on the way in stitching_task_7_folder (image_after_i)
