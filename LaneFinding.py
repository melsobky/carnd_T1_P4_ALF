import numpy as np
import cv2
import Support as Support

error_counter = 0
good_frames = 0
error_limit = 4

old_leftx=None
old_lefty=None
old_rightx=None
old_righty=None

old_left_fit = None
old_right_fit = None

def get_lane_lines(binary_warped,is_video_frame=False):
    global error_counter, old_leftx,old_lefty,old_rightx,old_righty,old_left_fit,old_right_fit,good_frames

    leftx = None
    lefty = None
    rightx = None
    righty = None
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # perform sanity checks for video frames
    if is_video_frame is True:
        if not (old_left_fit is None or old_right_fit is None):
            #perform guided lane finding based on old lanes data
            leftx, lefty, rightx, righty = perform_guided_lane_finding(binary_warped, out_img, old_left_fit, old_right_fit)

            if not (len(leftx) ==0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0):
                left_fit, right_fit, ploty = Support.fit_polylines(binary_warped.shape[0], leftx, lefty, rightx,
                                                                   righty)

                left_fitx, right_fitx = Support.get_polylines_points(ploty, left_fit, right_fit)

                y_mid = int(binary_warped.shape[0] / 2)
                # Checking that they have similar curvature
                # Checking that they are separated by approximately the right distance horizontally
                # Checking that they are roughly parallel
                if not is_lane_curvature_accepted(ploty, leftx, lefty, rightx, righty) or not (
                    3.4 / Support.xm_per_pix) < (right_fitx[y_mid] - left_fitx[y_mid]) < (
                    4.0 / Support.xm_per_pix):
                    # checks failed
                    # increase error counter
                    error_counter = error_counter + 1
                    # check that error counts is with in the accepted error limit
                    if error_counter >= error_limit:
                        # perform a blind lane detection
                        print("Error, Performing blind lane search")
                        leftx, lefty, rightx, righty = perform_blind_lane_search(binary_warped, out_img)
                        error_counter = 0
                    else:
                        # use lane data from last detection
                        leftx = old_leftx
                        lefty = old_lefty
                        rightx = old_rightx
                        righty = old_righty
                else:
                    # save lane data for future use
                    old_leftx = leftx
                    old_lefty = lefty
                    old_rightx = rightx
                    old_righty = righty
                    good_frames = good_frames + 1
            else:
                # use lane data from last detection
                leftx = old_leftx
                lefty = old_lefty
                rightx = old_rightx
                righty = old_righty


        else:
            leftx, lefty, rightx, righty = perform_blind_lane_search(binary_warped, out_img)
            # save lane data for future use
            old_leftx = leftx
            old_lefty = lefty
            old_rightx = rightx
            old_righty = righty

        print("Error Counter :",error_counter)
    else:
        leftx, lefty, rightx, righty = perform_blind_lane_search(binary_warped, out_img)
        error_counter = 0

    left_fit, right_fit, ploty = Support.fit_polylines(binary_warped.shape[0], leftx, lefty, rightx, righty)
    left_fitx, right_fitx = Support.get_polylines_points(ploty, left_fit, right_fit)

    old_left_fit = left_fit
    old_right_fit = right_fit

    draw_polylines(ploty, left_fitx, right_fitx, out_img)
    print('Good Frames : ', good_frames)

    return out_img,leftx,lefty,rightx,righty,ploty


def is_lane_curvature_accepted(ploty, leftx, lefty, rightx, righty):
    left_curverad, right_curverad = Support.get_real_lanes_curvature(ploty, leftx, lefty, rightx, righty)


    print(left_curverad,right_curverad)
    if left_curverad > 2000 and right_curverad > 2000:
        return True # Almost straight lanes
    elif (left_curverad < 80 or right_curverad < 80):
        return False
    # elif (left_fit[0] > 0 and right_fit[0] < 0) or (right_fit[0] > 0 and left_fit[0] < 0):
    #     return False # same curvature wit opposite direction
    else:
        return True


def perform_blind_lane_search(binary_warped, out_img):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    draw_scan_boxs(left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img)
    return leftx, lefty, rightx,righty


def draw_polylines(ploty, left_fitx, right_fitx, out_img):
    lpts = []
    rpts = []
    for i in range(0, len(ploty)):
        lpts.append([left_fitx[i], ploty[i]])
        rpts.append([right_fitx[i], ploty[i]])
    lpts = np.array(lpts, np.int32)
    lpts = lpts.reshape((-1, 1, 2))
    cv2.polylines(out_img, [lpts], False, (0, 255, 255))
    rpts = np.array(rpts, np.int32)
    rpts = rpts.reshape((-1, 1, 2))
    cv2.polylines(out_img, [rpts], False, (0, 255, 255))
    return


def perform_guided_lane_finding(binary_warped,out_img,left_fit,right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    draw_scan_boxs(left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img)
    return leftx, lefty, rightx,righty


def draw_scan_boxs(left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img):
    # draw scan windows on output image
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]