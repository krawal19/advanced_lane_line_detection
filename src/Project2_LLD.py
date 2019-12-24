'''
 * BSD 3-Clause License
 * @copyright (c) 2019, Krishna Bhatu, Hrishikesh Tawade, Kapil Rawal
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * @file    Project2_LLD.py
 * @author  Krishna Bhatu, Hrishikesh Tawade, Kapil Rawal
 * @version 1.0
 * @brief  Lane Line detection for project video and challenge video 
 *
 '''

import numpy as np
import cv2
from moviepy.editor import VideoFileClip

# unwarping an image
def unwarp_single_image(img):
    h,w = img.shape[:2]
    src = np.float32([ (0.44*w,0.64*h), 
              (0.58*w, 0.64*h),
              (0.10*w,h),
              (0.95*w,h)])
    dst = np.float32([(450,0),
          (w-450,0),
          (450,h),
          (w-450,h)])

    # use cv2.getPerspectiveTransform() to get M
    M = cv2.getPerspectiveTransform(src, dst)  
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped_image = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR) # get bird's eye view
    return unwarped_image,M,Minv

# HLS thershold
def hls_lthresh(img, thresh=(220, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

# Lab thershold
def lab_bthresh(img, thresh=(190,255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the B channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

# undistort an input image
def undistort_image(sample):
    mtx = np.array([[1.15422732e+03,0.00000000e+00,6.71627794e+02],[0.00000000e+00,1.14818221e+03,3.86046312e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    dist =  np.array([[-2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02]])
    undistorted_image = cv2.undistort(sample, mtx, dist, None, mtx)
    return undistorted_image

# Define method to fit polynomial to binary image with lines extracted, using sliding window
def sliding_window_polyfit(img):
    global kalman
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    points = np.array([leftx_base,rightx_base], np.float32)
    prediction = predictKalman(points)
    
    # Current positions to be updated for each window
    '''
    leftx_current = leftx_base
    rightx_current = rightx_base
    '''
    leftx_current = prediction[0]
    rightx_current = prediction[1]
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    visualization_data = (rectangle_data, histogram)
    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data ,leftx_base ,rightx_base 

# Method to determine radius of curvature and distance from lane center 
# based on binary image, polynomial fit, and L and R lane pixel indices
def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
  
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds] 
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        
    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1]/2
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist


def draw_lane(original_img, binary_img, l_fit, r_fit, Minv, leftx_base ,rightx_base):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    m_left = (ploty[0]- ploty[int(len(ploty)/2)])/(left_fitx[0]-left_fitx[int(len(left_fitx)/2)])
    m_right = (ploty[0]- ploty[int(len(ploty)/2)])/(right_fitx[0]-right_fitx[int(len(right_fitx)/2)])
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    
    return result,m_left,m_right

# Draws data over video frames
def draw_data(original_img, curv_rad, center_dist, m_left, m_right):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX

    global prev_curv_rad
    
    if(int(curv_rad) == 0):
       curv_rad = prev_curv_rad
       
    text = "Radius of Curvature: {} m".format(int(curv_rad))
    cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

    prev_curv_rad = curv_rad
    
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

    if (0 < m_left < 25):
        text = "Left Curve"
    elif(-25 < m_left < 0):
        text = "Right Curve"
    else:
        text = "Straight Road"
    cv2.putText(new_img, text, (40,170), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img

# Check for current, best fit fot the lane curve
def add_fit(fit,inds, best_fit, current_fit): #l_fit, l_lane_inds from sliding window
     diffs = np.array([0,0,0], dtype='float')
     if fit is not None:
         if best_fit is not None:
            # if we have a best fit, see how this new fit compares
            diffs = abs(fit- best_fit)
         if (diffs[0] > 0.001 or diffs[1] > 1.0 or diffs[2] > 100.) and len(current_fit) > 0:
            # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
             detected = False
         else:
            detected = True
            px_count = np.count_nonzero(inds) # What is this used for?
            current_fit.append(fit)
            if len(current_fit) > 5:
                # throw out old fits, keep newest n
                current_fit = current_fit[len(current_fit)-5:]
                best_fit = np.average(current_fit, axis=0)
        # or remove one from the history, if not found
     else:
         detected = False
         if len(current_fit) > 0:
            # throw out oldest fit
            current_fit = current_fit[:len(current_fit)-1]
         if len(current_fit) > 0:
            # if there are still any fits in the queue, best_fit is their average
            best_fit = np.average(current_fit, axis=0)
     return detected,current_fit,best_fit

# Kalman filter for lane estimation
def initKalman():
    global kalman
    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

    kalman.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]],np.float32)

    kalman.processNoiseCov = np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]],np.float32) * 0.03

    measurement = np.array((2,1), np.float32)
    prediction = np.zeros((2,1), np.float32)
    
# Predicts the kalman values
def predictKalman(points):
    kalman.correct(points)
    prediction = kalman.predict()
    return prediction

def pipeline(img):
    
    # Perspective Transform
    img_unwarp,M,Minv = unwarp_single_image(img)
    
    # HLS L-channel Threshold (using default parameters)
    img_LThresh = hls_lthresh(img_unwarp)
    
    # Lab B-channel Threshold (using default parameters)
    img_BThresh = lab_bthresh(img_unwarp)
    
    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_BThresh)
    combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1
    return combined, Minv

def process_image(img1):
    global l_line_detected
    global r_line_detected
    global l_best_fit
    global l_current_fit
    global r_best_fit
    global r_current_fit
    
    old_img = np.copy(img1)
    new_img = np.copy(img1)
    #new_img = undistort_image(new_img)
    
    img_bin, Minv = pipeline(new_img)

    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window  
    l_fit, r_fit, l_lane_inds, r_lane_inds, _, leftx_base ,rightx_base = sliding_window_polyfit(img_bin)

    # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)    
    if l_fit is not None and r_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        h = img1.shape[0]
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        x_int_diff = abs(r_fit_x_int-l_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            l_fit = None
            r_fit = None

    l_line_detected,l_line_current_fit,l_line_best_fit = add_fit(l_fit,l_lane_inds, l_best_fit, l_current_fit)
    r_line_detected,r_line_current_fit,r_line_best_fit = add_fit(r_fit,r_lane_inds, r_best_fit, r_current_fit)

    l_best_fit = l_line_best_fit
    l_current_fit = l_line_current_fit

    r_best_fit = r_line_best_fit
    r_current_fit = r_line_current_fit

    # draw the current best fit if it exists
    
    if l_line_best_fit is not None and r_line_best_fit is not None:
        
        img_out1, m_left, m_right= draw_lane(old_img, img_bin, l_line_best_fit, r_line_best_fit, Minv, leftx_base ,rightx_base)
        
        rad_l, rad_r, d_center= calc_curv_rad_and_center_dist(img_bin, l_line_best_fit, r_line_best_fit, 
                                                               l_lane_inds, r_lane_inds)
        img_out = draw_data(img_out1, (rad_l+rad_r)/2, d_center, m_left,m_right)
  
    else:
        
        img_out = old_img
    return img_out

# Takes input od src and dst of video path and starts the LLD
def testVideo1(srcPath,dstPath):
    initKalman()
    video_output1 = dstPath
    video_input1 = VideoFileClip(srcPath)
    processed_video = video_input1.fl_image(process_image)
    processed_video.write_videofile(video_output1, audio=False)

l_line_detected = False
r_line_detected = False
l_best_fit = None
l_current_fit = []
r_best_fit = None
r_current_fit = []
kalman  = None
prev_curv_rad = 0

# Lane Line detection for Challenge video
testVideo1('project_video.mp4', 'project_video_output.mp4')

# Lane Line detection for Challenge video 
testVideo1('challenge_video.mp4','challenge_video_output.mp4')

print("All videos Processed. Please check for output videos in folder of code")
