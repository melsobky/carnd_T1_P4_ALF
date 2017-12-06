import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os.path
import pickle

src = np.float32([[203, 720], [585, 460], [695, 460], [1127, 720]])

dst = np.float32([[320, 720], [320, 0], [960, 0], [960, 720]])

def get_camera_calibrations():
    cmtx = None
    cdist = None
    if os.path.isfile("camera_cal/camera_cal.p"):
        with open('camera_cal/camera_cal.p', 'rb') as f:
            out = pickle.load(f)
            cmtx = out["mtx"]
            cdist = out["dist"]
    else:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        img = cv2.imread(images[0])
        img_size = (img.shape[1], img.shape[0])
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

        # Do camera calibration given object points and image points
        ret, cmtx, cdist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = cmtx
        dist_pickle["dist"] = cdist
        pickle.dump(dist_pickle, open("camera_cal/camera_cal.p", "wb"))
    return cmtx, cdist


def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)
    return 'x=%1.4f, y=%1.4f' % (x, y)


def get_transform_matrix():
    M = cv2.getPerspectiveTransform(src, dst)
    return M

def get_inv_transform_matrix():
    invM = cv2.getPerspectiveTransform(dst, src)
    return invM


def get_thresholds(img, b_thresh=(0, 110),l_thresh=(225, 255),sx_thresh=(30,100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV).astype(np.float)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)

    l_channel = luv[:, :, 0]
    b_channel = lab[:, :, 2]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # Threshold color channel
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(l_binary)
    combined_binary[(b_binary == 1)|(l_binary == 1)|(sxbinary == 1)] = 1
    return combined_binary


def preprocess_image(img):
    ysize = img.shape[0]
    xsize = img.shape[1]

    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (dst.shape[1], dst.shape[0])
    binary = get_thresholds(dst)

    binary_warped = cv2.warpPerspective(binary, M, img_size, flags=cv2.INTER_LINEAR)

    return dst,binary,binary_warped
    # return dst,binary,None,None,None


mtx, dist = get_camera_calibrations()

M = get_transform_matrix()

dstimg = cv2.imread('./camera_cal/calibration1.jpg')
undstimg = cv2.undistort(dstimg, mtx, dist, None, mtx)
cv2.imwrite('./output_images/cal_out.jpg',undstimg)
