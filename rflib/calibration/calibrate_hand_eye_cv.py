"""
This script calculate the hand eye transformation by using the data collected by 'calibrate_collection.py'.
The result is a 4x4 matrix, will be saved as './result_calibration/eye_to_hand_mat.npy'
"""
import numpy as np 
import cv2 as cv 

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_pos = np.zeros((4*7, 3), np.float32)
obj_pos[:, :2] = np.mgrid[0:7, 0:4].T.reshape(-1, 2)
obj_pos *= 0.029

obj_pts = []
img_pts = []

num = 16

imgs = ['./save_calibration/{}.jpg'.format(i) for i in range(num)]

for idx, fname in enumerate(imgs):
    print(fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 4), None)

    if ret == True:
        obj_pts.append(obj_pos)

        corner2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_pts.append(corners)

        cv.drawChessboardCorners(img, (7, 4), corner2, ret)
        cv.imwrite('{}.jpg'.format(idx), img)
        
    else:
        print('failed')

_, mtx, _, rvecs, tvecs = cv.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)
# reading the tcp poses
tcp_r = []
tcp_t = []
for i in range(num):
    tcp_r.append(np.load('./save_calibration/r_{}.npy'.format(i)))
    tcp_t.append(np.load('./save_calibration/t_{}.npy'.format(i)))

result_r = np.zeros((3, 3))
result_t = np.zeros((3, 1))
cv.calibrateHandEye(tcp_r, tcp_t, rvecs, tvecs, result_r, result_t, cv.CALIB_HAND_EYE_PARK)
print(result_r)
print(result_t)
eye_to_hand_mat = np.zeros((4, 4))
eye_to_hand_mat[:3, :3] = result_r
eye_to_hand_mat[:3, 3] = result_t[:, 0]
eye_to_hand_mat[3, 3] = 1.
print(eye_to_hand_mat)
np.save('./result_calibration/eye_to_hand_mat.npy', eye_to_hand_mat)