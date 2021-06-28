"""
A script for calculating the 4x4 hand eye transformation by using the data collected by 'calibrate_collection.py' 
and the transformations from the camera to the marker calculated by the Matlab computer vision toolbox. 
The result will be saved as './result_calibration/eye_to_hand_matlab.npy'.
"""
import numpy as np 
import cv2 as cv 
import scipy.io as io 

mat_idx = 4
rot_path = './cam_calibration/{}_rot.mat'.format(mat_idx)
trans_path = './cam_calibration/{}_trans.mat'.format(mat_idx)
intr_path = './cam_calibration/{}_intr_mat.mat'.format(mat_idx)
rot = io.loadmat(rot_path)['rot'] 
tvecs = io.loadmat(trans_path)['trans'] / 1000. # convert from millimeters to meters
rvecs = rot.transpose(2, 1, 0)
num = len(tvecs)
# reading the tcp poses
tcp_r = []
tcp_t = []
for i in range(num):
    tcp_r.append(np.load('./save_calibration/data{}/r_{}.npy'.format(mat_idx, i)))
    tcp_t.append(np.load('./save_calibration/data{}/t_{}.npy'.format(mat_idx, i)))

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
np.save('./result_calibration/eye_to_hand_matlab.npy', eye_to_hand_mat)