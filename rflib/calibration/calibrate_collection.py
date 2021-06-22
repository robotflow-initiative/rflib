"""
A script used to collect robot arm poses and realsense images for hand eye calibration.
Move the arm to different poses and make sure the markers can be seen from the camera, 
then press 's' to save the data. Generally, 15-20 datas are enough for calibration.
"""
import os 
import sys 
import numpy as np
import time 

from pose import quat_to_mat
from flexiv import FlexivRobot

import pyrealsense2 as rs
import numpy as np
import cv2


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

def pause():
    print("Press Enter to continue...")
    input()

def get_pos_rot_from_xyzq(xyzq):
    pos = np.array([xyzq[0], xyzq[1], xyzq[2]])
    rot = quat_to_mat((xyzq[3],xyzq[4], xyzq[5], xyzq[6]))
    return pos, rot 

def get_img():
    frames = pipeline.wait_for_frames()
    # depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())    
    return color_image


def main(robot_ip_address, pc_ip_address):
    num_record = 0
    deg_to_rad = lambda x: x / 180. * np.pi
    flexiv = FlexivRobot(robot_ip_address=robot_ip_address,
                         pc_ip_address=pc_ip_address)
    print('Robot On: {}'.format(flexiv.emergency_state))
    print('Is Moving: {}'.format(flexiv.is_moving()))

    while(True):
        img = get_img()
        cv2.imshow('vis', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            print("saving the {}-th data".format(num_record))
            pos, rot = get_pos_rot_from_xyzq(flexiv.tcp_pose)
            np.save('./save_calibration/t_{}.npy'.format(num_record), pos)
            np.save('./save_calibration/r_{}.npy'.format(num_record), rot)
            cv2.imwrite('./save_calibration/{}.jpg'.format(num_record), img)
            num_record += 1
    pipeline.stop()
        

if __name__ == '__main__':
    main(robot_ip_address=sys.argv[1], pc_ip_address=sys.argv[2])
