"""
This is a demo for imitating the demonstration. 
For example, you can first create a demonstration folder, then run 'python demonstration.py 192.168.2.100 192.168.2.102 test' to 
record the demonstration in './demonstration/test_xxx.npy'. Press Ctrl+C to finishinig recording.
You can see the imitation result by running 'python imitate.py 192.168.2.100 192.168.2.102 test'
"""
import os 
import sys 
import numpy as np
import time 

from flexiv import FlexivRobot

save_path = os.path.join('.', 'demostrations')

def pause():
    print("Press Enter to continue...")
    input()

def main(robot_ip_address, pc_ip_address, filename):
    flexiv = FlexivRobot(robot_ip_address=robot_ip_address,
                         pc_ip_address=pc_ip_address)
    print('Robot On: {}'.format(flexiv.emergency_state))
    print('Is Moving: {}'.format(flexiv.is_moving()))

    pos_arr = np.load(os.path.join(save_path, filename + '_pos.npy'))
    vel_arr = np.load(os.path.join(save_path, filename + '_vel.npy'))
    ts_arr = np.load(os.path.join(save_path, filename + '_ts.npy'))
    
    print("This is a demo for imitate demonstrations. ")
    print("First, the robot will move to the initial pos of the demonstration. ")    
    pause()
    flexiv.move_joint(pos_arr[0], 5)

    rad2deg = lambda x: x / np.pi * 180
    pos_arr = rad2deg(pos_arr)
    vel_arr = rad2deg(vel_arr)
    acc_arr = (vel_arr[1:] - vel_arr[:-1])/(ts_arr[1:] - ts_arr[:-1])[:, None].repeat(7, 1)
    acc_arr = np.concatenate([acc_arr, acc_arr[-1][None, :]])

    pause() 
    # make sure the mode has been switched to 'pvat_download' before using send_joint_PVAT
    flexiv.switch_mode('pvat_download')
    flexiv.send_joint_PVAT(pos_arr, vel_arr, acc_arr, ts_arr)

    time.sleep(30)
    flexiv.switch_mode('idle')
    print("Imitation finished.")

if __name__ == '__main__':
    main(robot_ip_address=sys.argv[1], pc_ip_address=sys.argv[2], filename=sys.argv[3])