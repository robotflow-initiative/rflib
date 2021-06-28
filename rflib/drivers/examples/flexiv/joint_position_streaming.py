"""
This is an exmaple for joint position streaming.
The details of pvat streaming can be found in FlexivRobot.move_joint in flexiv.py
You can run this demo by the command 'python joint_position_streaming.py 192.168.2.100 192.168.2.102'
"""
import os 
import sys 
import numpy as np
import time 
import libwrapper

from flexiv import FlexivRobot

def pause():
    print("Press Enter to continue...")
    input()

def main(robot_ip_address, pc_ip_address):
    flexiv = FlexivRobot(robot_ip_address=robot_ip_address,
                         pc_ip_address=pc_ip_address)
    print('Robot On: {}'.format(flexiv.emergency_state))
    print('Is Moving: {}'.format(flexiv.is_moving))

    pause()
    cur_joint_pos = flexiv.joint_pos
    targ_joint_pos = cur_joint_pos + np.array([0., 0., 0., 0., 0., 0.5, 0.])
    flexiv.move_joint(targ_joint_pos, duration=3)
    flexiv.move_joint(cur_joint_pos, duration=3)

if __name__ == '__main__':
    main(robot_ip_address=sys.argv[1], pc_ip_address=sys.argv[2])
