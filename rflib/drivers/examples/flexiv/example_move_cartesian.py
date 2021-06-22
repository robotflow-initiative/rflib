"""
An examples showing that how to move the robot given a cartesian position
You can run this demo by the command 'python example_move_cartesian.py 192.168.2.100 192.168.2.102'
"""
import os 
import sys 
import numpy as np
import time 

from flexiv import FlexivRobot

def pause():
    print("Press Enter to continue...")
    input()

def main(robot_ip_address, pc_ip_address):
    deg_to_rad = lambda x: x / 180. * np.pi
    flexiv = FlexivRobot(robot_ip_address=robot_ip_address,
                         pc_ip_address=pc_ip_address)
    print('Robot On: {}'.format(flexiv.emergency_state))
    print('Is Moving: {}'.format(flexiv.is_moving()))
    flexiv.switch_mode('idle')

    cfg = [
        0.60, 0, 0.2, 0.004821024835109711, 0.01952599175274372,
        0.9997144937515259, 0.012901520356535912
    ]
    pause()
    flexiv.move_cartesian(cfg)

if __name__ == '__main__':
    main(robot_ip_address=sys.argv[1], pc_ip_address=sys.argv[2])
