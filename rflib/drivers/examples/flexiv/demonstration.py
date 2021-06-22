"""
This is a demo for recording the demonstration. 
For example, you can first create a demonstration folder, then run 'python demonstration.py 192.168.2.100 192.168.2.102 test' to 
record the demonstration in './demonstration/test_xxx.npy'. Press Ctrl+C to finishinig recording.
You can see the imitation result by running 'python imitate.py 192.168.2.100 192.168.2.102 test'
"""
import os 
import sys 
import numpy as np
import time 

from robotflow.control.flexiv import FlexivRobot

save_path = os.path.join('.', 'demostrations')

def pause():
    print("Press Enter to continue...")
    input()

def main(robot_ip_address, pc_ip_address, filename, interv=0.3):
    flexiv = FlexivRobot(robot_ip_address=robot_ip_address,
                         pc_ip_address=pc_ip_address)
    print('Robot On: {}'.format(flexiv.emergency_state))
    print('Is Moving: {}'.format(flexiv.is_moving()))

    print("This is a demo for recording demonstrations. ")
    pause()
    pos_arr = []
    vel_arr = []
    time_start = time.time()
    ts_arr = []
    try:
        while(True):
            pos_arr.append(flexiv.joint_pos)
            vel_arr.append(flexiv.joint_vel)
            ts_arr.append(time.time() - time_start)
            time.sleep(interv)
    except KeyboardInterrupt as e:
        length = len(pos_arr)
        np.save(os.path.join(save_path, filename + '_pos.npy'), np.stack(pos_arr)[:length])
        np.save(os.path.join(save_path, filename + '_vel.npy'), np.stack(vel_arr)[:length])
        np.save(os.path.join(save_path, filename + '_ts.npy'), np.array(ts_arr)[:length])
        
        print("This demonstration has been saved.")
        return 

if __name__ == '__main__':
    main(robot_ip_address=sys.argv[1], pc_ip_address=sys.argv[2], filename=sys.argv[3])