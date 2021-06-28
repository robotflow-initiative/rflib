# RFLib

RFLib is a foundational library for hardware interface, cuda interface (e.g. Ops). It is a part of the [RobotFlow](https://wenqiangx.github.io/robotflowproject/) project.

## ToDo
+ [ ] Add xacro2urdf converter
+ [ ] Add URDF parser, URDF2FBX converter
+ [ ] fill up calibration, and make it relevant to rflib
+ [ ] fill up drivers, and make it relevant to rflib
+ [ ] make N-D pose annotator relevant to rflib

## Installation
This line will install with ops, it will take a few minutes.
```
pip install -e .
```

If you want to make a quick install, because you only want to use image library or hardware drivers, you can:
```
RFLIB_WITH_OPS=0 pip install -e .
```

## Foundational Library/Drivers for
### Image Utils: [Docs](#)
1. 2D Vision: For this part, we build upon [mmcv](https://github.com/open-mmlab/mmcv).
2. 3D Vision
3. Multi-modal Perception

### CNN Utils [Docs](#)
For this part, we build upon [mmcv](https://github.com/open-mmlab/mmcv).

### Hardware Interface

We have tested with:
1. Robot
   + mobile dual arm
     + [Tobor 1](https://wenqiangx.github.io/robotflowproject/project/tobor_robot/): [tutorial](docs/real_robot_setup/tobor_tutorial.md)
     + [Tobor 2](#): in progress
   + robot arm
     + Flexiv Rizon 4: [tutorial](docs/real_robot_setup/flexiv_tutorial.md)
     + Franka Emika Panda: [tutorial](docs/real_robot_setup/franka_tutorial.md)
     + UR5: [tutorial](docs/real_robot_setup/ur5_tutorial.md)
     + Rokae xMate pro 3: [tutorial](docs/real_robot_setup/#)
   + robot hand
     + Robotiq 85: [tutorial](docs/real_robot_setup/#)
     + AG 95: [tutorial](docs/real_robot_setup/#)
     + DH-3: [tutorial](docs/real_robot_setup/#)
     + JQ3: [tutorial](docs/real_robot_setup/#)
     + Allegro: [tutorial](docs/real_robot_setup/#)
   + agv
     + Sunspeed: [tutorial](docs/real_robot_setup/#)
2. Sensors
   + camera
     + RealSense D415/D435/L515: [tutorial](docs/real_robot_setup/#)
     + Azure Kinect: [tutorial](docs/real_robot_setup/#)
   + tactile
     + [RFDigit](https://wenqiangx.github.io/robotflowproject/project/rfdigit/): [tutorial](docs/real_robot_setup/#)

## Projects in RobotFlow
### Software
+ [RFLib](https://github.com/WenqiangX/rflib): RobotFlow foundational library for Robot Vision, Planning and Control.
+ [RFVision](https://github.com/WenqiangX/rfvision): RobotFlow vision-related toolbox and benchmark.
+ [RFMove](https://github.com/WenqiangX/rfmove): RobotFlow planning toolbox.
+ [ReinForce](https://github.com/WenqiangX/ReinForce): RobotFlow reinforcement learning toolbox.
+ [RFController](https://github.com/WenqiangX/rfcontroller): RobotFlow controller toolbox.
+ [rFUniverse](https://github.com/WenqiangX/rfuniverse): A Unity-based Multi-purpose Simulation Environment.
+ [RFBulletT](https://github.com/WenqiangX/rfbullett): A Pybullet-based Multi-purpose Simulation Environment.
+ [RF_ROS](https://github.com/WenqiangX/rf_ros): ROS integration. Both ROS1 and ROS2 are supported.
+ [RobotFlow](https://github.com/WenqiangX/robotflow): The barebone of the whole system. It organizes all the functionalities.
### Hardware
+ [RFDigit](https://github.com/WenqiangX/rfdigit): A Customized Digit Tactile Sensor.
+ [RFNail](#): in progress.
### Open Ecosystem Based on RFLib
+ [N-D Pose Annotator](https://github.com/liuliu66/6DPoseAnnotator): support both rigid and articulated object pose annotation