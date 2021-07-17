# RFLib

RFLib is a foundational library for hardware interface, cuda interface (e.g. Ops). It is a part of the [RobotFlow](https://wenqiangx.github.io/robotflowproject/) project.

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
     + [Tobor](https://wenqiangx.github.io/robotflowproject/project/tobor_robot/): [tutorial](docs/real_robot_setup/tobor_tutorial.md)
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
+ [RFLib](https://github.com/mvig-robotflow/rflib): RobotFlow foundational library for Robot Vision, Planning and Control.
+ [RFVision](https://github.com/mvig-robotflow/rfvision): RobotFlow vision-related toolbox and benchmark.
+ [RFMove](https://github.com/mvig-robotflow/rfmove): RobotFlow planning toolbox.
+ [ReinForce](https://github.com/mvig-robotflow/ReinForce): RobotFlow reinforcement learning toolbox.
+ [RFController](https://github.com/mvig-robotflow/rfcontroller): RobotFlow controller toolbox.
+ [rFUniverse](https://github.com/mvig-robotflow/rfuniverse): A Unity-based Multi-purpose Simulation Environment.
+ [RFBulletT](https://github.com/mvig-robotflow/rfbullett): A Pybullet-based Multi-purpose Simulation Environment.
+ [RF_ROS](https://github.com/mvig-robotflow/rf_ros): ROS integration. Both ROS1 and ROS2 are supported.
+ [RobotFlow](https://github.com/mvig-robotflow/robotflow): The barebone of the whole system. It organizes all the functionalities.
### Hardware
+ [RFDigit](https://github.com/mvig-robotflow/rfdigit): A Customized Digit Tactile Sensor.
### Open Ecosystem
+ [N-D Pose Annotator](https://github.com/liuliu66/6DPoseAnnotator): support both rigid and articulated object pose annotation.
+ [model format converter](https://github.com/mvig-robotflow/model_format_converter): URDF and related model format converter.