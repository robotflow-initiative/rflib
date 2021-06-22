# Drivers
In this directory, we provide multiple real robot driver wrappers, including but not limited to: Flexiv Rizon, xMate, Franka.

To use these wrappers to control the real robot. Please refer to [here](./examples).

## Preparation For Different Robots
1. Flexiv Rizon
   
   Download [libwrapper](#), place it to /path/.

2. UR5
   
   We use [urx](https://github.com/SintefManufacturing/python-urx) to control UR5, before using UR5, please install urx first.

3. Franka Emika Panda

    If it is ok for you to use C++ to control the robot, then you should install libfranka as [here](https://frankaemika.github.io/docs/installation_linux.html) and then directly use demo from [here](./examples/franka/cpp).

    If you want to control it via Python, you should download `libfranka_wrapper` from [here](#), and examples are given in [here](./examples/franka/python).