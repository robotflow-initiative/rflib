# Install Guide for RFLib

## Installation
To note, the CNN ops are installed by default.
```
pip install -e .
```

## CUDA and GCC version
CUDA 10.2 is not compatible to GCC larger than 8, it is nvidia's restriction, we can do nothing about it.

a simple solution is to install the compatible GCC versions, for example for CUDA 10.2
```
sudo apt install gcc-8 g++-8
```
Then, change the default to 8
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
```