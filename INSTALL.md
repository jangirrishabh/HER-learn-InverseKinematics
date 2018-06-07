
## Table of Contents
- [Installation](#installation)
	- [Ubuntu 16.04](#ubuntu-1604)

## Installation


### Ubuntu 16.04
Basic requirements:
- ROS Kinetic (`/rosversion: 1.12.7`)
- Gazebo 8.1.1
- Python 3.5.2
- OpenAI gym

#### ROS Kinetic dependencies
```
sudo pip3 install rospkg catkin_pkg

sudo apt-get install python3-pyqt4

sudo apt-get install \
cmake gcc g++ qt4-qmake libqt4-dev \
libusb-dev libftdi-dev \
python3-defusedxml python3-vcstool \
ros-kinetic-control-toolbox     \
ros-kinetic-pluginlib	       \
ros-kinetic-trajectory-msgs     \
ros-kinetic-control-msgs	       \
ros-kinetic-std-srvs 	       \
ros-kinetic-nodelet	       \
ros-kinetic-urdf		       \
ros-kinetic-rviz		       \
ros-kinetic-kdl-conversions     \
ros-kinetic-eigen-conversions   \
ros-kinetic-tf2-sensor-msgs     \
ros-kinetic-pcl-ros \
ros-kinetic-navigation
```


#### Gazebo gym

```bash
git clone https://github.com/erlerobot/gym-gazebo
cd gym-gazebo
sudo pip3 install -e .
```

If successful, expect something like [this](https://gist.github.com/vmayoral/4a1e5999811ac8bfbf7875670c430d27).

#### Dependencies and libraries
```
sudo pip3 install h5py
sudo apt-get install python3-skimage

```


