# Learning Inverse Kinematics of a Barret WAM Robotic arm in Gazebo simulation


This work is based on the implementation of Hindsight Experience Replay from OpenAI Baselines adapted to this particular environment of a Barret WAM robotic arm in Gazebo simulation.



Credits for the Gym-Gazebo package for successful integration of ROS and Gazebo with the OpenAI gym - https://github.com/erlerobot/gym-gazebo


**`gym-gazebo` is a complex piece of software for roboticists that puts together simulation tools, robot middlewares (ROS, ROS 2), machine learning and reinforcement learning techniques. All together to create an environment whereto benchmark and develop behaviors with robots. Setting up `gym-gazebo` appropriately requires relevant familiarity with these tools.**




### Killing background processes

Sometimes, after ending or killing the simulation `gzserver` and `rosmaster` stay on the background, make sure you end them before starting new tests.

We recommend creating an alias to kill those processes.

```bash
echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> ~/.bashrc
```
