import gym
import rospy
import roslaunch
import time
import numpy as np
import random

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding
from numpy import linalg as LA
from iri_common_drivers_msgs.srv import QueryInverseKinematics
from iri_common_drivers_msgs.srv import QueryForwardKinematics


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



class GazeboWAMemptyEnvv2(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "iri_wam.launch")
        self.publishers = ['pub1', 'pub2', 'pub4', 'pub3', 'pub5', 'pub6', 'pub7'] #publishers for the motor commands

        self.publishers[0] = rospy.Publisher('/iri_wam/joint1_position_controller/command', Float64, queue_size=5)
        self.publishers[1] = rospy.Publisher('/iri_wam/joint2_position_controller/command', Float64, queue_size=5)
        self.publishers[2] = rospy.Publisher('/iri_wam/joint4_position_controller/command', Float64, queue_size=5)
        self.publishers[3] = rospy.Publisher('/iri_wam/joint3_position_controller/command', Float64, queue_size=5)
        self.publishers[4] = rospy.Publisher('/iri_wam/joint5_position_controller/command', Float64, queue_size=5)
        self.publishers[5] = rospy.Publisher('/iri_wam/joint6_position_controller/command', Float64, queue_size=5)
        self.publishers[6] = rospy.Publisher('/iri_wam/joint7_position_controller/command', Float64, queue_size=5) # discretely publishing motor actions for now
        self.pubMarker = rospy.Publisher('/goalPose', Marker, queue_size=5)  ##we might need multiple markers here

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        #self.minDisplacement = 0.02 #minimum discrete distance by a single action 
        #self.minDisplacementTolerance = 0.01
        #self.minDisplacementPose = self.minDisplacement + self.minDisplacementTolerance  # minimum distance to check for the goal reaching state
        self.distanceThreshold = 0.03
        self.baseFrame = 'iri_wam_link_base' 
        self.homingTime = 0.6 # time given for homing
        self.lenGoal = 3 # goal position list length
        self._max_episode_steps = 100
        self.reward_type = 'sparse'
        self.num_goals = 3
              
        

        self.home = np.zeros(4) # what position is the homing
        # self.Xacrohigh = np.array([2.6, 2.0, 2.8, 3.1, 1.24, 1.57, 2.96])
        # self.Xacrolow = np.array([-2.6, -2.0, -2.8, -0.9, -4.76, -1.57, -2.96])
        # self.IKlow = np.array([-2.6, -1.94, -2.73, -0.88, -4.74, -1.55, -2.98])
        # self.IKhigh = np.array([2.6, 1.94, 2.73, 3.08, 1.22, 1.55, 2.98])
        
        #self.low = np.array([-2.6, -1.42, -2.73, -0.88, -4.74, -1.55, -2.98])
        #self.high = np.array([2.6, 1.42, 2.73, 3.08, 1.22, 1.55, 2.98])

        #self.lowConc = np.array([-2.6, -1.42, -0.88, -2.6, -1.42, -0.88]) #1, 2, 4, 1, 2, 4
        #self.highConc = np.array([2.6, 1.42, 3.08, 2.6, 1.42, 3.08])

        self.lowConcObs = np.array([-2.6, -1.42, -0.88]) #1, 2, 4
        self.highConcObs = np.array([2.6, 1.42, 3.08])

        #self.samplelow = np.array([-2.4, -1.4, -2.03, -0.68, -4.04, -1.05, -2.08])
        #self.samplehigh = np.array([2.4, 1.4, 2.03, 2.78, 1.0, 1.05, 2.08])

        self.samplelow = np.array([-1.5, -0.8, -2.03, -0.48, -4.04, -1.05, -2.08])
        self.samplehigh = np.array([1.5, 0.8, 2.03, 2.48, 1.0, 1.05, 2.08])
        #self.high = np.array([5.2, 2.8, 5.4, 3.96, 6.96, 3.1, 5.96])        
        self.lowAction = [-1, -1, -1]
        self.highAction = [1, 1, 1]
        self.n_actions = len(self.highAction)

        self.lastObservation = None
        self.lastObservationJS = None
        self.goal = [None]*self.num_goals
        self.goalJS = [None]*self.num_goals

        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(len(self.highConcObs),), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(len(self.highConcObs),), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(len(self.highConcObs),), dtype='float32'),
        ))

        #self.reward_range = (-np.inf, np.inf)


    def getForwardKinematics(self, goalPosition): #get catesian coordinates for joint Positions
        rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_fk')
        try:
            getFK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_fk', QueryForwardKinematics)
            jointPoseReturned = getFK(goalPosition)
            return jointPoseReturned
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)


    def getGripperPosition(self, joints):
        frame_ID = self.baseFrame
        tempJointState = JointState()
        tempJointState.header.frame_id = self.baseFrame
        tempJointState.position = joints

        tempPoseFK = self.getForwardKinematics(tempJointState)

        return np.array([tempPoseFK.pose.pose.position.x, tempPoseFK.pose.pose.position.y, tempPoseFK.pose.pose.position.z])


    def _sample_goal(self): #sample from reachable positions
        frame_ID = self.baseFrame
        tempPoseFK = [None]*self.num_goals

        while np.array(tempPoseFK).any()==None:
            for x in range(self.num_goals):
                tempPosition = []
                for joint in range(4):
                    tempPosition.append(random.uniform(self.samplelow[joint], self.samplehigh[joint]))

                tempPosition.append(0)
                tempPosition.append(0)
                tempPosition.append(0)
                tempPosition[2] = 0
                tempJointState = JointState()
                tempJointState.header.frame_id = self.baseFrame
                tempJointState.position = tempPosition

                
                tempTemp = self.getForwardKinematics(tempJointState)
                tempPoseFK[x] = [tempTemp.pose.pose.position.x, tempTemp.pose.pose.position.y, tempTemp.pose.pose.position.z]
        
        self.goalJS = tempPoseFK
        return np.array(tempPoseFK)
                

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distanceThreshold).astype(np.float32)
        else:
            return -d

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        #if (d < self.distanceThreshold).astype(np.float32): print("SUCKSESSS")
        return (d < self.distanceThreshold).astype(np.float32)


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        #print ("action received ", action )
    

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        
        self._set_action(action)
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        #done = bool(self._is_success(obs['achieved_goal'], self.goal))
        done = False

        self.setMarkers(goal_distance(obs['achieved_goal'], self.goal))        

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
        
        return obs, reward, done, info

    

    def _set_action(self, action):

        action = action.copy()  # ensure that we don't change the action outside of this scope
        lastObs = self.lastObservationJS.copy()
        #print ("action received ", action)
        action *= 0.4 #limiting maximum displacement by actions
        #print ("action Gicen after augmenting ", action)
        for num, joint in enumerate(action):
            self.publishers[num].publish(lastObs[num] + joint)
            


    def _get_obs(self):
        data = None
        gripperPos = self.lastObservation
    
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=1)
                gripperPos = self.getGripperPosition(data.position) #cartesian coordinates of the gripper
                dataConc = np.array([data.position[0], data.position[1], data.position[3]]) # joint space coordinates of the robotic arm 1, 2, 4
                if ((np.array(dataConc)<=self.highConcObs).all()) and ((np.array(dataConc)>=self.lowConcObs).all()): #check if they lie inside allowed envelope
                    self.lastObservation = gripperPos.copy()
                    self.lastObservationJS = dataConc.copy()
                else:
                    data = None
                    print ("Bad observation data received STEP" )
                    for joint in range(3):
                        self.publishers[joint].publish(self.home[joint]) #homing at every reset
                    time.sleep(self.homingTime)
            except:
                pass

        return {
            'observation': gripperPos.copy(),
            'achieved_goal': gripperPos.copy(),
            'desired_goal': self.goal.copy(),
        }
        




    def reset(self):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")


        # rospy.wait_for_service('/iri_wam/controller_manager/list_controllers')
        # try:
        #     readController = rospy.ServiceProxy('/iri_wam/controller_manager/list_controllers', Empty)
        #     control = readController()
        #     #print (control)
        #     #onHai = bool(controllers.iri_wam_controller.state == "stopped")
        # except (rospy.ServiceException) as e:
        #     print ("Service call failed: %s"%e)

        #if not onHai:
        # rospy.wait_for_service('/iri_wam/controller_manager/switch_controller')
        # try:
        #     change_controller = rospy.ServiceProxy('/iri_wam/controller_manager/switch_controller', SwitchController)
        #     ret = change_controller(['joint1_position_controller', 'joint2_position_controller', 'joint3_position_controller', 'joint4_position_controller', 'joint5_position_controller', 'joint6_position_controller', 'joint7_position_controller'], ['iri_wam_controller'], 2)
        # except (rospy.ServiceException) as e:
        #     print ("Service call failed: %s"%e)

        
        for joint in range(3):
            self.publishers[joint].publish(self.home[joint]) #homing at every reset
        time.sleep(self.homingTime)

        self.goal = self._sample_goal().copy() # get a random goal every time you reset
        obs = self._get_obs()
                

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        return obs


    def setMarkers(self, difference):
        goalState = self.goal.copy()
        pointToPose = Point()
        pointToPose.x = goalState[0]
        pointToPose.y = goalState[1]
        pointToPose.z = goalState[2]
        markerObj = Marker()
        markerObj.header.frame_id = self.baseFrame
        #markerObj.header.stamp = rospy.get_rostime()
        markerObj.id = 0
        markerObj.ns = 'iri_wam'
        markerObj.type = markerObj.SPHERE
        markerObj.action = markerObj.ADD
        markerObj.pose.position = pointToPose
        markerObj.pose.orientation.w = 1.0
        markerObj.scale.x = 0.09
        markerObj.scale.y = 0.09
        markerObj.scale.z = 0.09



        if ( difference <= (self.distanceThreshold) ):
            markerObj.color.g = 1.0
            markerObj.color.a = 1.0

        else: 
            markerObj.color.r = 1.0
            markerObj.color.a = 1.0
            

        self.pubMarker.publish(markerObj)




















        





