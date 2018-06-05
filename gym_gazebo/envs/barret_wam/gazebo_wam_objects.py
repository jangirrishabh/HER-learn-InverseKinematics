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
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from iri_common_drivers_msgs.msg import tool_closeAction, tool_closeActionGoal, tool_openAction, tool_openActionGoal
from gz_gripper_plugin.srv import CheckGrasped
import actionlib


"""  WAM ENVIRONMENT FOR SINGLE BLOCK PICK AND PLACE LEARNING TASK  """


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def gripperClient(act):
        if act == 'close':
            client = actionlib.SimpleActionClient('/gripper/close_tool', tool_closeAction)
            goal = tool_closeActionGoal()
        elif act == 'open':
            client = actionlib.SimpleActionClient('/gripper/open_tool', tool_openAction)
            goal = tool_openActionGoal()

        client.wait_for_server()
        client.send_goal(goal)
        client.wait_for_result()
        return client.get_result() 

class GazeboWAMemptyEnvv2(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "iri_wam_HER.launch")
        self.publishers = ['pub1', 'pub2', 'pub4', 'pub3', 'pub5', 'pub6', 'pub7'] #publishers for the motor commands

        self.publishers[0] = rospy.Publisher('/iri_wam/joint1_position_controller/command', Float64, queue_size=1)
        self.publishers[1] = rospy.Publisher('/iri_wam/joint2_position_controller/command', Float64, queue_size=1)
        self.publishers[2] = rospy.Publisher('/iri_wam/joint4_position_controller/command', Float64, queue_size=1)
        # self.publishers[3] = rospy.Publisher('/iri_wam/joint3_position_controller/command', Float64, queue_size=5)
        # self.publishers[4] = rospy.Publisher('/iri_wam/joint5_position_controller/command', Float64, queue_size=5)
        self.publishers[3] = rospy.Publisher('/iri_wam/joint6_position_controller/command', Float64, queue_size=1)
        # self.publishers[6] = rospy.Publisher('/iri_wam/joint7_position_controller/command', Float64, queue_size=5) # discretely publishing motor actions for now
        
        self.pubMarker = ['marker1', 'marker2']
        self.pubMarker[0] = rospy.Publisher('/goalPose', Marker, queue_size=1)
        self.pubMarker[1] = rospy.Publisher('/goalPose', Marker, queue_size=1)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        #self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state",SetModelState)
        self.get_state = rospy.ServiceProxy("/gazebo/get_model_state",GetModelState)
        self.get_gripper = rospy.ServiceProxy('/check_grasped', CheckGrasped)

        self.distanceThreshold = 0.09
        self.distanceThresholdDemo = 0.05
        self.baseFrame = 'iri_wam_link_base'
        self.homingTime = 0.6 # time given for homing
        self.lenGoal = 3 # goal position list length
        self._max_episode_steps = 50
        self.reward_type = 'sparse'
        self.objModelNum = 1
        self.objectName = {
            "obj1" : 'obs_0', "objFixed" : 'obs_fixed'
        }
              

        self.home = np.array([0, 0.6, 1.4, 0, 0]) # what position is the homing
        # self.Xacrohigh = np.array([2.6, 2.0, 2.8, 3.1, 1.24, 1.57, 2.96])
        # self.Xacrolow = np.array([-2.6, -2.0, -2.8, -0.9, -4.76, -1.57, -2.96])
        # self.IKlow = np.array([-2.6, -1.94, -2.73, -0.88, -4.74, -1.55, -2.98])
        # self.IKhigh = np.array([2.6, 1.94, 2.73, 3.08, 1.22, 1.55, 2.98])
        
        #self.low = np.array([-2.6, -1.42, -2.73, -0.88, -4.74, -1.55, -2.98])
        #self.high = np.array([2.6, 1.42, 2.73, 3.08, 1.22, 1.55, 2.98])

        #self.lowConc = np.array([-2.6, -1.42, -0.88, -2.6, -1.42, -0.88]) #1, 2, 4, 1, 2, 4
        #self.highConc = np.array([2.6, 1.42, 3.08, 2.6, 1.42, 3.08])

        self.lowConcObs = np.array([-2.6, -1.94, -0.88, -1.55]) #1, 2, 4, 6 #check if they lie inside the envelope
        self.highConcObs = np.array([2.6, 1.94, 3.08, 1.55])

        self.samplelow = np.array([-0.6, 0.0, -2.8, 0.0, -4.70, -1.55, -2.96])
        self.samplehigh = np.array([0.6, 1.94, 2.8, 3.0, 1.2, 1.55, 2.96])

        #self.samplelow = np.array([-1.5, -0.8, -2.03, -0.48, -4.04, -1.05, -2.08])
        #self.samplehigh = np.array([1.5, 0.8, 2.03, 2.48, 1.0, 1.05, 2.08])
        #self.high = np.array([5.2, 2.8, 5.4, 3.96, 6.96, 3.1, 5.96])        
        self.lowAction = [-1, -1, -1, -1, -1]
        self.highAction = [1, 1, 1, 1, 1]
        self.n_actions = len(self.highAction)
        self.lenObs = 15

        self.lastObservation = None
        self.lastObservationJS = None
        self.lastObservationOrient = None
        self.goal = None
        self.goalJS = None
        self.object = None
        self.objInitial = None
        self.objInitialJS = None

        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(self.lenObs,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(self.lenGoal,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(self.lenGoal,), dtype='float32'),
        ))

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")


    def getForwardKinematics(self, goalPosition): #get catesian coordinates for joint Positions
        rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_fk')
        try:
            getFK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_fk', QueryForwardKinematics)
            jointPoseReturned = getFK(goalPosition)
            return jointPoseReturned
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)

    def getInverseKinematics(self, goalPose): #get joint angles for reaching the goal position
        tempPose = Pose()
        tempPose = goalPose
        tempPose.position.z -= 0.01 #offset to make the ik better

        goalPoseStamped = PoseStamped()
        goalPoseStamped.header.frame_id = self.baseFrame
        goalPoseStamped.pose = tempPose
        rospy.wait_for_service('/iri_wam/iri_wam_tcp_ik/get_wam_ik')
        try:
            getIK = rospy.ServiceProxy('/iri_wam/iri_wam_tcp_ik/get_wam_ik', QueryInverseKinematics)
            jointPositionsReturned = getIK(goalPoseStamped)
            return [jointPositionsReturned.joints.position[0], jointPositionsReturned.joints.position[1], jointPositionsReturned.joints.position[3], jointPositionsReturned.joints.position[5]]
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)


    def getArmPosition(self, joints):
        frame_ID = self.baseFrame
        tempJointState = JointState()
        tempJointState.header.frame_id = self.baseFrame
        tempJointState.position = joints

        tempPoseFK = self.getForwardKinematics(tempJointState)

        return [np.array([tempPoseFK.pose.pose.position.x, tempPoseFK.pose.pose.position.y, tempPoseFK.pose.pose.position.z]), np.array([tempPoseFK.pose.pose.orientation.x, tempPoseFK.pose.pose.orientation.y, tempPoseFK.pose.pose.orientation.z, tempPoseFK.pose.pose.orientation.w ])]


    def _sample_goal(self): #sample from reachable positions
        frame_ID = self.baseFrame
        tempPoseFK = None

        while tempPoseFK==None:
            tempPosition = []
            for joint in range(6):
                tempPosition.append(random.uniform(self.samplelow[joint], self.samplehigh[joint]))

            tempPosition.append(0)
            tempPosition[2] = 0
            tempPosition[4] = 0
            tempJointState = JointState()
            tempJointState.header.frame_id = self.baseFrame
            tempJointState.position = tempPosition

            self.goalJS = tempPosition # goal sampled in the joint space represenatation
            tempPoseFK = self.getForwardKinematics(tempJointState)
            if tempPoseFK!=None:
                tempTemp = [tempPoseFK.pose.pose.position.x, tempPoseFK.pose.pose.position.y, tempPoseFK.pose.pose.position.z]
                return np.array(tempTemp)

    def sample_goal_onTable(self): #sample from reachable positions

        if self.objInitial != None:
            sampledGoal = self.objInitial


            sampledGoal.position.x += random.uniform(0.0, 0.3) * np.random.choice([-1, 1], 1)
            if np.absolute(sampledGoal.position.x) < 0.2:
                sampledGoal.position.y += random.uniform(0.2, 0.4) * np.random.choice([-1, 1], 1)
            else :
                sampledGoal.position.y += random.uniform(0.1, 0.4) * np.random.choice([-1, 1], 1)
            sampledGoal.position.z += 0.065 + 0.15  # goal for arm

            

            sampledGoal.position.x = np.asscalar(np.clip(sampledGoal.position.x, 0.3, 0.65))
            sampledGoal.position.y = np.asscalar(np.clip(sampledGoal.position.y, -0.4 , 0.4))

            self.goalJS = self.getInverseKinematics(sampledGoal)
            return np.array([sampledGoal.position.x, sampledGoal.position.y, sampledGoal.position.z - 0.15 ]) # actual goal
        else:
            sampledGoal.position.x = 0.5001911647282589
            sampledGoal.position.y = 0.1004797189877992
            sampledGoal.position.z = -0.21252162794043228
            sampledGoal.orientation.x = 0.00470048637345294
            sampledGoal.orientation.y = 0.99998892605584
            sampledGoal.orientation.z = 9.419015715062839e-06
            sampledGoal.orientation.w = -0.00023044483691539005
            sampledGoal.position.z += 0.065 + 1.075 + 0.15
            self.goalJS = self.getInverseKinematics(sampledGoal)
            return np.array([sampledGoal.position.x, sampledGoal.position.y, sampledGoal.position.z - 0.15])

 
                

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward_HER(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        reward = np.zeros(achieved_goal.shape[0])
        for x in range(achieved_goal.shape[0]):
            d = goal_distance(achieved_goal[x][:3], desired_goal[x][:3])
            #d2 = goal_distance(achieved_goal[x][:3], desired_goal[x][3:])
            #reward[x] = -(d > self.distanceThreshold and (d2 <= 0.08)).astype(np.float32)
            reward[x] = -(d > self.distanceThreshold).astype(np.float32)
    
        return reward

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal[:3], desired_goal[:3])
        #d2 = goal_distance(achieved_goal[:3], desired_goal[3:])
        #return (d < self.distanceThreshold and (d2 <= 0.8)).astype(np.float32)
        return (d < self.distanceThreshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal[:3], desired_goal[:3])
        #d2 = goal_distance(achieved_goal[3:], desired_goal[3:])

        if self.reward_type == 'sparse':
            return -(d > self.distanceThreshold).astype(np.float32)
        else:
            return -d



    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        #print ("action received ", action )
    

        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")

        
        self._set_action(action)
        obs = self._get_obs()
        
        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        }

        
        #reward2  = goal_distance(np.array([self.object.position.x, self.object.position.y, self.object.position.z]), self.goal)
        #reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        done = bool(self._is_success(obs['achieved_goal'], obs['desired_goal'] ))
        #done = False

        self.setMarkers(goal_distance(obs['achieved_goal'][:3], obs['desired_goal'][:3] ), self.goal.copy(), 0)        

        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")
        
        return obs, reward, done, info

    

    def _set_action(self, action):

        action = action.copy()  # ensure that we don't change the action outside of this scope
        lastObs = self.lastObservationJS.copy()
        #action *= 0.4 #limiting maximum displacement by actions
        #print ("action Gicen after augmenting ", action)
        for num, joint in enumerate(action[:self.n_actions-1]):
            self.publishers[num].publish(lastObs[num] + joint)


        if (action[self.n_actions-1]) > 0.1: #close
            gripperClient('close')
            #print("CLOSE")
        elif (action[self.n_actions-1]) < -0.1: #open
            gripperClient('open')
            #print("OPEN")
        else: None
            #print("NONE")

            


    def _get_obs(self):
        data = None
        objectPos = np.zeros(3)
        fixedObjectPos = np.zeros(3)
        gripperPos = self.lastObservation
        gripperOrient = self.lastObservationOrient
        gripperState = 0

        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=1)
                objectPos = self.get_object_position(self.objectName['obj1'])
                self.setMarkers( 1.0, objectPos, 1)
                fixedObjectPos = self.get_object_position(self.objectName['objFixed'])
                [gripperPos, gripperOrient] = self.getArmPosition(data.position) #cartesian coordinates of the gripper
                dataConc = np.array([data.position[0], data.position[1], data.position[3], data.position[5]]) # joint space coordinates of the robotic arm 1, 2, 4, 6
                gripperState = self.get_gripper_state()

                if ((np.array(dataConc)<=self.highConcObs).all()) and ((np.array(dataConc)>=self.lowConcObs).all()): #check if they lie inside allowed envelope
                    self.lastObservation = gripperPos.copy()
                    self.lastObservationJS = dataConc.copy()
                    self.lastObservationOrient = gripperOrient.copy()
                else:
                    data = None
                    print ("Bad observation data received STEP" )
                    for joint in range(4):
                        self.publishers[joint].publish(self.home[joint]) #homing at every reset
                    #gripperClient('open')
                    time.sleep(self.homingTime)
            except:
                pass


        obs = np.concatenate([gripperPos, objectPos])
        obs = np.append(obs, gripperOrient)
        obs = np.append(obs, gripperState)
        obs = np.append(obs, np.linalg.norm(gripperPos - objectPos, axis=-1 ))
        #print(" DISTANCE ", np.linalg.norm(gripperPos - objectPos, axis=-1 ))
        obs = np.append(obs, fixedObjectPos)
        self.observations = obs.copy()

        return {
            'observation': obs.copy(),
            #'achieved_goal': np.concatenate([objectPos, fixedObjectPos]),
            #'desired_goal': np.concatenate([self.goal , fixedObjectPos]),
            'achieved_goal': objectPos.copy(),
            'desired_goal': self.goal.copy(),
        }
        

    def get_object_position(self, objectName):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            temp = self.get_state(objectName, None)
            temp.pose.position.z -= 1.075
            #temp.pose.position.y += 2.5
            objectPos = np.array([temp.pose.position.x, temp.pose.position.y, temp.pose.position.z])

            if objectName == 'obs_0' : self.object = temp.pose
            return objectPos

        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")


    def get_gripper_state(self):
        rospy.wait_for_service('/check_grasped')
        try:
            temp = self.get_gripper()
            
            if temp.grasped == '':
                return 0
            else: return 1 

        except (rospy.ServiceException) as e:
            print ("/check_grasped service call failed")




    def reset(self):
        # rospy.wait_for_service('/gazebo/reset_world') # Reset simulation was causing problems, do not reset simulation
        # try:
        #     #reset_proxy.call()
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_world service call failed")


        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")


        rospy.wait_for_service('/gazebo/set_model_state')
        try:

            pose = Pose()

            state = ModelState()
            state.model_name = self.objectName['obj1']
            #temp.reference_frame = "world"
            pose.position.x = 0.5001911647282589
            pose.position.y = 0.1004797189877992
            pose.position.z = 0.9
            pose.orientation.x = 0.00470048637345294
            pose.orientation.y = 0.99998892605584
            pose.orientation.z = 9.419015715062839e-06
            pose.orientation.w = -0.00023044483691539005

            state.pose = pose
            # temp.twist.linear.x = 0
            # temp.twist.linear.y = 0
            # temp.twist.linear.z = 0
            # temp.twist.angular.x = 0
            # temp.twist.angular.y = 0
            # temp.twist.angular.z = 0

            

            
            ret = self.set_state(state)

        except (rospy.ServiceException) as e:
            print ("/gazebo/set model pose service call failed")


        
        


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

        
        for joint in range(4):
            self.publishers[joint].publish(self.home[joint]) #homing at every reset
        gripperClient('open')
        time.sleep(self.homingTime)

    
        self.get_object_position(self.objectName['obj1'])
        self.objInitial = self.object
        self.objInitialJS = self.getInverseKinematics(self.objInitial)

        self.goal = self.sample_goal_onTable().copy() # get a random goal every time you reset

        rospy.wait_for_service('/gazebo/set_model_state')
        try:

            pose = Pose()

            state = ModelState()
            state.model_name = self.objectName['objFixed']
            #temp.reference_frame = "world"
            pose.position.x = self.goal[0]
            pose.position.y = self.goal[1]
            pose.position.z = 0.9
            pose.orientation.x = 0.00470048637345294
            pose.orientation.y = 0.99998892605584
            pose.orientation.z = 9.419015715062839e-06
            pose.orientation.w = -0.00023044483691539005

            state.pose = pose
            # temp.twist.linear.x = 0
            # temp.twist.linear.y = 0
            # temp.twist.linear.z = 0
            # temp.twist.angular.x = 0
            # temp.twist.angular.y = 0
            # temp.twist.angular.z = 0

            

            
            ret = self.set_state(state)

        except (rospy.ServiceException) as e:
            print ("/gazebo/set model pose service call failed")


        obs = self._get_obs()

        rospy.wait_for_service('/gazebo/set_model_state')
        try:

            pose = Pose()

            state = ModelState()
            state.model_name = self.objectName['obj1']
            #temp.reference_frame = "world"
            pose.position.x = 0.5001911647282589
            pose.position.y = 0.1004797189877992
            pose.position.z = 0.9
            pose.orientation.x = 0.00470048637345294
            pose.orientation.y = 0.99998892605584
            pose.orientation.z = 9.419015715062839e-06
            pose.orientation.w = -0.00023044483691539005

            state.pose = pose
            # temp.twist.linear.x = 0
            # temp.twist.linear.y = 0
            # temp.twist.linear.z = 0
            # temp.twist.angular.x = 0
            # temp.twist.angular.y = 0
            # temp.twist.angular.z = 0

            

            
            ret = self.set_state(state)

        except (rospy.ServiceException) as e:
            print ("/gazebo/set model pose service call failed")
        


        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        return obs


    def setMarkers(self, difference, point, objID):
        
        #goalState = np.array([self.objInitial.position.x, self.objInitial.position.y, self.objInitial.position.z])
        #goalState = np.array([0, 0, 0])
        pointToPose = Point()
        pointToPose.x = point[0]
        pointToPose.y = point[1]
        pointToPose.z = point[2]
        markerObj = Marker()
        markerObj.header.frame_id = self.baseFrame
        #markerObj.header.stamp = rospy.get_rostime()
        markerObj.id = objID
        markerObj.ns = 'iri_wam'
        markerObj.type = markerObj.SPHERE
        markerObj.action = markerObj.ADD
        markerObj.pose.position = pointToPose
        markerObj.pose.orientation.w = 1.0

        markerObj.scale.x = 0.065/2 + (0.09 - 0.065/2) * ( 1 - objID )
        markerObj.scale.y = 0.065/2 + (0.09 - 0.065/2) * ( 1 - objID )
        markerObj.scale.z = 0.065/2 + (0.09 - 0.065/2) * ( 1 - objID )



        if ( difference <= (self.distanceThreshold) ):
            markerObj.color.g = 1.0
            markerObj.color.a = 1.0

        else: 
            markerObj.color.r = 1.0
            markerObj.color.a = 1.0
            

        self.pubMarker[objID].publish(markerObj)



