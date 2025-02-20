#!/usr/bin/env python
import os
import rospy
import numpy as np
import math
from math import pi
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
#from gazebo_msgs.srv import SpawnModel, DeleteModel
from respawnGoal_ddpg import Respawn

#diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
#goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
#                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')


class Env_train():
    def __init__(self,is_training,m4):
        self.position = Pose()

        #self.goal_position = Pose()
        #self.goal_position.position.x = 0.
        #self.goal_position.position.y = 0.
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        #self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.respawn_goal = Respawn(m4)
        #self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        #self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.past_distance = 0.

        self.m4 = m4      
 
        if is_training:
            #stage2 =0.3
            #stage2_2m=0.1
            if self.m4=='4*4':
                self.threshold_arrive = 0.3
            elif self.m4=='2*2':
                self.threshold_arrive = 0.1
            elif self.m4=='2*1':
                self.threshold_arrive = 0.1
            else:
                self.threshold_arrive = 0.06
        else:
            if self.m4=='4*4':
                self.threshold_arrive = 0.3
            elif self.m4=='2*2':
                self.threshold_arrive = 0.1
            elif self.m4=='2*1':
                self.threshold_arrive = 0.1
            else:
                self.threshold_arrive = 0.06
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        #message
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)


    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation

        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        #q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        #yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)
        '''
        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle
        '''

    def getState(self, scan,past_action):
        scan_range = []
        #yaw = self.yaw
        #rel_theta = self.rel_theta
        #diff_angle = self.diff_angle
        #min_range = 0.2        
        #arrive = False
        scan_range = []
        heading = self.heading
        min_range = 0.13
        #min_range = 0.16
        done = False
        arrive = False
        self.get_goalbox=False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                if self.m4=="4*4":
                    scan_range.append(3.5)
                elif self.m4=="2*2":
                    scan_range.append(1.5)
                else:
                    scan_range.append(1)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)

        if min_range >= min(scan_range) > 0:
            done = True

        for pa in past_action:
            scan_range.append(pa)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        
        if current_distance <= self.threshold_arrive:
            ## done = True
            arrive = True
            self.get_goalbox = True

        #return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive
        #return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done
        return scan_range + [heading, current_distance], done,arrive
    #def setReward(self, done, arrive):
    def setReward(self, state, done,arrive):
	

        yaw_reward = []
        current_distance = state[-1]
        heading = state[-2]

        distance_rate = (self.past_distance - current_distance) 
            
        if current_distance<0.03:
            current_distance=0.03
        
        
        ########model stage2m_short##########################
        #reward = 3*(1/current_distance)-abs(heading) 

        ########model stage4m_short##########################
        #reward = 3*(1/current_distance)-abs(heading)
        if distance_rate > 0:
            reward = 5*(1/current_distance)-abs(heading)
        else:
            reward = -10.

        ########model stage3_test and stage2m_long########### 
        '''
        if distance_rate > 0:
            #reward = 200.*distance_rate
            
            reward = 200.*distance_rate+(1/current_distance)
           
            #print reward
        if distance_rate <= 0:
            reward = -10. #-8  
        '''
        ###########mix past_distnt and angle##################
        #reward= 200.*distance_rate+3*(1/current_distance)-abs(heading)
          
        self.past_distance = current_distance
                

        arrive =False
        if done:
            rospy.loginfo("Collision!!")
            reward = -150#-550
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200#500
            self.pub_cmd_vel.publish(Twist())
            
            #self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            #self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
            arrive = True
            self.reset()
        return reward ,arrive
	

    def step(self, action,past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        #state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        #state = [i / 3.5 for i in state]

        #for pa in past_action:
        #    state.append(pa)

        #state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        state, done,arrive = self.getState(data, past_action)

        #reward = self.setReward(done, arrive)
        reward,arr = self.setReward(state, done,arrive)
        
        return np.asarray(state), reward, done,arr

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            #self.respawn_goal.respawnModel()
            #self.goal_x, self.goal_y=0.65,-0.64
            self.initGoal = False
        
            

        self.goal_distance = self.getGoalDistace()
        state, done,arrive = self.getState(data,[0.,0.])

        return np.asarray(state)
