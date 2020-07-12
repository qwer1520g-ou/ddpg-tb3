#!/usr/bin/env python
import rospy
#import gym
#import gym_gazebo
import numpy as np
import tensorflow as tf
import time
from ddpg import *
from environment_train import Env_train
from environment_test import Env_test
from OUNoise import OUNoise

exploration_decay_start_step = 50000
#state_dim = 16
state_dim = 360+4  #laser 24 + past action 2 + current distance 1 + heading 1 = 28
action_dim = 2
action_linear_max = 0.22  # m/s
action_angular_max = 2.5  # rad/s
var_l=1#action_linear_max*.8
var_a=1#action_linear_max*.8
is_training = True

#m4 = '4*4' '2*2' '2*1' '2*1_complex'
m4='2*1' 


eps=100000

def main():

    
    rospy.init_node('ddpg_stage_1')
    
    summary_writer = tf.summary.FileWriter("../tensorboard/ddpg_route/env_2m_1m_complex")
    tensorboard_reward=0
    averg_reward=0

    if is_training:
        env = Env_train(is_training,m4)
    else:
        #env = Env_test(is_training,m4)
        env = Env_train(is_training,m4)

    #noise = OUNoise(2)

    agent = DDPG(env, state_dim, action_dim)
    past_action = np.array([0., 0.])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    if is_training:
        print('Training mode')
        avg_reward_his = []
        total_reward = 0
        
        var = 1.

        #while True:
        for e in range(eps):
            state = env.reset()
            one_round_step = 0
            tensorboard_reward=0

            while True:
                a = agent.action(state)
                a[0] = np.clip(np.random.normal(a[0], var_l), 0., action_linear_max)
                a[1] = np.clip(np.random.normal(a[1], var_a), -action_angular_max, action_angular_max)
                
                #noi=noise.noise()
                
                #a= act+noi
                #a[0] = np.clip(a[0], 0., 1.)
                #a[1] = np.clip(a[1], -0.5, 0.5)
                
                state_, r, done, arrive = env.step(a, past_action)
                time_step = agent.perceive(state, a, r, state_, done)
                
                if arrive:
                    result = 'Success'
                else:
                    result = 'Fail'

                if time_step > 0:
                    total_reward += r
                
                tensorboard_reward += r    

                if time_step % 10000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    avg_reward = total_reward / 10000
                    print('Average_reward = ', avg_reward)
                    avg_reward_his.append(round(avg_reward, 2))
                    print('Average Reward:',avg_reward_his)
                    total_reward = 0

                if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    var *= 0.9999

                past_action = a
                state = state_
                one_round_step += 1

                if arrive or done or one_round_step >= 500:
                    print('Step: %3i' % one_round_step, '| ep: %.2f' % e, '| Time step: %i' % time_step, '|', result)
                    if e % 10==0 and e>100:
                            averg_reward=averg_reward/10
                            agent.save_model(e)
		            summary_score=tf.Summary(value=[tf.Summary.Value(tag='score', simple_value=averg_reward)])
		            summary_writer.add_summary(summary_score, global_step=e)
		            summary_writer.flush()
                            averg_reward=0
                    else:
                            averg_reward=averg_reward+tensorboard_reward 
                    break

    else:
        print('Testing mode')
        for i in range(10):
            start_time = time.time()
            state = env.reset()
            one_round_step = 0

            while(1) :
                a = agent.action(state)
                a[0] = np.clip(a[0], 0., action_linear_max)
                a[1] = np.clip(a[1], -action_angular_max, action_angular_max)
                state_, r, done, arrive = env.step(a, past_action)
                past_action = a
                state = state_
                one_round_step += 1


                if arrive:
                    m, s = divmod(int(time.time() - start_time), 60)
	            h, m = divmod(m, 60)
                    print('ep:',i,"|",'Step: %3i' % one_round_step, '| Arrive!!!','time: %d:%02d:%02d'%(h, m, s))
                    one_round_step = 0
                    break

                if done:
                    m, s = divmod(int(time.time() - start_time), 60)
	            h, m = divmod(m, 60)
                    print('ep:',i,'|','Step: %3i' % one_round_step, '| Collision!!!','time: %d:%02d:%02d'%(h, m, s))
                    break


if __name__ == '__main__':
     main()
