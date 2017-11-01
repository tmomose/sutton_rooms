# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:57:52 2017

@author: momos_000
"""
# using tabular Q-Learning to learn a flat policy in RoomWorld
# referring to U.C. Berkeley DeepRL Bootcamp materials

import time
import numpy as np
from room_world import RoomWorld, Agent_Q
import learning_test_utilities as util


env          = RoomWorld()
state_space  = env.state_space
num_actions  = env.action_space.size
q_func       = util.QTable(state_space,num_actions)
agent_q      = Agent_Q(env,q_func)
cur_state    = env.reset(random_placement=True)
#training
iterations = 10000
max_steps  = 1000
epsilon, gamma, alpha = util.learning_parameters()
report_freq = iterations/20
hist = np.zeros((iterations,5)) #primitive step, avg_td, avg_ret, avg_greedy_ret, avg_greedy_steps
start_time = time.time()

for itr in range(iterations):
    tot_td = 0
    cur_state = env.reset(random_placement=True)
    stp = 0
    rewards = []
    done = False
    while not done and stp<max_steps:
        #epsilon = np.max([0.1,1.-itr/(iterations/2.)]) # linear epsilon-decay
        action  = agent_q.epsilon_greedy_action(cur_state,eps=epsilon)
        next_state, reward, done = env.step(action)
        rewards.append(reward)
        tde     = util.q_learning_update(gamma, alpha, agent_q.q_func.table, cur_state, action, next_state, reward)
        tot_td += tde
        stp += 1
        cur_state = next_state
    # record results for this iteration
    prev_steps = hist[itr-1,0]
    greedy_ret, greedy_steps = util.greedy_eval(agent_q,gamma,max_steps,10)
    hist[itr,:] = np.array([prev_steps+stp, tot_td/(stp), util.discounted_return(rewards,gamma)/stp, greedy_ret, greedy_steps])
    
    if itr % report_freq == 0: # evaluation
        print("Itr %i # Average reward: %.2f" % (itr, hist[itr,3]))

print("DONE. ({} seconds elapsed)".format(time.time()-start_time)) 