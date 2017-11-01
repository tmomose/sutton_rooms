# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:57:52 2017

@author: momos_000
"""
# using tabular Q-Learning to learn a flat policy in RoomWorld
# referring to U.C. Berkeley DeepRL Bootcamp materials

import time
import numpy as np
from room_world import RoomWorld, SmdpAgent_Q
import learning_test_utilities as util


iterations=1000

#setup
env          = RoomWorld()
state_space  = env.state_space
num_actions  = env.action_space.size
q_func       = util.QTable(state_space,num_actions) # as "goto hallway" options
options      = util.create_hallway_options(env)
agent_smdp   = SmdpAgent_Q(env,q_func,options)

#training
max_options = 5
gamma = 0.9
alpha = 0.05
report_freq = iterations/20
hist = np.zeros((iterations,4)) #primitive step, avg_td, avg_ret, avg_greedy_ret
start_time = time.time()

for itr in range(iterations):
    tot_td = 0
    cur_state = env.reset(random_placement=True)
    epsilon = 0.2
    done = False
    reward_record = []
    steps = 0
    for _ in range(max_options):
        #epsilon = np.max([0.1,1.-itr/(iterations/2.)]) # linear epsilon-decay
        opt  = agent_smdp.pick_option_greedy_epsilon(cur_state, eps=epsilon)
        states,actions,rewards,done = env.step_option(opt,agent_smdp.sebango)
        next_state = states[-1]
        tdes = util.q_learning_update_option_sequence(gamma, alpha, \
                                    agent_smdp.q_func.table, states, \
                                    rewards, opt.identifier)
        tot_td   += np.sum(tdes)
        reward_record.append(rewards)
        cur_state = next_state
        steps += len(states)
        if done:
            break
    prev_steps = hist[itr-1,0]
    ret = util.discounted_return(reward_record,gamma)
    hist[itr,:] = np.array([prev_steps+steps, tot_td/(steps), ret/(steps), util.greedy_eval(agent_smdp,gamma,max_options,10)])
    
    if itr % report_freq == 0: # evaluation
        print("Itr %i # Average reward: %.2f" % (itr, hist[itr,3]))

print("DONE. ({} seconds elapsed)".format(time.time()-start_time))