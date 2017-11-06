# -*- coding: utf-8 -*-
"""
Created on 20171106

@author: momos_000
"""
# using tabular Q-Learning to learn a hierarchical policy in RoomWorld
# referring to U.C. Berkeley DeepRL Bootcamp materials
# This version (11/6) adds intrapolicy learning over the LLC policy
# The LLC policy is initialized to give a reward of 1*(gamma)^(d-1) to 
# the option that would have been chosen in the original policy, and 
# 1*(gamma)^(d) for others where d is the Manhattan distance to the goal

import time
import numpy as np
from room_world import RoomWorld, SmdpAgent_Q
import learning_test_utilities as util


#setup
env          = RoomWorld()
state_space  = env.state_space
num_actions  = env.action_space.size
q_func       = util.QTable(state_space,num_actions) # as "goto hallway" options
options      = util.create_hallway_options(env)
agent_smdp   = SmdpAgent_Q(env,q_func,options)

#training
max_options = 200
iterations, epsilon, gamma, alpha = util.learning_parameters()
#alpha       = 1./16. # overwrite to match Sutton
report_freq = iterations/50
hist = np.zeros((iterations,8)) #training step, avg_td(HLC), avg_td(LLC), avg_ret, avg_greedy_ret, avg_greedy_successrate, avg_greedy_steps avg_greedy_choices
start_time = time.time()

for itr in range(iterations):
    tot_td = 0
    cur_state = env.reset(random_placement=True)
    epsilon = 0.2
    done = False
    reward_record = []
    steps = 0
    for _ in range(max_options):
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
    greedy_steps, greedy_choices, greedy_ret, greedy_success = util.greedy_eval(agent_smdp,gamma,max_options,100)
    hist[itr,:] = np.array([prev_steps+steps, tot_td/(steps), ret/(steps), greedy_ret, greedy_success, greedy_steps, greedy_choices])

    if itr % report_freq == 0: # evaluation
        print("Itr %i # Average reward: %.2f" % (itr, hist[itr,3]))

print("DONE. ({} seconds elapsed)".format(time.time()-start_time))
util.plot_and_pickle(env,agent_smdp,hist)
