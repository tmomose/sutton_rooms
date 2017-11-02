# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:57:52 2017

@author: momos_000
"""
# using tabular Q-Learning to learn a flat policy in RoomWorld
# referring to U.C. Berkeley DeepRL Bootcamp materials

import time
import numpy as np
from room_world import RoomWorld, SmdpPlanningAgent_Q
import learning_test_utilities as util

#setup
env          = RoomWorld()
state_space  = env.state_space
num_actions  = env.action_space.size
plan_length  = 2
q_func       = util.QTable(state_space,num_actions**plan_length) # as "goto hallway" options
options      = util.create_hallway_options(env)
agent_plan   = SmdpPlanningAgent_Q(env,q_func,options,plan_length=plan_length)  
#training
iterations, epsilon, gamma, alpha = util.learning_parameters()
#alpha       = 1./16. # overwrite to match Sutton
report_freq = iterations/50
hist        = np.zeros((iterations,6)) #primitive step, avg_td, avg_ret, avg_greedy_ret, avg_greedy_steps, avg_greedy_successrate
start_time  = time.time()

for itr in range(iterations):
    initial_state = env.reset(random_placement=True)
    done = False
    # Since plan length is 2, each "action" is evaluated once (no replan) 
    plan = agent_plan.make_plan_epsilon_greedy(initial_state,epsilon=epsilon)
    option_index = plan[0]*agent_plan.num_actions+plan[1]
    states, actions, rewards, done = env.step_plan(agent_plan.sebango)
    ret = util.discounted_return(rewards,gamma)
    # Add in a bonus at the end of the first option for subsequent options
    rewards2 = rewards[0]+[util.discounted_return(rewards[1:],gamma)*gamma**(len(rewards[0])+1)]
    steps = np.sum([len(s) for s in states]) 
    if actions[0] == [None]: # no valid action chosen
        states[0] = [initial_state,initial_state] # no transition
        
    # update q-table
    tdes    = util.q_learning_update_plan_options(gamma, alpha, \
                                agent_plan.q_func.table, states[0], \
                                rewards2, option_index)
    tot_td  = np.sum(tdes)
    prev_steps = hist[itr-1,0]
    greedy_ret, greedy_steps, greedy_success = util.greedy_eval(agent_plan,gamma,1,100)
    hist[itr,:] = np.array([prev_steps+steps, tot_td/(steps), ret/(steps), greedy_ret, greedy_steps, greedy_success])
    
    if itr % report_freq == 0: # evaluation
        print("Itr %i # Average reward: %.2f" % (itr, hist[itr,3]))
        
print("DONE. ({} seconds elapsed)".format(time.time()-start_time))