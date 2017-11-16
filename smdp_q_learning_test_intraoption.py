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


#training settings
max_options = 200
iterations, epsilon, gamma, alpha = util.learning_parameters()
report_freq = iterations/100
hist = np.zeros((iterations,8)) #training step, avg_td(HLC), avg_ret, avg_greedy_ret, avg_greedy_successrate, avg_greedy_steps, avg_greedy_choices, avg_td(LLC)

#setup
env          = RoomWorld()
state_space  = env.state_space
num_actions  = env.action_space.size
q_func       = util.QTable(state_space,num_actions) # as "goto hallway" options
options_q    = util.create_hallway_qtables(env,gamma,num_actions)
agent_smdp   = SmdpAgent_Q(env,q_func,options_q)
start_time = time.time()

# option update switching
switch_r           = 0.9 # success rate at which to start updating options
success_average_T  = 5   # number of episodes over which to average success
                         # rate for option update switching
last_success_rates = [0.0]*success_average_T
                     # used as a switch for starting option updates once
                     # average success over T iterations is > switch_r

for itr in range(iterations):
    tot_td = 0
    tot_tdo= 0
    cur_state = env.reset(random_placement=True)
    done = False
    reward_record = []
    steps = 0
    
    for _ in range(max_options):
        opt  = agent_smdp.pick_option_greedy_epsilon(cur_state, eps=epsilon)
        states,actions,rewards,done = env.step_option(opt,agent_smdp.sebango)
        next_state = states[-1]
        tdes = util.q_learning_update_option_sequence(gamma, alpha, \
                                    agent_smdp.q_func, states, \
                                    rewards, opt.identifier)
        if np.mean(last_success_rates) > switch_r: #update options
            if len(states)==1: # this happens if option was chosen in its termination state
                tdes_opt = [0.] # no update
            else:
                opt_rew = [env.step_reward]*np.max([len(rewards)-1,1])
                if states[-1].tolist() in opt.termination.tolist():
                    opt_rew[-1] = opt.success_reward # assumes all terminations states correspond to success
                tdes_opt = util.q_learning_update_intraoption(gamma, alpha, \
                            opt.policy, states, opt_rew, actions)
        else: # no update
            tdes_opt = [0.]
        tot_td   += np.sum(tdes)
        tot_tdo  += np.sum(tdes_opt)
        reward_record.append(rewards)
        cur_state = next_state
        steps += len(states)
        if done:
            break
    prev_steps = hist[itr-1,0]
    ret = util.discounted_return(reward_record,gamma)
    greedy_steps, greedy_choices, greedy_ret, greedy_success = util.greedy_eval(agent_smdp,gamma,max_options,100)
    hist[itr,:] = np.array([prev_steps+steps, tot_td/(steps), ret, greedy_ret, greedy_success, greedy_steps, greedy_choices, tot_tdo/(steps)])
    last_success_rates = last_success_rates[1:]+[greedy_success]
    
    if itr % report_freq == 0: # evaluation
        print("Itr %i # Average reward: %.2f" % (itr, hist[itr,3]))

print("DONE. ({} seconds elapsed)".format(time.time()-start_time))
util.plot_and_pickle(env,agent_smdp,hist)
