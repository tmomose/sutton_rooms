# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 2017

@author: momos_000
"""
# using tabular Q-Learning to learn a policy over options in RoomWorld
# referring to U.C. Berkeley DeepRL Bootcamp materials
# Nov 15: HRL without planning, with interruption

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
iterations=100
#alpha       = 1./16. # overwrite to match Sutton
report_freq = iterations/50
hist = np.zeros((iterations,7)) #training step, avg_td, avg_ret, avg_greedy_ret, avg_greedy_successrate, avg_greedy_steps, avg_greedy_choices
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
        states  = [cur_state]
        actions = []
        rewards = []
        done    = False
        switch  = False
        i       = 0
        while not switch and not done:
            actions.append(opt.act(states[-1]))
            ob, re, done = env.step(actions[-1],agent_smdp.sebango)
            rewards.append(re)
            states.append(ob)
            i+=1
            current_best_opt = agent_smdp.pick_option_greedy_epsilon(states[-1], eps=epsilon)
            # TODO: Add a margin so it doesn't get too trigger happy?
            if current_best_opt!=opt or opt.check_termination(states[-1]):
                switch = True
        next_state = states[-1]
        tdes = util.q_learning_update_option_sequence(gamma, alpha, \
                                    agent_smdp.q_func, states, \
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