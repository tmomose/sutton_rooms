import numpy as np
import copy


# Action definitions
RIGHT = 0
UP    = 1
LEFT  = 2
DOWN  = 3


class RoomWorld():
    """The environment for Sutton's semi-MDP HRL.
    """
    def __init__(self,goal_position=[7,9]):
        """Map of the rooms. -1 indicates wall, 0 indicates hallway,
           positive numbers indicate numbered rooms
        """
        self.numbered_map = np.array([
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1, 1, 1, 1, 1, 1,-1, 2, 2, 2, 2, 2,-1],
        [-1, 1, 1, 1, 1, 1,-1, 2, 2, 2, 2, 2,-1],
        [-1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2,-1],
        [-1, 1, 1, 1, 1, 1,-1, 2, 2, 2, 2, 2,-1],
        [-1, 1, 1, 1, 1, 1,-1, 2, 2, 2, 2, 2,-1],
        [-1,-1, 0,-1,-1,-1,-1, 2, 2, 2, 2, 2,-1],
        [-1, 3, 3, 3, 3, 3,-1,-1,-1, 0,-1,-1,-1],
        [-1, 3, 3, 3, 3, 3,-1, 4, 4, 4, 4, 4,-1],
        [-1, 3, 3, 3, 3, 3,-1, 4, 4, 4, 4, 4,-1],
        [-1, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4,-1],
        [-1, 3, 3, 3, 3, 3,-1, 4, 4, 4, 4, 4,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
        self.walkability_map = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.state_space   = np.argwhere(self.walkability_map)
        self.action_space  = np.arange(4) 
        self.goal_position = np.array(goal_position)
        self.action_success_rate = 0.666
        self.agents = [] # agents affect each other's observations, so should be included
        # Rewards
        self.step_reward      = 0.0 #-0.1 (Sutton used 0 and depended on discounting effect of gamma to push toward more efficient policies)
        self.collision_reward = 0.0 # was -0.1 at first, but spending a 
                                    # timestep without moving is a penalty
        self.goal_reward      = 1.#10.
        self.invalid_plan_reward = 0.0#-10.


    def add_agent(self,agent):
        """Adds an agent to the environment after giving it an identifier
        """
        agent.sebango = len(self.agents) + 2
        self.agents.append(agent)


    def move_agent(self,direction,sebango=2):
        """Attempts moving an agent in a specified direction.
           If the move would put the agent in a wall, the agent remains
           where he is and is given a negative reward value.
        """
        agent  = self.agents[sebango-2]
        new_pos = agent.move(direction)
        if self.walkability_map[tuple(new_pos)].all():
            agent.set_position(new_pos)
            collision = False
        else:
            collision = True
        return collision

    def evaluate_reward(self,sebango=2,collision=False):
        """Calculates the reward to be given for the current timestep after an
           action has been taken.
        """
        agent  = self.agents[sebango-2]
        reward = self.step_reward
        done   = False
        if collision:
            reward += self.collision_reward
        if (agent.get_position() == self.goal_position).all():
            reward += self.goal_reward
            done = True
        return reward, done

    def get_observation_map(self):
        """Returns the observation of the current state as a walkability map
           with agents (sebango) and goal position (-1) labeled
        """
        obs = copy.copy(self.walkability_map)
        for ag in self.agents:
            obs[tuple(ag.get_position())] = ag.sebango
        obs[tuple(self.goal_position)] = -1
        return obs
    
    def get_observation_pos(self,sebango):
        """Returns the observation of the current state as the position of the
           agent indicated by sebango.
           Assumes single agent and static goal location so only need agent pos
        """
        return self.agents[sebango-2].get_position()


    def step(self,direction,sebango=2):
        """Takes one timestep with a specific direction.
           Only deals with primitive actions.
           Determines the actual direction of motion stochastically
           Determines the reward and returns reward and observation.
           Observation is the walkability map + other info:
             - the agent indicated by its sebango (a number 2 or greater)
             - The goal is indicated as -1 in the observation map. 
        """
        roll   = np.random.random()
        sr = self.action_success_rate
        fr = 1.0 - sr
        if roll <= sr:
            coll = self.move_agent(direction,sebango)
        elif roll <= sr+fr/3.:
            coll = self.move_agent((direction+1)%4,sebango)
        elif roll <= sr+fr*2./3.:
            coll = self.move_agent((direction+2)%4,sebango)
        else:
            coll = self.move_agent((direction+3)%4,sebango)
        obs = self.get_observation_pos(2)
        reward, done = self.evaluate_reward(sebango,collision=coll)
        return obs, reward, done
    
    
    def step_option(self,option,sebango=2):
        """Steps through an option until termnation, then returns the final
           observation, reward history, and finishing evaluation.
        """
        obs  = [self.get_observation_pos(sebango)]
        acts = []
        rew  = []
        done = False
        while not done: # and not option.check_termination(obs[-1]):
            action = option.act(obs[-1])
            if action is not None: # option was valid
                acts.append(action)
                ob, re, done = self.step(acts[-1],sebango)
                rew.append(re)
                obs.append(ob)
        self.agents[sebango-2].current_option = None
        return obs, acts, rew, done
    
    
    def step_plan(self,sebango=2):
        """Steps through the plan set up in agent with the specified number,
           for n options where n is the plan length. 
           For now, no replanning partway through.
           Returns lists of observations, actions, and rewards together with a
           label for being done or not. Each is a list of lists, with the
           inner lists corresponding to the results of applying each option.
           If the goal is reached before the end of the plan is reached, then
           subsequent options are skipped and the corresponding lists are 
           recorded as [None].
        """
        obs  = []
        acts = []
        rew  = []
        done = []
        fin  = False
        agent = self.agents[sebango-2]
        for i in range(agent.plan_length):
            if not fin:
                valid = agent.options[agent.current_plan[i]].check_validity(self.get_observation_pos(sebango))
                if valid:
                    op = agent.current_plan[i]
                    obsi,actsi,rewi,donei = self.step_option(agent.options[op],sebango)
                    obs.append(obsi)
                    acts.append(actsi)
                    rew.append(rewi)
                    done.append(donei)
                    if donei:
                        fin = True
                else:
                    obs.append([self.get_observation_pos(sebango)])
                    acts.append([None])
                    rew.append([self.invalid_plan_reward])
                    _, donei = self.evaluate_reward()
                    done.append(donei)
                    if donei:
                        fin = True
            else:
                obs.append([self.get_observation_pos(sebango)])
                acts.append([None])
                rew.append([0.0])
                done.append(True)
        return obs, acts, rew, done


    def reset(self, random_placement=False):
        """Resets the state of the world, putting all registered  agents back
           to their initial positions (positions set at instantiation),
           unless random_placement = True 
        """
        if random_placement:
            random_index     = np.random.randint(low=0,
                    high=self.state_space.shape[0],size=len(self.agents))
            for i,ag in enumerate(self.agents):
                ag.set_position(self.state_space[random_index[i]])
        else:
            for ag in self.agents:
                ag.set_position(ag.initial_position)
        obs = self.get_observation_pos(2)    # CURRENTLY ASSUMING ONE AGENT!
        return obs



class Agent():
    """Base agent class for interacting with RoomWorld.
    """
    def __init__(self, env, initial_position=[1,1]):
        self.set_position(initial_position)
        self.initial_position = self.position
        self.sebango          = None
        self.num_actions      = 4 #UP,DOWN,LEFT,RIGHT
        self.environment      = env
        env.add_agent(self)


    def random_action(self):
        """Random action generator. State shape is (n_samples,i,j) where i,j
           are the rows and columns of a single observation.
        """
        return np.random.randint(low=0, high=self.num_actions)


    def move(self, direction):
        """Tries moving the agent in the specified direction.
           Returns new position, which must be checked against the map
           and approved or disapproved by the environment manager.
           direction is as specified at the start of this file.
                  numpy array has axis 0 vertical, so coordinates are (-y,x).
        """
        ax0_change  = (direction % 2) * (direction - 2)
        ax1_change = ((direction+1) % 2) * (1 - direction)
        return np.array(self.get_position() + [ax0_change,ax1_change])

    
    def set_position(self, new_pos):
        self.position = np.array(new_pos)

    def get_position(self):
        return self.position



class Agent_Q(Agent):
    """Agent class using a q-function to choose actions.
       q_func should be a callable object that takes observation as input and
       outputs an array of q-values for the various actions
    """
    def __init__(self, env, q_func, initial_position=[1,1]):
        super().__init__(env, initial_position=initial_position)
        self.q_func          = q_func
        self.q_func.sebango  = self.sebango


    def greedy_action(self, state):
        q_values = self.q_func(state)
        return np.argmax(q_values)

    def epsilon_greedy_action(self, state, eps=0.1):
        roll = np.random.random()
        if roll <= eps:
            return self.random_action()
        else:
            return self.greedy_action(state)



class SmdpAgent_Q(Agent_Q):
    """Agent class with q-function for choosing among predefined options.
       SmdpAgent_Q takes a list of trained options that number the same as 
       the number of outputs from q_func. The Q-values are used to determine 
       which option is used whenever a option is terminated.
       No interruption is encoded in this class, but the same effect could be
       attained by adding a critic to terminate the option when another option
       is deemed more beneficial.
    """
    def __init__(self, env, q_func, options, initial_position=[1,1]):
        super().__init__(env, q_func, initial_position=initial_position)
        if not len(options)==self.q_func.num_actions:
            print("WARNING: Number of options does not match Q-table dimensions")
        self.options        = options
        self.num_options    = len(self.options)
        self.current_option = None
        for i,opt in enumerate(self.options): # label the options for q-learning
            opt.identifier = i


    def pick_option_greedy_epsilon(self, state, eps=0.0):
        """Chooses a new option to apply at state
        """
        valid_options = [i for i in np.arange(self.num_options) if self.options[i].check_validity(state)]
        all_qs        = self.q_func(state)
        valid_qs      = [all_qs[i] for i in valid_options]
        roll = np.random.random()
        if roll <= eps:
            self.current_option = np.random.choice(valid_options)
        else:
            self.current_option = valid_options[np.argmax(valid_qs)]
        return self.options[self.current_option]



class SmdpPlanningAgent_Q(SmdpAgent_Q):
    """SmdpAgent_Q with a plan (option sequence) output rather than primitive
       actions."""
       
    def __init__(self, env, q_func, options, initial_position=[1,1], plan_length=2):
        super().__init__(env, q_func, options, initial_position=initial_position)
        self.plan_length = plan_length
        if not self.num_options**self.plan_length==self.q_func.num_actions:
            print("WARNING: plan dimensions do not match Q-table dimensions")
        self.plan = [None]*self.plan_length
        
    def make_plan(self,state):
        """Takes the current (starting) state and outputs a plan (sequence of
           options) to reach the goal. The plan is followed to completion 
           without re-evaluation (for now).
        """
        return self.make_plan_epsilon_greedy(state,epsilon=0.)
    
    def make_plan_epsilon_greedy(self,state,epsilon=0.):
        """Takes the current (starting) state and outputs a plan (sequence of
           options) to reach the goal. The plan is followed to completion 
           without re-evaluation (for now).
           For each part of the plan, a random option is chosen with 
           probability epsilon
        """
        roll = np.random.random()
        if roll <= epsilon:
            max_q = np.random.choice(np.arange(self.num_options**self.plan_length))
            #self.current_option = np.random.choice(valid_options)
        else:
            max_q = np.argmax(self.q_func(state))
        plan = np.array(np.unravel_index(max_q,[self.num_options]*self.plan_length))

        self.current_plan   = plan
        self.current_option = plan[0]
        return plan

    
    def advance_plan(self,state):
        """Advances the plan by one step, assuming that self.plan[0] has been
           completed/terminated. Advancement is achieved by shifting and
           back-filling with -(self.num_actions+1). (This value will return an
           IndexError if called)
        """
        self.plan[:-1] = self.plan[1:]
        self.plan[-1]  = -(self.num_actions+1)
        self.current_option = self.plan[0]
        if not self.plan[0].check_validity(state):
            valid_options = [i for i in np.arange(self.num_options) if self.options[i].check_validity(state)]
            self.plan[0] = np.random.choice(valid_options)
        return self.plan
    


class Option():
    """Semi-MDP option class. Deterministic policy. Callable.
       ATTRIBUTES:
           - num_actions: how many actions the option policy chooses among
           - policy: the deterministic option policy in the form of an array
                     matching the environment map shape, where entries at valid
                     states match the action for that state, other entries are
                     -1
       INPUTS:
           - state (observation)
       OUTPUTS:
           - next action (assumed to be greedy.) #TODO: intra-policy training
    """
    def __init__(self, policy, valid_states, termination_conditions, num_actions=4):
        self.policy      = policy
        self.num_actions = num_actions
        self.activation  = np.array(valid_states) # activation conditions (states)
        self.termination = np.reshape(termination_conditions,(-1,2)) # (states)
    
    
    def act(self,state):
        """The policy. Takes state (or observation) and returns action.
           This simply reads the necessary action from self.policy
           The action is applied to the agent in the arguments
        """
        if self.check_termination(state):
            return None
        else:
            return int(self.policy[tuple(state)])
            
    
    def greedy_action(self,state):
        """Would be used if non-deterministic. Included for compatibility,
           just in case if I add q-learning.
        """
        return self.act(state)
    
    
    def check_validity(self,state):
        """Returns boolean indicator of whether or not the state is among valid
           starting points for this option.
        """
        if type(state)==np.ndarray:
            state = state.tolist()
        return state in self.activation.tolist()
        
        
    def check_termination(self,state):
        """Returns boolean indicator of whether or not the policy is at a 
           termination state. (or not in a valid state to begin with)
        """
        if type(state)==np.ndarray:
            state = state.tolist()
        if state in self.termination.tolist() or not self.check_validity(state):
            return True
        else:
            return False
    

class Option_Q(Option):
    """This type of option stores the policy as a Q-table instead of an action
       lookup table. Use learning_test_utilities.QTable
    """
    def __init__(self, policy, valid_states, termination_conditions, num_actions=4, success_reward=1.0):
        super().__init__(policy, valid_states, termination_conditions, num_actions=num_actions)
        self.success_reward = success_reward
        
        
    def act(self,state):
        """Takes state (or observation) and returns action (argmax(Q)).
        """
        assert(not self.check_termination(state))
        q_values = self.policy(state)
        return np.argmax(q_values)