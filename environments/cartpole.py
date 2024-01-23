from environment import Environment
import numpy as np
from general import utils

class CartPoleEnvironment(Environment):
    def __init__(self, config):
        self.config = config
        scaling = config['scaling']
        margin = 0.1
        
        initial_pos_bounds = np.array([[-1, 1], [np.pi-0.1, np.pi+0.1], [0, 0], [0, 0]])

        # Cocatenate velocity ranges
        self.initial_state_space = utils.BoxSpace(bounds = np.concatenate((initial_pos_bounds,np.zeros((len(initial_pos_bounds)-1, 2))),0))

        # For additional customizations, implement "is_valid_goal" function
        max_range = 1
        goal_pos = 0 #0.25 * scaling
        goal_min = - max_range + margin # goal_pos - 0.1
        goal_max = max_range - margin # goal_pos + 0.1
        self.goal_space_train = utils.BoxSpace(bounds = [[goal_min, goal_max],[-np.pi/2, np.pi/2]])
        self.goal_space_test = utils.BoxSpace(bounds = [[goal_min, goal_max],[-np.pi/2, np.pi/2]])

        self.atomic_noise = [0.2 for i in range(1)]

        self.max_actions = 200 if self.config['max_actions'] is None else self.config['max_actions']
        self.timesteps_per_action = 10 if self.config['timesteps_per_action'] is None else self.config['timesteps_per_action']
        self.goal_thr = 0.01 if self.config['goal_thr'] is None else self.config['goal_thr']

        super().__init__(config)

    '''def step(self, action):
        self.sim.data.ctrl[:] = action
        total_reward = 0
        done = False
        info = dict()
        prev_state = self.state
        for _ in range(self.timesteps_per_action):
            self.sim.step()
            if self.visualize:
                self.render()
    
            if self.steps >= self.max_actions:
                done = True
                info['success'] = False
                break
        
        total_reward = self.get_reward(self.state)
        self.steps += 1
        #print(self.state, self.goal, total_reward, done, info)
        #print(self.state, total_reward, done, info)
        #print(self.state, self.goal, self.potential(self.state), total_reward, done, info)
        #print(total_reward, done, info, self.goal_reached_level)
        return self.state, total_reward, done, info


    def get_reward(self, state):
        angle = state[1]
        if self.dense_level == 0:
            angle_thr = np.deg2rad(5)
        elif self.dense_level == 1:
            angle_thr = np.deg2rad(10)
        elif self.dense_level == 2:
            angle_thr = np.deg2rad(20)
        
        if np.absolute(angle) <= angle_thr:
            return 1
        return 0'''

    def project_state_to_end_goal(self, state):
        return np.copy(state[:2])

    def initialize_state(self):
        while True:
            self.reset_pos_vel()
            goal_xyz = self.to_xyz(self.goal)
            state_xyz = self.to_xyz(self.state)
            if np.linalg.norm(goal_xyz - state_xyz) > 0.4:
                break

    def is_valid_goal(self, goal):
        return True

    def potential(self, state):
        state_xyz = self.to_xyz(state)
        goal_xyz = self.to_xyz(self.goal)
        return np.sqrt(5) - np.linalg.norm(state_xyz - goal_xyz)

    def check_goal(self):
        state_xyz = self.to_xyz(self.state)
        goal_xyz = self.to_xyz(self.goal)
        return np.all(np.absolute(goal_xyz - state_xyz) <= self.goal_thr)

    def to_xyz(self, s):
        x, theta = s[0], s[1]
        return np.array([x + 0.1, 0.05, 0.025]) + 0.5 * np.array([np.sin(theta), 0, np.cos(theta)])
    
    def display_goals(self):
        #print('Goal :', self.goal)
        x, y, z = self.to_xyz(self.goal)
        self.sim.data.mocap_pos[0] = [x, y, z]
        '''for i in range(len(self.fake_goals)):
            #print('Fake :', self.fake_goals[i])
            self.sim.data.mocap_pos[i+1][:len(self.goal)] = np.copy(self.fake_goals[i])'''

