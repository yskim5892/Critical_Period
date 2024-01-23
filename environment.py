import random
import time
import numpy as np
from environments_rllab.create_maze_env import create_maze_env
from mujoco_py import load_model_from_path, MjSim, MjViewer
from general import utils

class Environment():
    # Subgoal space can be the same as the state space or some other projection out of the state space
    def __init__(self, config):
        self.config = config
        self.name = config['env']

        if not hasattr(self, 'sim'):
            self.model = load_model_from_path("./mujoco_files/" + config['xml_filename'] + '.xml')
            self.sim = MjSim(self.model)

        self.state_dim = len(self.state)
        
        extents = self.sim.model.actuator_ctrlrange[:,1]
        self.action_space = utils.BoxSpace(center = np.zeros(len(extents)), extents = extents)
        
        #self.subgoal_colors = ["Magenta","Green","Red","Blue","Cyan","Orange","Maroon","Gray","White","Black"]
        self.subgoal_colors = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]])

        self.visualize = config['visualize']
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.goal = None
        self.is_testing = False
        self.dense_level = -1
        print(self.pdim, self.vdim)
        print(self.action_space.bounds)
        #self.generate_goals()

    @property
    def pdim(self):
        return len(self.sim.data.qpos)

    @property
    def vdim(self):
        return len(self.sim.data.qvel)

    @property
    def action_dim(self):
        return self.action_space.dim
    
    @property
    def goal_dim(self):
        return self.goal_space_test.dim

    @property
    def state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def reset_pos_vel(self):
        posvel = self.initial_state_space.random_sample()
        for i in range(self.pdim):
            self.sim.data.qpos[i] = posvel[i]
        for i in range(self.vdim):
            self.sim.data.qvel[i] = posvel[self.pdim + i]
    
    def initialize_state(self):
        self.reset_pos_vel()

    def render(self):
        self.viewer.render()

    def reset(self):
        if not self.config['fixed_goals']:
            self.generate_goals()

        self.steps = 0

        # goal_reached_level[i] == k : minimum distance to i-th goal ever was less than 
        #   dense_reward[dense_level][k][0]
        self.goal_reached_level = np.ones([len(self.goals)]).astype(int) * -1

        self.sim.data.ctrl[:] = 0
        self.initialize_state()
        if self.visualize:
            self.display_goals()
        self.sim.step()
        return self.state

    # Execute atomic action for number of frames specified by timesteps_per_action
    def step(self, action):
        self.sim.data.ctrl[:] = action
        total_reward = 0
        done = False
        info = dict()
        prev_state = self.state
        for _ in range(self.timesteps_per_action):
            self.sim.step()
            if self.config['simple']:
                pos = self.sim.data.qpos[:3] + action[:3] / (self.timesteps_per_action)
                pos = self.goal_space_train.clip(pos)
                for i in range(3):
                    self.sim.data.qpos[i] = pos[i]
            if self.visualize:
                self.render()
            
            reward = self.config['living_reward']
            if self.steps >= self.max_actions:
                reward += (-1.0) * self.config['timeout_penalty']
                done = True
                info['success'] = False
          
            if self.check_goal():
                info['success'] = True
                reward += self.config['goal_reward']
                if self.config['target_break']:
                    done = True
            for i in range(len(self.fake_goals)):
                if self.check_goal(goal=self.fake_goals[i]):
                    info['success'] = False
                    reward += (-1.0) * self.config['goal_penlaty']
                    if self.config['nontarget_break']:
                        done = True

            total_reward += reward
            if done:
                if self.visualize:
                    self.render()
                break
         
        if self.config['reward_setting'] == 'dense' and self.dense_level >= 1:
            total_reward += self.config['dense_reward_scale'] * (self.config['gamma'] * self.potential(self.state) - self.potential(prev_state))
        self.steps += 1
        if self.visualize:
            #print(self.state, self.goal, total_reward, done, info)
            print(self.state, self.goal, self.dist(self.state, self.goal), total_reward, done, info)
            #print(self.state, self.goal, self.potential(self.state), total_reward, done, info)
            #print(total_reward, done, info, self.goal_reached_level)
        return self.state, total_reward, done, info

    def set_dense_level(self, level):
        if level != self.dense_level: 
            print(f'Dense Level Set {self.dense_level} -> {level}')
        self.dense_level = level

    def add_noise(self, action):
        space = self.action_space
        noise = self.atomic_noise
        
        assert len(action) == space.dim, "Action must have same dimension as its space"
        assert len(action) == len(noise), "Noise percentage vector must have same dimension as action"

        action = space.clip(action + np.random.normal(0, noise * space.extents))
        return action

    def potential(self, state):
        state_g = self.project_state_to_end_goal(state)
        goal_space = self.goal_space_test
        diameter = np.linalg.norm(goal_space.bounds[:, 0] - goal_space.bounds[:, 1]) 
        return diameter - np.linalg.norm(state_g - self.goal)

    def check_goal(self, state=None, goal=None):
        if state is None:
            state = self.state
        if goal is None:
            goal = self.goal
        state_g = self.project_state_to_end_goal(state)
        return np.all(np.absolute(goal - state_g) <= self.goal_thr)

    def get_random_action(self):
        return self.action_space.random_sample()

    def is_valid_goal(self, end_goal):
        return True

    # Sets a goal and fake goals
    def generate_goals(self, n=None):
        goal_space = self.goal_space_test if self.is_testing else self.goal_space_train

        if n is None:
            n = self.config['n_goals']
        goals = []
        for i in range(n):
            while True:
                goal = goal_space.random_sample()
                all_far = True
                for other_goal in goals:
                    if np.all(np.absolute(goal[:2] - other_goal[:2]) <= self.config['goals_min_dist']):
                        all_far = False
                        break
                if all_far and self.is_valid_goal(goal):
                    break
            goals.append(goal)

        self.goals = np.array(goals)
        self.goal = self.goals[0]
        self.fake_goals = self.goals[1:]
    
    # Visualize goals. This function may need to be adjusted for new environments.
    def display_goals(self):
        #print('Goal :', self.goal)
        self.sim.data.mocap_pos[0][:len(self.goal)] = np.copy(self.goal)
        self.sim.model.site_rgba[0][3] = 1
        for i in range(len(self.fake_goals)):
            #print('Fake :', self.fake_goals[i])
            self.sim.data.mocap_pos[i+1][:len(self.goal)] = np.copy(self.fake_goals[i])

class EnvironmentRLLab(Environment):
    def __init__(self, config):
        self.base_env = create_maze_env(config['env'], config['scaling'])
        self.sim = self.base_env.wrapped_env.sim

        super().__init__(config)

    #def render(self):
    #    self.base_env.wrapped_env.render()

    '''def reset(self):
        self.goal = self.generate_goal()
        if self.visualize:
            self.display_goals()
        self.sim.data.ctrl[:] = 0
        #self.base_env.wrapped_env.reset()
        self.initialize_state()
        return self.base_env.wrapped_env._get_obs()'''

    # Execute atomic action for number of frames specified by timesteps_per_action
    '''def step(self, action):
        next_state, _, _, _ = self.base_env.wrapped_env.step(action, self.timesteps_per_action)
        #next_state, _, _, _ = self.base_env.wrapped_env.step(action)

        if self.visualize:
            self.viewer.render()
        return next_state'''
