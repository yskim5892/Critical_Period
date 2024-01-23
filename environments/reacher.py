from environment import Environment
import numpy as np
from general import utils

class ReacherEnvironment(Environment):
    def __init__(self, config):
        self.config = config
        initial_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        initial_pos_bounds = np.stack([initial_pos, initial_pos], axis=1)
        margin = 0.75
        initial_pos_bounds[0] = np.array([-np.pi, np.pi])
        initial_pos_bounds[1] = np.array([-np.pi, np.pi])

        # Cocatenate velocity ranges
        self.initial_state_space = utils.BoxSpace(bounds = np.concatenate((initial_pos_bounds,np.zeros((len(initial_pos_bounds)-1, 2))),0))

        self.r = 0.2
        # For additional customizations, implement "is_valid_goal" function
        goal_min = - self.r + 0.1
        goal_max = self.r - 0.1
        self.goal_space_train = utils.BoxSpace(bounds = [[goal_min, goal_max],[goal_min, goal_max]])
        self.goal_space_test = utils.BoxSpace(bounds = [[goal_min, goal_max],[goal_min, goal_max]])

        self.atomic_noise = [0.2 for i in range(8)]

        self.max_actions = 400 if self.config['max_actions'] is None else self.config['max_actions']
        self.timesteps_per_action = 10 if self.config['timesteps_per_action'] is None else self.config['timesteps_per_action']
        self.goal_thr = 0.05 if self.config['goal_thr'] is None else self.config['goal_thr']

        super().__init__(config)
    
    def dist(self, state, goal):
        v = self.sim.data.get_geom_xpos('fingertip')[:2] - self.goal
        return np.linalg.norm(v)

    def potential(self, state):
        return 0.4 - self.dist(state, self.goal)

    def check_goal(self, state=None, goal=None):
        if state is None:
            state = self.state
        if goal is None:
            goal = self.goal
        d = self.dist(self.state, self.goal)
        return d <= self.goal_thr

    @property
    def state(self):
        theta = self.sim.data.qpos[:2]
        return np.concatenate([np.cos(theta), np.sin(theta), self.sim.data.qpos[2:], self.sim.data.qvel[:2]])

    def project_state_to_end_goal(self, state):
        return np.array(state[1], state[3])

    def display_goals(self):
        g = self.goal
        self.sim.data.mocap_pos[0] = np.array([g[0], g[1], 0])
        self.sim.model.site_rgba[0][3] = 1

    def initialize_state(self):
        self.reset_pos_vel()

    def is_valid_goal(self, goal):
        return True
