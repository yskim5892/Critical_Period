from environment import Environment
import numpy as np
from general import utils

class PendulumEnvironment(Environment):
    def __init__(self, config):
        self.initial_state_space = utils.BoxSpace(bounds = np.array([[np.pi/4, 7*np.pi/4], [-0.05, 0.05]]))

        self.goal_space_train = utils.BoxSpace(bounds = [[np.deg2rad(-45), np.deg2rad(45)], [-15, 15]])
        self.goal_space_test = utils.BoxSpace(bounds = [[np.deg2rad(-10), np.deg2rad(10)], [-15, 15]])

        self.goal_thr = 0.1

        
        self.max_actions = 1000 if self.config['max_actions'] is None else self.config['max_actions']
        self.timesteps_per_action = 10 if self.config['timesteps_per_action'] is None else self.config['timesteps_per_action']
        self.goal_thr = 0.1 if self.config['goal_thr'] is None else self.config['goal_thr']

        super().__init__(config)
    @property
    def state(self):
        return np.concatenate([np.cos(self.sim.data.qpos), np.sin(self.sim.data.qpos), self.sim.data.qvel])

    def dist(self, state, goal):
        x, y, g = state[0], state[1], goal[0]
        return np.linalg.norm([state[0] - np.cos(g), state[1] - np.sin(g)]) / 2
        #return np.linalg.norm([np.cos(a) - np.cos(b), np.sin(a) - np.sin(b)]) / 2

    def potential(self, state):
        return 1 - self.dist(state, self.goal)

    def check_goal(self, state=None, goal=None):
        if state is None:
            state = self.state
        if goal is None:
            goal = self.goal
        d = self.dist(self.state, self.goal)
        return d <= self.goal_thr and np.absolute(state[2] - goal[1]) <= 1

    # Supplemental function that converts angle to between [-pi,pi]
    def _bound_angle(self, angle):
        bounded_angle = angle % (2*np.pi)

        if np.absolute(bounded_angle) > np.pi:
            bounded_angle = bounded_angle % np.pi - np.pi

        return bounded_angle

    '''@property
    def state(self):
        return np.concatenate([np.cos(self.sim.data.qpos), np.sin(self.sim.data.qpos), self.sim.data.qvel])'''

    def is_valid_goal(self, goal):
        return np.absolute(goal[1]) >= 8

    def display_goals(self):
        self.sim.data.mocap_pos[0] = np.array([0.5 * np.sin(self.goal[0]), 0, 0.5 * np.cos(self.goal[0]) + 0.6])

    def project_state_to_end_goal(self, state):
        pi = np.artan2(state[1], state[0])
        return np.array([pi, np.clip(state[2], -15, 15)])
