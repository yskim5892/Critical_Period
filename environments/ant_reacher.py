from environment import EnvironmentRLLab
import numpy as np
from general import utils

class AntReacherEnvironment(EnvironmentRLLab):
    def __init__(self, config):
        scaling = config['scaling']
        initial_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_pos_bounds = np.stack([initial_pos, initial_pos], axis=1)
        margin = 0.75
        min_range = -0.5 * scaling + margin
        max_range = 0.5 * scaling - margin
        initial_pos_bounds[0] = np.array([min_range, max_range])
        initial_pos_bounds[1] = np.array([min_range, max_range])

        # Cocatenate velocity ranges
        self.initial_state_space = utils.BoxSpace(bounds = np.concatenate((initial_pos_bounds,np.zeros((len(initial_pos_bounds)-1, 2))),0))

        # For additional customizations, implement "is_valid_goal" function
        goal_pos = 0 #0.25 * scaling
        goal_min = min_range # goal_pos - 0.1
        goal_max = max_range # goal_pos + 0.1
        self.goal_space_train = utils.BoxSpace(bounds = [[goal_min, goal_max],[goal_min, goal_max],[0.45,0.55]])
        self.goal_space_test = utils.BoxSpace(bounds = [[goal_min, goal_max],[goal_min, goal_max],[0.45,0.55]])

        self.atomic_noise = [0.2 for i in range(8)]

        self.max_actions = 100 * scaling
        self.timesteps_per_action = 15
        self.goal_thr = 1

        super().__init__(config)

    def project_state_to_end_goal(self, state):
        return np.copy(state[:3])

    def initialize_state(self):
        while True:
            self.reset_pos_vel()
            if np.linalg.norm(self.goal[:2] - self.sim.data.qpos[:2]) > self.config['scaling'] / 4:
                break

    def is_valid_goal(self, goal):
        return True
