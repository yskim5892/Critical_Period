from environment import EnvironmentRLLab, Environment
import numpy as np
from general import utils

class AntMazeEnvironment(EnvironmentRLLab):
    def __init__(self, config):
        self.config = config
        self.scaling = config['scaling']
        scaling = self.scaling
        initial_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_pos_bounds = np.stack([initial_pos, initial_pos], axis=1)
        initial_pos_bounds[0] = np.array([-0.25 * scaling, 0.25 * scaling])
        initial_pos_bounds[1] = np.array([-0.25 * scaling, 0.25 * scaling])

        # Cocatenate velocity ranges
        self.initial_state_space = utils.BoxSpace(bounds = np.concatenate((initial_pos_bounds,np.zeros((len(initial_pos_bounds)-1, 2))),0))

        self.margin = 0.75
        min_range_train_x = -0.5 * scaling + self.margin
        max_range_train_x = 2.5 * scaling - self.margin
        min_range_train_y = -0.5 * scaling + self.margin
        max_range_train_y = 2.5 * scaling - self.margin
        if config['env'] == 'ant_maze_hard':
            max_range_train_y = 4.5 * scaling - self.margin

        self.goal_space_train = utils.BoxSpace(bounds = [[min_range_train_x, max_range_train_x], [min_range_train_y, max_range_train_y],[0.45,0.55]])
        #self.goal_space_test = utils.BoxSpace(bounds = [[min_range_train_x, max_range_train_x], [min_range_train_y, max_range_train_y],[0.45,0.55]])

        if config['env'] == 'ant_maze_hard':
            self.goal_space_test = utils.BoxSpace(bounds = [[1.95 * scaling, 2.05 * scaling], [3.95 * scaling, 4.05 * scaling], [0.45, 0.55]])
        else:
            self.goal_space_test = utils.BoxSpace(bounds = [[-0.05 * scaling, 0.05 * scaling], [(2 - 0.05) * scaling, (2 + 0.05) * scaling], [0.45, 0.55]])


        self._max_h, self._max_v = 1, 3
        self.subgoal_space = utils.BoxSpace(bounds = [[min_range_train_x, max_range_train_x],[min_range_train_y, max_range_train_y],[0, self._max_h],[-self._max_v, self._max_v],[-self._max_v, self._max_v]])

        horizontal_threshold, vertical_threshold, vel_threshold = 0.2 * scaling, 0.5, 0.8
        self.end_goal_thresholds = np.array([horizontal_threshold, horizontal_threshold, vertical_threshold])
        self.subgoal_thresholds = np.array([horizontal_threshold, horizontal_threshold, vertical_threshold, vel_threshold, vel_threshold])

        self.atomic_noise = [0.2 for i in range(8)]
        self.subgoal_noise = [0.2 for i in range(len(self.subgoal_thresholds))]

        self.max_actions = 150 * scaling
        if config['env'] == 'ant_maze_hard':
            self.max_actions = 250 * scaling

        self.timesteps_per_action = 10

        super().__init__(config)

    def project_state_to_subgoal(self, state):
        projected_state = np.concatenate((state[:3], state[self.pdim : self.pdim + 2]))
        return self.subgoal_space.clip(projected_state, dims=np.arange(2, self.subgoal_space.dim))

    def project_state_to_end_goal(self, state):
        return np.copy(state[:3])

    def project_subgoal_to_end_goal(self, subgoal):
        return np.copy(subgoal[:3])

    def subgoal_to_nearest_state(self, subgoal, state_r = None):
        if state_r is not None:
            state = np.copy(state_r)
            state[:3] = subgoal[:3]
            state[self.pdim : self.pdim+2] = subgoal[3:]
            return state
        p = np.zeros(self.pdim)
        v = np.zeros(self.vdim)
        p[:3] = subgoal[:3]
        v[:2] = subgoal[3:]
        return np.concatenate((p, v))

    def end_goal_to_subgoal(self, end_goal):
        return np.concatenate((end_goal, [0, 0]))

    '''def initialize_state(self):
        while True:
            self.reset_pos_vel()
            self.goal = self.generate_goal()
            if np.linalg.norm(self.goal[:2] - self.sim.data.qpos[:2]) > self.scaling:
                break
    '''
    def is_valid_goal(self, goal):
        s = self.config['scaling']
        margin = self.margin
        g = goal
        if self.config['env'] == 'ant_maze_hard':
            return (g[0] > -0.5 * s + margin and g[0] < 2.5 * s - margin and g[1] > -0.5 * s + margin and g[1] < 4.5 * s - margin) and \
            not (g[0] < 1.5 * s + margin and g[1] > 0.5 * s - margin and g[1] < 1.5 * s + margin) and \
            not (g[0] > 0.5 * s - margin and g[1] > 2.5 * s - margin and g[1] < 3.5 * s + margin)
        else:
            return (g[0] > -0.5 * s + margin and g[0] < 2.5 * s - margin and g[1] > -0.5 * s + margin and g[1] < 2.5 * s - margin) and \
            not (g[0] < 1.5 * s + margin and g[1] > 0.5 * s - margin and g[1] < 1.5 * s + margin)
