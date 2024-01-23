from environment import EnvironmentRLLab
import numpy as np
from general import utils

class AntCrossEnvironment(EnvironmentRLLab):
    def __init__(self, config):
        scaling = config['scaling']
        initial_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
        initial_pos_bounds = np.stack([initial_pos, initial_pos], axis=1)
        margin = 1.0
        initial_pos_bounds[0] = np.array([-0.5*config['scaling'] + margin, 0.5*config['scaling'] - margin])
        initial_pos_bounds[1] = np.array([-0.5*config['scaling'] + margin, 1.5*config['scaling'] - margin])

        # Cocatenate velocity ranges
        self.initial_state_space = utils.BoxSpace(bounds = np.concatenate((initial_pos_bounds, np.zeros((len(initial_pos_bounds)-1, 2))), 0))

        # For additional customizations, implement "is_valid_goal" function
        min_range_x = -1.5 * scaling + margin
        max_range_x = 1.5 * scaling - margin
        min_range_y = -0.5 *scaling + margin
        max_range_y = 2.5 * scaling - margin
        self.goal_space_train = utils.BoxSpace(bounds = [[min_range_x, max_range_x], [min_range_y, max_range_y],[0.45,0.55]])
        self.goal_space_test = utils.BoxSpace(bounds = [[min_range_x, max_range_x], [min_range_y, max_range_y],[0.45,0.55]])

        size, self._max_h, self._max_v = 1.5 * scaling, 1, 3
        self.subgoal_space = utils.BoxSpace(bounds = [[min_range_x, max_range_x], [min_range_y, max_range_y],[0, self._max_h],[-self._max_v, self._max_v],[-self._max_v, self._max_v]])

        self.goal_thr = 0.5

        horizontal_threshold, vertical_threshold, vel_threshold = self.goal_thr, 0.2, 0.8
        self.end_goal_thresholds = np.array([horizontal_threshold, horizontal_threshold, vertical_threshold])
        self.subgoal_thresholds = np.array([horizontal_threshold, horizontal_threshold, vertical_threshold, vel_threshold, vel_threshold])

        self.atomic_noise = [0.2 for i in range(8)]
        self.subgoal_noise = [0.2 for i in range(len(self.subgoal_thresholds))]

        self.max_actions = 100 * scaling
        self.timesteps_per_action = 15

        super().__init__(config)

    def project_state_to_subgoal(self, state):
        projected_state = np.concatenate((state[:3], state[self.pdim : self.pdim + 2]))
        return self.subgoal_space.clip(projected_state, dims=np.arange(2, self.subgoal_space.dim))

    def project_state_to_end_goal(self, state):
        return np.copy(state[:3])

    def project_subgoal_to_end_goal(self, subgoal):
        return np.copy(subgoal[:3])

    def end_goal_to_subgoal(self, end_goal):
        return np.concatenate((end_goal, [0, 0]))

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

    def reset_sim(self, test, next_goal=None):
        self.sim.data.ctrl[:] = 0
        self.reset_pos_vel()
        scaling = self.config['scaling']
        if test:
            while True:
                self.reset_pos_vel()
                if (self.sim.data.qpos[0] > -0.5 * scaling and self.sim.data.qpos[0] < 0.5 * scaling and self.sim.data.qpos[1] > -0.5 * scaling and self.sim.data.qpos[1] < 0.5 * scaling):
                    break

        self.sim.step()
        return self.get_state()

    def is_valid_goal(self, goal):
        s = self.config['scaling']
        m = 1.0 # margin
        g = goal

        return (g[0] > -1.5 * s + m and g[0] < -0.5 * s - m and g[1] > 0.5 * s + m and g[1] < 1.5 * s - m) or\
               (g[0] > 0.5 * s + m and g[0] < 1.5 * s - m and g[1] > 0.5 * s + m and g[1] < 1.5 * s - m) or\
               (g[0] > -0.5 * s + m and g[0] < 0.5 * s - m and g[1] > 1.5 * s + m and g[1] < 2.5 * s - m)
        
