from environment import Environment
import numpy as np
from general import utils

class AntFourRoomsEnvironment(Environment):
    def __init__(self, config):
        scaling = config['scaling']
        initial_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_pos_bounds = np.stack([initial_pos, initial_pos], axis=1)
        initial_pos_bounds[0] = np.array([-6, 6])
        initial_pos_bounds[1] = np.array([-6, 6])

        # Cocatenate velocity ranges
        self.initial_state_space = utils.BoxSpace(bounds = np.concatenate((initial_pos_bounds, np.zeros((len(initial_pos_bounds)-1, 2))), 0))

        # For additional customizations, implement "is_valid_goal" function
        max_range = 6
        self.goal_space_train = utils.BoxSpace(bounds = [[-max_range,max_range],[-max_range,max_range],[0.45,0.55]])
        self.goal_space_test = utils.BoxSpace(bounds = [[-max_range,max_range],[-max_range,max_range],[0.45,0.55]])

        size, self._max_h, self._max_v = 8, 1, 3
        self.subgoal_space = utils.BoxSpace(bounds = [[-size, size], [-size, size],[0, self._max_h],[-self._max_v, self._max_v],[-self._max_v, self._max_v]])

        horizontal_threshold, vertical_threshold, vel_threshold = 0.4, 0.2, 0.8
        self.end_goal_thresholds = np.array([horizontal_threshold, horizontal_threshold, vertical_threshold])
        self.subgoal_thresholds = np.array([horizontal_threshold, horizontal_threshold, vertical_threshold, vel_threshold, vel_threshold])

        self.atomic_noise = [0.1 for i in range(8)]
        self.subgoal_noise = [0.1 for i in range(len(self.subgoal_thresholds))]

        self.max_actions = 500
        self.timesteps_per_action = 15

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

    def reset(self):
        self.goal = self.generate_goal()
        self.sim.data.ctrl[:] = 0
        goal_room = 0 if self.goal[1] >= 0 else 2
        if self.goal[0] * self.goal[1] <= 0:
            goal_room += 1

        # Place ant in room different than room containing goal
        initial_room = goal_room
        while initial_room == goal_room:
            if self.config['env'] == 'ant_four_rooms_half':
                if not self.is_testing:
                    initial_room = np.random.choice([2, 3])
                else:
                    initial_room = (goal_room + 2) % 4
            else:
                initial_room = np.random.randint(0,4)
                
        self.reset_pos_vel()

        # Move ant to correct room
        self.sim.data.qpos[0] = np.random.uniform(3,6.5)
        self.sim.data.qpos[1] = np.random.uniform(3,6.5)

        if initial_room == 1 or initial_room == 2:
            self.sim.data.qpos[0] *= -1
        elif initial_room == 2 or initial_room == 3:
            self.sim.data.qpos[1] *= -1

        #if self.visualize:
        #    self.display_end_goal()

        self.sim.step()
        return self.state

    def generate_goal(self):
        goal_space = self.goal_space_test if self.is_testing else self.goal_space_train
        end_goal = goal_space.random_sample()

        if self.config['env'] == "ant_four_rooms_half":
            room_num = np.random.randint(0,2)
        else:
            room_num = np.random.randint(0,4)

        end_goal[0] = np.random.uniform(3,6.5)
        end_goal[1] = np.random.uniform(3,6.5)
        end_goal[2] = np.random.uniform(0.45,0.55)

        if room_num == 1 or room_num == 2:
            end_goal[0] *= -1
        if room_num == 2 or room_num == 3:
            end_goal[1] *= -1

        # Visualize End Goal
        #self.display_end_goal()
        return end_goal.astype(float)

