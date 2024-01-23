from environment import Environment
import numpy as np
from general import utils

class UR5Environment(Environment):
    def __init__(self, config):
        self.config = config
        initial_pos = np.array([[5.96625837e-03], [3.22757851e-03], [-1.27944547e-01]])
        initial_pos_bounds = np.concatenate((initial_pos, initial_pos), 1)
        initial_pos_bounds[0] = np.array([-np.pi/8, np.pi/8])
        self.initial_state_space = utils.BoxSpace(bounds = np.concatenate((initial_pos_bounds, np.zeros((len(initial_pos_bounds), 2))), 0))

        self.goal_space_train = utils.BoxSpace(bounds = [[-np.pi, np.pi], [-np.pi/4, 0], [-np.pi/4, np.pi/4]])
        self.goal_space_test = utils.BoxSpace(bounds = [[-np.pi, np.pi], [-np.pi/4, 0], [-np.pi/4, np.pi/4]])

        self.subgoal_space = utils.BoxSpace(bounds = np.array([[-2*np.pi,2*np.pi],[-2*np.pi,2*np.pi],[-2*np.pi,2*np.pi],[-4,4],[-4,4],[-4,4]]))

        self.atomic_noise = [0.1, 0.1, 0.1]

        self.max_actions = 500 if self.config['max_actions'] is None else self.config['max_actions']
        self.timesteps_per_action = 15 if self.config['timesteps_per_action'] is None else self.config['timesteps_per_action']
        self.goal_thr = 0.02 if self.config['goal_thr'] is None else self.config['goal_thr']
        
        super().__init__(config)

    # Supplemental function that converts angle to between [-pi,pi]
    def _bound_angle(self, angle):
        bounded_angle = angle % (2*np.pi)

        if np.absolute(bounded_angle) > np.pi:
            bounded_angle = bounded_angle % np.pi - np.pi

        return bounded_angle

    def dist(self, state, goal):
        state_ = self.project_state_to_end_goal(state)
        def dist_(a, b):
            return np.linalg.norm([np.cos(a) - np.cos(b), np.sin(a) - np.sin(b)]) / 2
        p = 0
        for i in range(len(state_)):
            p += dist_(state_[i], goal[i]) / len(state_)
        return p

    def potential(self, state):
        return 1 - self.dist(state, self.goal)

    def check_goal(self):
        d = self.dist(self.state, self.goal)
        return d <= self.goal_thr

    def display_goals(self):
        poses = self.get_ur5_poses(self.goal[:3])
        for i in range(3):
            self.sim.data.mocap_pos[i] = poses[i]

    def project_state_to_end_goal(self, state):
        return np.array([self._bound_angle(state[i]) for i in range(3)])

    def is_valid_goal(self, end_goal):
        poses = self.get_ur5_poses(end_goal[:3])
        return np.absolute(end_goal[0]) > np.pi/4 and poses[1][2] > 0.05 and poses[2][2] > 0.15

    def subgoal_to_nearest_state(self, state, subgoal):
        result = []
        for i in range(3):
            if (subgoal[i] > 0 and state[i] < 0) or (subgoal[i] < 0 and state[i] > 0):
                result.append(subgoal[i])
                continue
            rest = state[i] % (2 * np.pi)
            q = state[i] // (2 * np.pi)
            if rest - subgoal[i] <= -np.pi:
                if q == 0:
                    result.append(subgoal[i])
                else:
                    result.append((q-1) * (2*np.pi) + subgoal[i])
            elif rest - subgoal[i] >= np.pi:
                if q == -1:
                    result.append(subgoal[i])
                else:
                    result.append((q+1) * (2*np.pi) + subgoal[i])
            else:
                result.append(q * (2*np.pi) + subgoal[i])

        for i in range(3, 6):
            if np.abs(subgoal[i]) <= 4:
                result.append(subgoal[i])
            else:
                result.append(state[i])
        return result

    def get_ur5_poses(self, joints):
        # B : Base, S : Shoulder, U : Upper arm, F : Forearm, Wn : nth Wrist
        # L_X_Y : Transformation matrix from X to Y reference frame
        # x_Y : coordinates of x in Y reference frame
        
        # s_S = np.array([0,0,0,1])
        u_U = np.array([0,0.13585,0,1])
        f_F = np.array([0.425,0,0,1])
        w1_W1 = np.array([0.39225,-0.1197,0,1])

        L_S_B = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])
        L_U_S = np.array([[np.cos(joints[0]), -np.sin(joints[0]), 0, 0],[np.sin(joints[0]), np.cos(joints[0]), 0, 0],[0,0,1,0],[0,0,0,1]])
        L_F_U = np.array([[np.cos(joints[1]),0,np.sin(joints[1]),0],[0,1,0,0.13585],[-np.sin(joints[1]),0,np.cos(joints[1]),0],[0,0,0,1]])
        L_W1_F = np.array([[np.cos(joints[2]),0,np.sin(joints[2]),0.425],[0,1,0,0],[-np.sin(joints[2]),0,np.cos(joints[2]),0],[0,0,0,1]])

        # s_B = L_S_B.dot(s_S)
        u_B = L_S_B.dot(L_U_S).dot(u_U)[:3]
        f_B = L_S_B.dot(L_U_S).dot(L_F_U).dot(f_F)[:3]
        w1_B = L_S_B.dot(L_U_S).dot(L_F_U).dot(L_W1_F).dot(w1_W1)[:3]

        return np.array([u_B, f_B, w1_B])

