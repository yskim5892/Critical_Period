from general import utils
from general.configs import Configs
import config_presets
import json, sys, os
import numpy as np
import glob
import matplotlib.pyplot as plt
from environments import get_env
from agent import MetaAgent
from MCTS import MCTSNode
from main import create_dir, train, test
import cv2

import matplotlib
matplotlib.use('tkagg')

class Visualizer():
    def __init__(self, meta_agent, img_size, d, center, ax1, ax2):
        self.meta_agent, self.env, self.config = meta_agent, meta_agent.env, meta_agent.config
        self.img_size, self.d, self.ax1, self.ax2 = img_size, d, ax1, ax2
        self.center = center
        
        self.imgs = [[] for _ in range(self.config['levels'])]

    def visualize(self):
        plt.rcParams['figure.figsize'] = (18, 9)

        self.draw_plans()
        utils.create_muldir('imgs', os.path.join('imgs', self.config['config_id']))
        for i in range(1, config['levels']):
            for j in range(len(self.imgs[i])):
                img = self.imgs[i][j].img
                print(img.shape)
                plt.imshow(img)

                img = cv2.cvtColor(np.float32(img * 255), cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join('imgs', self.config['config_id'], f'{i}_{j}.jpg'), img)
                plt.show()

    def draw_plans(self):
        initial_state = meta_agent.current_state
        goals = meta_agent.goal_by_level

        print('Initial State :', initial_state)
        print('Goals :', goals)

        for level in range(1, config['levels']):
            print('Level : %d'%level)
            agent = meta_agent.agents[level]
            goal = goals[level]
            root = MCTSNode(agent, agent.s2sg(initial_state), goal, 0, agent.state_with_timestep(initial_state, 0))

            best_action = root.best_action(500)

            current_node = root
            while not current_node._is_terminal_node():
                print('\nStep ', current_node.timestep)
                action_idx = current_node._rollout_policy()
                img = Image(self, self.center)
                img.draw_achievability_map(meta_agent.agents[level - 1], agent.change_timestep(current_node.real_state, 0))
                
                new_goals = np.copy(goals)
                new_goals[:level - 1] = None
                new_goals[level - 1] = current_node.actions[action_idx]
                img.visualize_node(current_node, new_goals)
                self.imgs[level].append(img)

                if action_idx not in current_node.children:
                    break
                else:
                    current_node = current_node.children[action_idx]
                                
            current_state = current_node.projected_state
            t = 0
            reward = 0
            time_limit = agent.time_limit
            while not current_node._check_goal(current_state) and current_node.timestep + t < time_limit:
                state_ = agent.sg2s(current_state, current_node.real_state, current_node.timestep + t)
                print('State :', state_)
                goal = env.end_goal_to_subgoal(current_node.goal) if agent.is_top else current_node.goal

                if config['AE_confident_action'] and agent.lower_agent.infer_AE([state_], [goal]).numpy() >= 1 - config['confidence_threshold']:
                    subgoal = goal
                else:
                    subgoal = agent.infer_actor([state_], [current_node.goal])[0]

                print('\nStep ', current_node.timestep + t)
                img = Image(self, self.center)
                img.draw_achievability_map(meta_agent.agents[level - 1], state_)
                new_goals = np.copy(goals)
                new_goals[:level] = None
                img.visualize_state_goals_actions(current_state, new_goals, [subgoal])
                self.imgs[level].append(img)
                new_state, goal_achieved, _ = current_node._predict_next_state_from_AE(state_, subgoal, current_node.timestep + t)

                t += 1
                if new_state is current_state:
                    print('Failed')
                    break
                current_state = new_state

                if goal_achieved:
                    current_state = subgoal
                    print('Goal Achieved ')
                    img = Image(self, self.center)
                    img.draw_achievability_map(meta_agent.agents[level - 1], agent.sg2s(current_state, current_node.real_state, current_node.timestep + t))
                    new_goals = np.copy(goals)
                    new_goals[:level] = None
                    img.visualize_state_goals_actions(current_state, new_goals, [])
                    self.imgs[level].append(img)
                    break

class Image():
    def __init__(self, visualizer, center):
        self.vis, self.img_size, self.d, self.ax1, self.ax2 = visualizer, visualizer.img_size, visualizer.d, visualizer.ax1, visualizer.ax2
        self.img = np.zeros([self.img_size, self.img_size, 3], dtype=float)
        self.center = center

    def pos_to_pixel(self, pos):
        result = [int(((pos[0] - self.center[0]) / self.d + 0.5) * self.img_size), int(((pos[1] - self.center[1]) / self.d + 0.5) * self.img_size)]
        if result[0] < 0 or result[0] >= self.img_size or result[1] < 0 or result[1] >= self.img_size:
            return None
        else:
            return result

    def color_pos(self, pos, color, pixel_size=1):
        pixel = self.pos_to_pixel(pos)
        if pixel is not None:
            start, end = - int((pixel_size - 1) / 2), int(pixel_size / 2) + 1
            for i in range(start, end):
                for j in range(start, end):
                    self.img[pixel[0] + i][pixel[1] + j] = color
        return pixel

    def draw_achievability_map(self, agent, state):
        center_subgoal = agent.env.project_state_to_subgoal(np.concatenate((self.center, state[2:])))
        for i in range(self.img_size):
            for j in range(self.img_size):
                goal = np.copy(center_subgoal)
                goal[self.ax1] += self.d * (i / self.img_size - 0.5)
                goal[self.ax2] += self.d * (j / self.img_size - 0.5)

                r = agent.infer_AE([state], [goal])[0]
                self.img[i][j] = [r, r, 0]

    def visualize_state_goals_actions(self, state, goals, actions):
        self.color_pos(state, [1, 1, 1], 3)
        print(f"State : {state}")
        for goal_level in range(len(goals)):
            if goals[goal_level] is None:
                continue
            goal_pixel_pos = self.color_pos(goals[goal_level], self.vis.env.subgoal_colors[goal_level], 3)
            print(f"Goal[{goal_level}] : {goals[goal_level]}")

        for idx in range(len(actions)):
            self.color_pos(actions[idx], [0, 1, 1])
            print(f"Action : {actions[idx]}")

    def visualize_node(self, node, goals):
        self.color_pos(node.projected_state, [1, 1, 1], 3)
        print(f"State : {node.projected_state}")
        for goal_level in range(len(goals)):
            if goals[goal_level] is None:
                continue
            goal_pixel_pos = self.color_pos(goals[goal_level], self.vis.env.subgoal_colors[goal_level], 3)
            print(f"Goal[{goal_level}] : {goals[goal_level]}")

        T = float(self.vis.config['time_scale'])
        for idx in range(node.b):
            r = 0 if node.N[idx] == 0 else np.clip((node.Q[idx] + T) / T, 0, 1)
            self.color_pos(node.actions[idx], [0, r, 1])
            print(f"Action : {node.actions[idx]} Q : {r} P : {node.P[idx]} N : {node.N[idx]}")

if __name__ == '__main__':
    config = Configs.load(preset_file_path='config_presets/default.json', argv=sys.argv)
    config_presets.postprocess_config(config)

    create_dir(config)
    utils.select_gpu(1)
    env = get_env(config)

    #config['config_id'] = '0.0015_True_8_0.75_0.5_25_0.5_False_False_True_0.6'
    print(config['config_id'])

    meta_agent = MetaAgent(config, env)
    meta_agent.is_testing = True
    env.is_testing = True
    meta_agent.current_state = env.reset()
    meta_agent.steps = 0

    agents = meta_agent.agents
    meta_agent.goal_by_level[2] = env.goal
    
    agents[2].current_state = agents[2].state_with_timestep(meta_agent.current_state, 0)
    agents[2].goal = meta_agent.goal_by_level[2]
    agents[2].steps = 0
    meta_agent.agents[1].AE_unseen_accuracy = 1
    meta_agent.goal_by_level[1], _ = agents[2].choose_action(True)
    
    agents[1].current_state = agents[1].state_with_timestep(meta_agent.current_state, 0)
    agents[1].goal = meta_agent.goal_by_level[1]
    agents[1].steps = 0
    meta_agent.agents[0].AE_unseen_accuracy = 1
    meta_agent.goal_by_level[0], _ = agents[1].choose_action(True)

    center = [0., 0.]
    if config['env'] == 'ant_maze':
        center = [config['scaling'], config['scaling']]
    elif config['env'] == 'ant_maze_hard':
        center = [config['scaling'], 2 * config['scaling']]
    vis = Visualizer(meta_agent, 64, 18, center, 0, 1)
    vis.visualize()
    '''plt.rcParams['figure.figsize'] = (18, 9)
    fig, axes = plt.subplots(config['levels'] - 1, config['time_scale'])

    fig.suptitle('Env : %s, MCTS node visualized'%config['env'])
    fig.tight_layout()

    size, d = 128, 32
    maps = achievability_map(None, meta_agent, meta_agent.current_state, size, d, 0, 1)
    imgs = plan_map(maps, meta_agent, meta_agent.current_state, meta_agent.goal_by_level, size, d, 0, 1)
    
    for i in range(config['levels'] - 1):
        for j in range(config['time_scale']):
            ax = axes[i, j]
            ax.imshow(imgs[i, j])
            ax.set_title('Level %d time %d'%(i + 1, j), fontdict={'color' : 'r'})
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()'''
