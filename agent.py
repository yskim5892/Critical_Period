import tensorflow as tf
import numpy as np
from general import utils
from general.stats import Stats
from replay_buffer import ReplayBuffer
from SAC import SACActor, SACCritic
from DDPG import DDPGActor, DDPGCritic
from PPO import PPOActor, PPOCritic
from actor import Actor
from critic import Critic
import os
import json
import time
import pdb

class Agent():
    def __init__(self, config, env):
        self.config = config
        self.env = env

        save_path = os.path.join(config['dir_name'], config['config_id'], str(config['run_id']))
        self.model_dir = os.path.join(save_path, 'models')

        utils.create_muldir(save_path, self.model_dir)

        self.logger = utils.Logger(save_path, enable_time_stamp=config['enable_time_stamp'])
        config_json = config.to_json()
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            f.write(config_json)
        self.logger.log('Config : \n' + config_json)
        self.logger.log(f'Save Path : {save_path}')

        for gpu_instance in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu_instance, True)

        self.buffer_size = min(self.config['episodes_to_store'] * self.env.max_actions, self.config['buffer_size_ceiling'])

        if self.config['algorithm'] == 'PPO':
            self.replay_buffer = ReplayBuffer(self.buffer_size, self.config['batch_size'], 9)
            self.transition_sample = ReplayBuffer(128, self.config['batch_size'], 9)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size, self.config['batch_size'], 6)
            self.transition_sample = ReplayBuffer(128, self.config['batch_size'], 6)
        
        if self.config['algorithm'] == 'SAC':
            actor_class = SACActor
            critic_class = SACCritic
        elif self.config['algorithm'] == 'DDPG':
            actor_class = DDPGActor
            critic_class = DDPGCritic
        else:
            actor_class = PPOActor
            critic_class = PPOCritic

        self.actor = actor_class(self)
        self.critic = critic_class(self)
        self.actor.build_model()
        self.critic.build_model()
        if self.config['use_target']:
            self.actor_target = actor_class(self, is_target=True)
            self.critic_target = critic_class(self, is_target=True)
            self.actor_target.build_model()
            self.critic_target.build_model()
            self.actor.update_target_network(self.actor_target, tau=1.)
            self.critic.update_target_network(self.critic_target, tau=1.)
        self.maxed_out = False
        self.stats = Stats()


        self.normalized_grads_sum = []
        self.normalized_grads_sq_sum = []
        self.normalized_grads_num = 0
        self.grads = []

        self.initialize()
        if config['load'] or config['load_nth_episode'] != -1:
            try:
                if config['load_nth_episode'] != -1:
                    self.restore(episode=config['load_nth_episode'])
                else:
                    self.restore()
            except ValueError:
                self.logger.log((f'Failed to load model from {self.model_dir}', 'red'))

    def initialize(self):
        self.total_train_episodes = tf.Variable(0, trainable=False, name='total_train_episodes')
        self.total_env_steps = tf.Variable(0, trainable=False, name='total_env_steps')
        self.train_steps = tf.Variable(0, trainable=False, name='train_steps')

        self.ckpt = tf.train.Checkpoint(total_train_episodes = self.total_train_episodes, total_env_steps = self.total_env_steps, train_steps = self.train_steps)

    
    def save(self, model_dir=None, save_by_episodes=False):
        if model_dir == None:
            model_dir = self.model_dir
        if save_by_episodes:
            model_dir = os.path.join(model_dir, 'epi_%d'%self.total_train_episodes.numpy())

        self.ckpt.save(os.path.join(model_dir, 'meta/ckpt'))
        self.actor.save_weights(os.path.join(model_dir, 'actor/ckpt'))
        self.critic.save_weights(os.path.join(model_dir, 'critic/ckpt'))
        self.logger.log('Model saved at %s'%model_dir)

    def restore(self, model_dir=None, episode=None):
        if model_dir == None:
            model_dir = self.model_dir
        if episode is not None:
            model_dir = model_dir + '/epi_%d'%episode

        self.ckpt.restore(tf.train.latest_checkpoint(os.path.join(model_dir, 'meta')))
        self.actor.load_weights(os.path.join(model_dir, 'actor/ckpt'))
        self.critic.load_weights(os.path.join(model_dir, 'critic/ckpt'))
        self.logger.log('Model loaded from %s'%model_dir)

    def reset_stats(self):
        for name in ['num_episodes', 'num_steps', 'num_successes', 'success_rate', 'actor_lr', 'critic_lr']:
            self.stats[name] = 0
        self.stats.register_var('q', ['max', 'min', 'avg'])
        self.stats.register_var('actor_loss', ['avg'])
        self.stats.register_var('critic_loss', ['avg'])
        self.stats.register_var('reward', ['max', 'min', 'avg'])
        
        if self.config['algorithm'] == 'SAC':
            self.stats.register_var('alpha', ['avg'])
            self.stats.register_var('log_prob', ['avg'])
            self.stats.register_var('action_std', ['avg'])

        elif self.config['algorithm'] == 'PPO':
            self.stats.register_var('ent_loss', ['avg'])
            self.stats.register_var('action_std', ['avg'])

        if self.config['record_sharpness']:
            self.stats.register_var('grad_var', ['avg'])
            self.stats.register_var('normalized_grad_var', ['avg'])
            
            self.stats.register_var('hessian_norm', ['avg'])
            self.stats.register_var('hessian_frobenius_norm', ['avg'])
            self.stats.register_var('SAM_sharpness', ['avg'])

    def update_stats(self):
        self.stats["success_rate"] = self.stats["num_successes"] / self.stats["num_episodes"]

    def record_stats(self, epoch, names):
        if self.config['no_record']:
            return
        prefix = 'test' if self.is_testing else 'train'
        self.stats.record_stats(names, f'{prefix}_', total_train_episodes = self.total_train_episodes.numpy(), total_env_steps = self.total_env_steps.numpy(), epoch = epoch)

    def learn(self):
        if self.total_train_episodes <= self.config['warmup_episodes']:
            return 
        if self.replay_buffer.size < self.config['warmup_size']:
            return
        self.logger.reset_time_stamp()
        algorithm = self.config['algorithm']
        for i in range(self.config['num_updates']):
            self.logger.time_stamp('Update %s'%i, 0)
            self.logger.time_stamp('Get Batch', 1)

            if algorithm == 'PPO':
                states, actions, _, goals, _, advs, returns, logps, _ = self.replay_buffer.get_batch(full=True)
            else:
                states, actions, rewards, next_states, goals, dones = self.replay_buffer.get_batch()

            self.logger.time_stamp('Get Next Actions', 1)
            
            if algorithm == 'SAC' or algorithm == 'PPO':
                next_actions = None
            else:
                next_actions = self.infer_actor(next_states, goals, use_target=self.config['use_target'])

            self.logger.time_stamp('Update Critic', 1)

            if algorithm == 'PPO':
                critic_batch = (states, goals, returns)
                critic_loss = self.critic.update(critic_batch)
            else:
                critic_batch = (states, actions, rewards, next_states, goals, next_actions, dones)
                critic_loss = self.critic.update(critic_batch)

            self.logger.time_stamp('Update Actor', 1)

            if algorithm == 'PPO':
                actor_batch = (states, goals, logps, advs, actions)
                actor_loss, grads = self.actor.update(actor_batch)
            else:
                actor_batch = (states, goals)
                actor_loss, grads = self.actor.update(actor_batch)
                #print('---------------------------')
                #print(grads)
           

            '''if self.config['record_sharpness']:
                grads_unraveled = np.concatenate([np.reshape(g, [-1]) for g in grads if g is not None], -1)
                self.grads.append(grads_unraveled)
                normalized_grads = [g / (np.linalg.norm(g) + 1e-8) for g in grads if g is not None]
                self.normalized_grads_unraveled =  np.concatenate([np.reshape(g, [-1]) for g in normalized_grads], -1)
                if self.normalized_grads_num == 0:
                    self.normalized_grads_sum = self.normalized_grads_unraveled
                    self.normalized_grads_sq_sum = np.square(self.normalized_grads_unraveled)
                else:
                    self.normalized_grads_sum += self.normalized_grads_unraveled
                    self.normalized_grads_sq_sum += np.square(self.normalized_grads_unraveled)
                self.normalized_grads_num += 1

                if i == self.config['num_updates'] - 1:
                    self.grads = np.array(self.grads)
                    grads_var = np.mean(np.var(self.grads, 0))
                    self.stats.record_var('grad_var', grads_var)
                    self.grads = []
                    
                    normalized_grads_var = np.mean(self.normalized_grads_sq_sum / self.normalized_grads_num - np.square(self.normalized_grads_sum / self.normalized_grads_num))
                    self.stats.record_var('normalized_grad_var', normalized_grads_var)'''

            self.logger.time_stamp('Update Targets', 1)
            if self.config['use_target']:
                self.critic.update_target_network(self.critic_target)
                self.actor.update_target_network(self.actor_target)

            self.logger.time_stamp('Record Stats', 1)
            self.stats.record_var('critic_loss', critic_loss.numpy())
            self.stats.record_var('actor_loss', actor_loss.numpy())
            self.logger.time_stamp('End', 1)

            self.train_steps = self.train_steps + 1

    def run(self):
        self.env.is_testing = self.is_testing
        self.state = self.env.reset()

        self.steps = 0
        self.goals = np.reshape(self.env.goals, [-1])
        total_reward = 0
        num_transition_sampled = 0

        while True:
            prev_state = self.state
            if (self.config['algorithm'] != 'PPO' and not self.is_testing and np.random.random_sample() < self.config['random_action_p']):
                action = self.env.get_random_action()

            else:
                if self.config['algorithm'] == 'PPO' or self.config['algorithm'] == 'SAC':
                    actions, logps = self.infer_actor([self.state], [self.goals])
                    action = actions[0].numpy()
                    logp = logps[0].numpy()
                else:
                    action = self.infer_actor([self.state], [self.goals])[0].numpy()
            if self.config['algorithm'] != 'PPO' and not self.is_testing:
                action = self.env.add_noise(action)

            self.state, reward, done, info = self.env.step(action)

            if self.config['algorithm'] == 'PPO':
                value = self.infer_critic([prev_state], [self.goals], None)[0].numpy()
                transition = [prev_state, action, reward, self.goals, value, None, None, logp, done]
            else:
                transition = [prev_state, action, reward, self.state, self.goals, done] 
            self.replay_buffer.add(transition)
            if np.random.rand() < 0.1 and num_transition_sampled <= 10 and not self.transition_sample.full():
                self.transition_sample.add(transition)
                num_transition_sampled += 1

            total_reward += reward

            if done:
                success = info['success'] if 'success' in info else False
                if success:
                    self.stats['num_successes'] += 1
                self.stats['num_steps'] += self.steps
                self.stats['num_episodes'] += 1
                self.stats.record_var('reward', total_reward)

                if self.config['HER']:
                    hindsight_trajectory = self.make_hindsight_trajectory(self.replay_buffer.data)
                if self.config['algorithm'] == 'PPO':
                    self.replay_buffer.data = self.finalize_trajectory(self.replay_buffer.data)
                    if self.config['HER']:
                        hindsight_trajectory = self.finalize_trajectory(hindsight_trajectory)
                    #pdb.set_trace()
                if self.config['HER']:
                    self.replay_buffer.data = np.concatenate([self.replay_buffer.data, hindsight_trajectory], 0)
                break

        if not self.is_testing:
            self.total_train_episodes = self.total_train_episodes + 1
            self.learn()

        if self.config['algorithm'] == 'PPO':
            self.replay_buffer.empty()

        return success
    
    def finalize_trajectory(self, traj):
        traj = np.copy(traj)
        gamma = self.config['gamma']
        lam = self.config['gae_lambda']

        gae = 0
        ret = 0
        advs = np.zeros([len(traj)])
        for t in reversed(range(len(traj))):
            state, action, reward, goal, value, _, _, logp, done = traj[t]
            last_traj = traj[min(t+1, len(traj) - 1)]
            next_value = last_traj[4]

            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * lam * (1 - done) * gae
            # traj[t][6] = gae + value # return = adv + value

            ret = reward + gamma * (1 - done) * ret
            traj[t][6] = ret

            advs[t] = gae

        advs = (advs - advs.mean()) / (advs.std() + 1e-10) # Normalize Adv
        for t in range(len(traj)):
            traj[t][5] = advs[t]
            #print(traj[t])
            #print('')
        return traj

    def make_hindsight_trajectory(self, traj):
        # Hindsight Trajectory
        traj = np.copy(traj)
        env = self.env
        goals = np.reshape(env.goals, [-1])

        goal = env.project_state_to_end_goal(traj[-1][0]) # Hindsight goal
        goals[:len(goal)] = goal

        dense_rwd_config = self.config['dense_reward'][env.dense_level]
        dense_penalty_config = self.config['dense_penalty'][env.dense_level]
        max_goal_level = len(dense_rwd_config) - 1
        max_fake_goal_level = len(dense_penalty_config) - 1
        goal_reached_level = np.ones([len(goals)//len(goal)]).astype(int) * -1
        for t in range(len(traj)):
            state, action, _, _, _, _, _, logp, done = traj[t]
            value = self.infer_critic([state], [goals], None)[0].numpy()
            traj[t][4] = value
            traj[t][3] = goals

            approached_goal = -1
            state_g = env.project_state_to_end_goal(state)
            reward = self.config['living_reward'] * env.timesteps_per_action
            for k in range(len(dense_rwd_config)):
                if np.all(np.absolute(goal - state_g) <= dense_rwd_config[k][0])\
                        and goal_reached_level[0] < k:
                    reward += dense_rwd_config[k][1]
                    goal_reached_level[0] = k
                    break
            to_break = False
            for k in range(len(dense_penalty_config)):
                if to_break:
                    break
                for i in range(len(env.fake_goals)):
                    if np.all(np.absolute(env.fake_goals[i] - state) <= dense_penalty_config[k][0])\
                            and goal_reached_level[i+1] < k:
                        reward += (-1.0) * dense_penalty_config[k][1]
                        goal_reached_level[i+1] = k
                        to_break = True
                        break

            traj[t][2] = reward
        return traj

    def infer_actor(self, states, goals, use_target = False):
        actor = self.actor_target if use_target else self.actor
        if self.config['algorithm'] == 'PPO' or self.config['algorithm'] == 'SAC':
            actions, logps = self.actor.action_logp(tf.concat([states, goals], 1))
            return actions, logps
        else:
            actions = actor(tf.concat([states, goals], 1))[0]
            return actions

    def infer_critic(self, states, goals, actions, use_target = False):
        if self.config['algorithm'] == 'PPO':
            vs, _ = self.critic(tf.concat([states, goals], 1))
            return vs
        else:
            critic = self.critic_target if use_target else self.critic
            qs, _ = critic(tf.concat([states, goals, actions], 1))
            return qs
    
    def s2g(self, state): # state to endgoal space
        return self.env.project_state_to_goal(state)

    def project_states_to_goals(self, states):
        return np.array([self.s2g(state) for state in states])
      
