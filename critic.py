import tensorflow as tf
import numpy as np
from general import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import time

class Critic(Model):
    def __init__(self, agent, is_target=False):
        super().__init__()
        self.agent = agent
        self.env, self.logger, self.config = agent.env, self.agent.logger, agent.config
        
        self._name = 'critic'
        if is_target:
            self._name = self._name + '_target'

        self.goal_dim = self.env.goal_dim
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        lr = utils.lr_decayed(self.config['critic_lr_scheduler'])

        self.q_min = self.config['living_reward'] * self.env.timesteps_per_action * self.env.max_actions - self.config['timeout_penalty']
        self.q_max = 25
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_min/self.q_init - 1)

        self.fcs = utils.create_layers(self.config['critic_model'], 'fc%s')

        self.fcs.append(Dense(1, name = 'last_layer', kernel_initializer=RandomUniform(-3e-3, 3e-3)))

        self.opt = Adam(lr)

    def build_model(self):
        dim = self.state_dim + self.goal_dim * self.config['n_goals'] + self.action_dim
        self.build(input_shape=(None, dim))

    @tf.function
    def last_layer(self, x):
        if self.config['unbound_q']:
            return tf.add(x, self.q_init)
        else:
            return tf.sigmoid(tf.add(x, self.q_offset)) * (self.q_max - self.q_min) + self.q_min

    def call(self, sga):
        raise NotImplementedError

    def update(self, critic_batch):
        raise NotImplementedError
            
    def update_target_network(self, target, tau=None):
        if tau is None:
            tau = self.config['tau']
        weights = self.get_weights()
        target_weights = target.get_weights()
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        target.set_weights(target_weights)
