import tensorflow as tf
import numpy as np
from general import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import time

class Actor(Model):
    def __init__(self, agent, is_target=False):
        super().__init__()
        self.agent = agent
        self.env, self.logger, self.config = agent.env, self.agent.logger, agent.config
        self.space = self.env.action_space
        self.action_dim = self.space.dim

        self._name = 'actor'
        if is_target:
            self._name = self._name + '_target'

        self.goal_dim = self.env.goal_dim
        self.state_dim = self.env.state_dim

        lr = utils.lr_decayed(self.config['actor_lr_scheduler'])

        self.fcs = utils.create_layers(self.config['actor_model'], 'fc%s')

        self.fcs.append(Dense(self.action_dim, activation = 'tanh', name = 'last_layer', kernel_initializer=RandomUniform(-3e-3, 3e-3)))
                
        self.opt = Adam(lr)
    
    def build_model(self):
        dim = self.state_dim + self.goal_dim * self.config['n_goals']
        self.build(input_shape=(None, dim))

    @tf.function
    def last_layer(self, x):
        return tf.add(tf.multiply(x, self.space.extents), self.space.center)

    def call(self, sg):
        raise NotImplementedError
        
    def update(self, actor_batch):
        raise NotImplementedError
        
    def update_target_network(self, target, tau=None):
        if tau is None:
            tau = self.config['tau']
        weights = self.get_weights()
        target_weights = target.get_weights()
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        target.set_weights(target_weights)
