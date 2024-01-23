from critic import Critic
import tensorflow as tf
import numpy as np
from general import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability.python.distributions import Normal

class PPOCritic(Critic):
    def __init__(self, agent, is_target=False):
        super().__init__(agent, is_target)
        self.clip_grad = self.config['PPO_clip_grad']
        
    def call(self, sg):
        self.logger.time_stamp('FC', 3)
        x = sg
        for i in range(len(self.fcs)):
            layer = self.fcs[i]
            x = layer(x)
            if i == len(self.fcs) - 3:
                feature = x

        self.logger.time_stamp('Compute Q', 3)
        #v = self.last_layer(x)
        self.logger.time_stamp('End', 3)
        return tf.squeeze(x, 1), feature

    def build_model(self):
        dim = self.state_dim + self.goal_dim * self.config['n_goals']
        self.build(input_shape=(None, dim))

    def update(self, critic_batch):
        states, goals, returns = critic_batch
        
        with tf.GradientTape() as tape:
            values, _ = self(tf.concat([states, goals], 1), training=True)
            loss = tf.reduce_mean(tf.square(returns - values))

        self.logger.time_stamp('Compute Grad', 2)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_grad)

        self.logger.time_stamp('Apply Grad', 2)
        self.opt.apply_gradients((grad, var) for (grad, var) in zip(grads, self.trainable_variables) if grad is not None)
        
        return loss
