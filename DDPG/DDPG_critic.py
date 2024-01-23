from critic import Critic
import tensorflow as tf
import numpy as np
from general import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability.python.distributions import Normal

class DDPGCritic(Critic):
    def __init__(self, agent, is_target=False):
        super().__init__(agent, is_target)
        
    def call(self, sga):
        self.logger.time_stamp('FC', 3)
        x = sga
        for i in range(len(self.fcs)):
            layer = self.fcs[i]
            x = layer(x)
            if i == len(self.fcs) - 3:
                feature = x

        self.logger.time_stamp('Compute Q', 3)
        q = self.last_layer(x)
        self.logger.time_stamp('End', 3)
        return tf.squeeze(q, 1), feature

    def update(self, critic_batch):
        old_states, old_actions, rewards, next_states, goals, next_actions, dones = critic_batch
        
        self.logger.time_stamp('Get Next Q', 2)
        if self.config['use_target']:
            wanted_qs = self.agent.critic_target(tf.concat([next_states, goals, next_actions], 1))[0].numpy()
        else:
            wanted_qs = self(tf.concat([next_states, goals, next_actions], 1))[0].numpy()

        self.logger.time_stamp('Compute Wanted Q', 2)
        wanted_qs = np.where(dones, rewards, rewards + self.config['gamma'] * wanted_qs)
        wanted_qs = np.maximum(np.minimum(wanted_qs, self.q_max), self.q_min)
        
        with tf.GradientTape() as tape:
            self.logger.time_stamp('Get Q', 2)
            q, _ = self(tf.concat([old_states, goals, old_actions], 1), training=True)
            self.logger.time_stamp('Compute Loss', 2)
            loss = tf.reduce_mean(tf.square(q - wanted_qs))

        self.logger.time_stamp('Compute Grad', 2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.logger.time_stamp('Apply Grad', 2)
        self.opt.apply_gradients((grad, var) for (grad, var) in zip(grads, self.trainable_variables) if grad is not None)
        
        return loss
