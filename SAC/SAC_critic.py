from critic import Critic
import tensorflow as tf
import numpy as np
from general import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability.python.distributions import Normal
import pdb

class SACCritic(Critic):
    def __init__(self, agent, is_target=False):
        super().__init__(agent)
        self.fcs_2 = utils.create_layers(self.config['critic_model'], 'fc%s_second')
        self.fcs_2.append(Dense(1, name = 'last_layer_second', kernel_initializer=RandomUniform(-3e-3, 3e-3)))

        self.weights_1 = None
        self.weights_2 = None
        lr = utils.lr_decayed(self.config['critic_lr_scheduler'])
        self.opt_2 = Adam(lr)
        self.count = 0
        
    def call(self, sga):
        self.logger.time_stamp('FC', 3)

        x = sga
        for i in range(len(self.fcs)):
            layer = self.fcs[i]
            x = layer(x)
        #q1 = self.last_layer(x)
        q1 = x

        x = sga
        for i in range(len(self.fcs_2)):
            layer = self.fcs_2[i]
            x = layer(x)
        #q2 = self.last_layer(x)
        q2 = x
        return tf.squeeze(q1), tf.squeeze(q2)

    def update(self, critic_batch):
        self.count += 1
        states, actions, rewards, next_states, goals, next_actions, dones = critic_batch
        #rewards = np.expand_dims(rewards, 1)
        #dones = np.expand_dims(dones, 1)

        next_actions, next_log_probs = self.agent.actor.action_logp(tf.concat([next_states, goals], 1))

        if self.config['use_target']:
            q1_target, q2_target = self.agent.critic_target(tf.concat([next_states, goals, next_actions], 1))
        else:
            q1_target, q2_target = self(tf.concat([next_states, goals, next_actions], 1))

        min_q_target = tf.minimum(q1_target, q2_target)
        soft_q_target = (min_q_target - self.agent.actor.alpha * next_log_probs).numpy()
        y = np.where(dones, rewards, rewards + self.config['gamma'] * soft_q_target)
        #y = np.maximum(np.minimum(y, 0), self.q_limit)

        with tf.GradientTape() as tape1:
            q1, q2 = self(tf.concat([states, goals, actions], 1), training=True)
            loss_1 = tf.reduce_mean(tf.square(q1 - y))

        with tf.GradientTape() as tape2: 
            _, q2 = self(tf.concat([states, goals, actions], 1), training=True)
            loss_2 = tf.reduce_mean(tf.square(q2 - y))
    
        if self.weights_1 is None:
            self.weights_1 = [var for var in self.trainable_variables if '_second' not in var.name]
        if self.weights_2 is None:
            self.weights_2 = [var for var in self.trainable_variables if '_second' in var.name]
        grads_1 = tape1.gradient(loss_1, self.weights_1)
        grads_2 = tape2.gradient(loss_2, self.weights_2)
        self.opt.apply_gradients((grad, var) for (grad, var) in zip(grads_1, self.weights_1) if grad is not None)
        self.opt_2.apply_gradients((grad, var) for (grad, var) in zip(grads_2, self.weights_2) if grad is not None)

        return loss_1 + loss_2
