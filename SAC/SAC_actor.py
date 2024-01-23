from actor import Actor
import tensorflow as tf
import numpy as np
from general import utils
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability.python.distributions import Normal, MultivariateNormalDiag
import pdb

class SACActor(Actor):
    def __init__(self, agent, is_target=False):
        super().__init__(agent, is_target)
        self.mu = Dense(self.action_dim, name='mu', kernel_initializer=RandomUniform(-3e-3, 3e-3), activation='tanh')
        self.std = Dense(self.action_dim, name='log_std', activation='softplus')
        #self.alpha = self.config['SAC_alpha']
        self.log_alpha = tf.Variable(np.log(self.config['SAC_alpha']), name='log_alpha', dtype=tf.float32)

        self.alpha_opt = Adam(self.config['SAC_alpha_lr'])
        self.count = 0
        
    def call(self, sg):
        self.logger.time_stamp('FC', 3)

        x = sg
        for i in range(len(self.fcs)):
            layer = self.fcs[i]
            x = layer(x)
            if i == len(self.fcs) - 3:
                feature = x

        mu = self.mu(x)
        mu = self.last_layer(mu)
        std = self.std(x)
        std = tf.clip_by_value(std, 1e-2, 1.0)
        
        #std = tf.exp(log_std)
        #normal = Normal(mu, std)
        return mu, std

    def action_logp(self, sg):
        mu, std = self.call(sg)
        
        normal = Normal(loc=mu, scale=std)
        action = normal.sample()
        action = tf.clip_by_value(action, self.space.bounds[:, 0], self.space.bounds[:, 1])

        self.agent.stats.record_var('action_std', tf.reduce_mean(std).numpy())

        '''x = normal.sample()
        y = tf.tanh(x)
        action = self.last_layer(y)
        log_prob = normal.log_prob(x)
        log_prob = log_prob - tf.reduce_sum(tf.math.log(1 - y**2 + 1e-16), axis=1)'''
        #log_prob -= tf.math.log(1 - y**2 + 1e-6)
        #log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)

        log_prob = normal.log_prob(action)
        log_prob = tf.reduce_sum(log_prob, 1)
        return action, log_prob 

    @property
    def alpha(self):
        #return self.config['SAC_alpha']
        return tf.exp(self.log_alpha)

    def compute_loss(self, actor_batch):
        states, goals = actor_batch
        actions, log_probs = self.action_logp(tf.concat([states, goals], 1))

        q1, q2 = self.agent.critic(tf.concat([states, goals, actions], 1))
        min_q = tf.minimum(q1, q2)
        loss = - tf.reduce_mean(min_q - tf.stop_gradient(self.alpha) * log_probs)
        self.agent.stats.record_var('log_prob', tf.reduce_mean(log_probs).numpy()) 
        return loss


    def update(self, actor_batch):
        self.count += 1
        states, goals = actor_batch
        with tf.GradientTape() as tape:
            loss = self.compute_loss(actor_batch)

        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients((grad, var) for (grad, var) in zip(grads, self.trainable_variables) if grad is not None)

        with tf.GradientTape() as tape:
            actions, log_probs = self.action_logp(tf.concat([states, goals], 1))
            alpha_loss = tf.reduce_mean(- self.alpha * (log_probs - self.action_dim))

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_opt.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        self.agent.stats.record_var('alpha', self.alpha.numpy())

        '''if self.config['record_sharpness'] and self.count % 10000 == 0:
            self.record_sharpness(actor_batch)'''

        return loss, grads
    
    def record_sharpness(self, transition_sample):
        states, _, _, _, goals, _ = transition_sample.get_batch(full=True)
        actor_batch = (states, goals)

        hessians = []
        jacobians = []
        variables = [var for var in self.trainable_variables if 'alpha' not in var.name]
        for i in range(len(variables)):
            with tf.GradientTape() as tape_:
                with tf.GradientTape() as tape:
                    loss = self.compute_loss(actor_batch)
                jac = tape.jacobian(loss, variables[i])
            hess = tape_.jacobian(jac, variables[i])
            size = tf.size(jac).numpy()
            jacobians.append(jac.numpy())
            hessians.append(np.reshape(hess.numpy(), [size, size]))
       
        for hess in hessians:
            hessian_vals, hessian_vecs = np.linalg.eig(hess)
            hessian_vals = np.absolute(hessian_vals)
            self.agent.stats.record_var('hessian_norm', hessian_vals.max()) 

            hessian_frob_norm = np.linalg.norm(hess)
            self.agent.stats.record_var('hessian_frobenius_norm', hessian_frob_norm)
        
        jacobian = np.concatenate([np.reshape(jac, [-1]) for jac in jacobians], -1)
        
        actor2 = SACActor(self.agent)
        actor2.build_model()
        actor2_variables = [var for var in actor2.trainable_variables if 'alpha' not in var.name]
        for i in range(len(variables)):
            epsilon_i = 0.02 * jacobians[i] / np.linalg.norm(jacobian)
            weight = variables[i] + epsilon_i
            actor2_variables[i].assign(weight)
        loss2 = actor2.compute_loss(actor_batch)
        self.agent.stats.record_var('SAM_sharpness', loss2 - loss)
