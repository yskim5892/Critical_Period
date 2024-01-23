from actor import Actor
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability.python.distributions import Normal, MultivariateNormalDiag
import numpy as np
from general import utils
import scipy
import pdb

class PPOActor(Actor):
    def __init__(self, agent):
        super().__init__(agent, False)
        #self.log_std = Dense(self.action_dim, name='log_std')
        self.log_std = tf.Variable(name='log_std', initial_value= np.log(self.env.action_space.extents).astype(float) - 1, dtype=tf.float32)

        self.clip_grad = self.config['PPO_clip_grad']
        self.clip_ratio = self.config['PPO_clip_ratio']
        self.ent_coef = self.config['PPO_ent_coef']
        self.count = 0

    def call(self, sg):
        x = sg
        for i in range(len(self.fcs)):
            layer = self.fcs[i]
            x = layer(x)
            if i == len(self.fcs) - 3:
                feature = x
        
        mean = self.last_layer(x)
        #log_std = self.log_std(x)
        log_std = tf.clip_by_value(self.log_std, -10 * np.ones([self.action_dim]), np.log(self.env.action_space.extents) - 0.5) 
        #log_std = self.log_std

        return mean, log_std

    def action_logp(self, sg):
        mean, log_std = self.call(sg)
        std = tf.exp(log_std)
        #normal = MultivariateNormalDiag(loc=mean, scale_diag=std)

        self.agent.stats.record_var('action_std', tf.reduce_mean(std).numpy())

        action = mean + tf.random.normal(tf.shape(mean)) * std
        action = tf.clip_by_value(action, self.space.bounds[:, 0], self.space.bounds[:, 1])
        
        logp = self.logp(log_std, mean, action)

        return action, logp

    def logp(self, log_std, mu, action):
        #p = -0.5 * (((action - mu) / (tf.exp(log_std) + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi))
        p = -0.5 * (((action - mu) / (tf.exp(log_std) + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi))

        return tf.reduce_sum(p, axis = -1)

    def compute_loss(self, actor_batch):
        states, goals, old_log_ps, advs, old_actions = actor_batch 
        actions, log_std = self(tf.concat([states, goals], 1), training=True)
        log_ps = self.logp(log_std, actions, old_actions)
        ratio = tf.exp(log_ps - old_log_ps)
        surr1 = ratio * advs

        clip_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surr2 = clip_ratio * advs

        act_loss = -tf.reduce_mean(tf.math.minimum(surr1, surr2))

        entropy = tf.reduce_sum(log_std + 0.5 * np.log(2 * np.pi * np.e), axis=-1)
        ent_loss = -tf.reduce_mean(entropy)
        
        loss = act_loss + self.ent_coef * ent_loss
        self.agent.stats.record_var('ent_loss', loss.numpy()) 
        return loss

    # values / returns
    def update(self, actor_batch):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(actor_batch) 
        grads = tape.gradient(loss, self.trainable_variables)
        
        clipped_grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_grad)

        self.opt.apply_gradients((grad, var) for (grad, var) in zip(clipped_grads, self.trainable_variables) if grad is not None)

        self.count += 1
        if self.config['record_sharpness'] and self.count % 1000 == 0:
            self.record_sharpness(actor_batch)
        return loss, grads

    def record_sharpness(self, actor_batch):
        hessians = []
        jacobians = []
        for i in range(len(self.trainable_variables)):
            with tf.GradientTape() as tape_:
                with tf.GradientTape() as tape:
                    loss = self.compute_loss(actor_batch)
                jac = tape.jacobian(loss, self.trainable_variables[i])
            hess = tape_.jacobian(jac, self.trainable_variables[i])
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
        
        actor2 = PPOActor(self.agent)
        actor2.build_model()
        for i in range(len(self.trainable_variables)):
            epsilon_i = 0.02 * jacobians[i] / np.linalg.norm(jacobian)
            weight = self.trainable_variables[i] + epsilon_i
            actor2.trainable_variables[i].assign(weight)
        loss2 = actor2.compute_loss(actor_batch)
        self.agent.stats.record_var('SAM_sharpness', loss2 - loss)
