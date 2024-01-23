from actor import Actor
import tensorflow as tf
import numpy as np
from general import utils

class DDPGActor(Actor):
    def __init__(self, agent, is_target=False):
        super().__init__(agent, is_target)
    
    def call(self, sg):
        self.logger.time_stamp('FC', 3)

        x = sg
        for i in range(len(self.fcs)):
            layer = self.fcs[i]
            x = layer(x)
            if i == len(self.fcs) - 3:
                feature = x

        action = self.last_layer(x)

        self.logger.time_stamp('End', 3)
        return action, feature

    def compute_loss(self, actor_batch):
        states, goals = actor_batch
        actions, _ = self(tf.concat([states, goals], 1), training=True)
        self.logger.time_stamp('Get Q', 2)
        critic_q = self.agent.infer_critic(states, goals, actions)
        self.logger.time_stamp('Compute Loss', 2)
        loss = - tf.reduce_mean(critic_q)
        return loss

    def update(self, actor_batch):
        # synchronously update actor network and AE network
        states, goals = actor_batch

        self.logger.time_stamp('Get Action', 2)
        with tf.GradientTape() as tape:
            loss = self.compute_loss(actor_batch)

        self.logger.time_stamp('Compute Grad', 2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.logger.time_stamp('Apply Grad', 2)
        self.opt.apply_gradients((grad, var) for (grad, var) in zip(grads, self.trainable_variables) if grad is not None)

        # self.sess.run(self.update_target_weights)
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
            #hessians.append(np.reshape(hess.numpy(), [size, size]))
       
        '''for hess in hessians:
            hessian_vals, hessian_vecs = np.linalg.eig(hess)
            hessian_vals = np.absolute(hessian_vals)
            self.agent.stats.record_var('hessian_norm', hessian_vals.max()) 

            hessian_frob_norm = np.linalg.norm(hess)
            self.agent.stats.record_var('hessian_frobenius_norm', hessian_frob_norm)'''
        
        jacobian = np.concatenate([np.reshape(jac, [-1]) for jac in jacobians], -1)
        
        actor2 = DDPGActor(self.agent)
        actor2.build_model()
        actor2_variables = [var for var in actor2.trainable_variables if 'alpha' not in var.name]
        for i in range(len(variables)):
            epsilon_i = 0.02 * jacobians[i] / np.linalg.norm(jacobian)
            weight = variables[i] + epsilon_i
            actor2_variables[i].assign(weight)
        loss2 = actor2.compute_loss(actor_batch)
        self.agent.stats.record_var('SAM_sharpness', loss2 - loss)

