from meta_controller.base_controller import EncoderNet, embedding
from meta_controller.rl_controller import RLNet2NetController
from meta_controller.rl_baseline_controller import ReinforceBaselineNet2NetController
import tensorflow as tf
import os
from tensorflow.python.ops import array_ops
from models.basic_model import BasicModel
import shutil
import numpy as np


class ReinforceAcerNet2NetController(ReinforceBaselineNet2NetController):

    def _define_subclass_input(self):
        self.advantages = tf.placeholder(
            tf.float32,
            shape=[None],
            name='advantages',
        )
        self.wider_qrets = tf.placeholder(
            tf.float32,
            shape=[None],
            name='wider_qrets'
        )
        self.deeper_qrets = tf.placeholder(
            tf.float32,
            shape=[None],
            name='deeper_qrets'
        )

    def build_training_process(self):
        c = tf.constant(0.8, dtype=tf.float32)
        # if self.wider_seg_deepr > 0, then get wide_side_obj, else wider_entropy = 0
        wider_side_obj, wider_entropy = tf.cond(
            tf.greater(self.wider_seg_deeper, 0),
            lambda: self.get_wider_side_obj(),
            lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        )
        wider_side_obj = tf.reduce_sum(wider_side_obj * tf.minimum(c, self.wider_rho))
        batch_size = array_ops.shape(self.reward)[0]
        deeper_side_obj, deeper_entropy = tf.cond(
            self.has_deeper,
            lambda: self.get_deeper_side_obj(),
            lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        )
        deeper_side_obj = tf.reduce_sum(deeper_side_obj * tf.minimum(c, self.deeper_rho))
        self.obj = wider_side_obj + deeper_side_obj
        entropy_term = wider_entropy * tf.cast(self.wider_seg_deeper, tf.float32) + \
                       deeper_entropy * tf.cast(batch_size - self.wider_seg_deeper, tf.float32)
        entropy_term /= tf.cast(batch_size, tf.float32)

        g = - self.obj - self.entropy_penalty * entropy_term

        optimizer = BasicModel.build_optimizer(self.learning_rate, self.opt_config[0], self.opt_config[1])
        self.train_step = [optimizer.minimize(g)]

        wq = tf.reshape(self.wider_actor.selected_q, [-1])
        w_loss = tf.losses.mean_squared_error(self.wider_qrets, wq)
        self.update_wider_q = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=w_loss)

        dq = tf.reshape(self.deeper_actor.selected_q, [-1])
        d_loss = tf.losses.mean_squared_error(self.deeper_qrets, dq)
        self.update_deeper_q = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=d_loss)
        # add baseline to training step
        if self.baseline_actor is not None:
            self.build_baseline_network()

    def get_wider_side_obj(self):
        wider_side_reward = self.reward[:(self.wider_seg_deeper*tf.constant(self.wider_actor.num_steps))]

        # obj from wider side
        wider_trajectory = tf.one_hot(self.wider_decision_trajectory, depth=max(self.wider_actor.out_dim, 2))
        wider_probs = tf.reduce_max(tf.multiply(wider_trajectory, self.wider_actor.probs), axis=2)
        wider_probs = tf.log(wider_probs)  # [wider_batch_size, num_steps]
        wider_probs = tf.multiply(wider_probs, self.wider_decision_mask)
        wider_side_obj = tf.multiply(tf.reshape(wider_probs, shape=[-1]), wider_side_reward)
        # shape [batch_size * nsteps, num_steps (50)]

        # wider_side_obj = tf.reduce_sum(wider_probs)
        return wider_side_obj, self.get_wider_entropy()

    def get_deeper_side_obj(self):
        deeper_side_reward = self.reward[(self.wider_seg_deeper*tf.constant(self.wider_actor.num_steps)):]

        # obj from deeper side
        deeper_side_obj = []
        for _i in range(self.deeper_actor.decision_num):
            decision_trajectory = self.deeper_decision_trajectory[:, _i]
            deeper_decision_mask = self.deeper_decision_mask[:, _i]
            decision_trajectory = tf.one_hot(decision_trajectory, depth=self.deeper_actor.out_dims[_i])
            deeper_probs = tf.reduce_max(tf.multiply(decision_trajectory, self.deeper_actor.probs[_i]), axis=1)
            deeper_probs = tf.log(deeper_probs)  # [deeper_batch_size]
            deeper_probs = tf.multiply(deeper_probs, deeper_decision_mask)

            deeper_side_obj.append(deeper_probs) # shape [decision_num, deeper_batch_size]
        deeper_side_obj = tf.reshape(tf.stack(deeper_side_obj, axis=1), shape=[-1])
        deeper_side_obj = tf.multiply(deeper_side_obj, deeper_side_reward)
        # deeper_side_obj = tf.reduce_sum(deeper_side_obj)
        return deeper_side_obj, self.get_deeper_entropy()

    def update_Q_function(self, wider_qrets, deeper_qrets, wider_seg_deeper, input_seq, seq_len, block_layer_num,
                          wider_decision_trajectory, wider_decision_mask, deeper_decision_trajectory, deeper_decison_mask,
                          learning_rate):
        self.sess.run([self.update_wider_q, self.update_deeper_q],
                      feed_dict={
                        self.wider_qrets: wider_qrets,
                        self.deeper_qrets: deeper_qrets,
                        self.wider_seg_deeper:wider_seg_deeper,
                        self.encoder.input_seq: input_seq,
                        self.encoder.seq_len: seq_len,
                        self.deeper_actor.block_layer_num: block_layer_num,
                        self.wider_decision_trajectory: wider_decision_trajectory,
                        self.wider_decision_mask: wider_decision_mask,
                        self.deeper_decision_trajectory: deeper_decision_trajectory,
                        self.deeper_decision_mask: deeper_decison_mask,
                        self.is_training: True,
                        self.learning_rate: learning_rate
                      })

