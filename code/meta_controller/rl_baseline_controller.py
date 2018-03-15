from meta_controller.base_controller import EncoderNet, embedding
from meta_controller.rl_controller import ReinforceNet2NetController
import tensorflow as tf
import os
from tensorflow.python.ops import array_ops
from models.basic_model import BasicModel
import shutil
import numpy as np


class ReinforceBaselineNet2NetController(ReinforceNet2NetController):
    # TODO how to call this function at train step?
    # need to pass encoder output seq to inputs, but
    def _define_subclass_input(self):
        self.advantages = tf.placeholder(
            tf.float32,
            shape=[None],
            name='advantages',
        )

    def calculate_advantage(self, reward, input_seq, input_len):
        # calculate advantage value for one set of input
        baseline = self.sess.run(self.baseline,
                                 feed_dict={self.reward: reward,
                                            self.baseline_actor.input_seq: input_seq,
                                            self.baseline_actor.seq_len: input_len})
        adv_val = reward-baseline
        # print "Calculating adv, adv_val shpae = {}, baselin shape = {}".format(adv_val.shape, baseline.shape)
        mean = np.mean(adv_val)
        std = np.std(adv_val)
        adv_val = (adv_val-mean)/std
        return baseline, adv_val

    # def save_to_replay(self, input_seq, seq_len, rewards):


    def build_baseline_network(self):
        self.baseline = self.baseline_actor.build()
        # print "Building baseline, baseline = {}".format(self.baseline)
        loss = tf.losses.mean_squared_error(self.baseline, self.reward)
        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # use the same learning rate as rl algorithm
        self.update_baseline_op = adam.minimize(loss=loss)



    def update_baseline_network(self, input_seq, seq_len, rewards, learning_rate):
        self.sess.run(self.update_baseline_op,
                      feed_dict={self.baseline_actor.input_seq: input_seq,
                                 self.baseline_actor.seq_len: seq_len,
                                 self.reward : rewards,
                                 self.learning_rate: learning_rate})

    def build_training_process(self):
        # if self.wider_seg_deepr > 0, then get wide_side_obj, else wider_entropy = 0
        wider_side_obj, wider_entropy = tf.cond(
            tf.greater(self.wider_seg_deeper, 0),
            lambda: self.get_wider_side_obj(),
            lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        )
        batch_size = array_ops.shape(self.reward)[0]
        deeper_side_obj, deeper_entropy = tf.cond(
            self.has_deeper,
            lambda: self.get_deeper_side_obj(),
            lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        )
        self.obj = wider_side_obj + deeper_side_obj
        entropy_term = wider_entropy * tf.cast(self.wider_seg_deeper, tf.float32) + \
                       deeper_entropy * tf.cast(batch_size - self.wider_seg_deeper, tf.float32)
        entropy_term /= tf.cast(batch_size, tf.float32)

        optimizer = BasicModel.build_optimizer(self.learning_rate, self.opt_config[0], self.opt_config[1])
        self.train_step = [optimizer.minimize(- self.obj - self.entropy_penalty * entropy_term)]
        # add baseline to training step
        if self.baseline_actor is not None:
            self.build_baseline_network()

    def sample_wider_decision_with_q(self, input_seq, seq_len):
        batch_size = len(seq_len)
        wider_decision, wider_probs, selected_prob, q_values, selected_q = self.sess.run(
            fetches=[self.wider_actor.decision, self.wider_actor.probs, self.wider_actor.selected_prob, 
                     self.wider_actor.q_values, self.wider_actor.selected_q],
            feed_dict={
                self.encoder.input_seq: input_seq,
                self.encoder.seq_len: seq_len,
                self.wider_seg_deeper: batch_size,
            }
        )  # [batch_size, num_steps]
        return wider_decision, wider_probs, selected_prob, q_values, selected_q

    def sample_deeper_decision_with_q(self, input_seq, seq_len, block_layer_num):
        deeper_decision, deeper_probs, selected_prob, q_values, selected_q = self.sess.run(
            fetches=[self.deeper_actor.decision, self.deeper_actor.probs, self.deeper_actor.selected_prob, self.deeper_actor.q_values, self.deeper_actor.selected_q],
            feed_dict={
                self.encoder.input_seq: input_seq,
                self.encoder.seq_len: seq_len,
                self.wider_seg_deeper: 0,
                self.is_training: False,
                self.deeper_actor.block_layer_num: block_layer_num,
                self.deeper_decision_trajectory: -np.ones([len(seq_len), self.deeper_actor.decision_num])
            }
        )  # [batch_size, decision_num]
        return deeper_decision, deeper_probs, selected_prob, q_values, selected_q

    
    # def get_wider_entropy_with_baseline(self):
    #     wider_entropy = -tf.multiply(tf.log(self.wider_actor.probs), self.advantages)
    #     wider_entropy = tf.reduce_sum(wider_entropy, axis=2)
    #     wider_entropy = tf.multiply(wider_entropy, self.wider_decision_mask)
    #     wider_entropy = tf.div(tf.reduce_sum(wider_entropy, axis=1), tf.reduce_sum(self.wider_decision_mask, axis=1))
    #     wider_entropy = tf.reduce_mean(wider_entropy)
    #     return wider_entropy

    # def get_deeper_entropy_with_baseline(self):
    #     deeper_entropy = []
    #     for _i in range(self.deeper_actor.decision_num):
    #         deeper_probs = self.deeper_actor.probs[_i]
    #         entropy = -tf.multiply(tf.log(deeper_probs + 1e-10), self.advantages)
    #         entropy = tf.reduce_sum(entropy, axis=1)
    #         deeper_entropy.append(entropy)
    #     deeper_entropy = tf.reduce_mean(deeper_entropy)
    #     return deeper_entropy
