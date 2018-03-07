from meta_controller.base_controller import EncoderNet, embedding
from meta_controller.rl_controller import RLNet2NetController
import tensorflow as tf
import os
from tensorflow.python.ops import array_ops
from models.basic_model import BasicModel
import shutil
import numpy as np


class ReinforceBaselineNet2NetController(RLNet2NetController):
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
        print "Calculating adv, adv_val shpae = {}, baselin shape = {}".format(adv_val.shape, baseline.shape)
        mean = np.mean(adv_val)
        std = np.std(adv_val)
        adv_val = (adv_val-mean)/std
        return adv_val

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
        # if has_deepr, then get deeper_side_obj, else deeper_entropy = 0
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

    def get_wider_entropy_with_baseline(self):
        wider_entropy = -tf.multiply(tf.log(self.wider_actor.probs), self.advantages)
        wider_entropy = tf.reduce_sum(wider_entropy, axis=2)
        wider_entropy = tf.multiply(wider_entropy, self.wider_decision_mask)
        wider_entropy = tf.div(tf.reduce_sum(wider_entropy, axis=1), tf.reduce_sum(self.wider_decision_mask, axis=1))
        wider_entropy = tf.reduce_mean(wider_entropy)
        return wider_entropy

    def get_deeper_entropy_with_baseline(self):
        deeper_entropy = []
        for _i in range(self.deeper_actor.decision_num):
            deeper_probs = self.deeper_actor.probs[_i]
            entropy = -tf.multiply(tf.log(deeper_probs + 1e-10), self.advantages)
            entropy = tf.reduce_sum(entropy, axis=1)
            deeper_entropy.append(entropy)
        deeper_entropy = tf.reduce_mean(deeper_entropy)
        return deeper_entropy


    def get_wider_side_obj(self):

        wider_side_reward = self.reward[:self.wider_seg_deeper]

        # obj from wider side
        wider_trajectory = tf.one_hot(self.wider_decision_trajectory, depth=max(self.wider_actor.out_dim, 2))
        wider_probs = tf.reduce_max(tf.multiply(wider_trajectory, self.wider_actor.probs), axis=2)
        wider_probs = tf.log(wider_probs)  # [wider_batch_size, num_steps]
        wider_probs = tf.multiply(wider_probs, self.wider_decision_mask)
        wider_probs = tf.multiply(wider_probs, tf.reshape(wider_side_reward, shape=[-1, 1]))

        wider_side_obj = tf.reduce_sum(wider_probs)
        return wider_side_obj, self.get_wider_entropy()


    def get_deeper_side_obj(self):
        deeper_side_reward = self.reward[self.wider_seg_deeper:]

        # obj from deeper side
        deeper_side_obj = []
        for _i in range(self.deeper_actor.decision_num):
            decision_trajectory = self.deeper_decision_trajectory[:, _i]
            deeper_decision_mask = self.deeper_decision_mask[:, _i]
            decision_trajectory = tf.one_hot(decision_trajectory, depth=self.deeper_actor.out_dims[_i])
            deeper_probs = tf.reduce_max(tf.multiply(decision_trajectory, self.deeper_actor.probs[_i]), axis=1)
            deeper_probs = tf.log(deeper_probs)  # [deeper_batch_size]
            deeper_probs = tf.multiply(deeper_probs, deeper_decision_mask)
            deeper_probs = tf.multiply(deeper_probs, deeper_side_reward)

            deeper_side_obj.append(tf.reduce_sum(deeper_probs))
        deeper_side_obj = tf.reduce_sum(deeper_side_obj)
        return deeper_side_obj, self.get_wider_entropy()

