from meta_controller.base_controller import EncoderNet, embedding
from meta_controller.rl_controller import RLNet2NetController
import tensorflow as tf
import os
from tensorflow.python.ops import array_ops
from models.basic_model import BasicModel
import shutil
import numpy as np

class BaselineNet(object):
    def __init__(self, num_steps, vocab, embedding_dim, rnn_units, rnn_type='bi_lstm', rnn_layers=1, name_prefix=""):
        self.num_steps = num_steps
        self.vocab = vocab
        self.embedding_dim = embedding_dim

        self.rnn_units = rnn_units
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.name_prefix = name_prefix
        # placeholder
        self.seq_len, self.input_seq = None, None
        # op
        self.encoder_output, self.encoder_state = None, None

    def _define_input(self):
        self.seq_len = tf.placeholder(
            tf.int32,
            [None],
            'seq_len'
        )  # length of each sequence, shape = [batch_size, ]

        self.input_seq = tf.placeholder(
            tf.int32,
            [None, self.num_steps],
            'input_seq'
        )  # input sequence, shape = [batch_size, num_steps]

    def build(self):
        self._define_input()

        output = self.input_seq
        output = embedding(output, self.vocab.size, self.embedding_dim, name=self.name_prefix + 'layer_embedding')
        input_dim = self.embedding_dim

        # Prepare data shape to match rnn function requirements
        # Current data input shape: [batch_size, num_steps, input_dim]
        # Required shape: 'num_steps' tensors list of shape [batch_size, input_dim]
        output = tf.transpose(output, [1, 0, 2])
        output = tf.reshape(output, [-1, input_dim])
        output = tf.split(output, self.num_steps, 0)

        if self.bidirectional:
            # 'num_steps' tensors list of shape [batch_size, rnn_units * 2]
            fw_cell = build_cell(self.rnn_units, self.cell_type, self.rnn_layers)
            bw_cell = build_cell(self.rnn_units, self.cell_type, self.rnn_layers)
            output, state_fw, state_bw = rnn.static_bidirectional_rnn(
                fw_cell, bw_cell, output, dtype=tf.float32, sequence_length=self.seq_len, scope=self.name_prefix + 'encoder')
            #TODO seperate variable by scope not by name?

            if isinstance(state_fw, tf.contrib.rnn.LSTMStateTuple):
                encoder_state_c = tf.concat([state_fw.c, state_bw.c], axis=1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat([state_fw.h, state_bw.h], axis=1, name='bidirectional_concat_h')
                state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            elif isinstance(state_fw, tf.Tensor):
                state = tf.concat([state_fw, state_bw], axis=1, name='bidirectional_concat')
            else:
                raise ValueError
        else:
            # 'num_steps' tensors list of shape [batch_size, rnn_units]
            cell = build_cell(self.rnn_units, self.cell_type, self.rnn_layers)
            output, state = rnn.static_rnn(cell, output, dtype=tf.float32, sequence_length=self.seq_len,
                                           scope='encoder')

        output = tf.stack(output, axis=0)  # [num_steps, batch_size, rnn_units]
        output = tf.transpose(output, [1, 0, 2])  # [batch_size, num_steps, rnn_units]
        self.encoder_output = output
        self.encoder_state = state
        return output, state

class ReinforceBaselineNet2NetController(RLNet2NetController):
    # TODO how to call this function at train step?
    # need to pass encoder output seq to inputs, but
    def _define_subclass_input(self):
        self.advantages = tf.placeholder(
            tf.float32,
            shape=[None],
            name='advantages',
        )
        self.baseline_input_seq = tf.placeholder(
            tf.int32,
            [None, self.rl_config["num_steps"]],
            'baseline_input_seq'
        )  # input sequence, shape = [batch_size, num_steps]

    def calculate_advantage(self, reward, input_seq, input_len):
        # calculate advantage value for one set of input
        baseline = self.sess.run(self.baseline,
                                 feed_dict={self.reward: reward,
                                            self.baseline_input_seq: input_seq})


        adv_val = reward-baseline
        print "Calculating adv, adv_val shpae = {}, baselin shape = {}".format(adv_val.shape, baseline.shape)
        mean = np.mean(adv_val)
        std = np.std(adv_val)
        adv_val = (adv_val-mean)/std
        return adv_val

    def build_baseline_network(self, size, n_layer, embedding_dim, vocab, num_steps):
        # add target place holder
        #self.baseline_input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.num_steps, self.rnn_units))
        # first build embedding layer to feed into baseline FC network
        with tf.variable_scope("rl_baseline"):
            out = self.baseline_input_seq
            out = embedding(out, vocab.size, embedding_dim, name='baseline_embedding')
            # input_dim = embedding_dim

            # Prepare data shape to match rnn function requirements
            # Current data input shape: [batch_size, num_steps, input_dim]
            # Required shape: 'num_steps' tensors list of shape [batch_size, input_dim]
            # out = tf.transpose(out, [1, 0, 2])
            out = tf.reshape(out, [-1, num_steps * embedding_dim])
            # out = tf.split(out, num_steps, 0)[-1] # take the last step as feature

        # TODO understand the dimenstion for encoder output
        # TODO figure out if we should feed in encoder state or not, don't feed in for now
        # print "Building baseline, encoder output = {}".format(out)
        #out = tf.reshape(encoder_output, [-1, int(encoder_output.get_shape()[2])]) #[batch_size * num_steps, rnn_units
        with tf.variable_scope("rl_baseline"):
            for i in range(n_layer):
                out = tf.contrib.layers.fully_connected(out, size, scope="rl_baseline_fc_{}".format(i))
                # use relu, xivar initialization by default
                # TODO do we need batch norm? prob not, look into if the og one has batch norm
            # build output layer
            out = tf.contrib.layers.fully_connected(out, 1, activation_fn=None)
        self.baseline = tf.squeeze(out)
        print "Building baseline, baseline = {}".format(self.baseline)
        loss = tf.losses.mean_squared_error(self.baseline, self.reward)
        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # use the same learning rate as rl algorithm
        self.update_baseline_op = adam.minimize(loss=loss)

    def update_baseline_network(self, input_seq, seq_len, rewards, learning_rate):
        self.sess.run(self.update_baseline_op,
                      feed_dict={self.baseline_input_seq: input_seq,
                                 self.reward : rewards,
                                 #self.baseline_seq_len: seq_len,
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
        if self.rl_config is not None:
            self.build_baseline_network(**self.rl_config)
            self.train_step.append(self.update_baseline_op)

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
        if self.rl_config is not None:
            return wider_side_obj, self.get_wider_entropy()
        else:
            return wider_side_obj, self.get_wider_entropy_with_baseline()


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
        if self.rl_config is not None:
            return deeper_side_obj, self.get_wider_entropy()
        else:
            return deeper_side_obj, self.get_deeper_entropy_with_baseline()

