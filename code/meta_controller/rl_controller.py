from meta_controller.base_controller import WiderActorNet, DeeperActorNet, EncoderNet, BaseController, embedding
import tensorflow as tf
import os
from tensorflow.python.ops import array_ops
from models.basic_model import BasicModel
import shutil
import numpy as np
import random


class RLNet2NetController(BaseController):
    def save(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load(self):
        if os.path.isfile('%s/model.ckpt.index' % self.path):
            try:
                self.saver.restore(self.sess, self.save_path)
            except Exception:
                print('Failed to to load model '
                                'from save path: %s' % self.save_path)
            print('Successfully load model from save path: %s' % self.save_path)
        else:
            print('No model files in ' + '%s/model.ckpt.index' % self.path)

    def __init__(self, path, entropy_penalty,
                 encoder, wider_actor, deeper_actor, opt_config, baseline_actor=None):
        assert(isinstance(encoder, EncoderNet))
        assert(isinstance(wider_actor, WiderActorNet))
        assert(isinstance(deeper_actor, DeeperActorNet))
        BaseController.__init__(self, path)
        self.entropy_penalty = entropy_penalty

        self.encoder = encoder
        self.wider_actor = wider_actor
        self.deeper_actor = deeper_actor
        self.opt_config = opt_config
        self.baseline_actor = baseline_actor
        # TODO put this in config
        self.replay_buf_size = 100
        self.graph = tf.Graph()
        self.obj, self.train_step = None, None
        with self.graph.as_default():
            self._define_input()
            self.build_forward()
            self.build_training_process()
            self.global_variables_initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()

        # replay buffers
        self.replay_buf = []
        # TODO change this to pass from config
        self.replay_size = 40

    def _define_input(self):
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.wider_seg_deeper = tf.placeholder(tf.int32, shape=[], name='wider_seg_deeper')

        self.wider_decision_trajectory = tf.placeholder(
            tf.int32,
            shape=[None, self.encoder.num_steps],
            name='wider_decision_trajectory',
        )  # [wider_batch_size, num_steps]
        self.wider_decision_mask = tf.placeholder(
            tf.float32,
            shape=[None, self.encoder.num_steps],
            name='wider_decision_mask',
        )  # [wider_batch_size, num_steps]

        self.deeper_decision_trajectory = tf.placeholder(
            tf.int32,
            shape=[None, self.deeper_actor.decision_num],
            name='deeper_decision_trajectory',
        )  # [deeper_batch_size, deeper_decision_num]

        self.deeper_decision_mask = tf.placeholder(
            tf.float32,
            shape=[None, self.deeper_actor.decision_num],
            name='deeper_decision_mask',
        )  # [deeper_batch_size, deeper_decision_num]

        self.reward = tf.placeholder(
            tf.float32,
            shape=[None],
            name='reward',
        )  # [batch_size]
        self.has_deeper = tf.placeholder(
            tf.bool,
            shape=[],
            name='has_deeper',
        )
        self._define_subclass_input()

    def _define_subclass_input(self):
        raise NotImplementedError

    def update_controller(self, learning_rate, wider_seg_deeper, wider_decision_trajectory, wider_decision_mask,
                          deeper_decision_trajectory, deeper_decison_mask, reward, block_layer_num, input_seq, seq_len):
        has_deeper = wider_seg_deeper < len(input_seq)
        feed_dict = {
            self.learning_rate: learning_rate,
            self.wider_seg_deeper: wider_seg_deeper,
            self.wider_decision_trajectory: wider_decision_trajectory,
            self.wider_decision_mask: wider_decision_mask,
            self.deeper_decision_trajectory: deeper_decision_trajectory,
            self.deeper_decision_mask: deeper_decison_mask,
            self.reward: reward,
            self.is_training: True and has_deeper,
            self.deeper_actor.block_layer_num: block_layer_num,
            self.encoder.input_seq: input_seq,
            self.encoder.seq_len: seq_len,
            self.has_deeper: has_deeper,
        }

        self.sess.run(self.train_step, feed_dict=feed_dict)

    def build_forward(self):
        encoder_output, encoder_state = self.encoder.build()
        feed2wider_output = encoder_output[:self.wider_seg_deeper]
        feed2deeper_output = encoder_output[self.wider_seg_deeper:]
        if isinstance(encoder_state, tf.contrib.rnn.LSTMStateTuple):
            encoder_state_c = encoder_state.c
            encoder_state_h = encoder_state.h

            feed2wider_c = encoder_state_c[:self.wider_seg_deeper]
            feed2wider_h = encoder_state_h[:self.wider_seg_deeper]
            feed2wider_state = tf.contrib.rnn.LSTMStateTuple(c=feed2wider_c, h=feed2wider_h)

            feed2deeper_c = encoder_state_c[self.wider_seg_deeper:]
            feed2deeper_h = encoder_state_h[self.wider_seg_deeper:]
            feed2deeper_state = tf.contrib.rnn.LSTMStateTuple(c=feed2deeper_c, h=feed2deeper_h)
        elif isinstance(encoder_state, tf.Tensor):
            feed2wider_state = encoder_state[:self.wider_seg_deeper]
            feed2deeper_state = encoder_state[self.wider_seg_deeper:]
        else:
            raise ValueError
        self.wider_input = feed2wider_output
        self.deeper_input = feed2deeper_output
        self.inputs = feed2wider_output + feed2deeper_output
        # TODO not sure what's the difference between output /state  and which one
        # to use to estimate baseline. maybe we can use both???
        self.wider_actor.build_forward(feed2wider_output)
        self.deeper_actor.build_forward(feed2deeper_output, feed2deeper_state, self.is_training,
                                        self.deeper_decision_trajectory)

    def build_training_process(self):
        raise NotImplementedError

    def sample_wider_decision(self, input_seq, seq_len):
        batch_size = len(seq_len)
        wider_decision, wider_probs = self.sess.run(
            fetches=[self.wider_actor.decision, self.wider_actor.probs],
            feed_dict={
                self.encoder.input_seq: input_seq,
                self.encoder.seq_len: seq_len,
                self.wider_seg_deeper: batch_size,
            }
        )  # [batch_size, num_steps]
        return wider_decision, wider_probs

    def sample_deeper_decision(self, input_seq, seq_len, block_layer_num):
        deeper_decision, deeper_probs = self.sess.run(
            fetches=[self.deeper_actor.decision, self.deeper_actor.probs],
            feed_dict={
                self.encoder.input_seq: input_seq,
                self.encoder.seq_len: seq_len,
                self.wider_seg_deeper: 0,
                self.is_training: False,
                self.deeper_actor.block_layer_num: block_layer_num,
                self.deeper_decision_trajectory: -np.ones([len(seq_len), self.deeper_actor.decision_num])
            }
        )  # [batch_size, decision_num]
        return deeper_decision, deeper_probs

    def _initialize_session(self):
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.global_variables_initializer)
        shutil.rmtree(self.logs_path, ignore_errors=True)
        self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

    def get_wider_entropy(self):
        wider_entropy = -tf.multiply(tf.log(self.wider_actor.probs), self.wider_actor.probs)
        wider_entropy = tf.reduce_sum(wider_entropy, axis=2)
        wider_entropy = tf.multiply(wider_entropy, self.wider_decision_mask)
        wider_entropy = tf.div(tf.reduce_sum(wider_entropy, axis=1), tf.reduce_sum(self.wider_decision_mask, axis=1))
        wider_entropy = tf.reduce_mean(wider_entropy)
        return wider_entropy

    def get_deeper_entropy(self):
        deeper_entropy = []
        for _i in range(self.deeper_actor.decision_num):
            deeper_probs = self.deeper_actor.probs[_i]
            entropy = -tf.multiply(tf.log(deeper_probs + 1e-10), deeper_probs)
            entropy = tf.reduce_sum(entropy, axis=1)
            deeper_entropy.append(entropy)
        deeper_entropy = tf.reduce_mean(deeper_entropy)
        return deeper_entropy



class ReinforceNet2NetController(RLNet2NetController):
    def _define_subclass_input(self):
        pass

    def build_training_process(self):
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
        print "in build, reward = {}".format(self.reward)
        self.train_step = optimizer.minimize(- self.obj - self.entropy_penalty * entropy_term)

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
        return deeper_side_obj, self.get_deeper_entropy()

    def add_to_replay(self, rewards,
                     encoder_input_seq, encoder_seq_len,
                        wider_decision_trajectory, deeper_decision_trajectory,
                        wider_probs_trajectory, deeper_probs_trajectory,
                      wider_decision_mask, deeper_decision_mask, deeper_block_layer_num,
                      wider_action_num, deeper_action_num, batch_size):
        # each term is [batch size * n_step, x]
        n_step = wider_action_num + deeper_action_num

        # transpose to shape ( batch_size, n_step, -1]
        print encoder_seq_len.reshape((n_step, batch_size, -1))
        encoder_seq_len = np.transpose(encoder_seq_len.reshape((n_step, batch_size, -1)), (1,0,2))
        encoder_input_seq = np.transpose(encoder_input_seq.reshape((n_step, batch_size, -1)),(1, 0, 2))
        wider_decision_trajectory = np.transpose(wider_decision_trajectory.reshape((wider_action_num, batch_size, -1)), (1,0,2))
        wider_probs_trajectory = np.transpose(wider_probs_trajectory.reshape((wider_action_num, batch_size, -1)), (1,0,2))
        deeper_decision_trajectory = np.transpose(deeper_decision_trajectory.reshape((deeper_action_num, batch_size, -1)), (1,0,2))
        deeper_probs_trajectory = np.transpose(deeper_probs_trajectory.reshape((deeper_action_num, batch_size, -1)), (1,0,2))
        deeper_decision_mask = np.transpose(deeper_decision_mask.reshape((deeper_action_num, batch_size, -1)), (1,0,2))
        wider_decision_mask = np.transpose(wider_decision_mask.reshape((wider_action_num, batch_size, -1)), (1,0,2))
        deeper_block_layer_num = np.transpose(deeper_block_layer_num.reshape((deeper_action_num, batch_size, -1)), (1,0,2))
        rewards = np.transpose(rewards.reshape((n_step, batch_size, -1)), (1,0,2))

        # prob trajectory are in shape [batch_size, n_step, n_decision] already

        for i in range(batch_size):
            self.replay_buf.append((rewards[i],
                         encoder_input_seq[i], encoder_seq_len[i],
                            wider_decision_trajectory[i], deeper_decision_trajectory[i],
                            wider_probs_trajectory[i], deeper_probs_trajectory[i],
                                   wider_decision_mask[i], deeper_decision_mask[i], deeper_block_layer_num[i]))
        if len(self.replay_buf) >= self.replay_buf_size:
            del self.replay_buf[:batch_size]

    def replay_buf_len(self):
        return len(self.replay_buf)

    def sample_from_replay(self, n):
        # sample n trajectory from the buffer, might have duplicate
        trajectories = [list(random.choice(self.replay_buf)) for _ in range(n)]
        #np.concatenate(trajectories, axis=0)

        # [batchsize, step_size, -1]
        rewards, encoder_input_seq, encoder_seq_len,\
        wider_decision_trajectory, deeper_decision_trajectory, \
        wider_probs_trajectory, deeper_probs_trajectory,\
        wider_decision_mask, deeper_decision_mask,\
        deeper_block_layer_num = map(list, zip(*trajectories))

        # convert all to np
        rewards = np.array(rewards)
        encoder_input_seq = np.array(encoder_input_seq)
        encoder_seq_len = np.array(encoder_seq_len)
        wider_decision_trajectory = np.array(wider_decision_trajectory)
        wider_probs_trajectory = np.array(wider_probs_trajectory)
        deeper_decision_trajectory = np.array(deeper_decision_trajectory)
        deeper_probs_trajectory = np.array(deeper_probs_trajectory)
        deeper_block_layer_num = np.array(deeper_block_layer_num)
        # calcuate importance sampling ratio
        wider_ratio_trajectory = []  # batch size, steps, layer (50)
        deeper_ratio_trajectory = [] # batch size, steps, decision num
        #print "wider shape {}".format(wider_decision_trajectory.shape)
        #print "deeper decision shape {}".format(deeper_decision_trajectory.shape)
        #print "reward shape {}".format(rewards.shape)
        #print "wider prob shape {}".format(wider_probs_trajectory.shape)
        #print "deeper prob shape {}".format(deeper_probs_trajectory.shape)
        #print "wider prob {}".format(wider_probs_trajectory.shape)
        #print "seq shape {}".format(encoder_input_seq.shape)
        #print "deeper block layer shape {}".format(deeper_block_layer_num.shape)
        wider_action_num = wider_decision_trajectory.shape[1]
        deeper_action_num = deeper_decision_trajectory.shape[1]
        for _j in range(wider_action_num):
            # get current policy for each step
            replay_actions = wider_decision_trajectory[:, _j,:] # batch size, # layer
            replay_probs = wider_probs_trajectory[:, _j,:] # batch size. # layer
            _, cur_wider_probs = self.sample_wider_decision(encoder_input_seq[:, _j,:], np.squeeze(encoder_seq_len[:, _j, :]))
            wider_ratio_per_step = np.array([])

            # loop through batch
            print replay_actions.shape
            batch_ratio = []
            for b in range(n):
                layer_ratio = []
                for layer in range(self.wider_actor.num_steps):
                    layer_ratio.append(cur_wider_probs[b][layer][replay_actions[b][layer]] / replay_probs[b][layer])
                batch_ratio.append(layer_ratio)
            wider_ratio_trajectory.append(batch_ratio)

        ## sample deeper actions
        for _j in range(deeper_action_num):

            replay_actions = deeper_decision_trajectory[:,_j, :]
            replay_probs = deeper_probs_trajectory[:, _j, :]
            deeper_start = wider_action_num
            _, deeper_probs_all = self.sample_deeper_decision(encoder_input_seq[:, _j + deeper_start,:],
                                                                         np.squeeze(encoder_seq_len[:, _j + deeper_start, :]), deeper_block_layer_num[:,_j,:])
            # cur_deeper_probs = #decision, #batch
            # replay_actions = #batch, #decision
            batch_ratio = []
            for b in range(n):
                # for each batch
                decision_ratio = []
                for decision in range(self.deeper_actor.decision_num):
                    # for each decision
                    print replay_actions
                    print replay_probs
                    print deeper_probs_all
                    decision_ratio.append(deeper_probs_all[decision][b][replay_actions[b][decision]] / replay_probs[b][decision])
                batch_ratio.append(decision_ratio)
            # deeper_ratio_trajectory = #batch, #decision
            deeper_ratio_trajectory.append(batch_ratio)

        # all return terms in shape [n_step, batch_size, -1]
        return rewards, encoder_input_seq, np.squeeze(encoder_seq_len),\
        wider_decision_trajectory, deeper_decision_trajectory, \
        wider_decision_mask, deeper_decision_mask, \
        deeper_block_layer_num, \
        wider_ratio_trajectory, deeper_ratio_trajectory



