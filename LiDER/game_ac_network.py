#!/usr/bin/env python3
import logging
import numpy as np
import tensorflow as tf
import pygame

from abc import ABC
from abc import abstractmethod
from termcolor import colored
from time import sleep

logger = logging.getLogger("game_ac_network")


class GameACNetwork(ABC):
    """Actor-Critic Network Base Class."""

    use_mnih_2015 = False

    def __init__(self, action_size, thread_index, device="/cpu:0"):
        """Initialize GameACNetwork class."""
        self._action_size = action_size
        self._thread_index = thread_index
        self._device = device

    def prepare_loss(self, entropy_beta=0.01, critic_lr=0.5):
        """Prepare loss function of actor-critic.

        Keyword arguments:
        entropy_beta -- value multiplied to entropy
        critic_lr -- value multiplied to critic loss

        Reference:
        Based from A2C OpenAI Baselines. We use the loss function explained in
        the PAAC paper Clemente et al 2017.
        Efficient Parallel Methods for Deep Reinforcement Learning
        """
        def cat_entropy(logits):
            a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

        with tf.name_scope("Loss"):
            # taken action (input for policy)
            self.a = tf.placeholder(
                tf.float32, shape=[None, self._action_size], name="action")

            # temporal difference (R-V) (input for policy)
            self.advantage = tf.placeholder(
                tf.float32, shape=[None], name="advantage")
            self.cumulative_reward = tf.placeholder(
                tf.float32, shape=[None], name="cumulative_reward")

            assert self.a.shape.as_list() == self.logits.shape.as_list()
            neglogpac = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.a)
            pg_loss = tf.reduce_mean(self.advantage * neglogpac)
            vf_loss = tf.reduce_mean(tf.squared_difference(
                tf.squeeze(self.v), self.cumulative_reward) / 2.0)
            entropy = tf.reduce_mean(cat_entropy(self.logits))
            self.total_loss = pg_loss - entropy * entropy_beta \
                + vf_loss * critic_lr

    def prepare_sil_loss(self, entropy_beta=0, w_loss=1.0, critic_lr=0.5,
                         min_batch_size=1):
        """Prepare self-imitation loss.

        Keyword arguments:
        critic_lr -- value multiplied to critic loss

        Reference:
        Based from A2C-SIL
        """
        def cat_entropy(logits):
            a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

        with tf.name_scope("SIL_Loss"):
            # taken action (input for policy)
            self.a_sil = tf.placeholder(tf.float32,
                                        shape=[None, self._action_size],
                                        name="action_sil")

            # temporal difference (R-V)+ (input for policy)
            # (.)+ = max(-, 0)
            self.returns = tf.placeholder(tf.float32, shape=[None],
                                          name="returns")

            self.weights = tf.placeholder(tf.float32, shape=[None],
                                          name="memory_weights")

            # Our implementation with no reward clipping
            mask = tf.where(
                self.returns - tf.squeeze(self.v) > 0.0,
                tf.ones_like(self.returns), tf.zeros_like(self.returns))
            self.num_valid_samples = tf.reduce_sum(mask)
            self.num_samples = tf.maximum(self.num_valid_samples,
                                          min_batch_size)

            neglogpac = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.a_sil)

            v_estimate = tf.squeeze(self.v)

            self.advs = self.returns - v_estimate
            self.clipped_advs = tf.stop_gradient(
                tf.maximum(tf.zeros_like(self.advs), self.advs))

            # sil_pg_loss = self.weights * neglogpac * self.clipped_advs
            sil_pg_loss = neglogpac * self.clipped_advs
            sil_pg_loss = tf.reduce_sum(sil_pg_loss) / self.num_samples

            # entropy = self.weights * cat_entropy(self.logits) * mask
            entropy = cat_entropy(self.logits) * mask
            entropy = tf.reduce_sum(entropy) / self.num_samples

            val_error = v_estimate - self.returns
            delta = tf.stop_gradient(
                tf.minimum(val_error, tf.zeros_like(self.returns)) * mask)

            # sil_val_loss = self.weights * val_error * delta
            sil_val_loss = val_error * delta
            sil_val_loss = tf.reduce_sum(sil_val_loss) / self.num_samples
            sil_val_loss *= 0.5

            self.total_loss_sil = sil_pg_loss - entropy * entropy_beta \
                + sil_val_loss * critic_lr
            self.total_loss_sil *= w_loss

    def build_grad_cam_grads(self):
        """Compute gradients for Grad-CAM.

        Reference:
        https://github.com/hiveml/tensorflow-grad-cam/blob/master/main.py
        """
        with tf.name_scope("GradCAM_Loss"):
            # We only care about target visualization class.
            signal = tf.multiply(self.logits, self.a)
            y_c = tf.reduce_sum(signal, axis=1)

            if self.use_mnih_2015:
                self.conv_layer = self.h_conv3
            else:
                self.conv_layer = self.h_conv2

            grads = tf.gradients(y_c, self.conv_layer)[0]
            # Normalizing the gradients
            self.grad_cam_grads = tf.div(
                grads,
                tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    @abstractmethod
    def run_policy_and_value(self, sess, s_t):
        raise NotImplementedError()

    @abstractmethod
    def run_policy(self, sess, s_t):
        raise NotImplementedError()

    @abstractmethod
    def run_value(self, sess, s_t):
        raise NotImplementedError()

    @abstractmethod
    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_network, name=None, upper_layers_only=False):
        if upper_layers_only:
            src_vars = src_network.get_vars_upper()
            dst_vars = self.get_vars_upper()
        else:
            src_vars = src_network.get_vars()
            dst_vars = self.get_vars()

        sync_ops = []
        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def conv_variable(self, shape, layer_name='conv', gain=1.0):
        """Return weights and biases for convolutional 2D layer.

        Keyword arguments:
        shape -- [kernel_height, kernel_width, in_channel, out_channel]
        layer_name -- name of variables in the layer
        gain -- argument for orthogonal initializer (default 1.0)
        """
        with tf.variable_scope(layer_name):
            weight = tf.get_variable(
                'weights', shape,
                initializer=tf.orthogonal_initializer(gain=gain))
            bias = tf.get_variable(
                'biases', [shape[3]],
                initializer=tf.zeros_initializer())
        return weight, bias

    def fc_variable(self, shape, layer_name='fc', gain=1.0):
        """Return weights and biases for dense layer.

        Keyword arguments:
        shape -- [# of units in, # of units out]
        layer_name -- name of variables in the layer
        gain -- argument for orthogonal initializer (default 1.0)
        """
        with tf.variable_scope(layer_name):
            weight = tf.get_variable(
                'weights', shape,
                initializer=tf.orthogonal_initializer(gain=gain))
            bias = tf.get_variable(
                'biases', [shape[1]], initializer=tf.zeros_initializer())
        return weight, bias

    def conv2d(self, x, W, stride, data_format='NHWC', padding="VALID",
               name=None):
        """Return convolutional 2d layer.

        Keyword arguments:
        x -- input
        W -- weights of layer with the shape
            [kernel_height, kernel_width, in_channel, out_channel]
        stride -- stride
        data_format -- NHWC or NCHW (default NHWC)
        padding -- SAME or VALID (default VALID)
        """
        return tf.nn.conv2d(
            x, W, strides=[1, stride, stride, 1], padding=padding,
            # use_cudnn_on_gpu=self.use_gpu,
            data_format=data_format, name=name)

    def load_transfer_model(self, sess, folder=None, not_transfer_fc2=False,
                            not_transfer_fc1=False, not_transfer_conv3=False,
                            not_transfer_conv2=False, var_list=None):
        """Load model from pre-trained network."""
        assert folder is not None
        print(folder)
        assert folder.is_dir()

        assert self._thread_index == -1  # only load model to global network
        # assert self._thread_index < 0  # only load model to global network

        logger.info("Initialize network from a pretrain"
                    " model in {}".format(folder))

        transfer_all = False
        if not_transfer_conv2:
            folder /= 'noconv2'
        elif not_transfer_conv3:
            folder /= 'noconv3'
        elif not_transfer_fc1:
            folder /= 'nofc1'
        elif not_transfer_fc2:
            folder /= 'nofc2'
        else:
            transfer_all = True
            # max_value_file = folder / "max_output_value"
            # with max_value_file.open('r') as f_max_value:
            #     transfer_max_output_val = float(f_max_value.readline().split()[0])
            folder /= 'all'

        saver_transfer_from = tf.train.Saver(var_list=var_list)
        checkpoint_transfer_from = tf.train.get_checkpoint_state(str(folder))

        if checkpoint_transfer_from and checkpoint_transfer_from.model_checkpoint_path:
            saver_transfer_from.restore(sess, checkpoint_transfer_from.model_checkpoint_path)
            logger.info("Successfully loaded: {}".format(checkpoint_transfer_from.model_checkpoint_path))

            global_vars = tf.global_variables()
            is_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if f]
            for var in initialized_vars:
                logger.info("    {} loaded".format(var.op.name))
                sleep(1)

            if transfer_all:
                # scale down last layer if it's transferred
                # logger.info("Normalizing output layer with max value"
                #            " {}...".format(transfer_max_output_val))
                # W_fc2_norm = tf.div(self.W_fc2, transfer_max_output_val)
                # b_fc2_norm = tf.div(self.b_fc2, transfer_max_output_val)

                logger.info("Normalizing fc2 output layer...")
                maxW = tf.abs(tf.reduce_max(self.W_fc2))
                minW = tf.abs(tf.reduce_min(self.W_fc2))
                maxAbsW = tf.maximum(maxW, minW)
                W_fc2_norm = tf.div(self.W_fc2, maxAbsW)

                maxb = tf.abs(tf.reduce_max(self.b_fc2))
                minb = tf.abs(tf.reduce_min(self.b_fc2))
                maxAbsb = tf.maximum(maxb, minb)
                b_fc2_norm = tf.div(self.b_fc2, maxAbsb)
                sess.run([
                    self.W_fc2.assign(W_fc2_norm),
                    self.b_fc2.assign(b_fc2_norm),
                    ])

                # if '_slv' in str(folder):
                #     logger.info("Normalizing fc3 output layer...")
                #     maxW = tf.abs(tf.reduce_max(self.W_fc3))
                #     minW = tf.abs(tf.reduce_min(self.W_fc3))
                #     maxAbsW = tf.maximum(maxW, minW)
                #     W_fc3_norm = tf.div(self.W_fc3, maxAbsW)
                #
                #     maxb = tf.abs(tf.reduce_max(self.b_fc3))
                #     minb = tf.abs(tf.reduce_min(self.b_fc3))
                #     maxAbsb = tf.maximum(maxb, minb)
                #     b_fc3_norm = tf.div(self.b_fc3, maxAbsb)
                #
                #     sess.run([
                #         self.W_fc3.assign(W_fc3_norm),
                #         self.b_fc3.assign(b_fc3_norm),
                #         ])

                logger.info("Output layer(s) normalized")


class GameACFFNetwork(GameACNetwork):
    """Actor-Critic Feedforward Network class."""

    def __init__(self, action_size, thread_index,  # -1 for global
                 device="/cpu:0", padding="VALID", in_shape=(84, 84, 4)):
        """Initialize GameACFFNetwork class."""
        GameACNetwork.__init__(self, action_size, thread_index, device)
        logger.info("use_mnih_2015: {}".format(
            colored(self.use_mnih_2015,
                    "green" if self.use_mnih_2015 else "red")))
        logger.info("padding: {}".format(padding))
        logger.info("in_shape: {}".format(in_shape))
        scope_name = "net_" + str(self._thread_index)
        self.last_hidden_fc_output_size = 512
        self.in_shape = in_shape

        # state (input)
        self.s = tf.placeholder(
            tf.float32, [None] + list(in_shape), name="state")
        self.s_n = tf.div(self.s, 255.)

        with tf.device(self._device), tf.variable_scope(scope_name):
            if self.use_mnih_2015:
                self.W_conv1, self.b_conv1 = self.conv_variable(
                    [8, 8, 4, 32], layer_name='conv1', gain=np.sqrt(2))
                self.W_conv2, self.b_conv2 = self.conv_variable(
                    [4, 4, 32, 64], layer_name='conv2', gain=np.sqrt(2))
                self.W_conv3, self.b_conv3 = self.conv_variable(
                    [3, 3, 64, 64], layer_name='conv3', gain=np.sqrt(2))

                # 3136 for VALID padding and 7744 for SAME padding
                fc1_size = 3136 if padding == 'VALID' else 7744
                self.W_fc1, self.b_fc1 = self.fc_variable(
                    [fc1_size, self.last_hidden_fc_output_size],
                    layer_name='fc1', gain=np.sqrt(2))
            else:
                logger.warn("Does not support SAME padding")
                assert padding == 'VALID'
                self.W_conv1, self.b_conv1 = self.conv_variable(
                    [8, 8, 4, 16], layer_name='conv1', gain=np.sqrt(2))
                self.W_conv2, self.b_conv2 = self.conv_variable(
                    [4, 4, 16, 32], layer_name='conv2', gain=np.sqrt(2))
                fc1_size = 2592
                self.W_fc1, self.b_fc1 = self.fc_variable(
                    [fc1_size, self.last_hidden_fc_output_size],
                    layer_name='fc1', gain=np.sqrt(2))

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self.fc_variable(
                [self.last_hidden_fc_output_size, action_size],
                layer_name='fc2')

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self.fc_variable(
                [self.last_hidden_fc_output_size, 1], layer_name='fc3')

            if self.use_mnih_2015:
                self.h_conv1 = tf.nn.relu(
                    self.conv2d(self.s_n,  self.W_conv1, 4, padding=padding)
                    + self.b_conv1)
                self.h_conv2 = tf.nn.relu(
                    self.conv2d(self.h_conv1, self.W_conv2, 2, padding=padding)
                    + self.b_conv2)
                self.h_conv3 = tf.nn.relu(
                    self.conv2d(self.h_conv2, self.W_conv3, 1, padding=padding)
                    + self.b_conv3)

                self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, fc1_size])
                self.h_fc1 = tf.nn.relu(
                    tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
            else:
                self.h_conv1 = tf.nn.relu(
                    self.conv2d(self.s_n,  self.W_conv1, 4, padding=padding)
                    + self.b_conv1)
                self.h_conv2 = tf.nn.relu(
                    self.conv2d(self.h_conv1, self.W_conv2, 2, padding=padding)
                    + self.b_conv2)

                h_conv2_flat = tf.reshape(self.h_conv2, [-1, fc1_size])
                self.h_fc1 = tf.nn.relu(
                    tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            # policy (output)
            self.logits = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
            self.pi = tf.nn.softmax(self.logits)
            # value (output)
            self.v = tf.matmul(self.h_fc1, self.W_fc3) + self.b_fc3
            self.v0 = self.v[:, 0]

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out, logits = sess.run(
            [self.pi, self.v0, self.logits], feed_dict={self.s: [s_t]})
        return (pi_out[0], v_out[0], logits[0])

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def run_value(self, sess, s_t):
        v_out = sess.run(self.v0, feed_dict={self.s: [s_t]})
        return v_out[0]

    def evaluate_grad_cam(self, sess, state, action):
        activations, gradients = sess.run(
            [self.conv_layer, self.grad_cam_grads],
            feed_dict={self.s: [state], self.a: [action]})
        return activations[0], gradients[0]

    def get_vars(self):
        if self.use_mnih_2015:
            return [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3,
                ]
        else:
            return [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3,
                ]

    def get_vars_upper(self):
        return [
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3,
            ]
