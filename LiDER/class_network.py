#!/usr/bin/env python3
"""Network model.

This module defines the network architecture used in the classification
training of the human demonstration data.

"""
import logging
import numpy as np
import tensorflow as tf

from abc import ABC
from abc import abstractmethod
from termcolor import colored

logger = logging.getLogger("network")


class Network(ABC):
    """Network base class."""

    use_mnih_2015 = False
    l1_beta = 0.
    l2_beta = 0.
    use_gpu = True

    def __init__(self, action_size, thread_index, device=None):
        """Initialize Network base class."""
        self.action_size = action_size
        self._thread_index = thread_index
        self._device = device


    @abstractmethod
    def prepare_loss(self):
        """Prepare tf operations training loss."""
        raise NotImplementedError()

    @abstractmethod
    def prepare_evaluate(self):
        """Prepare tf operations for evaluation."""
        raise NotImplementedError()

    @abstractmethod
    def load(self, sess, checkpoint):
        """Load existing model."""
        raise NotImplementedError()

    @abstractmethod
    def run_policy(self, sess, s_t):
        """Infer network output based on input s_t."""
        raise NotImplementedError()

    @abstractmethod
    def get_vars(self):
        """Return list of variables in the network."""
        raise NotImplementedError()

    def conv_variable(self, shape, layer_name='conv', gain=1.0, decoder=False):
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
                'biases', [shape[3] if not decoder else shape[2]],
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

    def conv2d_transpose(self, x, W, output_shape, stride, data_format='NHWC',
                         padding='VALID', name=None):
        """Return transpose convolutional 2d layer.

        Keyword arguments:
        x -- input
        W -- weights of layer with the shapes
            [kernel_height, kernel_width, in_channel, out_channel]
        output_shape -- shape of the output
        stride -- stride
        data_format -- NHWC or NCHW (default NHWC)
        padding -- SAME or VALID (default VALID)
        """
        return tf.nn.conv2d_transpose(
            x, W, output_shape, strides=[1, stride, stride, 1],
            padding=padding, data_format=data_format, name=name)

    def sync_from(self, src_network, name=None, upper_layers_only=False):
        if upper_layers_only:
            src_vars = src_network.get_vars_upper()
            dst_vars = self.get_vars_upper()
        else:
            src_vars = src_network.get_vars()
            dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "Network", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def build_grad_cam_grads(self):
        """Compute Grad-CAM from last convolutional layer after activation.

        Source:
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
                grads, tf.sqrt(tf.reduce_mean(tf.square(grads)))
                + tf.constant(1e-5))

    def evaluate_grad_cam(self, sess, state, action):
        """Return activation and Grad-CAM of last convolutional layer.

        Keyword arguments:
        sess -- tf session
        state -- network input image
        action -- class label
        """
        activations, gradients = sess.run(
            [self.conv_layer, self.grad_cam_grads],
            feed_dict={self.s: [state], self.a: [action]})
        return activations[0], gradients[0]

    def prepare_slv_loss(self, sl_xentropy, sl_loss_weight, val_weight):
        """Prepare self-imitation loss."""
        self.returns = tf.placeholder(
            tf.float32, shape=[None], name="sil_loss")

        v_estimate = tf.squeeze(self.v)
        advs = self.returns - v_estimate
        clipped_advs = tf.stop_gradient(
            tf.maximum(tf.zeros_like(advs), advs))

        # sl_loss = sl_xentropy
        sl_loss = sl_xentropy * clipped_advs

        val_loss = tf.squared_difference(
            tf.squeeze(self.v), self.returns) / 2.0

        loss = tf.reduce_mean(sl_loss) * sl_loss_weight
        loss += tf.reduce_mean(val_loss) * val_weight

        return sl_loss, val_loss, loss


class MultiClassNetwork(Network):
    """Multi-class Classification Network."""

    def __init__(self, pretrain_graph, action_size, thread_index,
                 padding="VALID", in_shape=(84, 84, 4), sae=False,
                 tied_weights=False, use_denoising=False, noise_factor=0.3,
                 loss_function='mse', use_slv=False, device=None):
        """Initialize MultiClassNetwork class."""
        Network.__init__(self, action_size, thread_index, device)
        self.graph = pretrain_graph
        logger.info("network: MultiClassNetwork")
        logger.info("device: {}".format(self._device))
        logger.info("action_size: {}".format(self.action_size))
        logger.info("use_mnih_2015: {}".format(
            colored(self.use_mnih_2015,
                    "green" if self.use_mnih_2015 else "red")))
        logger.info("L1_beta: {}".format(
            colored(self.l1_beta, "green" if self.l1_beta > 0. else "red")))
        logger.info("L2_beta: {}".format(
            colored(self.l2_beta, "green" if self.l2_beta > 0. else "red")))
        logger.info("padding: {}".format(padding))
        logger.info("in_shape: {}".format(in_shape))
        logger.info("use_slv: {}".format(
            colored(use_slv, "green" if use_slv else "red")))
        scope_name = "net_" + str(self._thread_index)
        self.last_hidden_fc_output_size = 512
        self.in_shape = in_shape
        self.use_slv = use_slv

        with self.graph.as_default():
            # state (input)
            self.s = tf.placeholder(tf.float32, [None] + list(self.in_shape))
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
                    # logger.warn("Does not support SAME padding")
                    # assert self.padding == 'VALID'
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

                if self.use_slv:
                    # weight for value output layer
                    self.W_fc3, self.b_fc3 = self.fc_variable(
                        [self.last_hidden_fc_output_size, 1], layer_name='fc3')

                if self.use_mnih_2015:
                    h_conv1 = tf.nn.relu(self.conv2d(
                        self.s_n,  self.W_conv1, 4, padding=padding)
                        + self.b_conv1)

                    h_conv2 = tf.nn.relu(self.conv2d(
                        h_conv1, self.W_conv2, 2, padding=padding)
                        + self.b_conv2)

                    self.h_conv3 = tf.nn.relu(self.conv2d(
                        h_conv2, self.W_conv3, 1, padding=padding)
                        + self.b_conv3)

                    h_conv3_flat = tf.reshape(self.h_conv3, [-1, fc1_size])
                    h_fc1 = tf.nn.relu(tf.matmul(
                        h_conv3_flat, self.W_fc1) + self.b_fc1)
                else:
                    h_conv1 = tf.nn.relu(self.conv2d(
                        self.s_n,  self.W_conv1, 4, padding=padding)
                        + self.b_conv1)

                    self.h_conv2 = tf.nn.relu(self.conv2d(
                        h_conv1, self.W_conv2, 2, padding=padding)
                        + self.b_conv2)

                    h_conv2_flat = tf.reshape(self.h_conv2, [-1, fc1_size])
                    h_fc1 = tf.nn.relu(tf.matmul(
                        h_conv2_flat, self.W_fc1) + self.b_fc1)

                # policy (output)
                self.logits = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
                self.pi = tf.nn.softmax(self.logits)
                self.max_value = tf.reduce_max(self.logits, axis=None)

                if self.use_slv:
                    # value (output)
                    self.v = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
                    self.v0 = self.v[:, 0]

                self.saver = tf.train.Saver()

    def prepare_loss(self, sl_loss_weight=1.0, val_weight=0.01, min_batch_size=4):
        """Prepare tf operations training loss."""
        with self.graph.as_default():
            with tf.device(self._device), tf.name_scope("class-Loss"):
                # taken action (input for policy)
                self.a = tf.placeholder(tf.float32,
                                        shape=[None, self.action_size])

                sl_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.a, logits=self.logits)

                if self.use_slv:
                    sl_loss, val_loss, self.sl_loss = \
                        self.prepare_slv_loss(sl_xentropy, sl_loss_weight,
                                              val_weight)
                else:
                    self.sl_loss = tf.reduce_mean(sl_xentropy)

                self.total_loss = self.sl_loss

                net_vars = self.get_vars_no_bias()
                if self.l1_beta > 0:
                    l1_loss = tf.add_n(
                        [tf.reduce_sum(tf.abs(net_vars[i]))
                         for i in range(len(net_vars))]) * self.l1_beta
                    self.total_loss += l1_loss

                if self.l2_beta > 0:
                    l2_loss = tf.add_n(
                        [tf.nn.l2_loss(net_vars[i])
                         for i in range(len(net_vars))]) * self.l2_beta
                    self.total_loss += l2_loss


    def run_policy(self, sess, s_t):
        """Infer network output based on input s_t."""
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]


    def get_vars(self):
        """Return list of variables in the network."""
        if self.use_mnih_2015:
            vars = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                ]
        else:
            vars = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                ]

        if self.use_slv:
            vars.extend([self.W_fc3, self.b_fc3])

        return vars


    def get_vars_no_bias(self):
        """Return list of variables in the network excluding bias."""
        if self.use_mnih_2015:
            vars = [
                self.W_conv1, self.W_conv2,
                self.W_conv3, self.W_fc1, self.W_fc2,
                ]
        else:
            vars = [self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2]

        if self.use_slv:
            vars.extend([self.W_fc3])

        return vars


    def load(self, sess=None, checkpoint_dir=''):
        """Load existing model."""
        assert sess is not None
        assert checkpoint_dir != ''
        self.saver.restore(sess, checkpoint_dir)
        logger.info("Successfully loaded pretrained model from: {}".format(checkpoint_dir))


    def prepare_evaluate(self):
        """Prepare tf operations for evaluation."""
        with tf.device(self._device), self.graph.as_default():
            correct_prediction = tf.equal(
                tf.argmax(self.logits, 1), tf.argmax(self.a, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
