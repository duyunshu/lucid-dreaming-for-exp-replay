#!/usr/bin/env python3
import cv2
import logging
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from common.game_state import GameState
from common.game_state import get_wrapper_by_name
from common.util import convert_onehot_to_a
from termcolor import colored
from queue import Queue, PriorityQueue
from copy import deepcopy
from common_worker import CommonWorker
from sil_memory import SILReplayMemory
from datetime import datetime

import random

logger = logging.getLogger("SIL_training_thread")


class SILTrainingThread(CommonWorker):
    """Asynchronous Actor-Critic Training Thread Class."""

    entropy_beta = 0.01
    gamma = 0.99
    finetune_upper_layers_only = False
    transformed_bellman = False
    clip_norm = 0.5
    use_grad_cam = False

    def __init__(self, thread_index, global_net, local_net,
                 initial_learning_rate, learning_rate_input, grad_applier,
                 device=None, batch_size=None,
                 use_rollout=False, one_buffer=False, sampleR=False):
        """Initialize A3CTrainingThread class."""
        assert self.action_size != -1

        self.is_sil_thread = True
        self.thread_idx = thread_index
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_input = learning_rate_input
        self.local_net = local_net
        self.batch_size = batch_size
        self.use_rollout = use_rollout
        self.one_buffer = one_buffer
        self.sampleR = sampleR

        logger.info("===SIL thread_index: {}===".format(self.thread_idx))
        logger.info("device: {}".format(device))
        logger.info("action_size: {}".format(self.action_size))
        logger.info("entropy_beta: {}".format(self.entropy_beta))
        logger.info("gamma: {}".format(self.gamma))
        logger.info("reward_type: {}".format(self.reward_type))
        logger.info("transformed_bellman: {}".format(
            colored(self.transformed_bellman,
                    "green" if self.transformed_bellman else "red")))
        logger.info("clip_norm: {}".format(self.clip_norm))
        logger.info("use_grad_cam: {}".format(colored(self.use_grad_cam,
                    "green" if self.use_grad_cam else "red")))

        reward_clipped = True if self.reward_type == 'CLIP' else False

        local_vars = self.local_net.get_vars

        with tf.device(device):
            critic_lr = 0.1
            entropy_beta = 0
            w_loss = 1.0
            logger.info("sil batch_size: {}".format(self.batch_size))
            logger.info("sil w_loss: {}".format(w_loss))
            logger.info("sil critic_lr: {}".format(critic_lr))
            logger.info("sil entropy_beta: {}".format(entropy_beta))
            self.local_net.prepare_sil_loss(entropy_beta=entropy_beta,
                                            w_loss=w_loss,
                                            critic_lr=critic_lr)
            var_refs = [v._ref() for v in local_vars()]

            self.sil_gradients = tf.gradients(
                self.local_net.total_loss_sil, var_refs)

        global_vars = global_net.get_vars

        with tf.device(device):
            if self.clip_norm is not None:
                self.sil_gradients, grad_norm = tf.clip_by_global_norm(
                    self.sil_gradients, self.clip_norm)
            sil_gradients_global = list(
                zip(self.sil_gradients, global_vars()))
            sil_gradients_local = list(
                zip(self.sil_gradients, local_vars()))
            self.sil_apply_gradients = grad_applier.apply_gradients(
                sil_gradients_global)
            self.sil_apply_gradients_local = grad_applier.apply_gradients(
                sil_gradients_local)

        self.sync = self.local_net.sync_from(global_net)

        self.episode = SILReplayMemory(
            self.action_size, max_len=None, gamma=self.gamma,
            clip=reward_clipped,
            height=self.local_net.in_shape[0],
            width=self.local_net.in_shape[1],
            phi_length=self.local_net.in_shape[2],
            reward_constant=self.reward_constant)

        # temp_buffer for mixing and re-sample (brown arrow in Figure 1)
        # initial only when needed (A3CTBSIL & LiDER-OneBuffer does not need temp_buffer)
        self.temp_buffer = None
        if (self.use_rollout) and (not self.one_buffer):
            self.temp_buffer = SILReplayMemory(
                self.action_size, max_len=self.batch_size*2, gamma=self.gamma,
                clip=reward_clipped,
                height=self.local_net.in_shape[0],
                width=self.local_net.in_shape[1],
                phi_length=self.local_net.in_shape[2],priority=True,
                reward_constant=self.reward_constant)


    def record_sil(self, sil_ctr=0, total_used=0, num_a3c_used=0, a3c_used_return=0,
                   rollout_used=0, rollout_used_return=0,
                   old_used=0, global_t=0, mode='SIL'):
        """Record SIL."""
        summary = tf.Summary()
        summary.value.add(tag='{}/sil_ctr'.format(mode),
                          simple_value=float(sil_ctr))
        summary.value.add(tag='{}/total_num_sample_used'.format(mode),
                          simple_value=float(total_used))

        summary.value.add(tag='{}/num_a3c_used'.format(mode),
                          simple_value=float(num_a3c_used))
        summary.value.add(tag='{}/a3c_used_return'.format(mode),
                          simple_value=float(a3c_used_return))

        summary.value.add(tag='{}/num_rollout_used'.format(mode),
                          simple_value=float(rollout_used_return))
        summary.value.add(tag='{}/rollout_used_return'.format(mode),
                          simple_value=float(rollout_used))

        summary.value.add(tag='{}/num_old_used'.format(mode),
                          simple_value=float(old_used))

        self.writer.add_summary(summary, global_t)
        self.writer.flush()


    def sil_train(self, sess, global_t, sil_memory, m, rollout_buffer=None):
        """Self-imitation learning process."""
        # copy weights from shared to local
        sess.run(self.sync)
        cur_learning_rate = self._anneal_learning_rate(global_t,
                                                       self.initial_learning_rate)

        local_sil_ctr = 0
        local_sil_a3c_used, local_sil_a3c_used_return = 0, 0
        local_sil_rollout_used, local_sil_rollout_used_return = 0, 0
        local_sil_old_used = 0

        total_used = 0
        num_a3c_used = 0
        num_rollout_used = 0
        num_old_used = 0

        for _ in range(m):
            d_batch_size, r_batch_size = 0, 0
            # A3CTBSIL
            if not self.use_rollout:
                d_batch_size = self.batch_size
            # or LiDER-OneBuffer
            elif self.use_rollout and self.one_buffer:
                d_batch_size = self.batch_size
            # or LiDER
            else:
                assert rollout_buffer is not None
                assert self.temp_buffer is not None
                self.temp_buffer.reset()
                if not self.sampleR: # otherwise, LiDER-SampleR
                    d_batch_size = self.batch_size
                r_batch_size = self.batch_size

            batch_state, batch_action, batch_returns, batch_fullstate, \
                batch_rollout, batch_refresh, weights = ([] for i in range(7))

            # sample from buffer D
            if d_batch_size > 0 and len(sil_memory) > d_batch_size:
                d_sample = sil_memory.sample(d_batch_size , beta=0.4)
                d_index_list, d_batch, d_weights = d_sample
                d_batch_state, d_action, d_batch_returns, \
                    d_batch_fullstate, d_batch_rollout, d_batch_refresh = d_batch
                # update priority of sampled experiences
                self.update_priorities_once(sess, sil_memory, d_index_list,
                                            d_batch_state, d_action, d_batch_returns)

                if self.temp_buffer is not None: # when LiDER
                    d_batch_action = convert_onehot_to_a(d_action)
                    self.temp_buffer.extend_one_priority(d_batch_state, d_batch_fullstate,
                        d_batch_action, d_batch_returns, d_batch_rollout, d_batch_refresh)
                else: # when A3CTBSIL or LiDER-OneBuffer
                    batch_state.extend(d_batch_state)
                    batch_action.extend(d_action)
                    batch_returns.extend(d_batch_returns)
                    batch_fullstate.extend(d_batch_fullstate)
                    batch_rollout.extend(d_batch_rollout)
                    batch_refresh.extend(d_batch_refresh)
                    weights.extend(d_weights)

            # sample from buffer R
            if r_batch_size > 0 and len(rollout_buffer) > r_batch_size:
                r_sample = rollout_buffer.sample(r_batch_size, beta=0.4)
                r_index_list, r_batch, r_weights = r_sample
                r_batch_state, r_action, r_batch_returns, \
                    r_batch_fullstate, r_batch_rollout, r_batch_refresh = r_batch
                # update priority of sampled experiences
                self.update_priorities_once(sess, rollout_buffer, r_index_list,
                                            r_batch_state, r_action, r_batch_returns)

                if self.temp_buffer is not None: # when LiDER
                    r_batch_action = convert_onehot_to_a(r_action)
                    self.temp_buffer.extend_one_priority(r_batch_state, r_batch_fullstate,
                        r_batch_action, r_batch_returns, r_batch_rollout, r_batch_refresh)
                else: # when A3CTBSIL or LiDER-OneBuffer
                    batch_state.extend(r_batch_state)
                    batch_action.extend(r_action)
                    batch_returns.extend(r_batch_returns)
                    batch_fullstate.extend(r_batch_fullstate)
                    batch_rollout.extend(r_batch_rollout)
                    batch_refresh.extend(r_batch_refresh)
                    weights.extend(r_weights)

            # LiDER only: pick 32 out of mixed
            # (at the beginning the 32 could all from buffer D since rollout has no data yet)
            # make sure the temp_buffer has been filled with at least size of one batch before sampling
            if self.temp_buffer is not None and len(self.temp_buffer) >= self.batch_size:
                sample = self.temp_buffer.sample(self.batch_size, beta=0.4)
                index_list, batch, weights = sample
                # overwrite the initial empty list
                batch_state, batch_action, batch_returns, \
                    batch_fullstate, batch_rollout, batch_refresh = batch

            # sil policy update (if one full batch is sampled)
            if len(batch_state) == self.batch_size:
                feed_dict = {
                    self.local_net.s: batch_state,
                    self.local_net.a_sil: batch_action,
                    self.local_net.returns: batch_returns,
                    self.local_net.weights: weights,
                    self.learning_rate_input: cur_learning_rate,
                    }
                fetch = [
                    self.local_net.clipped_advs,
                    self.local_net.advs,
                    self.sil_apply_gradients,
                    self.sil_apply_gradients_local,
                    ]
                adv_clip, adv, _, _ = sess.run(fetch, feed_dict=feed_dict)
                pos_idx = [i for (i, num) in enumerate(adv) if num > 0]
                neg_idx = [i for (i, num) in enumerate(adv) if num <= 0]
                # log number of samples used for SIL updates
                total_used += len(pos_idx)
                num_rollout_used += np.sum(np.take(batch_rollout, pos_idx))
                num_a3c_used += (len(pos_idx) - np.sum(np.take(batch_rollout, pos_idx)))
                num_old_used += np.sum(np.take(batch_refresh, pos_idx))
                # return for used rollout samples
                rollout_idx = [i for (i, num) in enumerate(batch_rollout) if num > 0]
                pos_rollout_idx = np.intersect1d(rollout_idx, pos_idx)
                if len(pos_rollout_idx) > 0:
                    local_sil_rollout_used_return += np.sum(np.take(adv, pos_rollout_idx))
                # return for used a3c samples
                a3c_idx = [i for (i, num) in enumerate(batch_rollout) if num <= 0]
                pos_a3c_idx = np.intersect1d(a3c_idx, pos_idx)
                if len(pos_a3c_idx) > 0:
                    local_sil_a3c_used_return += np.sum(np.take(batch_returns, pos_a3c_idx))

                local_sil_ctr += 1

        local_sil_a3c_used += num_a3c_used
        local_sil_rollout_used += num_rollout_used
        local_sil_old_used += num_old_used

        return local_sil_ctr, local_sil_a3c_used, local_sil_a3c_used_return, \
               local_sil_rollout_used, local_sil_rollout_used_return, \
               local_sil_old_used

    def update_priorities_once(self, sess, memory, index_list, batch_state,
                               batch_action, batch_returns):
        """Self-imitation update priorities once."""
        # copy weights from shared to local
        sess.run(self.sync)

        feed_dict = {
            self.local_net.s: batch_state,
            self.local_net.a_sil: batch_action,
            self.local_net.returns: batch_returns,
            }
        fetch = self.local_net.clipped_advs
        adv_clip = sess.run(fetch, feed_dict=feed_dict)
        memory.set_weights(index_list, adv_clip)
