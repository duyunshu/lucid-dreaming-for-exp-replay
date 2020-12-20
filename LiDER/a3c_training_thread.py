#!/usr/bin/env python3
import cv2
import logging
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from common.game_state import GameState
from common.game_state import get_wrapper_by_name
from common.replay_memory import ReplayMemory
from common.util import generate_image_for_cam_video
from common.util import grad_cam
from common.util import make_movie
from common.util import transform_h
from common.util import transform_h_inv
from common.util import visualize_cam
from sil_memory import SILReplayMemory
from termcolor import colored
from queue import Queue
from copy import deepcopy
from common_worker import CommonWorker

logger = logging.getLogger("a3c_training_thread")


class A3CTrainingThread(CommonWorker):
    """Asynchronous Actor-Critic Training Thread Class."""
    log_interval = 100
    perf_log_interval = 1000
    local_t_max = 20
    entropy_beta = 0.01
    gamma = 0.99
    shaping_actions = -1  # -1 all actions, 0 exclude noop
    transformed_bellman = False
    clip_norm = 0.5
    use_grad_cam = False
    use_sil = False
    log_idx = 0
    reward_constant = 0

    def __init__(self, thread_index, global_net, local_net,
                 initial_learning_rate, learning_rate_input, grad_applier,
                 device=None, no_op_max=30):
        """Initialize A3CTrainingThread class."""
        assert self.action_size != -1

        self.is_sil_thread = False
        self.is_refresh_thread = False

        self.thread_idx = thread_index
        self.learning_rate_input = learning_rate_input
        self.local_net = local_net

        self.no_op_max = no_op_max
        self.override_num_noops = 0 if self.no_op_max == 0 else None

        logger.info("===A3C thread_index: {}===".format(self.thread_idx))
        logger.info("device: {}".format(device))
        logger.info("use_sil: {}".format(
            colored(self.use_sil, "green" if self.use_sil else "red")))
        logger.info("local_t_max: {}".format(self.local_t_max))
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
            self.local_net.prepare_loss(
                entropy_beta=self.entropy_beta, critic_lr=0.5)
            var_refs = [v._ref() for v in local_vars()]

            self.gradients = tf.gradients(
                self.local_net.total_loss, var_refs)

        global_vars = global_net.get_vars

        with tf.device(device):
            if self.clip_norm is not None:
                self.gradients, grad_norm = tf.clip_by_global_norm(
                    self.gradients, self.clip_norm)
            self.gradients = list(zip(self.gradients, global_vars()))
            self.apply_gradients = grad_applier.apply_gradients(self.gradients)

        self.sync = self.local_net.sync_from(global_net)

        self.game_state = GameState(env_id=self.env_id, display=False,
                                    no_op_max=self.no_op_max, human_demo=False,
                                    episode_life=True,
                                    override_num_noops=self.override_num_noops)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0
        self.episode_steps = 0

        # variable controlling log output
        self.prev_local_t = 0

        with tf.device(device):
            if self.use_grad_cam:
                self.action_meaning = self.game_state.env.unwrapped \
                    .get_action_meanings()
                self.local_net.build_grad_cam_grads()

        if self.use_sil:
            self.episode = SILReplayMemory(
                self.action_size, max_len=None, gamma=self.gamma,
                clip=reward_clipped,
                height=self.local_net.in_shape[0],
                width=self.local_net.in_shape[1],
                phi_length=self.local_net.in_shape[2],
                reward_constant=self.reward_constant)


    def train(self, sess, global_t, train_rewards):
        """Train A3C."""
        states = []
        fullstates=[]
        actions = []
        rewards = []
        values = []
        rho = []

        terminal_pseudo = False  # loss of life
        terminal_end = False  # real terminal

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        # t_max times loop
        for i in range(self.local_t_max):
            state = cv2.resize(self.game_state.s_t,
                               self.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            fullstate = self.game_state.clone_full_state()

            pi_, value_, logits_ = self.local_net.run_policy_and_value(sess,
                                                                       state)
            action = self.pick_action(logits_)

            states.append(state)
            fullstates.append(fullstate)
            actions.append(action)
            values.append(value_)

            if self.thread_idx == self.log_idx \
               and self.local_t % self.log_interval == 0:
                log_msg1 = "lg={}".format(np.array_str(
                    logits_, precision=4, suppress_small=True))
                log_msg2 = "pi={}".format(np.array_str(
                    pi_, precision=4, suppress_small=True))
                log_msg3 = "V={:.4f}".format(value_)
                logger.debug(log_msg1)
                logger.debug(log_msg2)
                logger.debug(log_msg3)

            # process game
            self.game_state.step(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward

            if self.use_sil:
                # save states in episode memory
                self.episode.add_item(self.game_state.s_t, fullstate,
                                      action, reward, terminal)

            if self.reward_type == 'CLIP':
                reward = np.sign(reward)

            rewards.append(reward)

            self.local_t += 1
            self.episode_steps += 1
            global_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_pseudo = True

                env = self.game_state.env
                name = 'EpisodicLifeEnv'
                if get_wrapper_by_name(env, name).was_real_done:
                    # reduce log freq
                    if self.thread_idx == self.log_idx:
                        log_msg = "train: worker={} global_t={} local_t={}".format(
                            self.thread_idx, global_t, self.local_t)
                        score_str = colored("score={}".format(
                            self.episode_reward), "magenta")
                        steps_str = colored("steps={}".format(
                            self.episode_steps), "blue")
                        log_msg += " {} {}".format(score_str, steps_str)
                        logger.debug(log_msg)

                    train_rewards['train'][global_t] = (self.episode_reward,
                                                        self.episode_steps)
                    self.record_summary(
                        score=self.episode_reward, steps=self.episode_steps,
                        episodes=None, global_t=global_t, mode='Train')
                    self.episode_reward = 0
                    self.episode_steps = 0
                    terminal_end = True

                self.game_state.reset(hard_reset=False)
                break

        cumsum_reward = 0.0
        if not terminal:
            state = cv2.resize(self.game_state.s_t,
                               self.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            cumsum_reward = self.local_net.run_value(sess, state)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_state = []
        batch_action = []
        batch_adv = []
        batch_cumsum_reward = []

        # compute and accumulate gradients
        for(ai, ri, si, vi) in zip(actions, rewards, states, values):
            if self.transformed_bellman:
                ri = np.sign(ri) * self.reward_constant + ri
                cumsum_reward = transform_h(
                    ri + self.gamma * transform_h_inv(cumsum_reward))
            else:
                cumsum_reward = ri + self.gamma * cumsum_reward
            advantage = cumsum_reward - vi

            # convert action to one-hot vector
            a = np.zeros([self.action_size])
            a[ai] = 1

            batch_state.append(si)
            batch_action.append(a)
            batch_adv.append(advantage)
            batch_cumsum_reward.append(cumsum_reward)

        cur_learning_rate = self._anneal_learning_rate(global_t,
                self.initial_learning_rate )

        feed_dict = {
            self.local_net.s: batch_state,
            self.local_net.a: batch_action,
            self.local_net.advantage: batch_adv,
            self.local_net.cumulative_reward: batch_cumsum_reward,
            self.learning_rate_input: cur_learning_rate,
            }

        sess.run(self.apply_gradients, feed_dict=feed_dict)

        t = self.local_t - self.prev_local_t
        if (self.thread_idx == self.log_idx and t >= self.perf_log_interval):
            self.prev_local_t += self.perf_log_interval
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            logger.info("worker-{}, log_worker-{}".format(self.thread_idx,
                                                          self.log_idx))
            logger.info("Performance : {} STEPS in {:.0f} sec. {:.0f}"
                        " STEPS/sec. {:.2f}M STEPS/hour.".format(
                            global_t,  elapsed_time, steps_per_sec,
                            steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t, terminal_end, terminal_pseudo
