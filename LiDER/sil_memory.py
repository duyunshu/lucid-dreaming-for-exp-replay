#!/usr/bin/env python3
import cv2
import logging
import numpy as np
import random
import time

from common.replay_memory import PrioritizedReplayBuffer
from common.util import transform_h
from common.util import transform_h_inv
from copy import deepcopy

logger = logging.getLogger("sil_memory")


class SILReplayMemory(object):

    def __init__(self, num_actions, max_len=None, gamma=0.99, clip=False,
                 height=84, width=84, phi_length=4, priority=False,
                 reward_constant=0, fs_size=0):
        """Initialize SILReplayMemory class."""
        if priority:
            self.buff = PrioritizedReplayBuffer(max_len, alpha=0.6)
        else:
            self.states = []
            self.actions = []
            self.rewards = []
            self.terminal = []
            self.returns = []
            self.fullstates = []  # for the purpose of restore to a state
            self.from_rollout = [] # record if an experience is from rollout
            self.refreshed = [] # record if an experience has been refreshed (in buffer R)

        self.priority = priority
        self.num_actions = num_actions
        self.maxlen = max_len
        self.gamma = gamma
        self.clip = clip
        self.height = height
        self.width = width
        self.phi_length = phi_length
        self.reward_constant = reward_constant
        self.fs_size = fs_size


    def log(self):
        """Log memory information."""
        logger.info("priority: {}".format(self.priority))
        logger.info("maxlen: {}".format(self.maxlen))
        logger.info("gamma: {}".format(self.gamma))
        logger.info("clip: {}".format(self.clip))
        logger.info("h x w: {} x {}".format(self.height, self.width))
        logger.info("phi_length: {}".format(self.phi_length))
        logger.info("reward_constant: {}".format(self.reward_constant))
        logger.info("memory size: {}".format(self.__len__()))


    def add_item(self, s, fs, a, rew, t, from_rollout=False, refreshed=False):
        """Use only for episode memory."""
        assert len(self.returns) == 0
        assert not self.priority
        if np.shape(s) != self.shape():
            s = cv2.resize(s, (self.height, self.width),
                           interpolation=cv2.INTER_AREA)
        self.states.append(s)
        self.fullstates.append(fs)
        self.actions.append(a)
        self.rewards.append(rew)
        self.terminal.append(t)
        self.from_rollout.append(from_rollout)
        self.refreshed.append(refreshed)


    def get_data(self):
        """Get data."""
        assert not self.priority
        return (deepcopy(self.states),
                deepcopy(self.fullstates),
                deepcopy(self.actions),
                deepcopy(self.rewards),
                deepcopy(self.terminal),
                deepcopy(self.from_rollout),
                deepcopy(self.refreshed))


    def set_data(self, s, fs, a, r, t, from_rollout, refreshed):
        """Set data."""
        assert not self.priority
        self.states = s
        self.fullstates = fs
        self.actions = a
        self.rewards = r
        self.terminal = t
        self.from_rollout = from_rollout
        self.refreshed = refreshed


    def reset(self):
        """Reset memory."""
        if self.priority:
            self.buff .reset()
        else:
            self.states.clear()
            self.fullstates.clear()
            self.actions.clear()
            self.rewards.clear()
            self.terminal.clear()
            self.returns.clear()
            self.from_rollout.clear()
            self.refreshed.clear()


    def shape(self):
        """Return shape of state."""
        return (self.height, self.width, self.phi_length)


    def extend(self, x):
        """Use only in SIL memory."""
        assert x.terminal[-1]  # assert that last state is a terminal state
        if self.fs_size == 0:
            self.fs_size = len(x.fullstates[0])

        x_returns = self.__class__.compute_returns(
            x.rewards, x.terminal, self.gamma, self.clip, self.reward_constant)

        if self.priority:
            data = zip(x.states, x.fullstates, x.actions, x_returns,
                       x.from_rollout, x.refreshed)
            for feature in data:
                self.buff.add(*feature)
        else:
            self.states.extend(x.states)
            self.fullstates.extend(x.fullstates)
            self.actions.extend(x.actions)
            self.returns.extend(x_returns)
            self.from_rollout.extend(x.from_rollout)
            self.refreshed.extend(x.refreshed)

            if len(self) > self.maxlen:
                st_slice = len(self) - self.maxlen
                self.states = self.states[st_slice:]
                self.fullstates = self.fullstates[st_slice:]
                self.actions = self.actions[st_slice:]
                self.returns = self.returns[st_slice:]
                self.from_rollout = self.from_rollout[st_slice:]
                self.refreshed = self.refreshed[st_slice:]
                assert len(self) == self.maxlen

            assert len(self) == len(self.returns) <= self.maxlen

        x.reset()
        assert len(x) == 0


    def extend_one_priority(self, x_states, x_fullstates, x_actions, x_returns, x_rollout, x_refresh):
        """Use only in SIL memory."""
        assert self.priority
        if self.fs_size == 0:
            self.fs_size = len(x_fullstates[0])

        data = zip(x_states, x_fullstates, x_actions, x_returns, x_rollout, x_refresh)
        for feature in data:
            self.buff.add(*feature)


    def extend_one(self, x_states, x_fullstates, x_actions, x_returns, x_rollout, x_refresh):
        """Use only in exp_buffer memory, add a single batch"""
        assert not self.priority

        self.states.extend(x_states)
        self.fullstates.extend(x_fullstates)
        self.actions.extend(x_actions)
        self.returns.extend(x_returns)
        self.from_rollout.extend(x_rollout)
        self.refreshed.extend(x_refresh)

        if len(self) > self.maxlen:
            st_slice = len(self) - self.maxlen
            self.states = self.states[st_slice:]
            self.fullstates = self.fullstates[st_slice:]
            self.actions = self.actions[st_slice:]
            self.returns = self.returns[st_slice:]
            self.from_rollout = self.from_rollout[st_slice:]
            self.refreshed = self.refreshed[st_slice:]
            assert len(self) == self.maxlen

        assert len(self) == len(self.returns) <= self.maxlen


    @staticmethod
    def compute_returns(rewards, terminal, gamma, clip=False, c=1.89):
        """Compute expected return."""
        length = np.shape(rewards)[0]
        returns = np.empty_like(rewards, dtype=np.float32)

        if clip:
            rewards = np.clip(rewards, -1., 1.)
        else:
            # when reward is 1, t(r=1) = 0.412 which is less than half of
            # reward which slows down the training with Atari games with
            # raw rewards at range (-1, 1). To address this down scaled reward,
            # we add the constant c=sign(r) * 1.89 to ensure that
            # t(r=1 + sign(r) * 1.89) ~ 1
            rewards = np.sign(rewards) * c + rewards

        for i in reversed(range(length)):
            if terminal[i]:
                returns[i] = rewards[i] if clip else transform_h(rewards[i])
            else:
                if clip:
                    returns[i] = rewards[i] + gamma * returns[i+1]
                else:
                    # apply transformed expected return
                    exp_r_t = gamma * transform_h_inv(returns[i+1])
                    returns[i] = transform_h(rewards[i] + exp_r_t)
        return returns


    def __len__(self):
        """Return length of memory using states."""
        if self.priority:
            return len(self.buff)
        return len(self.states)


    def sample(self, batch_size, beta=0.4):
        """Return a random batch sample from the memory."""
        assert len(self) >= batch_size

        shape = (batch_size, self.height, self.width, self.phi_length)
        states = np.zeros(shape, dtype=np.uint8)
        actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        returns = np.zeros(batch_size, dtype=np.float32)
        assert self.fs_size != 0
        fullstates = np.zeros((batch_size, self.fs_size), dtype=np.uint8)
        from_rollout = np.zeros(batch_size, dtype=bool)
        refreshed = np.zeros(batch_size, dtype=bool)

        if self.priority:
            sample = self.buff.sample(batch_size, beta)
            states, fullstates, acts, returns, from_rollout, refreshed, \
                weights, idxes = sample
            for i, a in enumerate(acts):
                actions[i][a] = 1  # one-hot vector
            batch = (states, actions, returns, fullstates, from_rollout, refreshed)
        else:
            weights = np.ones(batch_size, dtype=np.float32)
            idxes = random.sample(range(0, len(self.states)), batch_size)
            for i, rand_i in enumerate(idxes):
                states[i] = np.copy(self.states[rand_i])
                actions[i][self.actions[rand_i]] = 1  # one-hot vector
                returns[i] = self.returns[rand_i]
                fullstates[i] = np.copy(self.fullstates[rand_i])
                from_rollout[i] = self.from_rollout[rand_i]
                refreshed[i] = self.refreshed[rand_i]
            batch = (states, actions, returns, fullstates, from_rollout, refreshed)
        return idxes, batch, weights


    def sample_one_random(self):
        """Return one random sample from shared_buffer without priority.
           Used in prioritized memory when sampling without priority only
           Take one sample at a time
        """
        assert len(self) >= 1
        assert self.priority

        sample = self.buff.sample_rand(1)
        states, fullstates, acts, returns, from_rollout, refreshed = sample
        batch = (states, fullstates, acts, returns, from_rollout, refreshed)
        return batch


    def set_weights(self, indexes, priors):
        """Set weights of the samples according to index."""
        if self.priority:
            self.buff.update_priorities(indexes, priors)


    def __del__(self):
        """Clean up."""
        if not self.priority:
            del self.states
            del self.actions
            del self.rewards
            del self.terminal
            del self.returns
