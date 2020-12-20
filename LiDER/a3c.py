#!/usr/bin/env python3
import logging
import numpy as np
import os
import pathlib
import signal
import sys
import threading
import time
import pandas as pd
import math

from threading import Event, Thread
from common_worker import CommonWorker
from a3c_training_thread import A3CTrainingThread
from sil_training_thread import SILTrainingThread
from refresh_thread import RefreshThread
from class_network import MultiClassNetwork as PretrainedModelNetwork
from common.game_state import GameState
from common.util import prepare_dir
from game_ac_network import GameACFFNetwork
from queue import Queue
from sil_memory import SILReplayMemory
from copy import deepcopy
from setup_functions import *


logger = logging.getLogger("a3c")

try:
    import cPickle as pickle
except ImportError:
    import pickle


def run_a3c(args):
    """Run A3C experiment."""
    GYM_ENV_NAME = args.gym_env.replace('-', '_')
    GAME_NAME = args.gym_env.replace('NoFrameskip-v4','')

    # setup folder name and path to folder
    folder = pathlib.Path(setup_folder(args, GYM_ENV_NAME))

    # setup GPU (if applicable)
    import tensorflow as tf
    gpu_options = setup_gpu(tf, args.use_gpu, args.gpu_fraction)

    ######################################################
    # setup default device
    device = "/cpu:0"

    global_t = 0
    rewards = {'train': {}, 'eval': {}}
    best_model_reward = -(sys.maxsize)

    # setup logging info for analysis, see Section 4.2 of the paper
    sil_dict = {
                # count number of SIL updates
                "sil_ctr":{},
                # total number of buffer D samples (i.e., generated by A3C workers) used during SIL (i.e., passed max op)
                "sil_a3c_used":{},
                # the return of used samples for buffer D
                "sil_a3c_used_return":{},
                # total number of buffer R samples (i.e., generated by refresher worker) used during SIL (i.e., passed max op)
                "sil_rollout_used":{},
                # the return of used samples for buffer R
                "sil_rollout_used_return":{},
                # number of old samples still used (even after refreshing)
                "sil_old_used":{}
                }
    sil_ctr, sil_a3c_used, sil_a3c_used_return = 0, 0, 0
    sil_rollout_used, sil_rollout_used_return = 0, 0
    sil_old_used = 0


    rollout_dict = {
                    # total number of rollout performed
                    "rollout_ctr": {},
                    # total number of successful rollout (i.e., Gnew > G)
                    "rollout_added_ctr":{},
                    # the return of Gnew
                    "rollout_new_return":{},
                    # the return of G
                    "rollout_old_return":{}
                    }
    rollout_ctr, rollout_added_ctr = 0, 0
    rollout_new_return = 0 # this records the total, avg = this / rollout_added_ctr
    rollout_old_return = 0 # this records the total, avg = this / rollout_added_ctr

    # setup file names
    reward_fname = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
    sil_fname = folder / '{}-a3c-dict-sil.pkl'.format(GYM_ENV_NAME)
    rollout_fname = folder / '{}-a3c-dict-rollout.pkl'.format(GYM_ENV_NAME)
    if args.load_pretrained_model:
        class_reward_fname = folder / '{}-class-rewards.pkl'.format(GYM_ENV_NAME)

    sharedmem_fname = folder / '{}-sharedmem.pkl'.format(GYM_ENV_NAME)
    sharedmem_params_fname = folder / '{}-sharedmem-params.pkl'.format(GYM_ENV_NAME)
    sharedmem_trees_fname = folder / '{}-sharedmem-trees.pkl'.format(GYM_ENV_NAME)

    rolloutmem_fname = folder / '{}-rolloutmem.pkl'.format(GYM_ENV_NAME)
    rolloutmem_params_fname = folder / '{}-rolloutmem-params.pkl'.format(GYM_ENV_NAME)
    rolloutmem_trees_fname = folder / '{}-rolloutmem-trees.pkl'.format(GYM_ENV_NAME)

    # for removing older ckpt, save mem space
    prev_ckpt_t = -1

    stop_req = False

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.action_space.n
    game_state.close()
    del game_state.env
    del game_state

    input_shape = (args.input_shape, args.input_shape, 4)
    #######################################################
    # setup global A3C
    GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
    global_network = GameACFFNetwork(
        action_size, -1, device, padding=args.padding,
        in_shape=input_shape)
    logger.info('A3C Initial Learning Rate={}'.format(args.initial_learn_rate))

    # setup pretrained model
    global_pretrained_model = None
    local_pretrained_model = None
    pretrain_graph = None

    # if use pretrained model to refresh
    # then must load pretrained model
    # otherwise, don't load model
    if args.use_lider and args.nstep_bc > 0:
        assert args.load_pretrained_model, "refreshing with other policies, must load a pre-trained model (TA or BC)"
    else:
        assert not args.load_pretrained_model, "refreshing with the current policy, don't load pre-trained models"

    if args.load_pretrained_model:
        pretrain_graph, global_pretrained_model = setup_pretrained_model(tf,
            args, action_size, input_shape,
            device="/gpu:0" if args.use_gpu else device)
        assert global_pretrained_model is not None
        assert pretrain_graph is not None

    time.sleep(2.0)

    # setup experience memory
    shared_memory = None # => this is BufferD
    rollout_buffer = None # => this is BufferR
    if args.use_sil:
        shared_memory = SILReplayMemory(
            action_size, max_len=args.memory_length, gamma=args.gamma,
            clip=False if args.unclipped_reward else True,
            height=input_shape[0], width=input_shape[1],
            phi_length=input_shape[2], priority=args.priority_memory,
            reward_constant=args.reward_constant)

        if args.use_lider and not args.onebuffer:
            rollout_buffer = SILReplayMemory(
                action_size, max_len=args.memory_length, gamma=args.gamma,
                clip=False if args.unclipped_reward else True,
                height=input_shape[0], width=input_shape[1],
                phi_length=input_shape[2], priority=args.priority_memory,
                reward_constant=args.reward_constant)

        # log memory information
        shared_memory.log()
        if args.use_lider and not args.onebuffer:
            rollout_buffer.log()

    ############## Setup Thread Workers BEGIN ################
    # 17 total number of threads for all experiments
    assert args.parallel_size ==17, "use 17 workers for all experiments"

    startIndex = 0
    all_workers = []

    # a3c and sil learning rate and optimizer
    learning_rate_input = tf.placeholder(tf.float32, shape=(), name="opt_lr")
    grad_applier = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_input,
        decay=args.rmsp_alpha,
        epsilon=args.rmsp_epsilon)

    setup_common_worker(CommonWorker, args, action_size)

    # setup SIL worker
    sil_worker = None
    if args.use_sil:
        _device = "/gpu:0" if args.use_gpu else device

        sil_network = GameACFFNetwork(
            action_size, startIndex, device=_device,
            padding=args.padding, in_shape=input_shape)

        sil_worker = SILTrainingThread(startIndex, global_network, sil_network,
            args.initial_learn_rate,
            learning_rate_input,
            grad_applier, device=_device,
            batch_size=args.batch_size,
            use_rollout=args.use_lider,
            one_buffer=args.onebuffer,
            sampleR=args.sampleR)

        all_workers.append(sil_worker)
        startIndex += 1

    # setup refresh worker
    refresh_worker = None
    if args.use_lider:
        _device = "/gpu:0" if args.use_gpu else device

        refresh_network = GameACFFNetwork(
            action_size, startIndex, device=_device,
            padding=args.padding, in_shape=input_shape)

        refresh_local_pretrained_model = None
        # if refreshing with other polies
        if args.nstep_bc > 0:
            refresh_local_pretrained_model = PretrainedModelNetwork(
                pretrain_graph, action_size, startIndex,
                padding=args.padding,
                in_shape=input_shape, sae=False,
                tied_weights=False,
                use_denoising=False,
                noise_factor=0.3,
                loss_function='mse',
                use_slv=False, device=_device)

        refresh_worker = RefreshThread(
            thread_index=startIndex, action_size=action_size, env_id=args.gym_env,
            global_a3c=global_network, local_a3c=refresh_network,
            update_in_rollout=args.update_in_rollout, nstep_bc=args.nstep_bc,
            global_pretrained_model=global_pretrained_model,
            local_pretrained_model=refresh_local_pretrained_model,
            transformed_bellman = args.transformed_bellman,
            device=_device,
            entropy_beta=args.entropy_beta, clip_norm=args.grad_norm_clip,
            grad_applier=grad_applier,
            initial_learn_rate=args.initial_learn_rate,
            learning_rate_input=learning_rate_input)

        all_workers.append(refresh_worker)
        startIndex += 1

    # setup a3c workers
    setup_a3c_worker(A3CTrainingThread, args, startIndex)
    for i in range(startIndex, args.parallel_size):
        local_network = GameACFFNetwork(
            action_size, i, device="/cpu:0",
            padding=args.padding,
            in_shape=input_shape)

        a3c_worker = A3CTrainingThread(
            i, global_network, local_network,
            args.initial_learn_rate, learning_rate_input, grad_applier,
            device="/cpu:0", no_op_max=30)

        all_workers.append(a3c_worker)
    ############## Setup Thread Workers END ################

    # setup config for tensorflow
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)

    # prepare sessions
    sess = tf.Session(config=config)
    pretrain_sess = None
    if global_pretrained_model:
        pretrain_sess = tf.Session(config=config, graph=pretrain_graph)

    # initial pretrained model
    if pretrain_sess:
        assert args.pretrained_model_folder is not None
        global_pretrained_model.load(
            pretrain_sess,
            args.pretrained_model_folder)

    sess.run(tf.global_variables_initializer())
    if global_pretrained_model:
        initialize_uninitialized(tf, pretrain_sess,
                                 global_pretrained_model)
    if local_pretrained_model:
        initialize_uninitialized(tf, pretrain_sess,
                                 local_pretrained_model)

    # summary writer for tensorboard
    summ_file = args.save_to+'log/a3c/{}/'.format(GYM_ENV_NAME) + str(folder)[58:] # str(folder)[12:]
    summary_writer = tf.summary.FileWriter(summ_file, sess.graph)

    # init or load checkpoint with saver
    root_saver = tf.train.Saver(max_to_keep=1)
    saver = tf.train.Saver(max_to_keep=1)
    best_saver = tf.train.Saver(max_to_keep=1)

    checkpoint = tf.train.get_checkpoint_state(str(folder)+'/model_checkpoints')
    if checkpoint and checkpoint.model_checkpoint_path:
        root_saver.restore(sess, checkpoint.model_checkpoint_path)
        logger.info("checkpoint loaded:{}".format(
            checkpoint.model_checkpoint_path))
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[-1])
        logger.info(">>> global step set: {}".format(global_t))

        tmp_t = (global_t // args.eval_freq) * args.eval_freq
        logger.info(">>> tmp_t: {}".format(tmp_t))

        # set wall time
        wall_t = 0.

        # set up reward files
        best_reward_file = folder / 'model_best/best_model_reward'
        with best_reward_file.open('r') as f:
            best_model_reward = float(f.read())

        # restore rewards
        rewards = restore_dict(reward_fname, global_t)
        logger.info(">>> restored: rewards")

        # restore loggings
        sil_dict = restore_dict(sil_fname, global_t)
        sil_ctr = sil_dict['sil_ctr'][tmp_t]
        sil_a3c_used = sil_dict['sil_a3c_used'][tmp_t]
        sil_a3c_used_return = sil_dict['sil_a3c_used_return'][tmp_t]
        sil_rollout_used = sil_dict['sil_rollout_used'][tmp_t]
        sil_rollout_used_return = sil_dict['sil_rollout_used_return'][tmp_t]
        sil_old_used = sil_dict['sil_old_used'][tmp_t]
        logger.info(">>> restored: sil_dict")

        rollout_dict = restore_dict(rollout_fname, global_t)
        rollout_ctr = rollout_dict['rollout_ctr'][tmp_t]
        rollout_added_ctr = rollout_dict['rollout_added_ctr'][tmp_t]
        rollout_new_return = rollout_dict['rollout_new_return'][tmp_t]
        rollout_old_return = rollout_dict['rollout_old_return'][tmp_t]
        logger.info(">>> restored: rollout_dict")

        if args.load_pretrained_model:
            class_reward_file = folder / '{}-class-rewards.pkl'.format(GYM_ENV_NAME)
            class_rewards = restore_dict(class_reward_file, tmp_t)

        # restore replay buffers (if saved)
        if args.checkpoint_buffer:
            # restore buffer D
            if args.use_sil and args.priority_memory:
                shared_memory = restore_buffer(sharedmem_fname, shared_memory, global_t)
                shared_memory = restore_buffer_trees(sharedmem_trees_fname, shared_memory, global_t)
                shared_memory = restore_buffer_params(sharedmem_params_fname, shared_memory, global_t)
                logger.info(">>> restored: shared_memory (Buffer D)")
                shared_memory.log()
                # restore buffer R
                if args.use_lider and not args.onebuffer:
                    rollout_buffer = restore_buffer(rolloutmem_fname, rollout_buffer, global_t)
                    rollout_buffer = restore_buffer_trees(rolloutmem_trees_fname, rollout_buffer, global_t)
                    rollout_buffer = restore_buffer_params(rolloutmem_params_fname, rollout_buffer, global_t)
                    logger.info(">>> restored: rollout_buffer (Buffer R)")
                    rollout_buffer.log()

        # if all restores okay, remove old ckpt to save storage space
        prev_ckpt_t = global_t

    else:
        logger.warning("Could not find old checkpoint")
        wall_t = 0.0
        prepare_dir(folder, empty=True)
        prepare_dir(folder / 'model_checkpoints', empty=True)
        prepare_dir(folder / 'model_best', empty=True)
        prepare_dir(folder / 'frames', empty=True)

    lock = threading.Lock()

    # next saving global_t
    def next_t(current_t, freq):
        return np.ceil((current_t + 0.00001) / freq) * freq

    next_global_t = next_t(global_t, args.eval_freq)
    next_save_t = next_t(
        global_t, args.eval_freq*args.checkpoint_freq)

    step_t = 0

    def train_function(parallel_idx, th_ctr, ep_queue, net_updates):
        nonlocal global_t, step_t, rewards, class_rewards, lock, \
                 next_save_t, next_global_t, prev_ckpt_t
        nonlocal shared_memory, rollout_buffer
        nonlocal sil_dict, sil_ctr, sil_a3c_used, sil_a3c_used_return, \
                 sil_rollout_used, sil_rollout_used_return, \
                 sil_old_used
        nonlocal rollout_dict, rollout_ctr, rollout_added_ctr, \
                 rollout_new_return, rollout_old_return

        parallel_worker = all_workers[parallel_idx]
        parallel_worker.set_summary_writer(summary_writer)

        with lock:
            # Evaluate model before training
            if not stop_req and global_t == 0 and step_t == 0:
                rewards['eval'][step_t] = parallel_worker.testing(
                    sess, args.eval_max_steps, global_t, folder,
                    worker=all_workers[-1])

                # testing pretrained TA or BC in game
                if args.load_pretrained_model:
                    assert pretrain_sess is not None
                    assert global_pretrained_model is not None
                    class_rewards['class_eval'][step_t] = \
                        parallel_worker.test_loaded_classifier(global_t=global_t,
                                                    max_eps=50, # testing 50 episodes
                                                    sess=pretrain_sess,
                                                    worker=all_workers[-1],
                                                    model=global_pretrained_model)
                    # log pretrained model performance
                    class_eval_file = pathlib.Path(args.pretrained_model_folder[:41]+\
                        str(GAME_NAME)+"/"+str(GAME_NAME)+'-eval.txt')
                    class_std = np.std(class_rewards['class_eval'][step_t][-1])
                    class_mean = np.mean(class_rewards['class_eval'][step_t][-1])
                    with class_eval_file.open('w') as f:
                        f.write("class_mean: \n" + str(class_mean) + "\n")
                        f.write("class_std: \n" + str(class_std) + "\n")
                        f.write("class_rewards: \n" + str(class_rewards['class_eval'][step_t][-1]) + "\n")

                checkpt_file = folder / 'model_checkpoints'
                checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                saver.save(sess, str(checkpt_file), global_step=global_t)
                save_best_model(rewards['eval'][global_t][0])

                # saving worker info to dicts for analysis
                sil_dict['sil_ctr'][step_t] = sil_ctr
                sil_dict['sil_a3c_used'][step_t] = sil_a3c_used
                sil_dict['sil_a3c_used_return'][step_t] = sil_a3c_used_return
                sil_dict['sil_rollout_used'][step_t] = sil_rollout_used
                sil_dict['sil_rollout_used_return'][step_t] = sil_rollout_used_return
                sil_dict['sil_old_used'][step_t] = sil_old_used

                rollout_dict['rollout_ctr'][step_t] = rollout_ctr
                rollout_dict['rollout_added_ctr'][step_t] = rollout_added_ctr
                rollout_dict['rollout_new_return'][step_t] = rollout_new_return
                rollout_dict['rollout_old_return'][step_t] = rollout_old_return

                # dump pickle
                dump_pickle([rewards, sil_dict, rollout_dict],
                            [reward_fname, sil_fname, rollout_fname],
                            global_t)
                if args.load_pretrained_model:
                    dump_pickle([class_rewards], [class_reward_fname], global_t)

                logger.info('Dump pickle at step {}'.format(global_t))

                # save replay buffer (only works under priority mem)
                if args.checkpoint_buffer:
                    if shared_memory is not None and args.priority_memory:
                        params = [shared_memory.buff._next_idx, shared_memory.buff._max_priority]
                        trees = [shared_memory.buff._it_sum._value,
                                 shared_memory.buff._it_min._value]
                        dump_pickle([shared_memory.buff._storage, params, trees],
                                    [sharedmem_fname, sharedmem_params_fname, sharedmem_trees_fname],
                                    global_t)
                        logger.info('Saving shared_memory')

                    if rollout_buffer is not None and args.priority_memory:
                        params = [rollout_buffer.buff._next_idx, rollout_buffer.buff._max_priority]
                        trees = [rollout_buffer.buff._it_sum._value,
                                 rollout_buffer.buff._it_min._value]
                        dump_pickle([rollout_buffer.buff._storage, params, trees],
                                    [rolloutmem_fname, rolloutmem_params_fname, rolloutmem_trees_fname],
                                    global_t)
                        logger.info('Saving rollout_buffer')

                prev_ckpt_t = global_t

                step_t = 1

        # set start_time
        start_time = time.time() - wall_t
        parallel_worker.set_start_time(start_time)

        if parallel_worker.is_sil_thread:
            sil_interval = 0  # bigger number => slower SIL updates
            m_repeat = 4
            min_mem = args.batch_size * m_repeat
            sil_train_flag = len(shared_memory) >= min_mem

        while True:
            if stop_req:
                return

            if global_t >= (args.max_time_step * args.max_time_step_fraction):
                return

            if parallel_worker.is_sil_thread:
                # before sil starts, init local count
                local_sil_ctr = 0
                local_sil_a3c_used, local_sil_a3c_used_return = 0, 0
                local_sil_rollout_used, local_sil_rollout_used_return = 0, 0
                local_sil_old_used = 0

                if net_updates.qsize() >= sil_interval \
                   and len(shared_memory) >= min_mem:
                    sil_train_flag = True

                if sil_train_flag:
                    sil_train_flag = False

                    th_ctr.get()

                    train_out = parallel_worker.sil_train(
                        sess, global_t, shared_memory, m_repeat,
                        rollout_buffer=rollout_buffer)

                    local_sil_ctr, local_sil_a3c_used, local_sil_a3c_used_return, \
                        local_sil_rollout_used, local_sil_rollout_used_return, \
                        local_sil_old_used = train_out

                    th_ctr.put(1)

                    with net_updates.mutex:
                        net_updates.queue.clear()

                    if args.use_lider:
                        parallel_worker.record_sil(sil_ctr=sil_ctr,
                                              total_used=(sil_a3c_used + sil_rollout_used),
                                              num_a3c_used=sil_a3c_used,
                                              a3c_used_return=sil_a3c_used_return/(sil_a3c_used+1),#add one in case divide by zero
                                              rollout_used=sil_rollout_used,
                                              rollout_used_return=sil_rollout_used_return/(sil_rollout_used+1),
                                              old_used=sil_old_used,
                                              global_t=global_t)

                        if sil_ctr % 200 == 0 and sil_ctr > 0:
                            rollout_buffsize = 0
                            if not args.onebuffer:
                                rollout_buffsize = len(rollout_buffer)
                            log_data = (sil_ctr, len(shared_memory),
                                        rollout_buffsize,
                                        sil_a3c_used+sil_rollout_used,
                                        args.batch_size*sil_ctr,
                                        sil_a3c_used,
                                        sil_a3c_used_return/(sil_a3c_used+1),
                                        sil_rollout_used,
                                        sil_rollout_used_return/(sil_rollout_used+1),
                                        sil_old_used)
                            logger.info("SIL: sil_ctr={0:}"
                                        " sil_memory_size={1:}"
                                        " rollout_buffer_size={2:}"
                                        " total_sample_used={3:}/{4:}"
                                        " a3c_used={5:}"
                                        " a3c_used_return_avg={6:.2f}"
                                        " rollout_used={7:}"
                                        " rollout_used_return_avg={8:.2f}"
                                        " old_used={9:}".format(*log_data))
                    else:
                        parallel_worker.record_sil(sil_ctr=sil_ctr,
                                                   total_used=(sil_a3c_used + sil_rollout_used),
                                                   num_a3c_used=sil_a3c_used,
                                                   rollout_used=sil_rollout_used,
                                                   global_t=global_t)
                        if sil_ctr % 200 == 0 and sil_ctr > 0:
                            log_data = (sil_ctr, sil_a3c_used+sil_rollout_used,
                                        args.batch_size*sil_ctr,
                                        sil_a3c_used,
                                        len(shared_memory))
                            logger.info("SIL: sil_ctr={0:}"
                                        " total_sample_used={1:}/{2:}"
                                        " a3c_used={3:}"
                                        " sil_memory_size={4:}".format(*log_data))

                # Adding episodes to SIL memory is centralize to ensure
                # sampling and updating of priorities does not become a problem
                # since we add new episodes to SIL at once and during
                # SIL training it is guaranteed that SIL memory is untouched.
                max = args.parallel_size
                while not ep_queue.empty():
                    data = ep_queue.get()
                    parallel_worker.episode.set_data(*data)
                    shared_memory.extend(parallel_worker.episode)
                    parallel_worker.episode.reset()
                    max -= 1
                    if max <= 0: # This ensures that SIL has a chance to train
                        break

                diff_global_t = 0

                # centralized rollout counting
                local_rollout_ctr, local_rollout_added_ctr = 0, 0
                local_rollout_new_return, local_rollout_old_return = 0, 0

            elif parallel_worker.is_refresh_thread:
                # before refresh starts, init local count
                diff_global_t = 0
                local_rollout_ctr, local_rollout_added_ctr = 0, 0
                local_rollout_new_return, local_rollout_old_return = 0, 0

                if len(shared_memory) >= 1:
                    th_ctr.get()
                    # randomly sample a state from buffer D
                    sample = shared_memory.sample_one_random()
                    # after sample, flip refreshed to True
                    # TODO: fix this so that only *succesful* refresh is flipped to True
                    assert sample[-1] == True

                    train_out = parallel_worker.rollout(sess, folder, pretrain_sess,
                                                   global_t, sample,
                                                   args.addall,
                                                   args.max_ep_step,
                                                   args.nstep_bc,
                                                   args.update_in_rollout)

                    diff_global_t, episode_end, part_end, local_rollout_ctr, \
                        local_rollout_added_ctr, add, local_rollout_new_return, \
                        local_rollout_old_return = train_out

                    th_ctr.put(1)

                    if rollout_ctr % 20 == 0 and rollout_ctr > 0:
                        log_msg = "ROLLOUT: rollout_ctr={} added_rollout_ct={} worker={}".format(
                        rollout_ctr, rollout_added_ctr, parallel_worker.thread_idx)
                        logger.info(log_msg)
                        logger.info("ROLLOUT Gnew: {}, G: {}".format(local_rollout_new_return,
                                                                     local_rollout_old_return))

                    # should always part_end, i.e., end of episode
                    # and only add if new return is better (if not LiDER-AddAll)
                    if part_end and add:
                        if not args.onebuffer:
                            # directly put into Buffer R
                            rollout_buffer.extend(parallel_worker.episode)
                        else:
                            # Buffer D add sample is centralized when OneBuffer
                            ep_queue.put(parallel_worker.episode.get_data())

                    parallel_worker.episode.reset()

                # centralized SIL counting
                local_sil_ctr = 0
                local_sil_a3c_used, local_sil_a3c_used_return = 0, 0
                local_sil_rollout_used, local_sil_rollout_used_return = 0, 0
                local_sil_old_used = 0

            # a3c training thread worker
            else:
                th_ctr.get()

                train_out = parallel_worker.train(sess, global_t, rewards)
                diff_global_t, episode_end, part_end = train_out

                th_ctr.put(1)

                if args.use_sil:
                    net_updates.put(1)
                    if part_end:
                        ep_queue.put(parallel_worker.episode.get_data())
                        parallel_worker.episode.reset()

                # centralized SIL counting
                local_sil_ctr = 0
                local_sil_a3c_used, local_sil_a3c_used_return = 0, 0
                local_sil_rollout_used, local_sil_rollout_used_return = 0, 0
                local_sil_old_used = 0
                # centralized rollout counting
                local_rollout_ctr, local_rollout_added_ctr = 0, 0
                local_rollout_new_return, local_rollout_old_return = 0, 0

            # ensure only one thread is updating global_t at a time
            with lock:
                global_t += diff_global_t

                # centralize increasing count for SIL and Rollout
                sil_ctr += local_sil_ctr
                sil_a3c_used += local_sil_a3c_used
                sil_a3c_used_return += local_sil_a3c_used_return
                sil_rollout_used += local_sil_rollout_used
                sil_rollout_used_return += local_sil_rollout_used_return
                sil_old_used += local_sil_old_used

                rollout_ctr += local_rollout_ctr
                rollout_added_ctr += local_rollout_added_ctr
                rollout_new_return += local_rollout_new_return
                rollout_old_return += local_rollout_old_return

                # if during a thread's update, global_t has reached a evaluation interval
                if global_t > next_global_t:
                    next_global_t = next_t(global_t, args.eval_freq)
                    step_t = int(next_global_t - args.eval_freq)

                    # wait for all threads to be done before testing
                    while not stop_req and th_ctr.qsize() < len(all_workers):
                        time.sleep(0.001)

                    step_t = int(next_global_t - args.eval_freq)

                    rewards['eval'][step_t] = parallel_worker.testing(
                        sess, args.eval_max_steps, global_t, folder,
                        worker=all_workers[-1])
                    save_best_model(rewards['eval'][step_t][0])
                    last_reward = rewards['eval'][step_t][0]

                    # saving worker info to dicts
                    # SIL
                    sil_dict['sil_ctr'][step_t] = sil_ctr
                    sil_dict['sil_a3c_used'][step_t] = sil_a3c_used
                    sil_dict['sil_a3c_used_return'][step_t] = sil_a3c_used_return
                    sil_dict['sil_rollout_used'][step_t] = sil_rollout_used
                    sil_dict['sil_rollout_used_return'][step_t] = sil_rollout_used_return
                    sil_dict['sil_old_used'][step_t] = sil_old_used
                    # ROLLOUT
                    rollout_dict['rollout_ctr'][step_t] = rollout_ctr
                    rollout_dict['rollout_added_ctr'][step_t] = rollout_added_ctr
                    rollout_dict['rollout_new_return'][step_t] = rollout_new_return
                    rollout_dict['rollout_old_return'][step_t] = rollout_old_return

                    # save after done with eval
                    if global_t > next_save_t:
                        next_save_t = next_t(global_t, args.eval_freq*args.checkpoint_freq)

                        # dump pickle
                        dump_pickle([rewards, sil_dict, rollout_dict],
                                    [reward_fname, sil_fname, rollout_fname],
                                    global_t)
                        if args.load_pretrained_model:
                            dump_pickle([class_rewards], [class_reward_fname], global_t)
                        logger.info('Dump pickle at step {}'.format(global_t))

                        # save replay buffer (only works for priority mem for now)
                        if args.checkpoint_buffer:
                            if shared_memory is not None and args.priority_memory:
                                params = [shared_memory.buff._next_idx, shared_memory.buff._max_priority]
                                trees = [shared_memory.buff._it_sum._value,
                                         shared_memory.buff._it_min._value]
                                dump_pickle([shared_memory.buff._storage, params, trees],
                                            [sharedmem_fname, sharedmem_params_fname, sharedmem_trees_fname],
                                            global_t)
                                logger.info('Saved shared_memory')

                            if rollout_buffer is not None and args.priority_memory:
                                params = [rollout_buffer.buff._next_idx, rollout_buffer.buff._max_priority]
                                trees = [rollout_buffer.buff._it_sum._value,
                                         rollout_buffer.buff._it_min._value]
                                dump_pickle([rollout_buffer.buff._storage, params, trees],
                                            [rolloutmem_fname, rolloutmem_params_fname, rolloutmem_trees_fname],
                                            global_t)
                                logger.info('Saved rollout_buffer')

                        # save a3c after saving buffer -- in case saving buffer OOM
                        # so that at least we can revert back to the previous ckpt
                        checkpt_file = folder / 'model_checkpoints'
                        checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                        saver.save(sess, str(checkpt_file), global_step=global_t,
                                   write_meta_graph=False)
                        logger.info('Saved model ckpt')

                        # if everything saves okay, clean up previous ckpt to save space
                        remove_pickle([reward_fname, sil_fname, rollout_fname],
                                      prev_ckpt_t)
                        if args.load_pretrained_model:
                            remove_pickle([class_reward_fname], prev_ckpt_t)

                        remove_pickle([sharedmem_fname, sharedmem_params_fname,
                                       sharedmem_trees_fname],
                                      prev_ckpt_t)
                        if rollout_buffer is not None and args.priority_memory:
                            remove_pickle([rolloutmem_fname, rolloutmem_params_fname,
                                           rolloutmem_trees_fname],
                                          prev_ckpt_t)

                        logger.info('Removed ckpt from step {}'.format(prev_ckpt_t))

                        prev_ckpt_t = global_t


    def signal_handler(signal, frame):
        nonlocal stop_req
        logger.info('You pressed Ctrl+C!')
        stop_req = True

        if stop_req and global_t == 0:
            sys.exit(1)

    def save_best_model(test_reward):
        nonlocal best_model_reward
        if test_reward > best_model_reward:
            best_model_reward = test_reward
            best_reward_file = folder / 'model_best/best_model_reward'

            with best_reward_file.open('w') as f:
                f.write(str(best_model_reward))

            best_checkpt_file = folder / 'model_best'
            best_checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
            best_saver.save(sess, str(best_checkpt_file))


    train_threads = []
    th_ctr = Queue()
    for i in range(args.parallel_size):
        th_ctr.put(1)

    episodes_queue = None
    net_updates = None
    if args.use_sil:
        episodes_queue = Queue()
        net_updates = Queue()

    for i in range(args.parallel_size):
        worker_thread = Thread(
            target=train_function,
            args=(i, th_ctr, episodes_queue, net_updates,))
        train_threads.append(worker_thread)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # set start time
    start_time = time.time() - wall_t

    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop')

    for t in train_threads:
        t.join()

    logger.info('Now saving data. Please wait')

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = folder / 'wall_t.{}'.format(global_t)
    with wall_t_fname.open('w') as f:
        f.write(str(wall_t))

    # save final model
    checkpoint_file = str(folder / '{}_checkpoint_a3c'.format(GYM_ENV_NAME))
    root_saver.save(sess, checkpoint_file, global_step=global_t)

    dump_final_pickle([rewards, sil_dict, rollout_dict],
                      [reward_fname, sil_fname, rollout_fname])

    logger.info('Data saved!')

    # if everything saves okay & is done training (not because of pressed Ctrl+C),
    # clean up previous ckpt to save space
    if global_t >= (args.max_time_step * args.max_time_step_fraction):
        remove_pickle([reward_fname, sil_fname, rollout_fname],
                      prev_ckpt_t)
        if args.load_pretrained_model:
            remove_pickle([class_reward_fname], prev_ckpt_t)

        remove_pickle([sharedmem_fname, sharedmem_params_fname, sharedmem_trees_fname],
                      prev_ckpt_t)
        if rollout_buffer is not None and args.priority_memory:
            remove_pickle([rolloutmem_fname, rolloutmem_params_fname, rolloutmem_trees_fname],
                          prev_ckpt_t)

        logger.info('Done training, removed ckpt from step {}'.format(prev_ckpt_t))


    sess.close()
    if pretrain_sess:
        pretrain_sess.close()