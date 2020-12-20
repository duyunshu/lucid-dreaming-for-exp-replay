import os
import logging
import numpy as np
import pathlib

from common.util import load_memory

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger("setup_functions")

def setup_folder(args, env_name):
    if not os.path.exists(args.save_to+"a3c/"):
        os.makedirs(args.save_to+"a3c/")

    if args.folder is not None:
        folder = args.save_to+'a3c/{}_{}'.format(env_name, args.folder)
    else:
        folder = args.save_to+'a3c/{}'.format(env_name)
        end_str = ''

        if args.unclipped_reward:
            end_str += '_rawreward'
        if args.transformed_bellman:
            end_str += '_transformedbell'

        if args.use_sil:
            end_str += '_sil'
            if args.priority_memory:
                end_str += '_prioritymem'

        if args.use_lider:
            end_str+='_lider'
            if args.onebuffer:
                end_str+="_onebuffer"
            if args.memory_length != 100000:
                t="_"+str(args.memory_length / 100000)+"memlength"
                end_str+=t
            if args.addall:
                end_str+='_addall'
            if args.sampleR:
                end_str+='_sampleR'
            if args.load_TA:
                end_str+='_TA'
            elif args.load_BC:
                end_str+='_BC'

        folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    return folder


def setup_gpu(tf, use_gpu, gpu_fraction):
    gpu_options = None
    if use_gpu:
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
    return gpu_options


def setup_pretrained_model(tf, args, action_size, in_shape, device=None):
    pretrain_model = None
    pretrain_graph = tf.Graph()
    if args.classify_demo:
        from class_network import MultiClassNetwork \
            as PretrainedModelNetwork
    else:
        logger.error("Classification type Not supported yet!")
        assert False

    PretrainedModelNetwork.use_mnih_2015 = args.use_mnih_2015
    PretrainedModelNetwork.l1_beta = args.class_l1_beta
    PretrainedModelNetwork.l2_beta = args.class_l2_beta
    PretrainedModelNetwork.use_gpu = args.use_gpu
    # pretrained_model thread has to be -1!
    pretrain_model = PretrainedModelNetwork(
        pretrain_graph, action_size, -1,
        padding=args.padding,
        in_shape=in_shape, sae=False,
        tied_weights=False,
        use_denoising=False,
        noise_factor=0.3,
        loss_function='mse',
        use_slv=False, device=device)

    return pretrain_graph, pretrain_model


def setup_common_worker(CommonWorker, args, action_size):
    CommonWorker.action_size = action_size
    CommonWorker.env_id = args.gym_env
    CommonWorker.reward_constant = args.reward_constant
    CommonWorker.max_global_time_step = args.max_time_step
    if args.unclipped_reward:
        CommonWorker.reward_type = "RAW"
    else:
        CommonWorker.reward_type = "CLIP"


def setup_a3c_worker(A3CTrainingThread, args, log_idx):
    A3CTrainingThread.log_interval = args.log_interval
    A3CTrainingThread.perf_log_interval = args.performance_log_interval
    A3CTrainingThread.local_t_max = args.local_t_max
    A3CTrainingThread.entropy_beta = args.entropy_beta
    A3CTrainingThread.gamma = args.gamma
    A3CTrainingThread.use_mnih_2015 = args.use_mnih_2015
    A3CTrainingThread.transformed_bellman = args.transformed_bellman
    A3CTrainingThread.clip_norm = args.grad_norm_clip
    A3CTrainingThread.use_sil = args.use_sil
    A3CTrainingThread.use_lider = args.use_lider
    A3CTrainingThread.log_idx = log_idx
    A3CTrainingThread.reward_constant = args.reward_constant


def initialize_uninitialized(tf, sess, model=None):
    if model is not None:
        with model.graph.as_default():
            global_vars=tf.global_variables()
    else:
        global_vars = tf.global_variables()

    is_not_initialized = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = \
        [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def restore_buffer(fn, buffer, global_t):
    filename = str(fn) + "-" + str(global_t)
    temp = pickle.load(pathlib.Path(filename).open('rb'))
    x_states = []
    x_fullstates = []
    x_actions = []
    x_returns = []
    x_rollout = []
    x_refresh = []
    for data in temp:
        s, fs, a, r, roll, refresh = data
        x_states.append(s)
        x_fullstates.append(fs)
        x_actions.append(a)
        x_returns.append(r)
        x_rollout.append(roll)
        x_refresh.append(refresh)
    if len(x_fullstates) > 0:
        buffer.extend_one_priority(x_states, x_fullstates, x_actions,
                                   x_returns, x_rollout, x_refresh)
    del temp
    return buffer


def restore_buffer_trees(fn, buffer, global_t):
    filename = str(fn) + "-" + str(global_t)
    temp = pickle.load(pathlib.Path(filename).open('rb'))
    assert len(temp) == 2
    buffer.buff._it_sum._value = temp[0]
    buffer.buff._it_min._value = temp[1]
    del temp
    return buffer


def restore_buffer_params(fn, buffer, global_t):
    filename = str(fn) + "-" + str(global_t)
    temp = pickle.load(pathlib.Path(filename).open('rb'))
    assert len(temp) == 2
    buffer.buff._next_idx = temp[0]
    buffer.buff._max_priority = temp[1]
    del temp
    return buffer


def restore_dict(dict_name, global_t):
    fn = str(dict_name) + "-" + str(global_t)
    file = pickle.load(pathlib.Path(fn).open('rb'))
    return file


def dump_pickle(dict_list, fn_list, global_t=""):
    assert len(dict_list) == len(fn_list)
    for i in range(len(dict_list)):
        fn = str(fn_list[i]) + "-" + str(global_t)
        pickle.dump(dict_list[i], pathlib.Path(fn).open('wb'), pickle.HIGHEST_PROTOCOL)


def dump_final_pickle(dict_list, fn_list):
    assert len(dict_list) == len(fn_list)
    for i in range(len(dict_list)):
        fn = str(fn_list[i])
        pickle.dump(dict_list[i], pathlib.Path(fn).open('wb'), pickle.HIGHEST_PROTOCOL)


def remove_pickle(fn_list, global_t):
    for i in range(len(fn_list)):
        fn = str(fn_list[i]) + "-" + str(global_t)
        os.remove(fn)
