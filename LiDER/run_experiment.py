#!/usr/bin/env python3
import argparse
import coloredlogs
import logging

from a3c import run_a3c
from time import sleep

logger = logging.getLogger()


def main():
    fmt = "%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s"
    coloredlogs.install(level='DEBUG', fmt=fmt)
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    # general setups
    parser.add_argument('--folder', type=str, default=None, help='name the result folder')
    parser.add_argument('--append-experiment-num', type=str, default=None,
                        help='experiment identifier (date)')
    parser.add_argument('--parallel-size', type=int, default=17,
                        help='parallel thread size')
    parser.add_argument('--gym-env', type=str, default='MsPacmanNoFrameskip-v4',
                        help='OpenAi Gym environment ID')
    parser.add_argument('--save-to', type=str, default='results/',
                        help='where to save results')
    parser.add_argument('--log-interval', type=int, default=500, help='logging info frequency')
    parser.add_argument('--performance-log-interval', type=int, default=1000, help='logging info frequency')

    # checkpoint replay buffer, note that saving replay buffer is memory-intensive
    # can increase 'checkpoint-freq' to reduce mem usage;
    # default is 1, meaning saving every 1 million steps
    # can use e.g. --checkpoint-freq=5 to save every 5 million steps
    parser.add_argument('--checkpoint-buffer', action='store_true',
                        help='checkpointing replay buffer')
    parser.set_defaults(checkpoint_buffer=False)
    parser.add_argument('--checkpoint-freq', type=int, default=1,
                        help='checkpoint frequency, default to every eval-freq*checkpoint-freq steps')

    # enable gpu (default cpu only)
    parser.add_argument('--use-gpu', action='store_true', help='use GPU')
    parser.set_defaults(use_gpu=False)
    parser.add_argument('--gpu-fraction', type=float, default=0.6)
    parser.add_argument('--cuda-devices', type=str, default='')

    # setup network architecture
    parser.add_argument('--use-mnih-2015', action='store_true',
                        help='use Mnih et al [2015] network architecture, if a3c will add value output layer')
    parser.set_defaults(use_mnih_2015=False)
    parser.add_argument('--input-shape', type=int, default=84,
                        help='84x84 as default')
    parser.add_argument('--padding', type=str, default='VALID',
                        help='VALID or SAME')

    # setup A3C components
    parser.add_argument('--local-t-max', type=int, default=20,
                        help='repeat step size')
    parser.add_argument('--rmsp-alpha', type=float, default=0.99,
                        help='decay parameter for RMSProp')
    parser.add_argument('--rmsp-epsilon', type=float, default=1e-5,
                        help='epsilon parameter for RMSProp')
    parser.add_argument('--initial-learn-rate', type=float, default=7e-4,
                        help='initial learning rate for RMSProp')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards')
    parser.add_argument('--entropy-beta', type=float, default=0.01,
                        help='entropy regularization constant')
    parser.add_argument('--max-time-step', type=float, default=10 * 10**7,
                        help='maximum time step, use to anneal learning rate')
    parser.add_argument('--max-time-step-fraction', type=float, default=1.,
                        help='overides maximum time step by a fraction')
    parser.add_argument('--grad-norm-clip', type=float, default=0.5,
                        help='gradient norm clipping')
    parser.add_argument('--eval-freq', type=int, default=1000000,
                        help='how often to evaluate the agent')
    parser.add_argument('--eval-max-steps', type=int, default=125000,
                        help='number of steps for evaluation')
    parser.add_argument('--l2-beta', type=float, default=0.,
                        help='L2 regularization beta')

    parser.add_argument('--max-ep-step', type=float, default=10000,
                        help='maximum time step for an episode (lower than ALE default)')
    #ALE default see: https://github.com/openai/gym/blob/54f22cf4db2e43063093a1b15d968a57a32b6e90/gym/envs/__init__.py#L635

    # Alternatives to reward clipping
    parser.add_argument('--unclipped-reward', action='store_true',
                        help='use raw reward')
    parser.set_defaults(unclipped_reward=False)
    # Transformed Bellman: Ape-X Pohlen, et. al 2018
    parser.add_argument('--transformed-bellman', action='store_true',
                        help='use transformed bellman equation')
    parser.set_defaults(transformed_bellman=False)
    parser.add_argument('--reward-constant', type=float, default=2.0,
                        help='value added to all non-zero rewards when using'
                             ' transformed bellman operator')

    # sil parameters
    parser.add_argument('--use-sil', action='store_true',
                        help='self imitation learning loss (SIL)')
    parser.set_defaults(use_sil=False)
    parser.add_argument('--batch-size', type=int, default=512,
                        help='SIL batch size')
    parser.add_argument('--priority-memory', action='store_true',
                        help='Use Prioritized Replay Memory')
    parser.set_defaults(priority_mem=False)
    parser.add_argument('--memory-length', type=int, default=100000,
                        help='SIL memory size')

    # refresher parameters
    parser.add_argument('--use-lider', action='store_true',
                        help='use the current policy to refresh')
    parser.set_defaults(use_rollout=True)
    parser.add_argument('--nstep-bc', type=int, default=100000,
                        help='rollout using TA/BC for n steps then thereafter follow a3c till terminal, '
                        'if==0 means no TA/BC used, i.e. LiDER ')
    parser.add_argument('--update-in-rollout', action='store_true',
                        help='make immediate update using rollout data')
    parser.set_defaults(update_in_rollout=False)

    # ablation studies
    parser.add_argument('--onebuffer', action='store_true',
                        help='use one buffer for all workers, no separete rollout buffer')
    parser.set_defaults(onebuffer=False)
    parser.add_argument('--addall', action='store_true',
                        help='add all rollout data, otherwise, add only when new return is better')
    parser.set_defaults(addall=False)
    parser.add_argument('--sampleR', action='store_true',
                        help='sample only from buffer R')
    parser.set_defaults(sampleR=False)

    # extensions: load TA/BC parameters
    parser.add_argument('--load-pretrained-model', action='store_true', help='use different policy for refresh')
    parser.set_defaults(load_pretrained_model=False)
    parser.add_argument('--load-TA', action='store_true', help='refresh with TA')
    parser.set_defaults(load_TA=False)
    parser.add_argument('--load-BC', action='store_true', help='refresh with BC')
    parser.set_defaults(load_BC=False)
    parser.add_argument('--pretrained-model-folder', type=str, default=None)
    parser.add_argument('--classify-demo', action='store_true',
                        help='Load pretrained classifier')
    parser.set_defaults(classify_demo=False)


    args = parser.parse_args()

    if args.onebuffer or args.addall or args.sampleR:
        assert args.use_lider, 'must use_lider for addall, onebuffer, or sampleR'

    if args.load_pretrained_model:
        assert args.use_lider, 'must use_lider for LiDER-TA or LiDER-BC'
        assert args.pretrained_model_folder is not None, 'must provide pretrained model path'
        assert not (args.load_TA and args.load_BC), 'must choose between load TA OR BC, cannot load both'

    if args.use_lider:
        args.update_in_rollout=True
        args.roll_random=True
        args.nstep_bc=0
        logger.info('Running LiDER...')
        if args.onebuffer:
            logger.info('Ablation LiDER-OneBuffer')
        elif args.addall:
            logger.info('Ablation LiDER-AddAll')
        elif args.sampleR:
            logger.info('Ablation LiDER-SampleR')
        elif args.load_TA:
            logger.info('Extention: LiDER-TA')
        elif args.load_BC:
            logger.info('Extention: LiDER-BC')
    else:
        logger.info('Running A3CTBSIL...')

    run_a3c(args)

if __name__ == "__main__":
    main()
