#!/usr/bin/env python
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import pandas as pd
import seaborn as sns
import argparse
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

parser = argparse.ArgumentParser()
# general parameters
parser.add_argument('--env', type=str, required=True, help="which game to plot")
parser.add_argument('--max-timesteps', type=int, default=50)
parser.add_argument('--deep-rl-algo', type=str, default='a3c')
parser.add_argument('--result-folder', type=str, default='a3c')
parser.add_argument('--saveplot', action='store_true', help='save plot or not')
parser.set_defaults(saveplot=False)
parser.add_argument('--nolegend', action='store_true', help='dont show legend')
parser.set_defaults(nolegend=False)
parser.add_argument('--folder', type=str, default='plots', help="where to save the plot")
parser.add_argument('--fn', type=str, default='', help="plot name")


# A3CTBSIL and LiDER
parser.add_argument('--baseline', action='store_true', help='plot A3CTBSIL')
parser.set_defaults(baseline=False)
parser.add_argument('--lidera3c', action='store_true', help='plot LiDER')
parser.set_defaults(lidera3c=False)

# Ablation studies
parser.add_argument('--sampler', action='store_true', help='LiDER-SampleR')
parser.set_defaults(sampler=False)
parser.add_argument('--onebuffer', action='store_true', help='LiDER-OneBuffer')
parser.set_defaults(onebuffer=False)
parser.add_argument('--addall', action='store_true', help='LiDER-AddAll')
parser.set_defaults(addall=False)

# Extentions
parser.add_argument('--liderta', action='store_true', help='LiDER-TA')
parser.set_defaults(liderta=False)
parser.add_argument('--liderbc', action='store_true', help='LiDER-BC')
parser.set_defaults(liderbc=False)

args = parser.parse_args()

game = args.env
deep_rl_algo = args.deep_rl_algo
game_env_type = 'NoFrameskip_v4'
location = 'lower right'
ncol = 1
num_data = [1, 2] # detault three trials

# plot the horizontal line for TA and BC
# NOTE: their values need to be set manually
# when running LiDER-TA or LiDER-BC, the pretrained models are evaluated for 50 episodes at step 0
# their results are saved under pretrained_models/TA(orBC)/[game]/[game]-modeleval.txt
plot_ta = True if args.liderta else False
plot_bc = True if args.liderbc else False

# set legend location (if needed)
if game == 'Freeway':
    pass
    # location = "upper left"
elif game == 'Gopher':
    location = "upper left"
elif game == 'MsPacman':
    # pass
    location = "upper left"
elif game == 'NameThisGame':
    # pass
    location = "upper left"
elif game =='Alien':
    location = "upper left"
elif game == 'MontezumaRevenge':
    location = "upper left"
else:
    print("Invalid game!!!")
    sys.exit()


gym_env = game + game_env_type
print(gym_env)

#sns.set_style("darkgrid")
sns.set(context='paper', style='darkgrid', rc={'figure.figsize':(7,5)})
LW = 1.5
ALPHA = 0.1
MARKERSIZE = 12

plt.figure(figsize=(6.6,4.8))

N = 1

if deep_rl_algo == 'a3c':
    MAX_TIMESTEPS = args.max_timesteps*1000000
else:
    print("Invalid algorithm!!!")
    sys.exit()

PER_EPOCH = 1000000


def create_dataframe(rewards_all_trials, per_epoch=0, max_timesteps=0, label=''):
    if per_epoch == 0:
        per_epoch = PER_EPOCH
    if max_timesteps == 0:
        max_timesteps = MAX_TIMESTEPS
    timestep = [ t/per_epoch for t in range(0, (max_timesteps+per_epoch), per_epoch) ]
    d = {}

    print("steps," + ','.join(map(str, sorted(rewards_all_trials[0]['eval'].keys()))))
    for data_idx, reward_data in enumerate(rewards_all_trials):
        rewards = []
        all_rewards = []
        for r in sorted(reward_data['eval'].keys()):
            all_rewards.append(reward_data['eval'][r][0])
            if r <= max_timesteps and r % per_epoch == 0:
                rewards.append(reward_data['eval'][r][0]) ### <=== (reward, steps, num_episodes)
        print(label + '-' + str(data_idx+1) + '-' + str(len(all_rewards)) + ':' + ','.join(map(str, all_rewards)) + '\n')
        d['Rewards{}'.format(data_idx+1)] = rewards
    df = pd.DataFrame(data=d, index=timestep)
    df.index.name = 'Timestep'
    df = df.iloc[::N]
    return df



def plot_fun(
    ex_type='', color='green', result_folder='a3c',
    marker='o', markersize=MARKERSIZE,
    lw=LW, linestyle=None, label='test', num_data=[1,2,3]):
    ''' creates dataframe and plots graph '''
    rewards_all_trials = []
    for data_idx in num_data:
        folder=('results/{}/'.format(result_folder) + gym_env + \
        '{}_{}/'.format(ex_type, data_idx) + \
        gym_env + '-{}-rewards.pkl'.format(deep_rl_algo))
        print(folder)
        r_data = pickle.load(open(folder, 'rb'))
        rewards_all_trials.append(r_data)

    df_rewards = create_dataframe(rewards_all_trials, label=label)
    while len(rewards_all_trials):
        del rewards_all_trials[0]

    df_rewards_mean = df_rewards.mean(axis=1)
    print ("MEAN: ", df_rewards.mean(axis=0))
    df_rewards_std = df_rewards.std(axis=1)

    plt.plot(
        df_rewards_mean.index,
        df_rewards_mean,
        color=sns.xkcd_rgb[color],
        marker=marker,
        markersize=markersize,
        markevery=3,
        lw=lw,
        linestyle=linestyle,
        label=label)
    plt.fill_between(
        df_rewards_std.index,
        df_rewards_mean - df_rewards_std,
        df_rewards_mean + df_rewards_std,
        color=sns.xkcd_rgb[color],
        alpha=ALPHA)


def plot_ta_bc_fun(timestep, mean_val, std_val, color='purple', lw=2,
                   linestyle='dotted', alpha=0.1, label=None):
    plt.plot(
        TIMESTEP,
        [mean_val]*len(TIMESTEP),
        color=sns.xkcd_rgb[color],
        lw=2,
        linestyle=linestyle,
        label=label)
    plt.fill_between(
        TIMESTEP,
        mean_val - std_val,
        mean_val + std_val,
        color=sns.xkcd_rgb[color],
        alpha=0.08)


################plotting start here##########################
# A3CTBSIL
if args.baseline:
    ex_type = '_rawreward_transformedbell_sil_prioritymem'
    plot_fun(
        ex_type=ex_type,
        color='violet red',
        marker='P',
        result_folder=args.result_folder,
        label='{}TBSIL'.format(deep_rl_algo.upper()),
        num_data=num_data)
    ncol += 1

# LiDER
if args.lidera3c:
    ex_type = '_rawreward_transformedbell_sil_prioritymem_lider'
    plot_fun(
        ex_type=ex_type,
        color='green',#'kelly green',
        marker='^',#'x',
        result_folder=args.result_folder,
        label='LiDER',
        num_data=num_data)
    ncol += 1

# LiDER-AddAll
if args.addall:
    ex_type = '_rawreward_transformedbell_sil_prioritymem_lider_addall'
    plot_fun(
        ex_type=ex_type,
        color='dark sky blue',
        marker='d',
        result_folder=args.result_folder,
        label='LiDER-AddAll',
        num_data=num_data)
    ncol += 1

# LiDER-OneBuffer
if args.onebuffer:
    ex_type = '_rawreward_transformedbell_sil_prioritymem_lider_onebuffer'
    plot_fun(
        ex_type=ex_type,
        color='forest green',
        marker='.',
        result_folder=args.result_folder,
        label='LiDER-OneBuffer',
        num_data=num_data)
    ncol += 1

# LiDER-SampleR
if args.sampler:
    ex_type = '_rawreward_transformedbell_sil_prioritymem_lider_sampleR'
    plot_fun(
        ex_type=ex_type,
        color='purple',
        marker='x',
        result_folder=args.result_folder,
        label='LiDER-SampleR',
        num_data=num_data)
    ncol += 1

# LiDER-TA
if args.liderta:
    ex_type='_rawreward_transformedbell_sil_prioritymem_lider_TA'
    plot_fun(
        ex_type=ex_type,
        color='orange',
        marker='o',
        result_folder=args.result_folder,
        label='LiDER-TA',
        num_data=num_data)
    ncol += 1

# LiDER-BC
if args.liderbc:
    ex_type = '_rawreward_transformedbell_sil_prioritymem_lider_BC'
    plot_fun(
        ex_type=ex_type,
        color='cobalt',
        marker='v',
        result_folder=args.result_folder,
        label='LiDER-BC',
        num_data=num_data)
    ncol += 1

TIMESTEP = [ t/PER_EPOCH  for t in range(0, (MAX_TIMESTEPS+PER_EPOCH), PER_EPOCH) ]
# horizontal lines
# values taken from the paper; replace with your experiment results
if plot_ta:
    ncol += 1
    if args.env == 'MsPacman':
        plot_ta_bc_fun(TIMESTEP, mean_val=9145.42, std_val=955.94, color='purple', lw=2, linestyle='dotted')
        plt.text(x=30, y=10500, s='Trained Agent', fontsize='xx-large',color='purple')
    elif args.env == 'Alien':
        plot_ta_bc_fun(TIMESTEP, mean_val=7190.4, std_val=1251.27, color='purple', lw=2, linestyle='dotted')
        # plt.text(x=30, y=7100, s='Trained Agent', fontsize='xx-large',color='purple')
    elif args.env == 'MontezumaRevenge':
        plot_ta_bc_fun(TIMESTEP, mean_val=1108.0, std_val=1057.14, color='purple', lw=2, linestyle='dotted')
        # plt.text(x=32, y=1400, s='Trained Agent', fontsize='xx-large',color='purple')
    elif args.env == 'Freeway':
        plot_ta_bc_fun(TIMESTEP, mean_val=32.92, std_val=0.27, color='purple', lw=2, linestyle='dotted')
        # plt.text(x=30, y=33.45, s='Trained Agent', fontsize='xx-large',color='purple')
    elif args.env == 'NameThisGame':
        plot_ta_bc_fun(TIMESTEP, mean_val=9969.0, std_val=1910.91, color='purple', lw=2, linestyle='dotted')
        # plt.text(x=30, y=10000, s='Trained Agent', fontsize='xx-large',color='purple')
    elif args.env == 'Gopher':
        plot_ta_bc_fun(TIMESTEP, mean_val=6972.4, std_val=2190.26, color='purple', lw=2, linestyle='dotted')
        # plt.text(x=34, y=7100, s='Trained Agent', fontsize='xx-large',color='purple')
    else:
        print("Invalid game!!!")
        sys.exit()

if plot_bc:
    ncol += 1
    if args.env == 'MsPacman':
        plot_ta_bc_fun(TIMESTEP, mean_val=1776.6, std_val=993.94, color='black', lw=2, linestyle='--')
        plt.text(x=30, y=1100, s='Behavior Cloning', fontsize='xx-large',color='black')
    elif args.env == 'Alien':
        plot_ta_bc_fun(TIMESTEP, mean_val=839.2, std_val=718.72, color='black', lw=2, linestyle='--')
        # plt.text(x=30, y=900, s='Behavior Cloning', fontsize='xx-large',color='black')
    elif args.env == 'MontezumaRevenge':
        plot_ta_bc_fun(TIMESTEP, mean_val=174.0, std_val=205.73, color='black', lw=2, linestyle='--')
        # plt.text(x=32, y=250, s='Behavior Cloning', fontsize='xx-large',color='black')
    elif args.env == 'Freeway':
        plot_ta_bc_fun(TIMESTEP, mean_val=25.06, std_val=1.48, color='black', lw=2, linestyle='--')
        # plt.text(x=30, y=25.5, s='Behavior Cloning', fontsize='xx-large',color='black')
    elif args.env == 'NameThisGame':
        plot_ta_bc_fun(TIMESTEP, mean_val=1491.2, std_val=530.55, color='black', lw=2, linestyle='--')
        # plt.text(x=30, y=1500, s='Behavior Cloning', fontsize='xx-large',color='purple')
    elif args.env == 'Gopher':
        plot_ta_bc_fun(TIMESTEP, mean_val=450.8, std_val=393.27, color='black', lw=2, linestyle='--')
        # plt.text(x=30, y=700, s='Behavior Cloning', fontsize='xx-large',color='black')
    else:
        print("Invalid game!!!")
        sys.exit()


plt.tick_params(axis='y',labelsize='x-large')
plt.xlabel('Steps (in millions)', fontsize='x-large')
plt.ylabel('Reward', fontsize='x-large')
if args.env=="NameThisGame": # make sure all plots start from 0
    plt.gca().set_ylim(bottom=0)

if not args.nolegend:
    plt.legend(loc=location, fontsize='xx-large', ncol=1)

header = game
plt.title('{}'.format(game), fontsize='xx-large')
if args.saveplot:
    assert args.fn != '', "must provide file names to save the plot"
    figname = '{}_{}_'.format(game, deep_rl_algo)
    figname += args.fn
    path = args.folder + '/' + args.env
    if not os.path.exists(path):
        os.makedirs(path)
    plt.tight_layout()
    plt.savefig('{}/{}/{}.pdf'.format(args.folder, args.env, figname))
    plt.savefig('{}/{}/{}.png'.format(args.folder, args.env, figname))
plt.show()
