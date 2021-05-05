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
import pathlib
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

parser = argparse.ArgumentParser()
# general parameters
parser.add_argument('--env', type=str, default='MsPacman')
parser.add_argument('--max-timesteps', type=int, default=50)
parser.add_argument('--deep-rl-algo', type=str, default='a3c')
parser.add_argument('--result-folder', type=str, default='results')
parser.add_argument('--dict', type=str, default='sil', help="plot which dictionary")
parser.add_argument('--key', type=str, default='sil_ctr', help='dict keys')
parser.add_argument('--saveplot', action='store_true', help='save plot or not')
parser.set_defaults(saveplot=False)
parser.add_argument('--nolegend', action='store_true', help='dont show legend')
parser.set_defaults(nolegend=False)
parser.add_argument('--folder', type=str, default='plots', help="where to save the plot")
parser.add_argument('--fn', type=str, default='', help="plot name")

# Fig 3
parser.add_argument('--refresh-success', action='store_true', help='refresher worker success rate')
parser.set_defaults(refresh_success=False)
parser.add_argument('--refresh-gnew', action='store_true', help='comparing G with Gnew')
parser.set_defaults(refresh_gnew=False)

# Fig 4
parser.add_argument('--sil-old-used', action='store_true', help='old samples used')
parser.set_defaults(sil_old_used=False)
parser.add_argument('--batch-sample-usage-ratio', action='store_true', help='batch sample usage ratio')
parser.set_defaults(batch_sample_usage_ratio=False)
parser.add_argument('--sil-sample-usage-ratio', action='store_true', help='sil sample usage ratio')
parser.set_defaults(sil_sample_usage_ratio=False)
parser.add_argument('--sil-return-of-used-samples', action='store_true', help='return of used samples')
parser.set_defaults(sil_return_of_used_samples=False)

# Fig 5
parser.add_argument('--total-batch-usage', action='store_true', help='A3CTBSIL')
parser.set_defaults(total_batch_usage=False)
parser.add_argument('--total-used-return', action='store_true', help='A3CTBSIL')
parser.set_defaults(total_used_return=False)

args = parser.parse_args()

games = args.env.split(",")
deep_rl_algo = args.deep_rl_algo
game_env_type = 'NoFrameskip_v4'
location = 'lower right'
ncol = 1
num_data = [1, 2, 3] # detault three trials

gym_envs={}
for game in games:
    gym_envs[game] = game + game_env_type
print(gym_envs)

#sns.set_style("darkgrid")
sns.set(context='paper', style='darkgrid', rc={'figure.figsize':(7,5)})
LW = 2
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

def create_dataframe(rewards_all_trials, per_epoch=0, max_timesteps=0, key=''):
    if per_epoch == 0:
        per_epoch = PER_EPOCH
    if max_timesteps == 0:
        max_timesteps = MAX_TIMESTEPS
    timestep = [ t/per_epoch for t in range(0, (max_timesteps+per_epoch), per_epoch) ]
    print(timestep)
    d = {}

    all_rewards = []
    for data_idx, reward_data in enumerate(rewards_all_trials):
        rewards = []
        for r in sorted(reward_data[key].keys()):
            if r <= max_timesteps and r % per_epoch == 0:
                rewards.append(reward_data[key][r])
        all_rewards.append(rewards)
        d['{}'.format(data_idx+1)] = rewards
    df = pd.DataFrame(data=d, index=timestep)
    df.index.name = 'Timestep'
    df = df.iloc[::N]

    return df


def aggregate_fun(ex_type='', dictionary='sil', key='sil_ctr', result_folder='a3c',
                  num_data=[1,2,3], gym_env='MspacmanNoFrameskip_v4', game='MsPacman'):
    ''' creates dataframe and plots graph '''
    rewards_all_trials = []
    print(num_data)
    for data_idx in num_data:
        folder=('{}/{}/'.format(result_folder, args.deep_rl_algo) + gym_env + \
        '{}_{}/'.format(ex_type, data_idx) + \
        gym_env + '-{}-dict-{}.pkl'.format(deep_rl_algo, dictionary))
        print(folder)
        r_data = pickle.load(open(folder, 'rb'))
        rewards_all_trials.append(r_data)

    df = create_dataframe(rewards_all_trials, key=key)
    while len(rewards_all_trials):
        del rewards_all_trials[0]

    return df

def plot_fun(df, color='black', marker=None, markevery=3,
             markersize=MARKERSIZE, lw=LW, linestyle=None, label='test'):
    df_rewards_mean = df.mean(axis=1)
    df_rewards_std = df.std(axis=1)
    print("STD:", df_rewards_std)
    print("mean:", df_rewards_mean)

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

################################################
ex_type = '_rawreward_transformedbell_sil_prioritymem'
lider_ex_type = '_rawreward_transformedbell_sil_prioritymem_lider'

### refresher worker
if args.refresh_success:
    args.fn = "refresher_success_rate"
    added_refresh = aggregate_fun(ex_type=lider_ex_type, dictionary='rollout',
                                  key="rollout_added_ctr",
                                  result_folder=args.result_folder,
                                  num_data=num_data, gym_env=gym_envs[game],
                                  game=game)
    total_refresh = aggregate_fun(ex_type=lider_ex_type, dictionary='rollout',
                                  key="rollout_ctr",
                                  result_folder=args.result_folder,
                                  num_data=num_data, gym_env=gym_envs[game],
                                  game=game)
    rate_success_refresh = (added_refresh.div(total_refresh)) * 100

    plot_fun(df=rate_success_refresh, color='forest green',
             label='Successful Refresh')

    title = game+" Refresher Success Rate"
    y_label = "Percentage (%)"
    location = "best"

if args.refresh_gnew:
    args.fn = "Gnew_vs_G"
    avg_new_return = aggregate_fun(ex_type=lider_ex_type, dictionary='rollout',
                                   key="rollout_new_return",
                                   result_folder=args.result_folder,
                                   num_data=num_data, gym_env=gym_envs[game],
                                   game=game)

    avg_old_return = aggregate_fun(ex_type=lider_ex_type, dictionary='rollout',
                                   key="rollout_old_return",
                                   result_folder=args.result_folder,
                                   num_data=num_data, gym_env=gym_envs[game],
                                   game=game)

    count = aggregate_fun(ex_type=lider_ex_type, dictionary='rollout',
                          key="rollout_added_ctr", result_folder=args.result_folder,
                          num_data=num_data, gym_env=gym_envs[game], game=game)

    avg_new_return = avg_new_return.div(count)
    avg_old_return = avg_old_return.div(count)

    plot_fun(df=avg_new_return, color="forest green",
             label='Gnew')
    plot_fun(df=avg_old_return, color="orange", linestyle="dashed",
             label='G')

    title = game+ " Gnew vs. G"
    y_label = "Average TB Return"
    location = "best"


################################################
### SIL in LiDER
if args.sil_old_used:
    old_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil', key="sil_old_used",
                               result_folder=args.result_folder, num_data=num_data, gym_env=gym_envs[game], game=game)
    total_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil', key="sil_a3c_used",
                               result_folder=args.result_folder, num_data=num_data, gym_env=gym_envs[game], game=game)
    total_used = total_used.add(aggregate_fun(ex_type=lider_ex_type, dictionary='sil', key="sil_rollout_used",
                                result_folder=args.result_folder, num_data=num_data, gym_env=gym_envs[game], game=game))
    df=old_used.div(total_used)*100
    point1 = df.iloc[[1]]
    point25 = df.iloc[[25]]
    point50 = df.iloc[[50]]

    print("============================")
    avg = float(point1.mean(axis=1))
    std = float(point1.std(axis=1))
    print("Old samples used (%) by the SIL worker in LiDER at 1 million training steps: {}% (std {})".format(avg, std))
    avg = float(point25.mean(axis=1))
    std = float(point25.std(axis=1))
    print("Old samples used (%) by the SIL worker in LiDER at 25 million training steps: {}% (std {})".format(avg, std))
    avg = float(point50.mean(axis=1))
    std = float(point50.std(axis=1))
    print("Old samples used (%) by the SIL worker in LiDER at 50 million training steps: {}% (std {})".format(avg, std))
    sys.exit() # only display values here, no plot

if args.batch_sample_usage_ratio:
    args.fn = "batch_sample_usage_ratio"
    sil_a3c_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                 key="sil_a3c_used",
                                 result_folder=args.result_folder,
                                 num_data=num_data, gym_env=gym_envs[game],
                                 game=game)
    sil_rollout_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                     key="sil_rollout_used",
                                     result_folder=args.result_folder,
                                     num_data=num_data, gym_env=gym_envs[game],
                                     game=game)
    total_used = sil_a3c_used.add(sil_rollout_used)

    sil_a3c_sampled = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                    key="sil_a3c_sampled",
                                    result_folder=args.result_folder,
                                    num_data=num_data, gym_env=gym_envs[game],
                                    game=game)
    sil_rollout_sampled = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                        key="sil_rollout_sampled",
                                        result_folder=args.result_folder,
                                        num_data=num_data, gym_env=gym_envs[game],
                                        game=game)
    total_sampled = sil_a3c_sampled.add(sil_rollout_sampled)

    plot_fun(df=sil_rollout_used.div(total_sampled)*100,
             color='forest green', label='Buffer R')

    plot_fun(df=sil_a3c_used.div(total_sampled)*100,
             color='orange', linestyle="dashed", label="Buffer D")

    title = game+" Batch Sample Usage Ratio: \n Buffer D vs. Buffer R"
    y_label = "Percentage (%)"
    location = 'upper right'

if args.sil_sample_usage_ratio:
    args.fn = "sil_sample_usage_ratio"
    sil_a3c_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                 key="sil_a3c_used",
                                 result_folder=args.result_folder,
                                 num_data=num_data, gym_env=gym_envs[game],
                                 game=game)
    sil_rollout_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                     key="sil_rollout_used",
                                     result_folder=args.result_folder,
                                     num_data=num_data, gym_env=gym_envs[game],
                                     game=game)
    total_used = sil_a3c_used.add(sil_rollout_used)

    lider_R_userate = sil_rollout_used.div(total_used)*100
    lider_D_userate = sil_a3c_used.div(total_used)*100

    plot_fun(df=lider_R_userate, color='forest green',
             label='Buffer R')

    plot_fun(df=lider_D_userate, color='orange',
             linestyle="dashed", label="Buffer D")

    title = game+" SIL Sample Usage Ratio: \n Buffer D vs. Buffer R"
    y_label = "Percentage (%)"
    location = 'center right'

if args.sil_return_of_used_samples:
    args.fn = "sil_return_of_used_samples"
    sil_a3c_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                 key="sil_a3c_used",
                                 result_folder=args.result_folder,
                                 num_data=num_data, gym_env=gym_envs[game],
                                 game=game)
    sil_a3c_used_return = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                        key="sil_a3c_used_return",
                                        result_folder=args.result_folder,
                                        num_data=num_data, gym_env=gym_envs[game],
                                        game=game)
    a3c_used_return = sil_a3c_used_return.div(sil_a3c_used)

    sil_rollout_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                     key="sil_rollout_used",
                                     result_folder=args.result_folder,
                                     num_data=num_data, gym_env=gym_envs[game],
                                     game=game)
    sil_rollout_used_return = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                            key="sil_rollout_used_return",
                                            result_folder=args.result_folder,
                                            num_data=num_data, gym_env=gym_envs[game],
                                            game=game)
    rollout_used_return = sil_rollout_used_return.div(sil_rollout_used)

    plot_fun(df=rollout_used_return, color='forest green',
             label='Buffer R')

    plot_fun(df=a3c_used_return, color='orange',
             linestyle="dashed", label='Buffer D')

    y_label = "Average TB Return"
    location = 'center right'
    title = game+" Return of Used Samples: \n Buffer D vs. Buffer R"

################################################
### SIL in A3CTBSIL vs. LiDER
if args.total_batch_usage:
    args.fn = 'total_batch_usage'
    sil_a3c_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                 key="sil_a3c_used", result_folder=args.result_folder,
                                 num_data=num_data, gym_env=gym_envs[game],
                                 game=game)
    sil_rollout_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                     key="sil_rollout_used",
                                     result_folder=args.result_folder, num_data=num_data,
                                     gym_env=gym_envs[game],
                                     game=game)
    total_used = sil_a3c_used.add(sil_rollout_used)
    sil_ctr = aggregate_fun(ex_type=lider_ex_type, dictionary='sil', key="sil_ctr",
                            result_folder=args.result_folder, num_data=num_data,
                            gym_env=gym_envs[game], game=game)
    total_sampled = sil_ctr*32
    plot_fun(df=total_used.div(total_sampled)*100, color='green',
             linestyle="dashdot", label='LiDER')


    base_used = aggregate_fun(ex_type=ex_type, dictionary='sil', key="sil_a3c_used",
                              result_folder=args.result_folder, num_data=num_data,
                              gym_env=gym_envs[game], game=game)
    base_sampled = aggregate_fun(ex_type=ex_type, dictionary='sil', key="sil_ctr",
                                 result_folder=args.result_folder, num_data=num_data,
                                 gym_env=gym_envs[game], game=game)
    base_sampled=base_sampled*32
    plot_fun(df=base_used.div(base_sampled)*100, color='violet red',
             linestyle="dotted", label='A3CTBSIL')


    title = game+" Batch Sample Usage Ratio: \n A3CTBSIL vs. LiDER"
    y_label = "Percentage (%)"
    location = 'best'

if args.total_used_return:
    args.fn = "total_used_return"
    base_used = aggregate_fun(ex_type=ex_type, dictionary='sil',
                              key="sil_a3c_used",
                              result_folder=args.result_folder, num_data=num_data,
                              gym_env=gym_envs[game], game=game)
    base_used_return = aggregate_fun(ex_type=ex_type, dictionary='sil',
                                     key="sil_a3c_used_return",
                                     result_folder=args.result_folder, num_data=num_data,
                                     gym_env=gym_envs[game], game=game)
    base_used_return = base_used_return.div(base_used)

    sil_rollout_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                     key="sil_rollout_used",
                                     result_folder=args.result_folder, num_data=num_data,
                                     gym_env=gym_envs[game], game=game)
    sil_rollout_used_return = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                            key="sil_rollout_used_return",
                                            result_folder=args.result_folder, num_data=num_data,
                                            gym_env=gym_envs[game], game=game)
    sil_rollout_used_return = sil_rollout_used_return.div(sil_rollout_used)

    sil_a3c_used = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                 key="sil_a3c_used",
                                 result_folder=args.result_folder, num_data=num_data,
                                 gym_env=gym_envs[game], game=game)
    sil_a3c_used_return = aggregate_fun(ex_type=lider_ex_type, dictionary='sil',
                                        key="sil_a3c_used_return",
                                        result_folder=args.result_folder, num_data=num_data,
                                        gym_env=gym_envs[game], game=game)
    sil_a3c_used_return = sil_a3c_used_return.div(sil_a3c_used)

    # count LiDER together
    lider_used_return = (sil_rollout_used_return.add(sil_a3c_used_return))/2

    plot_fun(df=lider_used_return, color='green',
             linestyle="dashdot", label='LiDER')

    plot_fun(df=base_used_return, color='violet red',
             linestyle="dotted", label='A3CTBSIL')

    y_label = "Averge TB Return"

    location = "best"
    title = game+" Return of Used Samples: \n A3CTBSIL vs. LiDER"
################################################

plt.tick_params(axis='y',labelsize='x-large')
plt.xlabel('Steps (in millions)', fontsize='x-large')
plt.ylabel(y_label, fontsize='x-large')

if not args.nolegend:
    plt.legend(loc=location, fontsize='xx-large', ncol=1)

plt.title('{}'.format(title), fontsize='xx-large')
if args.saveplot:
    assert args.fn != '', "must provide file names to save the plot"
    figname = args.fn
    path = args.folder + '/analysis'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.tight_layout()
    plt.savefig('{}/{}/{}.pdf'.format(args.folder, "analysis", figname))
    plt.savefig('{}/{}/{}.png'.format(args.folder, "analysis", figname))
plt.show()
