# Lucid Dreaming for Experience Replay (LiDER)
Implementation of our paper [Lucid Dreaming for Experience Replay: Refreshing Past States with the Current Policy](https://arxiv.org/abs/2009.13736)

Tested in Ubuntu 18.04, Tensorflow 1.11, Python 3.6

Citation:
```
@article{Du2020LiDER,
  title={Lucid Dreaming for Experience Replay: Refreshing Past States with the Current Policy},
  author={Yunshu Du and Garrett Warnell and A. Gebremedhin and P. Stone and Matthew E. Taylor},
  journal={ArXiv},
  year={2020},
  volume={abs/2009.13736}
}
```

#### Currently available
A3CTBSIL, LiDER, three LiDER ablations, LiDER-TA, LiDER-BC

Data and plots

#### Upcoming
Analyses, running on a cluster

## Installation

It is recommended to use a virtual environment for installation. Using ```conda``` (version 4.7.10) as an example here.

Create a virtual environment (name "liderenv") from the ```env.yml``` file:

    $ conda env create -f env.yml

Activate the environment

    $ conda activate liderenv

This "liderenv" environment installs a _CPU-only_ TensorFlow version ```tensorflow==1.11``` to avoid the complexity of setting up [CUDA](https://docs.nvidia.com/cuda/) (our implementation can run in either CPU or GPU).

If you have a supported GPU and have already set up CUDA, install tensorflow-gpu by replacing ```tensorflow==1.11.0``` with ```tensorflow-gpu==1.11.0``` in the ```env.yml``` file. For more informatioon about GPU suppoert, see [Tensorflow website](https://www.tensorflow.org/install/gpu).

_Note: you may use a newer version of Tensorflow (e.g., 1.14) but there will be warning messages. However, this implementation has not been tested in TensorFlow 1.14 or higher_

## How to run
We provide several bash files that have been configured to run corresponding experiments in the paper (i.e., using the same parameters as in the paper). The bash file takes the first argument as the input to choose which game to train. Valid games are: ```Gopher```, ```NameThisGame```, ```Alien```, ```MsPacman```, ```Freeway```, and ```MontezumaRevenge```.

_Note: If you installed ```tensorflow-gpu```, uncomment line ```--use-gpu --cuda-devices=0``` in the bash file to enable running on GPU._

####  A3CTBSIL, LiDER, and LiDER ablation studies

For example, to train MsPacman in baseline A3CTBSIL:

    $./run_a3ctbsil.sh MsPacman

To run LiDER, our proposed framework:

    $./run_lider.sh MsPacman

To run the ablation LiDER-AddAll:

    $./run_lider_addall.sh MsPacman

To run the ablation LiDER-OneBuffer:

    $./run_lider_onebuffer.sh MsPacman

To run the ablation LiDER-SampleR:

    $./run_lider_sampleR.sh MsPacman

####  LiDER-TA and LiDER-BC
First, download [pretrained_models.tar.gz](https://drive.google.com/file/d/1Zz0VoB5hWf7OpHWoG-nPInCpX36gscBU/view?usp=sharing) (file size ~2GB). Then, unzip to obtain all pre-trained models used in the paper:

    $ tar -xzvf pretrained_models.tar.gz

This will produce folder `pretrained_models`, which contains the checkpoints for all pre-trained models. This folder should be placed under the main directory:
```
lucid-dreaming-for-exp-replay
└──LiDER
└──pretrained_models
   └── TA
        └── Alien
        └── Freeway
        └── Gopher
        └── MontezumaRevenge
        └── MsPacman
        └── NameThisGame
   └── BC
        └── Alien
        └── Freeway
        └── Gopher
        └── MontezumaRevenge
        └── MsPacman
        └── NameThisGame

```

Then, to run the extension LiDER-TA:

    $./run_lider_ta MsPacman

To run the extension LiDER-BC:

    $./run_lider_bc MsPacman

## <a name="ckpt"></a>Checkpointing

By default, our implementation checkpoints the training process every 1 million training steps and dumps results to the ```results``` folder (which will be created if not already exists), including:
* Current model parameters: folder `model_checkpoints`

* Best model parameters: folder `model_best`

* Testing rewards: pickle file with pattern `[gamename]_v4-a3c-rewards.pkl`)

* Additional statistics used for analyses, see Section 4.2 of the paper: pickle files with pattern `[gamename]_v4-a3c-dict-*.pkl`

* Replay buffers: pickle files with pattern `[gamename]_v4-*mem-*.pkl`

At the beginning of a run, our implementation looks for whether a folder for the same experiment setting already exists for the current experiment. If so, it will resume from the last saved checkpoint; otherwise, it will create a new experiment folder and start training from scratch.

The experiment folder will be created under `results` during training based on the current setting. For example. the experiment folder for training LiDER on MsPacman will be named `MsPacmanNoFrameskip_v4_rawreward_transformedbell_sil_prioritymem_lider_[trial_number]`.

`[trial_number]` is set by parameter `--append-experiment-num` in the bach file. By default, `--append-experiment-num=1`. To run more trials, set it to a different number, such as `--append-experiment-num=2` or `--append-experiment-num=3`.

The checkpointing frequency can be changed by modifying argument ```--checkpoint-freq=1``` to any number between 0 and 50. For example, ```--checkpoint-freq=5``` means to checkpoint every 5 million training steps.

The argument ```--checkpoint-buffer``` controls whether to save the entire replay buffer. When enabled, the replay buffer(s) will be saved at each checkpoint; previously collected experiences will be loaded when training is resumed.

Note that saving replay buffers can be memory-intensive. If you don't have enough capacity, you can either increase ```--checkpoint-freq``` to checkpoint less frequently, or remove line ```--checkpoint-buffer``` to save only the model parameters. If replay buffers were not saved, when resuming training you will start with empty buffer(s).  


## Data and plots

####  Reproduce figures and analyses in the paper
See folder [data_and_plots](https://github.com/duyunshu/lucid-dreaming-for-exp-replay/tree/master/data_and_plots).

####  Plot your experiment results
When running your own experiments, a `results` folder will be created to save results. For example, after training MsPacman using LiDER for three trials, the `results` folder will structure as follows:
```
lucid-dreaming-for-exp-replay
└──LiDER
└──pretrained_models
└──results
   └── a3c
        └── MsPacmanNoFrameskip_v4_rawreward_transformedbell_sil_prioritymem_lider_1
        └── MsPacmanNoFrameskip_v4_rawreward_transformedbell_sil_prioritymem_lider_2
        └── MsPacmanNoFrameskip_v4_rawreward_transformedbell_sil_prioritymem_lider_3
   └── log
        └── a3c
            └── MsPacmanNoFrameskip_v4
```

`results/a3c` saves model checkpoints and testing rewards (see descriptions under [Checkpointing](#ckpt)). `results/log` saves tensorboard events so that you can monitor the training process. To launch tensorboard, follow the standard procedure and set `--logdir` to corresponding log folders. See [TensorBoard website](https://www.tensorflow.org/tensorboard/get_started) for more instructions.

The `.pkl` files saved under `results/a3c/[experiment_name]` are used for plotting the testing results. Use the plotting script `generate_plots.py` to generate plots. For example, to plot A3CTBSIL and LiDER:

    $ python3 generate_plots.py --env=MsPacman --baseline --lidera3c --saveplot

Here is a list of available parameters in the plotting script:

* Parameters for general settings:
  * `--env`: which game to plot. For example, `--env=MsPacman`.
  * `--max-timesteps`: the number of time steps (in million) to plot. For example,`--max-timesteps=30` will plot results for 30 million steps. The default value is 50 million steps.
  * `--saveplot`: when enabled, save the plot; otherwise, only display the plot without saving. The default value is `False` (i.e., not saving).
  * `--nolegend`: when enabled, no legend is shown. The default value is `False` (i.e., show legend).
  * `--folder`: where to save the plot. By default, a new folder `plots` will be created.
  * `--fn`: file name of the plot. When `--saveplot` is enabled, `--fn` must be provided to save the plot.


* Parameters for plotting each algorithm:
  * `--baseline`: A3CTBSIL
  * `--lidera3c`: LiDER
  * `--addall`: LiDER-AddAll
  * `--onebuffer`: LiDER-OneBuffer
  * `--sampler`: LiDER-SampleR
  * `--liderta`: LiDER-TA
  * `--liderbc`: LiDER-BC

##### Note on plotting LiDER-TA and LiDER-BC
The values of the horizontal lines showing pretrained TAs and BCs' performance (Figure 7 of our paper) need to be supplied manually. The current values are taken from our paper. When running your own experiments, the TA and BC's performance will be evaluated for 50 episodes at the beginning of training. Their evaluation results will be stored under `pretrained_models/TA (or BC)/[game]/[game]-model-eval.txt`, including the episodic mean reward, the standard deviation, and the reward for each episode.


####  Plot your analyses
Analyses can be generated for your A3CTBSIL and LiDER experiments (_currently not supporting analysis for other extensions_). The results are also stored under the `results` folder. Use script `generate_analyses.py` to generate analyses plot. For example, to look at the "success rate" in refresher worker (Section 4.2.1):

    $ python3 generate_analyses.py --env=MsPacman --refresh-success --saveplot

Here is a list of available parameters in the plotting script:
  * Parameters for general settings are the same as above

  * Parameters for plotting each analysis:
    * Section 4.2.1:
      * `--refresh-success`: Success rate
      * `--refresh-gnew`: Gnew vs. G
    * Section 4.2.2:
      * `--sil-old-used`: Old samples used (_this does not generate plots, only displays values of the table_)
      * `--batch-sample-usage-ratio`: Batch sample usage ratio (LiDER)
      * `--sil-sample-usage-ratio`: SIL sample usage ratio
      * `--sil-return-of-used-samples`: Return of used samples (LiDER)
    * Section 4.2.3:
      * `--total-batch-usage`: batch sample usage ratio (A3CTBSIL & LiDER)
      * `--total-used-return`: Return of used samples (A3CTBSIL & LiDER)


## Acknowledgements
We thank Gabriel V. de la Cruz Jr. for helpful discussions; his open-source code at [github.com/gabrieledcjr/DeepRL](github.com/gabrieledcjr/DeepRL) is used for training the behavior cloning models in this work. This research used resources of [Kamiak](https://hpc.wsu.edu/), Washington State University’s high-performance computing cluster. Assefaw Gebremedhin is supported by the NSF award IIS-1553528. Part of this work has taken place in the [Intelligent Robot Learning (IRL) Lab](https://irll.ca/) at the University of Alberta, which is supported in part by research grants from the Alberta Machine Intelligence Institute (Amii), CIFAR, and NSERC. Part of this work has taken place in the [Learning Agents Research Group (LARG)](https://www.cs.utexas.edu/users/pstone/) at UT Austin. LARG research is supported in part by NSF (CPS-1739964, IIS1724157, NRI-1925082), ONR (N00014-18-2243), FLI (RFP2-000), ARL, DARPA, Lockheed Martin, GM, and Bosch. Peter Stone serves as the Executive Director of Sony AI America and receives financial compensation for this work. The terms of this arrangement have been reviewed and approved by the University of Texas at Austin in accordance with its policy on objectivity in research.
