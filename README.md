# Lucid Dreaming for Experience Replay (LiDER)
Implementation of paper [Lucid Dreaming for Experience Replay: Refreshing Past States with the Current Policy](https://arxiv.org/abs/2009.13736)

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

##### Currently available
A3CTBSIL, LiDER, and three LiDER ablations

##### Upcoming
LiDER-TA, LiDER-BC, data and plots, running on a cluster

## Installation

It is recommended to use a virtual environment for installation. Using ```conda``` as an example here.

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

For example, to train MsPacman in baseline A3CTBSIL:

    $./run_a3ctbsil.sh MsPacman

To run LiDER, our proposed framework:

    $./run_lider.sh MsPacman

To run the ablation LiDER-AddAll,

    $./run_lider_addall.sh MsPacman

To run the ablation LiDER-OneBuffer,

    $./run_lider_onebuffer.sh MsPacman

To run the ablation LiDER-SampleR

    $./run_lider_sampleR.sh MsPacman

Coming soon:

To run the extension LiDER-TA

To run the extension LiDER-BC

## Checkpointing

By default, our implementation checkpoints the training process every 1 million training steps and dumps results to the ```results``` folder (which will be created if not already exists), including:
* Current model parameters: folder `model_checkpoints`

* Best model parameters: folder `model_best`

* Testing rewards: pickle file with pattern `[gamename]_v4-a3c-rewards.pkl`)

* Additional statistics used for analyses, see Section 4.2 of the paper: pickle files with pattern `[gamename]_v4-a3c-dict-*.pkl`

* Replay buffers: pickle files with pattern `[gamename]_v4-*mem-*.pkl`

At the beginning of a run, our implementation looks for whether a folder for the same experiment setting already exists for the current experiment. If so, it will resume from the last saved checkpoint; otherwise, it will create a new experiment folder and start training from scratch.

The experiment folder will be created under `results` during training based on the current setting. For example. the experiment folder for training LiDER on MsPacman will be named `MsPacmanNoFrameskip_v4_rawreward_transformedbell_sil_prioritymem_lider_[today's date]`

The checkpointing frequency can be changed by modifying argument ```--checkpoint-freq=1``` to any number between 0 and 50. For example, ```--checkpoint-freq=5``` means to checkpoint every 5 million training steps.

The argument ```--checkpoint-buffer``` controls whether to save the entire replay buffer. When enabled, the replay buffer(s) will be saved at each checkpoint; previously collected experiences will be loaded when training is resumed.

Note that saving replay buffers can be memory-intensive. If you don't have enough capacity, you can either increase ```--checkpoint-freq``` to checkpoint less frequently, or remove line ```--checkpoint-buffer``` to save only the model parameters. If replay buffers were not saved, when resuming training you will start with an empty buffer(s).  


## Data and plots

Coming soon

## Acknowledgements
We thank Gabriel V. de la Cruz Jr. for helpful discussions; his open-source code at [github.com/gabrieledcjr/DeepRL](github.com/gabrieledcjr/DeepRL) is used for training the behavior cloning models in this work. This research used resources of [Kamiak](https://hpc.wsu.edu/), Washington State University’s high-performance computing cluster. Assefaw Gebremedhin is supported by the NSF award IIS-1553528. Part of this work has taken place in the [Intelligent Robot Learning (IRL) Lab](https://irll.ca/) at the University of Alberta, which is supported in part by research grants from the Alberta Machine Intelligence Institute (Amii), CIFAR, and NSERC. Part of this work has taken place in the [Learning Agents Research Group (LARG)](https://www.cs.utexas.edu/users/pstone/) at UT Austin. LARG research is supported in part by NSF (CPS-1739964, IIS1724157, NRI-1925082), ONR (N00014-18-2243), FLI (RFP2-000), ARL, DARPA, Lockheed Martin, GM, and Bosch. Peter Stone serves as the Executive Director of Sony AI America and receives financial compensation for this work. The terms of this arrangement have been reviewed and approved by the University of Texas at Austin in accordance with its policy on objectivity in research.
