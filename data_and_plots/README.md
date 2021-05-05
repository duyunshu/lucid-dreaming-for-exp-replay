# Data and Plots for LiDER

Data and plots for [Lucid Dreaming for Experience Replay: Refreshing Past States with the Current Policy](https://arxiv.org/abs/2009.13736),

We provide here all data used in our paper for reproducibility.

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
####  Reproduce figures

First, download [LiDER_data.tar.gz](https://drive.google.com/file/d/1q2mXe2LIgE2g_nwCR2JVpwS0GA6IIAdU/view?usp=sharing) (file size ~100MB). Then, unzip to obtain all data used in the paper:

    $ tar -xzvf LiDER_data.tar.gz

This will produce folder `LiDER_data`, which contains reward pickle files for all experiments in the paper. This folder should be placed under folder `data_and_plots` (i.e., *this* folder):
```
lucid-dreaming-for-exp-replay
└──LiDER
└──data_and_plots
   └── LiDER_data
       └── Alien
       └── Freeway
       └── Gopher
       └── MontezumaRevenge
       └── MsPacman
       └── NameThisGame
   └── reproduce_fig.py
   └── ...
```

Reproduce figures in the paper with script `reproduce_fig.py`. Using `MsPacman` in examples below.

Figure 1

    $ python3 reproduce_fig.py --env=MsPacman --baseline --lidera3c --saveplot --fn=LiDER

Figure 6 (ablation studies)

    $ python3 reproduce_fig.py --env=MsPacman --baseline --lidera3c --addall --onebuffer --sampler --saveplot --fn=ablation

Figure 7 (extensions)

    $ python3 reproduce_fig.py --env=MsPacman --baseline --lidera3c --liderta --liderbc --saveplot --fn=extension

####  Reproduce analyses

Analyses were performed in `MsPacman` only, the data is located in folder `MsPacman`:
```
└── LiDER_data
   └── MsPacman
       └── analyses
       └── ...       
```
Reproduce analyses in the paper with script `reproduce_analyses.py`. Parameter `--env=MsPacman` by default; there are no data for other games. Parameter `--fn` is also set in the script for saving plots, no need to specify.

Figure 3

    $ python3 reproduce_analyses.py --refresh-success --saveplot
    $ python3 reproduce_analyses.py --refresh-gnew --saveplot

Table 1

    $ python3 reproduce_analyses.py --sil-old-used

Figure 4

    $ python3 reproduce_analyses.py --batch-sample-usage-ratio --saveplot
    $ python3 reproduce_analyses.py --sil-sample-usage-ratio --saveplot
    $ python3 reproduce_analyses.py --sil-return-of-used-samples --saveplot

Figure 5

    $ python3 reproduce_analyses.py --total-batch-usage --saveplot
    $ python3 reproduce_analyses.py --total-used-return --saveplot
