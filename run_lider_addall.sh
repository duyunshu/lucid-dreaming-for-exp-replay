#!/bin/bash
currentDate=`date +"%m-%d-%Y"`

# Four games to choose from: MsPacman, Alien, Freeway, MontezumaRevenge
game=$1

if [ "$game" == "Gopher" ]; then
    echo "Training Gopher..."
elif [ "$game" == "NameThisGame" ]; then
    echo "Training NameThisGame..."
elif [ "$game" == "MsPacman" ]; then
    echo "Training Ms.Pac-Man..."
elif [ "$game" == "Alien" ]; then
    echo "Training Alien..."
elif [ "$game" == "Freeway" ]; then
    echo "Training Freeway..."
elif [ "$game" == "MontezumaRevenge" ]; then
    echo "Training Montezuma's Revenge..."
else
    echo "invalid game!"
    echo "Choose from [Gopher, NameThisGame, MsPacman, Alien, Freeway, MontezumaRevenge]."
    echo "For example: ./run_lider_addall.sh MsPacman"
    echo "Exiting..."; exit
fi

python3 LiDER/run_experiment.py \
  --gym-env=${game}NoFrameskip-v4 \
  --parallel-size=17 \
  --max-time-step-fraction=0.5 \
  --use-mnih-2015 --input-shape=88 --padding=SAME \
  --unclipped-reward --transformed-bellman \
  --use-sil --priority-memory \
  --batch-size=32 \
  --checkpoint-buffer --checkpoint-freq=1 \
  --append-experiment-num=${currentDate} \
  --use-lider \
  --addall \
  # --use-gpu --cuda-devices=0 \ # uncomment for gpu support
