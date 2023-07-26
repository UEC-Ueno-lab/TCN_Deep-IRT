#!/bin/bash

folds=(1 2 3 4 5)

for i in ${folds[@]}
do
  python main.py \
  --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/nishio_v2data/ \
  --model_type convmem01 --epochs 250 --dataset assist2017_pid \
  --batch_size=1024 --fold $i --lr 0.01 --skill_item 1

  echo $i
done