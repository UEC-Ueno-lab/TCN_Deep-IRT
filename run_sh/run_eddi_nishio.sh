#!/bin/bash
folds=(1 2 3 4 5)
# folds=(5)

# for fold in ${folds[@]}; do
#   python main.py \
#   --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/exp01/nishio/ \
#   --model_type convmem01 --epochs 60 --dataset assist2017_pid \
#   --batch_size=512 --fold $fold --lr 0.01 --skill_item 1
#         echo simu_item${item}_learner2000_sigma${sigma}
# done
for fold in ${folds[@]}; do
  python main.py \
  --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/eight_layer1219/ \
  --model_type convmem01 --epochs 80 --dataset Eddi \
  --batch_size=512 --fold $fold --lr 0.01 --skill_item 1
done