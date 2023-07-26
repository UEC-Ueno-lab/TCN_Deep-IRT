#!/bin/bash
items=(50 100 200 300)
# items=(50)
learners=(2000)
sigmas=(0.1 0.3 0.5 1.0)
# sigmas=(0.3 0.5 1.0)
# sigmas=(0.1)


for item in ${items[@]}; do
    for sigma in ${sigmas[@]}; do
  python main.py \
  --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/sim_exp02/nishio_conv2/ \
  --model_type convmem01 --epochs 250 --dataset simu_item${item}_learner2000_sigma${sigma} \
  --batch_size=512 --fold 1 --lr 0.01 --skill_item 0
        echo simu_item${item}_learner2000_sigma${sigma}
    done
done