#!/bin/bash
folds=(1 2 3 4 5)
layers=(6 5 4 3 2 1)
# folds=(5)
for layer in ${layers[@]}; do
    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_${layer}_last_1/ \
    --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer $layer\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 1
    done
    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_${layer}_last_1/ \
    --model_type window_exp --epochs 80 --dataset assist2009_pid --conv_nlayer $layer\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 1
    done

    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_${layer}_last_1/ \
    --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer $layer\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 1
    done

    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_${layer}_last_1/ \
    --model_type window_exp --epochs 80 --dataset statics --conv_nlayer $layer\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 0
    done
done

for layer in ${layers[@]}; do
    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_${layer}_last_1/ \
    --model_type window_exp --epochs 80 --dataset Eddi --conv_nlayer $layer\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 1
    done

    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_${layer}_last_1/ \
    --model_type window_exp --epochs 100 --dataset junyi --conv_nlayer $layer\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 0
    done
done
# for fold in ${folds[@]}; do
#   python main.py \
#   --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_1_last_1/ \
#   --model_type convmem01 --epochs 80 --dataset assist2015 \
#   --batch_size=512 --fold $fold --lr 0.01 --skill_item 0
# done