#!/bin/bash
folds=(1 2 3 4 5)
layers=(6 5 4 3 2 1)
kernel_size7=(3)
# folds=(5)
for kernel in ${kernel_size7[@]}; do
    for fold in ${folds[@]}; do
    echo "layer: kernel: "$kernel
    echo "=============="
    done
done

kernel_size6=(3, 4)
for kernel in ${kernel_size6[@]}; do
    for fold in ${folds[@]}; do
    echo $fold
    echo $kernel
    echo "=============="
    done
done

kernel_size5=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
# folds=(5)
for kernel in ${kernel_size5[@]}; do
    for fold in ${folds[@]}; do
    echo $fold
    echo $kernel
    echo "=============="
    done
done

kernel_size4=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
# folds=(5)
for kernel in ${kernel_size4[@]}; do
    for fold in ${folds[@]}; do
    echo $fold
    echo $kernel
    echo "=============="
    done
done

kernel_size3=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50)
for kernel in ${kernel_size3[@]}; do
    for fold in ${folds[@]}; do
    echo $fold
    echo $kernel
    echo "=============="
    done
done


# for layer in ${layers[@]}; do
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_${layer}_last_1/ \
#     --model_type window_exp --epochs 80 --dataset Eddi --conv_nlayer $layer\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_${layer}_last_1/ \
#     --model_type window_exp --epochs 100 --dataset junyi --conv_nlayer $layer\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 0
#     done
# done
# for fold in ${folds[@]}; do
#   python main.py \
#   --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_1_last_1/ \
#   --model_type convmem01 --epochs 80 --dataset assist2015 \
#   --batch_size=512 --fold $fold --lr 0.01 --skill_item 0
# done