#!/bin/bash
folds=(1 2 3 4 5)
layers=(6 5 4 3 2 1)
kernel_size7=(3)
# folds=(5)
# for kernel in ${kernel_size7[@]}; do
#     echo "layer: 7, kernel: "$kernel
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_7_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 7\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_7_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2009_pid --conv_nlayer 7\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_7_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 7\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_7_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset statics --conv_nlayer 7\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 0 --kernel_size $kernel
#     done
# done

# kernel_size6=(3, 4)
# kernel_size6=(3)
# for kernel in ${kernel_size6[@]}; do
#     echo "layer: 6, kernel: "$kernel
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_6_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 6\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_6_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2009_pid --conv_nlayer 6\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_6_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 6\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_6_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset statics --conv_nlayer 6\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 0 --kernel_size $kernel
#     done
# done

# kernel_size5=(3 4 5 6 7 8 9 10 11 12 13)
kernel_size5=(8 9 10 11 12 13)
# folds=(5)
# for kernel in ${kernel_size5[@]}; do
#     echo "layer: 5, kernel: "$kernel
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_5_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 5\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_5_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2009_pid --conv_nlayer 5\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_5_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 5\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_5_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset statics --conv_nlayer 5\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 0 --kernel_size $kernel
#     done
# done

# kernel_size4=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
# kernel_size4=(15 16 17 18 19 20 21 22 23 24 25)
# # folds=(5)
# for kernel in ${kernel_size4[@]}; do
#     echo "layer: 4, kernel: "$kernel
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_4_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 4\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_4_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2009_pid --conv_nlayer 4\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_4_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 4\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_4_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset statics --conv_nlayer 4\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 0 --kernel_size $kernel
#     done
# done

# kernel_size3=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
# kernel_size3=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
# for kernel in ${kernel_size3[@]}; do
#     echo "layer: 3, kernel: "$kernel
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_3_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 3\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done
#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_3_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2009_pid --conv_nlayer 3\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_3_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 3\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
#     done

#     for fold in ${folds[@]}; do
#     python main.py \
#     --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_3_last_${kernel}/ \
#     --model_type window_exp --epochs 80 --dataset statics --conv_nlayer 3\
#     --batch_size=512 --fold $fold --lr 0.01 --skill_item 0 --kernel_size $kernel
#     done
# done

# kernel_size2=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
kernel_size2=(27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
for kernel in ${kernel_size2[@]}; do
    echo "layer: 2, kernel: "$kernel
    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_2_last_${kernel}/ \
    --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 2\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
    done
    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_2_last_${kernel}/ \
    --model_type window_exp --epochs 80 --dataset assist2009_pid --conv_nlayer 2\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
    done

    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_2_last_${kernel}/ \
    --model_type window_exp --epochs 80 --dataset assist2017_pid --conv_nlayer 2\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 1 --kernel_size $kernel
    done

    for fold in ${folds[@]}; do
    python main.py \
    --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/window_exp/layer_2_last_${kernel}/ \
    --model_type window_exp --epochs 80 --dataset statics --conv_nlayer 2\
    --batch_size=512 --fold $fold --lr 0.01 --skill_item 0 --kernel_size $kernel
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