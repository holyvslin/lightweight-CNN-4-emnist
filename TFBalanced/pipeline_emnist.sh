#! bin/bash



date
python train_model_capacity_normal_emnist.py | tee model/train_model_capacity_normal_emnist_balanced.log


date

python train_model_1x1_4_Conv3x3_decay_6w_relu_normal_emnist.py | tee model/train_model_1x1_4_Conv3x3_decay_6w_relu_normal_emnist_balanced.log
date
