#!/bin/bash -l

# Activate virtual environment
source ../pue-env/bin/activate

# Define variables
export perturb_type=classwise
export base_version=resnet18
export config_path=configs/cifar100
export data_type=CIFAR100
export attack_type=min-min
export num_steps=1
export universal_stop_error=0.1
export epsilon=8
export step_size=0.8
export train_step=20
export robust_noise=0.25
export u_p=10
export universal_train_target='train_subset'
export exp_name=./pue_exp/${perturb_type}/${data_type}/noise=${robust_noise}-u_p=${u_p}

# Train an online surrogate and generate PUEs
python make_pue.py    --config_path          ${config_path} \
                      --train_data_type      ${data_type} \
                      --test_data_type       ${data_type} \
                      --version              ${base_version} \
                      --noise_shape          100 3 32 32 \
                      --epsilon              ${epsilon} \
                      --num_steps            ${num_steps} \
                      --step_size            ${step_size} \
                      --train_step           ${train_step}     \
                      --attack_type          ${attack_type} \
                      --perturb_type         ${perturb_type} \
                      --robust_noise         ${robust_noise}\
                      --avgtimes_perturb     ${u_p}  \
                      --exp_name             ${exp_name} \
                      --universal_stop_error ${universal_stop_error} \
                      --data_parallel \
                      --universal_train_target ${universal_train_target} \
                      --use_subset \
                      --pue
