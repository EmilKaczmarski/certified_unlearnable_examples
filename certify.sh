#!/bin/bash -l

# Activate virtual environment
source ../pue-env/bin/activate

# Define variables
export perturb_type=classwise
export noise_type=PUE
export base_version=resnet18
export config_path=configs/cifar10
export data_type=CIFAR10
export attack_type=min-min
export N=1000
export u_p=1
export sigma=0.25
export q=0.9
export train_noise=0.25

# Path the experiment name and the path to the PUE surrogate weights
export exp_name=./cert_exp/${perturb_type}/${data_type}/${noise_type}/train_noise=${train_noise}/u_p=${u_p}/q=${q}-sigma=${sigma}
export surrogate_path=./checkpoints/online_surrogate/${noise_type}/${perturb_type}/${data_type}/noise=${train_noise}-u_p=${u_p}

# Provide the experiment name and the path to the baseline surrogate weights
#export exp_name=./cert_exp/${perturb_type}/${data_type}/${noise_type}/q=${q}-sigma=${sigma}
#export surrogate_path=./checkpoints/online_surrogate/${noise_type}/${perturb_type}/${data_type}

# Certify
python certify.py --exp_name ${exp_name} \
                  --config_path ${config_path} \
                  --surrogate_path ${surrogate_path} \
                  --train_data_type ${data_type} \
                  --test_data_type ${data_type} \
                  --version ${base_version} \
                  --perturb_type ${perturb_type} \
                  --sigma ${sigma} \
                  --q ${q} \
                  --N ${N} \
                  --data_parallel
