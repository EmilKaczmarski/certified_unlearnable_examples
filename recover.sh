#!/bin/bash -l

# Activate virtual environment
source ../pue-env/bin/activate

export config_path=configs/cifar100
export data_type=CIFAR100  #{CIFAR10, CIFAR100}
export method=PUE
export train_mode=normally_trained #{online_surrogate, normally_trained}
export recovery_rate=0.2

# Path to trained classifiers 
export surrogate_path=./checkpoints/${train_mode}/${method}/classwise/${data_type}

# Path to PUE surrogate:
#export surrogate_path=./checkpoints/${train_mode}/${method}/classwise/${data_type}/noise=0.25-u_p=10

for eta_loop in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
export eta=${eta_loop}
# Train the model
python recover.py  --version                 resnet18_recover \
                        --model_version           resnet18 \
                        --exp_name                recover_exp_best/${method}/${data_type}/recovery_rate=${recovery_rate}/eta=${eta} \
                        --config_path             ${config_path} \
                        --train_data_type         ${data_type} \
                        --test_data_type          ${data_type} \
                        --train_batch_size        128 \
                        --eta                     ${eta} \
                        --recover_rate            ${recovery_rate}\
                        --surrogate_path          ${surrogate_path} \
                        --load_model   \
                        --project      
                        #--use_train_subset   
done
