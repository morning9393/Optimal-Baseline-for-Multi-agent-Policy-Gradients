#!/bin/sh
env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="6x1"
agent_obsk=3
algo="mappo"
exp="vanilla"
seed_max=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --lr 5e-6 --std_x_coef 5 --std_y_coef 5e-1 --lr_decay 0.99 --seed 50 --n_training_threads 8 --n_rollout_threads 4 --num_mini_batch 40 --episode_length 1000 --num_env_steps 2000000 --ppo_epoch 5 --use_value_active_masks --use_eval --add_center_xy --use_state_agent
done
