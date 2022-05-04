#!/bin/bash

# echo "Running experiment... outputing here > push_ppo_dense_handcrafted.txt"
# python main.py --domain push --algorithm ppo --reward_type dense_handcrafted --pddl_type grid_based --granularity 5 --episode_steps 100 &> push_ppo_dense_handcrafted.txt

echo "Running experiment... outputing here > reach_sac_dense_handcrafted.txt"
python main.py --domain reach --algorithm sac --reward_type dense_handcrafted --pddl_type grid_based --granularity 5 &> reach_sac_dense_handcrafted.txt