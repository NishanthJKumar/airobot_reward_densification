#!/bin/bash

echo "Running experiment... outputing here > reach_ppo_sparse.txt"
python main.py --domain reach --algorithm ppo --reward_type sparse_handcrafted --pddl_type single_subgoal &> reach_ppo_sparse.txt

echo "Running experiment... outputing here > reach_ppo_single_subgoal.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type single_subgoal &> reach_ppo_single_subgoal.txt

echo "Running experiment... outputing here > reach_ppo_multi_subgoal.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type multi_subgoal &> reach_ppo_multi_subgoal.txt

echo "Running experiment... outputing here > reach_ppo_grid_based.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type grid_based &> reach_ppo_grid_based.txt

echo "Running experiment... outputing here > reach_ppo_single_subgoal_basic.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type single_subgoal --dynamic_shaping basic &> reach_ppo_single_subgoal_basic.txt

echo "Running experiment... outputing here > reach_ppo_multi_subgoal_basic.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping basic &> reach_ppo_multi_subgoal_basic.txt

echo "Running experiment... outputing here > reach_ppo_grid_based_basic.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type grid_based --dynamic_shaping basic &> reach_ppo_grid_based_basic.txt