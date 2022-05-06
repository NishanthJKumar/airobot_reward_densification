#!/bin/bash

echo "Running experiment... outputing here > push_ppo_sparse.txt"
python main.py --domain push --algorithm ppo --reward_type sparse_handcrafted --pddl_type single_subgoal &> push_ppo_sparse.txt

echo "Running experiment... outputing here > push_ppo_single_subgoal.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type single_subgoal &> push_ppo_single_subgoal.txt

echo "Running experiment... outputing here > push_ppo_multi_subgoal.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type multi_subgoal &> push_ppo_multi_subgoal.txt

echo "Running experiment... outputing here > push_ppo_grid_based.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type grid_based &> push_ppo_grid_based.txt

echo "Running experiment... outputing here > push_ppo_single_subgoal_basic.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type single_subgoal --dynamic_shaping basic &> push_ppo_single_subgoal_basic.txt

echo "Running experiment... outputing here > push_ppo_multi_subgoal_basic.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping basic &> push_ppo_multi_subgoal_basic.txt

echo "Running experiment... outputing here > push_ppo_grid_based_basic.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type grid_based --dynamic_shaping basic &> push_ppo_grid_based_basic.txt