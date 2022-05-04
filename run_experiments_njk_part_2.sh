#!/bin/bash

echo "Running experiment... outputing here > push_sac_single_subgoal.txt"
python main.py --domain push --algorithm sac --reward_type pddl --pddl_type single_subgoal &> push_sac_single_subgoal.txt

echo "Running experiment... outputing here > push_sac_multi_subgoal.txt"
python main.py --domain push --algorithm sac --reward_type pddl --pddl_type multi_subgoal &> push_sac_multi_subgoal.txt

echo "Running experiment... outputing here > push_sac_grid_based.txt"
python main.py --domain push --algorithm sac --reward_type pddl --pddl_type grid_based &> push_sac_grid_based.txt

echo "Running experiment... outputing here > push_ppo_single_subgoal.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type single_subgoal &> push_ppo_single_subgoal.txt
