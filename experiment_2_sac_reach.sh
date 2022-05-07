#!/bin/bash

echo "Running experiment... outputing here > reach_sac_spase.txt"
python main.py --domain reach --algorithm sac --reward_type sparse_handcrafted --pddl_type single_subgoal &> reach_sac_sparse.txt

echo "Running experiment... outputing here > reach_sac_single_subgoal.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type single_subgoal &> reach_sac_single_subgoal.txt

echo "Running experiment... outputing here > reach_sac_multi_subgoal.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type multi_subgoal &> reach_sac_multi_subgoal.txt

echo "Running experiment... outputing here > reach_sac_grid_based.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type grid_based &> reach_sac_grid_based.txt

echo "Running experiment... outputing here > reach_sac_single_subgoal_basic.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type single_subgoal --dynamic_shaping basic &> reach_sac_single_subgoal_basic.txt

echo "Running experiment... outputing here > reach_sac_multi_subgoal_basic.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping basic &> reach_sac_multi_subgoal_basic.txt

echo "Running experiment... outputing here > reach_sac_grid_based_basic.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type grid_based --dynamic_shaping basic &> reach_sac_grid_based_basic.txt