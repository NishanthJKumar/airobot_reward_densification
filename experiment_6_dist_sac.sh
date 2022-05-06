echo "Running experiment... outputing here > reach_sac_dense.txt"
python main.py --domain reach --algorithm sac --reward_type dense_handcrafted --pddl_type single_subgoal &> reach_sac_dense.txt

echo "Running experiment... outputing here > push_sac_dense.txt"
python main.py --domain push --algorithm sac --reward_type dense_handcrafted --pddl_type single_subgoal &> push_sac_dense.txt

echo "Running experiment... outputing here > reach_sac_single_subgoal_dist.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type single_subgoal --dynamic_shaping dist &> reach_sac_single_subgoal_dist.txt

echo "Running experiment... outputing here > reach_sac_multi_subgoal_dist.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping dist &> reach_sac_multi_subgoal_dist.txt

echo "Running experiment... outputing here > reach_sac_grid_based_dist.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type grid_based --dynamic_shaping dist &> reach_sac_grid_based_dist.txt

echo "Running experiment... outputing here > push_sac_single_subgoal_dist.txt"
python main.py --domain push --algorithm sac --reward_type pddl --pddl_type single_subgoal --dynamic_shaping dist &> push_sac_single_subgoal_dist.txt

echo "Running experiment... outputing here > push_sac_multi_subgoal_dist.txt"
python main.py --domain push --algorithm sac --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping dist &> push_sac_multi_subgoal_dist.txt

echo "Running experiment... outputing here > push_sac_grid_based_dist.txt"
python main.py --domain push --algorithm sac --reward_type pddl --pddl_type grid_based --dynamic_shaping dist &> push_sac_grid_based_dist.txt