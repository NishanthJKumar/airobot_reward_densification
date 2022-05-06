echo "Running experiment... outputing here > reach_ppo_dense.txt"
python main.py --domain reach --algorithm ppo --reward_type dense_handcrafted --pddl_type single_subgoal &> reach_ppo_dense.txt

echo "Running experiment... outputing here > push_ppo_dense.txt"
python main.py --domain push --algorithm ppo --reward_type dense_handcrafted --pddl_type single_subgoal &> push_ppo_dense.txt

echo "Running experiment... outputing here > reach_ppo_single_subgoal_dist.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type single_subgoal --dynamic_shaping dist &> reach_ppo_single_subgoal_dist.txt

echo "Running experiment... outputing here > reach_ppo_multi_subgoal_dist.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping dist &> reach_ppo_multi_subgoal_dist.txt

echo "Running experiment... outputing here > reach_ppo_grid_based_dist.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type grid_based --dynamic_shaping dist &> reach_ppo_grid_based_dist.txt

echo "Running experiment... outputing here > push_ppo_single_subgoal_dist.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type single_subgoal --dynamic_shaping dist &> push_ppo_single_subgoal_dist.txt

echo "Running experiment... outputing here > push_ppo_multi_subgoal_dist.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping dist &> push_ppo_multi_subgoal_dist.txt

echo "Running experiment... outputing here > push_ppo_grid_based_dist.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type grid_based --dynamic_shaping dist &> push_ppo_grid_based_dist.txt