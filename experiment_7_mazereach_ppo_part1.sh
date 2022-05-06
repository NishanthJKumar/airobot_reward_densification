echo "Running experiment... outputing here > mazereach_ppo_sparse.txt"
python main.py --domain mazereach --algorithm ppo --reward_type sparse_handcrafted --pddl_type single_subgoal --granularity 6 &> mazereach_ppo_sparse.txt

echo "Running experiment... outputing here > mazereach_ppo_dense.txt"
python main.py --domain mazereach --algorithm ppo --reward_type dense_handcrafted --pddl_type single_subgoal --granularity 6 &> mazereach_ppo_dense.txt

echo "Running experiment... outputing here > mazereach_ppo_single_subgoal_dist.txt"
python main.py --domain mazereach --algorithm ppo --reward_type pddl --pddl_type single_subgoal --granularity 6 --dynamic_shaping dist &> mazereach_ppo_single_subgoal_dist.txt

echo "Running experiment... outputing here > mazereach_ppo_multi_subgoal_dist.txt"
python main.py --domain mazereach --algorithm ppo --reward_type pddl --pddl_type multi_subgoal ---granularity 6 -dynamic_shaping dist &> mazereach_ppo_multi_subgoal_dist.txt

echo "Running experiment... outputing here > mazereach_ppo_grid_based_dist.txt"
python main.py --domain mazereach --algorithm ppo --reward_type pddl --pddl_type grid_based --granularity 6 --dynamic_shaping dist &> mazereach_ppo_grid_based_dist.txt