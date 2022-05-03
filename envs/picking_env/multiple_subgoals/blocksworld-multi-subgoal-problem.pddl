(define (problem block-pick-task)
	(:domain blocksworld-multi-subgoal)
	(:objects claw - gripper subgoal0 subgoal1 subgoal2 subgoal3 goal - location box1 - box)
	(:init
		(gripper_open claw)
		(not (holding claw box1))
		(is_goal goal)
		(is_subgoal1 subgoal1)
		(is_subgoal2 subgoal2)
		(is_subgoal3 subgoal3)
		(robot_at claw subgoal0)
		
	)
	(:goal (object_at box1 goal))
)