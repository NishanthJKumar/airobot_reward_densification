(define (problem block-task)
	(:domain blocksworld)
	(:objects claw - gripper subgoal0 subgoal1 goal - location box1 - box)
	(:init
		(gripper_open claw)
		(not (holding claw box1))
		(is_goal goal)
		(is_subgoal1 subgoal1)
		(robot_at claw subgoal0)
		
	)
	(:goal (object_at box1 goal))
)