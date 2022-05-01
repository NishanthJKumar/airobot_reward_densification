(define (problem block-task)
	(:domain blocksworld)
	(:objects claw - gripper subgoal0 subgoal goal - location obj - box)
	(:init
		(gripper_open claw)
		(object_below obj subgoal)
		(not (holding claw obj))
		(is_goal goal)
		(is_subgoal subgoal)
		(robot_at claw subgoal0)
		
	)
	(:goal (object_at obj goal))
)