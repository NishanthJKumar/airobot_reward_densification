(define (problem task)
	(:domain pushing)
	(:objects claw - gripper box1 - box subgoal0 subgoal2 goal - location)
	(:init
		(is_goal goal)
        (robot_at claw subgoal0)
        (object_at box1 subgoal2)
    )
	(:goal (and 
		(object_at box1 goal))
	)
)
