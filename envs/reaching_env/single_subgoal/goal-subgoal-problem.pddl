(define (problem task)
	(:domain reaching)
	(:objects claw - gripper subgoal goal - location)
	(:init
		(is_goal goal)
	)
	(:goal (and 
		(at claw goal))
	)
)
