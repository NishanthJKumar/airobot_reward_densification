(define (problem task)
	(:domain reaching-multi-subgoal)
	(:objects claw - gripper subgoal1 subgoal2 subgoal3 goal - location)
	(:init
		(is_goal goal)
        (is_subgoal1 subgoal1)
        (is_subgoal2 subgoal2)
		(is_subgoal3 subgoal3)
	)
	(:goal (and 
		(at claw goal))
	)
)
