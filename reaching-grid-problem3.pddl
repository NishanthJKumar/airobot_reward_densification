(define (problem task)
	(:domain reaching-task)
	(:objects claw - gripper loc0 loc1 loc2 loc3 loc4 loc5 loc6 loc7 goal - location)
	(:init
		(is_goal goal)
		(at claw loc0)
		
	)
	(:goal (and 
		(at claw goal))
	)
)