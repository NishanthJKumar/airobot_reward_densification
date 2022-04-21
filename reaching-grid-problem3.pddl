(define (problem task)
	(:domain reaching-grid)
	(:objects claw - gripper loc0 loc1 loc2 loc3 loc4 loc5 loc6 loc7 goal - location)
	(:init
		(is_goal goal)
		(at claw loc0)
		(neighbors goal loc3)
		(neighbors loc3 goal)
		
	)
	(:goal (and 
		(at claw goal))
	)
)