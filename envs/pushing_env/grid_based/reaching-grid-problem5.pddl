(define (problem task)
	(:domain reaching-grid)
	(:objects claw - gripper box1 - box loc0 loc1 loc2 loc3 loc4 loc5 loc6 loc7 goal - location)
	(:init
		(is_goal goal)
		(robot_at claw loc4)
        (object_at box1 loc1)
		(neighbors loc0 loc1)
		(neighbors loc0 loc4)
		(neighbors loc1 loc0)
		(neighbors loc1 loc2)
		(neighbors loc1 loc5)
		(neighbors loc2 loc1)
		(neighbors loc2 loc3)
		(neighbors loc2 loc6)
		(neighbors loc3 loc2)
		(neighbors loc3 loc7)
		(neighbors loc4 loc0)
		(neighbors loc4 loc5)
		(neighbors loc5 loc1)
		(neighbors loc5 loc4)
		(neighbors loc5 loc6)
		(neighbors loc6 loc2)
		(neighbors loc6 loc5)
		(neighbors loc6 loc7)
		(neighbors loc7 loc3)
		(neighbors loc7 loc6)
		(neighbors goal loc7)
		(neighbors loc7 goal)
		
	)
	(:goal (and 
		(object_at box1 goal))
	)
)