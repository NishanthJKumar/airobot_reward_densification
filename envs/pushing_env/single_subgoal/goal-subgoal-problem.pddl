(define (problem task)
	(:domain pushing)
	(:objects claw - gripper box1 - box box_start claw_start goal - location)
	(:init
		(is_goal goal)
        (object_at box1 box_start)
        (robot_at claw claw_start)
    )
	(:goal (and 
		(object_at box goal))
	)
)
