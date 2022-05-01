(define (domain blocksworld)
    (:requirements :strips :equality :typing :negative-preconditions)
    (:types 
        location gripper box - object
    )
    (:predicates 
        (object_below ?o - box ?loc - location)
        (object_at ?o - box ?loc - location)
        (robot_at ?g - gripper ?loc - location)
        (gripper_open ?g - gripper)
        (holding ?g - gripper ?x - object)
        (is_subgoal ?loc - location)
        (is_goal ?loc - location)
    )

    (:action move-to-subgoal
        :parameters (?g - gripper ?from ?to - location)
        :precondition (and
            (not (is_goal ?from))
            (not (is_goal ?to))
            (robot_at ?g ?from)
            (not (robot_at ?g ?to))
            (is_subgoal ?to)
        )
        :effect (and 
            (not (robot_at ?g ?from))
            (robot_at ?g ?to)
        )
    )

    (:action pickup
        :parameters (?claw - gripper ?obj - box ?loc - location)
        :precondition (and
            (robot_at ?claw ?loc)
            (object_below ?obj ?loc)
            (gripper_open ?claw)
        )
        :effect (and 
            (holding ?claw ?obj) 
            (not (gripper_open ?claw))
            (not (object_below ?obj ?loc))
        )
    )

    (:action move-obj-to-goal
        :parameters (?claw - gripper ?obj - box ?from ?to - location)
        :precondition (and 
            (holding ?claw ?obj)
            (not (gripper_open ?claw))
            (robot_at ?claw ?from)
            (not (robot_at ?claw ?to))
            (not (object_at ?obj ?to))
            (is_goal ?to)
        )
        :effect (and
            (robot_at ?claw ?to)
            (object_at ?obj ?to)
        )
    )

)