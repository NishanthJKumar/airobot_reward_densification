(define (domain blocksworld-multi-subgoal)
    (:requirements :strips :equality :typing :negative-preconditions)
    (:types 
        location gripper box - object
    )
    (:predicates 
        (object_at ?o - box ?loc - location)
        (robot_at ?g - gripper ?loc - location)
        (gripper_open ?g - gripper)
        (holding ?g - gripper ?x - object)
        (is_subgoal1 ?loc - location)
        (is_subgoal2 ?loc - location)
        (is_subgoal3 ?loc - location)
        (is_goal ?loc - location)
    )

    (:action move-to-subgoal1
        :parameters (?g - gripper ?from ?to - location)
        :precondition (and
            (not (is_goal ?from))
            (not (is_goal ?to))
            (not (is_subgoal2 ?from))
            (not (is_subgoal2 ?to))
            (not (is_subgoal3 ?from))
            (not (is_subgoal3 ?to))
            (robot_at ?g ?from)
            (not (robot_at ?g ?to))
            (is_subgoal1 ?to)
        )
        :effect (and 
            (not (robot_at ?g ?from))
            (robot_at ?g ?to)
        )
    )

    (:action move-to-subgoal2
        :parameters (?g - gripper ?from ?to - location)
        :precondition (and
            (not (is_goal ?from))
            (not (is_goal ?to))
            (not (is_subgoal1 ?from))
            (not (is_subgoal1 ?to))
            (not (is_subgoal3 ?from))
            (not (is_subgoal3 ?to))
            (robot_at ?g ?from)
            (not (robot_at ?g ?to))
            (is_subgoal2 ?to)
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
            (is_subgoal2 ?loc)
            (gripper_open ?claw)
        )
        :effect (and 
            (holding ?claw ?obj) 
            (not (gripper_open ?claw))
        )
    )

    (:action move-to-subgoal3
        :parameters (?g - gripper ?from ?to - location ?obj - box)
        :precondition (and
            (not (is_goal ?from))
            (not (is_goal ?to))
            (not (is_subgoal2 ?from))
            (not (is_subgoal2 ?to))
            (not (is_subgoal1 ?from))
            (not (is_subgoal1 ?to))
            (robot_at ?g ?from)
            (not (robot_at ?g ?to))
            (holding ?g ?obj)
            (is_subgoal3 ?to)
        )
        :effect (and 
            (not (robot_at ?g ?from))
            (robot_at ?g ?to)
            (object_at ?obj ?to)
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