(define (domain pushing-multi-subgoal)
    
    (:requirements :typing :negative-preconditions :strips :equality)
    
    (:types 
        gripper - object 
        location - object
        box - object
    )

    (:predicates
        (is_goal ?loc - location)
        (is_subgoal1 ?loc - location)
        (is_subgoal2 ?loc - location)
        (is_subgoal3 ?loc - location)
        (robot_at ?g - gripper ?loc - location)
        (object_at ?g - box ?loc - location)
    )

    (:action move-to-subgoal1
        :parameters (?g - gripper ?loc - location)
        :precondition (and
            (is_subgoal1 ?loc)
            (not (robot_at ?g ?loc))
        )
        :effect (and
            (robot_at ?g ?loc)
        )
    )

    (:action move-to-subgoal2
        :parameters (?g - gripper ?loc1 ?loc2 - location)
        :precondition (and
            (is_subgoal1 ?loc1)
            (is_subgoal2 ?loc2)
            (robot_at ?g ?loc1)
            (not (robot_at ?g ?loc2))
        )
        :effect (and
            (robot_at ?g ?loc2)
            (not (robot_at ?g ?loc1))
        )
    )

    (:action move-to-subgoal3
        :parameters (?g - gripper ?loc1 ?loc2 - location ?obj - box)
        :precondition (and
            (is_subgoal2 ?loc1)
            (is_subgoal3 ?loc2)
            (robot_at ?g ?loc1)
            (not (robot_at ?g ?loc2))
            (object_at ?obj ?loc1)
            (not (object_at ?obj ?loc2))

        )
        :effect (and
            (robot_at ?g ?loc2)
            (not (robot_at ?g ?loc1))
            (object_at ?obj ?loc2)
            (not (object_at ?obj ?loc1))
        )
    )

    (:action move-to-goal
        :parameters (?g - gripper ?loc ?goal - location ?obj - box)
        :precondition (and
            (robot_at ?g ?loc)
            (is_goal ?goal)
            (is_subgoal3 ?loc)
            (not (robot_at ?g ?goal))
            (object_at ?obj ?loc)
            (not (object_at ?obj ?goal))
        )
        :effect (and
            (robot_at ?g ?goal)
            (not (robot_at ?g ?loc))
            (object_at ?obj ?goal)
            (not (object_at ?obj ?loc))
        )
    )
)
