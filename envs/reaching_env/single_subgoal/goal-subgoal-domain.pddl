(define (domain reaching)
    
    (:requirements :typing :negative-preconditions :strips :equality)
    
    (:types 
        gripper - object 
        location - object
    )

    (:predicates
        (is_goal ?loc - location)
        (at ?g - gripper ?loc - location)
    )

    (:action move-to-subgoal
        :parameters (?g - gripper ?loc - location)
        :precondition (and
            (not (is_goal ?loc))
            (not (at ?g ?loc))
        )
        :effect (and
            (at ?g ?loc)
        )
    )

    (:action move-to-goal
        :parameters (?g - gripper ?loc ?goal - location)
        :precondition (and
            (at ?g ?loc)
            (is_goal ?goal)
            (not (is_goal ?loc))
            (not (at ?g ?goal))
        )
        :effect (and
            (at ?g ?goal)
            (not (at ?g ?loc))
        )
    )
)
