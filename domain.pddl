(define (domain reaching)
    
    (:requirements :typing :strips :equality)
    
    (:types 
        gripper - object 
        location - object
    )

    (:predicates
        (is_goal ?loc - location)
        (at ?g - gripper ?loc - location)
    )

    (:action move-to-subgoal
        :parameter (?g - gripper ?loc - location)
        :precondtition (and
            (not (is_goal ?loc))
            (not (at ?g ?loc))
        )
        :effect (and
            (at ?g ?loc)
        )
    )

    (:action move-to-goal
        :parameter (?g - gripper ?loc ?goal - location)
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
