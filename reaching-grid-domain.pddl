(define (domain reaching-grid)
    
    (:requirements :typing :negative-preconditions :strips :equality)
    
    (:types 
        gripper - object 
        location - object
    )

    (:predicates
        (is_goal ?loc - location)
        (at ?g - gripper ?loc - location)
        (neighbors ?loc1 ?loc2 - location)
    )

    (:action move-to-location
        :parameters (?g - gripper ?from ?to - location)
        :precondition (and
            (at ?g ?from)
            (not (at ?g ?to))
            (not (is_goal ?to))
            (neighbors ?from ?to)
        )
        :effect (and
            (at ?g ?to)
            (not (at ?g ?from))
        )
    )

    (:action move-to-goal
        :parameters (?g - gripper ?from ?to - location)
        :precondition (and
            (at ?g ?from)
            (is_goal ?to)
            (not (at ?g ?to))
            (not (is_goal ?from))
            (neighbors ?from ?to)
        )
        :effect (and 
            (at ?g ?to)
        )
    )
)
