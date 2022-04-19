(define (domain reaching)
    
    (:requirements :typing :negative-preconditions :strips :equality)
    
    (:types 
        gripper - object 
        location - object
    )

    (:predicates
        (at ?g - gripper ?loc - location)
        (neighbors ?loc1 ?loc2 - location)
    )

    (:action move-to-location
        :parameters (?g - gripper ?from ?to - location)
        :precondition (and
            (at ?g ?from)
            (not (at ?g ?to))
            (neighbors ?from ?to)
        )
        :effect (and
            (at ?g ?to)
            (not (at ?g ?from))
        )
    )
)
