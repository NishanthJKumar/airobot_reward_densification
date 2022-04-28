(define (domain pushing)
    
    (:requirements :typing :negative-preconditions :strips :equality)
    
    (:types 
        box - object
        gripper - object
        location - object
    )

    (:predicates
        (is_goal ?loc - location)
        (object_at ?o - box ?loc - location)
        (robot_at ?g - gripper ?loc - location)
    )

    (:action move-to-location
        :parameters (?g - gripper ?from ?to - location)
        :precondition (and
            (not (is_goal ?from))
            (not (is_goal ?to))
            (robot_at ?g ?from)
            (not (robot_at ?g ?to))
        )
        :effect (and 
            (not (robot_at ?g ?from))
            (robot_at ?g ?to)
        )
    )

    (:action push-box
        :parameter (?g - gripper ?b - box ?from ?to - location)
        :precondition (and
            (not (object_at ?b ?to))
            (not (robot_at ?g ?to))
            (robot_at ?g ?from)
            (object_at ?b ?from)
        )
        :effect (and
            (not (robot_at ?g ?from))
            (not (object_at ?b ?from))
            (robot_at ?g ?to)
            (object_at ?b ?to)
        )
    )
)
