(define (domain pushing-grid)
    
    (:requirements :typing :negative-preconditions :strips :equality)
    
    (:types
        box - object
        gripper - object 
        location - object
    )

    (:predicates
        (is_goal ?loc - location)
        (neighbors ?loc1 ?loc2 - location)
        (object_at ?o - box ?loc - location)
        (robot_at ?g - gripper ?loc - location)
    )

    (:action move-to-location
        :parameters (?g - gripper ?from ?to - location)
        :precondition (and
            (not (is_goal ?to))
            (robot_at ?g ?from)
            (not (is_goal ?from))
            (neighbors ?from ?to)
            (not (robot_at ?g ?to))
        )
        :effect (and
            (robot_at ?g ?to)
            (not (robot_at ?g ?from))
        )
    )

    (:action push-box
        :parameters (?g - gripper ?b - box ?from ?to - location)
        :precondition (and
            (robot_at ?g ?from)
            (object_at ?b ?from)
            (neighbors ?from ?to)
            (not (robot_at ?g ?to))
            (not (object_at ?b ?to))
        )
        :effect (and 
            (robot_at ?g ?to)
            (object_at ?b ?to)
            (not (robot_at ?g ?from))
            (not (object_at ?b ?from))
        )
    )
    
    (:action move-to-goal
        :parameters (?g - gripper ?b - box ?from ?to - location)
        :precondition (and
            (is_goal ?to)
            (robot_at ?g ?from)
            (object_at ?b ?from)
            (not (is_goal ?from))
            (neighbors ?from ?to)
            (not (robot_at ?g ?to))
            (not (object_at ?b ?to))
        )
        :effect (and 
            (robot_at ?g ?to)
            (object_at ?b ?to)
            (not (robot_at ?g ?from))
            (not (object_at ?b ?from))
        )
    )
)
