(define (domain reaching-multi-subgoal)
    
    (:requirements :typing :negative-preconditions :strips :equality)
    
    (:types 
        gripper - object 
        location - object
    )

    (:predicates
        (is_goal ?loc - location)
        (is_subgoal1 ?loc - location)
        (is_subgoal2 ?loc - location)
        (is_subgoal3 ?loc - location)
        (at ?g - gripper ?loc - location)
    )

    (:action move-to-subgoal1
        :parameters (?g - gripper ?loc - location)
        :precondition (and
            (is_subgoal1 ?loc)
            (not (at ?g ?loc))
        )
        :effect (and
            (at ?g ?loc)
        )
    )

    (:action move-to-subgoal2
        :parameters (?g - gripper ?loc1 ?loc2 - location)
        :precondition (and
            (is_subgoal1 ?loc1)
            (is_subgoal2 ?loc2)
            (at ?g ?loc1)
            (not (at ?g ?loc2))
        )
        :effect (and
            (at ?g ?loc2)
            (not (at ?g ?loc1))
        )
    )

    (:action move-to-subgoal3
        :parameters (?g - gripper ?loc1 ?loc2 - location)
        :precondition (and
            (is_subgoal2 ?loc1)
            (is_subgoal3 ?loc2)
            (at ?g ?loc1)
            (not (at ?g ?loc2))
        )
        :effect (and
            (at ?g ?loc2)
            (not (at ?g ?loc1))
        )
    )

    (:action move-to-goal
        :parameters (?g - gripper ?loc ?goal - location)
        :precondition (and
            (at ?g ?loc)
            (is_goal ?goal)
            (is_subgoal3 ?loc)
            (not (at ?g ?goal))
        )
        :effect (and
            (at ?g ?goal)
            (not (at ?g ?loc))
        )
    )
)
