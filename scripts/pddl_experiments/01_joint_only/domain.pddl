(define (domain joint_only)
    (:requirements :strips :negative-preconditions)
    (:predicates
        ; static predicate, type
        (GroundedBeam ?beam)
        (Joint ?earlierbeam ?laterbeam)
        (Robot ?robot)

        ; fluent predicate (state of the world)
        (BeamAtStorage ?beam)
        (BeamAtAssembled ?beam)

        (RobotHold ?robot ?beam)
        (RobotFree ?robot)
    )
    (:action assemble_beam_grounded
        :parameters (?beam ?robot)
        :precondition (and
            (BeamAtStorage ?beam)
            (GroundedBeam ?beam)
            (Robot ?robot)
            (RobotFree ?robot)
        )
        :effect(and 
            (not (BeamAtStorage ?beam)) 
            (BeamAtAssembled ?beam)
            )
    )
    (:action assemble_beam_and_hold
        :parameters (?beam ?robot)
        :precondition (and
            (BeamAtStorage ?beam)
            (Robot ?robot)
            (RobotFree ?robot)
            (not (GroundedBeam ?beam))
            ; at least one neighboring beam must be assembled
            (exists (?earlierbeam) (and
                (Joint ?earlierbeam ?beam)
                (BeamAtAssembled ?earlierbeam)
            ))
        )
        :effect(and 
            (not (BeamAtStorage ?beam)) 
            (BeamAtAssembled ?beam)
            (RobotHold ?robot ?beam)
            (not (RobotFree ?robot))
            )
    )
    (:action release_hold
        :parameters (?beam ?robot)
        :precondition (and
            (Robot ?robot)
            (RobotHold ?robot ?beam)
            (BeamAtAssembled ?beam)
            ; two built joints have been built
            (exists (?earlierbeam1 ?earlierbeam2)
                (and 
                    (not (= ?earlierbeam1 ?earlierbeam2))
                    (Joint ?earlierbeam1 ?beam)
                    (Joint ?earlierbeam2 ?beam)
                    (BeamAtAssembled ?earlierbeam1)
                    (BeamAtAssembled ?earlierbeam2)
                )
            )
        )
        :effect(and 
            (not (RobotHold ?robot ?beam))
            (RobotFree ?robot)
        )
    )
)