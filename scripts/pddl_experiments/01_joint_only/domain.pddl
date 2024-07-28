(define (domain joint_only)
    (:requirements :strips :negative-preconditions)
    (:predicates
        (GroundedBeam ?beam)

        (BeamAtStorage ?beam)
        (BeamAtAssembled ?beam)
        (Joint ?earlierbeam ?laterbeam)

        (Robot ?robot)
        (RobotHold ?robot)
    )
    (:action assemble_beam_and_hold
        :parameters (?beam ?robot)
        :precondition (and
            (BeamAtStorage ?beam)
            (Robot ?robot)
            (not (RobotHold ?robot))
            ; at least one neighboring beam must be assembled
            (or
                (GroundedBeam ?beam)
                (exists (?earlierbeam) (and
                    (Joint ?earlierbeam ?beam)
                    (BeamAtAssembled ?earlierbeam)
                ))
            )
        )
        :effect(and 
            (not (BeamAtStorage ?beam)) 
            (BeamAtAssembled ?beam)
            (RobotHold ?robot)
            )
    )
    (:action release_hold
        :parameters (?beam ?robot)
        :precondition (and
            (Robot ?robot)
            (RobotHold ?robot)
            ; two built joints have been built
            (exists (?earlierbeam1 ?earlierbeam2)
                (and 
                    (or (Joint ?earlierbeam1 ?beam) (Joint ?beam ?earlierbeam1))
                    (or (Joint ?earlierbeam2 ?beam) (Joint ?beam ?earlierbeam2))
                    (BeamAtAssembled ?earlierbeam1)
                    (BeamAtAssembled ?earlierbeam2)
                )
            )
        )
        :effect(and 
            (not (RobotHold ?robot))
        )
    )
)