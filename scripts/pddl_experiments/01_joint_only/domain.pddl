(define (domain joint_only)
    (:requirements :strips :negative-preconditions)
    (:predicates
        (GroundedBeam ?beam)

        (BeamAtStorage ?beam)
        (BeamAtAssembled ?beam)
        (Joint ?earlierbeam ?laterbeam)
    )
    (:action assemble_beam
        :parameters (?beam)
        :precondition (and
            (BeamAtStorage ?beam)
            ; at least one neighboring beam must be assembled
            (or
                (GroundedBeam ?beam)
                (exists (?earlierbeam) (and
                    (Joint ?earlierbeam ?beam)
                    (BeamAtAssembled ?earlierbeam)
                ))
            )
        )
        :effect(and (not (BeamAtStorage ?beam)) (BeamAtAssembled ?beam))
    )
)