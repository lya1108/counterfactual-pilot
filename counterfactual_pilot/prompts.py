"""Prompt for the counterfactual action-choice pilot.

The model sees ONE pre-failure image and must pick exactly one of ACTIONS. It is deliberately
told nothing about the failure type, the original action, or the reference correction — the
point of the pilot is to see what it proposes on its own.

Frame of reference: the note below describes the fixed `agentview` camera and is the same for
every image, so it leaks nothing about a particular failure. It is included because the axis
mapping is otherwise ambiguous (computed from robosuite's table_arena.xml, camera at
pos=(0.5, 0, 1.35) looking along (-0.708, 0, -0.706)):

    world +y -> image right (coefficient 1.000)
    world +z -> image up    (0.708)
    world +x -> image down  (-0.706) and toward the camera (+0.708)

so LEFT/RIGHT are unambiguous in this view while FORWARD/BACKWARD run along the depth axis and
are visually entangled with UP/DOWN. Pilot 1 only injects lateral errors along y for that reason.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

# Known fragility of v1, measured 2026-09-10 on the 10 agentview frames. Permuting the label list
# *inside* USER_PROMPT (original / reversed / shuffled / removed entirely), with the
# structured-output choice list held fixed, changed the answer on 5 of 10 images. Permuting the
# structured-output list instead, with the prompt text fixed, changed nothing (0 of 10). So the
# in-text enumeration below — not the API-level constraint — is what the model is anchoring on.
# What survived every permutation: MOVE_LEFT on both lateral_offset_y_pos frames, MOVE_RIGHT on
# ep02, and CLOSE_GRIPPER on ep07 (never OPEN_GRIPPER anywhere). `close_too_high` -> MOVE_DOWN did
# not survive. Treat any v1 result that is not in that stable set as prompt-dependent, and keep
# v1 unchanged so the recorded runs stay comparable.

# The 8 candidate actions. This exact list is also handed to vLLM as a structured-output
# constraint, so the order and spelling here are what the server will allow.
ACTIONS: list[str] = [
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_UP",
    "MOVE_DOWN",
    "OPEN_GRIPPER",
    "CLOSE_GRIPPER",
]

SYSTEM_PROMPT = (
    "You are analysing a robot manipulation attempt that is about to fail. "
    "You answer with exactly one action label and nothing else."
)

USER_PROMPT = """This image is a single frame from a Franka Panda robot trying to grasp and lift the small cube on the table. The attempt shown here fails.

The frame is taken at the last moment when a different action could still have changed the outcome.

Viewpoint: a fixed camera in front of the table, looking down at it. In this image, moving RIGHT means moving toward the right side of the image, LEFT toward the left side, UP away from the table surface, and DOWN toward the table surface. FORWARD means toward the camera, BACKWARD means away from the camera.

Question: instead of what the robot actually did next, which single action should it have taken at this moment to avoid the failure?

Answer with exactly one of these labels:
MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, MOVE_BACKWARD, MOVE_UP, MOVE_DOWN, OPEN_GRIPPER, CLOSE_GRIPPER"""


# --------------------------------------------------------------------------- perception probe
# Pilot run 20260910_124400 answered CLOSE_GRIPPER on 3/3 `gripper_closed_early` frames — the
# opposite of the plausible correction. That is consistent with two very different causes: the
# gripper's open/closed state is not legible in the image, or it is legible and simply not used.
# This probe asks the factual question directly, so the two can be told apart. Unlike the
# counterfactual task, this one has a real ground truth: the gripper command at the captured step
# (`original_action[6]`), so it is scored, and that is a perception measurement, not task accuracy.
# v2. v1 wrapped the question in a scene preamble and a trailing "Answer with exactly one word:
# OPEN or CLOSED" line, and scored 3/6 on wrist frames — it answered OPEN for everything. Ablation
# on 2026-09-10 (3 closed + 3 open wrist frames, choice order fixed):
#     bare question                          6/6
#     bare question + system prompt          6/6
#     preamble + question                    5/6
#     preamble + question + "Answer ..."     3/6   <- degenerate, always the first in-text option
# The structured-output constraint already guarantees the format, so restating the options inside
# the prompt adds nothing and measurably biases the answer. The bare form scores 10/10 on BOTH
# agentview and wrist frames (7/7 open, 3/3 closed — a constant answer cannot do that), and 9/10
# with the system prompt added.
#
# But do not read that 10/10 as "gripper state is reliably legible". Running --robustness over the
# three variants below gives 19/30 with 0/10 images unanimous: `swapped` answers OPEN for all ten
# and `paraphrase` answers CLOSED for all ten, i.e. they stop looking at the image entirely. Only
# phrasings whose wording matches the label vocabulary (OPEN / CLOSED) discriminate. So this probe
# supports "the model can read gripper state under aligned phrasing", not "it reads it reliably".
GRIPPER_PROBE_VERSION = "gripper_v2"

GRIPPER_CHOICES: list[str] = ["OPEN", "CLOSED"]

GRIPPER_SYSTEM_PROMPT = "You answer with exactly one word and nothing else."

GRIPPER_PROBE_PROMPT = (
    "Are the gripper's two fingers currently OPEN (apart from each other) "
    "or CLOSED (pressed together)?"
)


# --------------------------------------------------------------------------- prompt variants
# A single answer says what the model proposed; it does not say whether the proposal is stable.
# Since permuting the in-text label list moved 5 of 10 answers, one prompt is not enough evidence
# to act on. These variants span the axis that actually moved things (the in-text enumeration)
# plus a wording paraphrase, so each image gets an agreement score instead of a lone answer.
_LABEL_HEADER = "Answer with exactly one of these labels:"

# Derived from USER_PROMPT rather than retyped, so `action_prompt(ACTIONS)` reproduces v1 exactly
# and the canonical variant stays comparable with the runs already recorded.
PROMPT_STEM = USER_PROMPT.split(_LABEL_HEADER)[0].rstrip()

PROMPT_STEM_ALT = """The picture below is one frame from an attempt by a Franka Panda robot to pick up the small cube resting on the table. This attempt does not succeed.

The frame is taken at the last instant where choosing differently would still have mattered.

Camera: fixed, in front of the table, angled downward. RIGHT and LEFT mean the right and left of the picture; UP means away from the table surface and DOWN means toward it; FORWARD means nearer to the camera and BACKWARD means further from it.

What single action would have been better than the one the robot actually took next?"""


def action_prompt(labels: list[str] | None, stem: str = PROMPT_STEM) -> str:
    """`labels=None` omits the in-text list entirely and leans on the structured-output constraint."""
    if labels is None:
        return stem
    return f"{stem}\n\n{_LABEL_HEADER}\n" + ", ".join(labels)


# Dropping the depth axis. When the correct answer is MOVE_RIGHT the model is right 72% of the time
# and its errors go to MOVE_FORWARD (16 of 22), never to MOVE_LEFT — it abandons the lateral axis
# rather than getting the sign wrong. Two readings fit that: the depth axis is visually ambiguous in
# this camera and MOVE_FORWARD is an escape hatch, or the model genuinely reads the offset as depth.
# Removing FORWARD/BACKWARD separates them — an escape hatch means those calls move to MOVE_RIGHT,
# a genuine depth reading means they move to another non-lateral label.
ACTIONS_6: list[str] = [a for a in ACTIONS if a not in ("MOVE_FORWARD", "MOVE_BACKWARD")]


def action_prompt_variants(seed: int = 7, actions: list[str] | None = None) -> list[tuple[str, str]]:
    import random

    actions = list(actions or ACTIONS)
    shuffled = list(actions)
    random.Random(seed).shuffle(shuffled)
    return [
        ("canonical", action_prompt(actions)),  # byte-identical to USER_PROMPT for the full set
        ("labels_reversed", action_prompt(list(reversed(actions)))),
        ("labels_shuffled", action_prompt(shuffled)),
        ("labels_omitted", action_prompt(None)),
        ("paraphrase", action_prompt(actions, PROMPT_STEM_ALT)),
    ]


def gripper_prompt_variants() -> list[tuple[str, str]]:
    return [
        ("canonical", GRIPPER_PROBE_PROMPT),
        ("swapped", "Are the gripper's two fingers currently CLOSED (pressed together) "
                    "or OPEN (apart from each other)?"),
        ("paraphrase", "Look at the two fingers of the robot gripper. Is there a gap between "
                       "them, or are they pressed against each other?"),
    ]


# --------------------------------------------------------------------------- decomposed probes
# The 8-way task is asymmetric: 98% when the answer is MOVE_LEFT, 72% when MOVE_RIGHT, and the
# failures abandon the lateral axis rather than flipping sign. That could break at three places —
# seeing which side the gripper is on, turning that into a direction, or choosing the lateral axis
# at all among eight options. These two binary probes cut the first two out of the third.
#
#   side     : pure perception, no action vocabulary at all
#   lateral  : the direction, with the axis already chosen for the model
#
# Both are scored against the recorded offset, and the frame sets are balanced left/right, so a
# probe that collapses to a constant answer scores exactly 50% instead of looking half-right.
# Phrasing stays bare — the ablation above showed a trailing "answer with one of ..." line is what
# degrades a binary probe, not naming the alternatives inside the question itself.
SIDE_VERSION = "side_v1"
SIDE_CHOICES: list[str] = ["LEFT", "RIGHT"]
SIDE_PROMPT = "Is the robot's gripper to the LEFT or to the RIGHT of the red cube?"


def side_prompt_variants() -> list[tuple[str, str]]:
    return [
        ("canonical", SIDE_PROMPT),
        ("swapped", "Is the robot's gripper to the RIGHT or to the LEFT of the red cube?"),
        ("paraphrase", "Which side of the red cube is the robot's gripper on?"),
    ]


LATERAL_VERSION = "lateral_v1"
LATERAL_CHOICES: list[str] = ["MOVE_LEFT", "MOVE_RIGHT"]
LATERAL_PROMPT = ("The robot's gripper is not lined up with the red cube. To line it up, should the "
                  "gripper move left or move right?")


def lateral_prompt_variants() -> list[tuple[str, str]]:
    return [
        ("canonical", LATERAL_PROMPT),
        ("swapped", "The robot's gripper is not lined up with the red cube. To line it up, should "
                    "the gripper move right or move left?"),
        ("paraphrase", "Which way must the gripper travel horizontally to sit directly over the "
                       "red cube?"),
    ]


def build_messages(
    image_data_url: str,
    user_prompt: str = USER_PROMPT,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """OpenAI-compatible chat messages for one image.

    Image block format verified against the remote vLLM 0.23.0 server serving `gemma4`
    (google/gemma-4-31b-it) on 2026-09-10.
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

# --------------------------------------------------------------------------- height probe
# The lead-time run (2026-09-18) found the outcome prediction blind on two of the four faults:
# `close_too_high` 0/36 across all three conditions and `gripper_closed_early` 5/36, while the two
# lateral faults were 41/48. The thinking traces say why -- on a frame where the fingertips are
# 8.5 cm above the cube the model states "the gripper is very close to the cube, ready to close",
# and on a frame with the fingers pressed together it states "the fingers are open". So the errors
# are asserted facts, not judgements.
#
# That leaves two readings, and they need different fixes: the vertical gap is not legible in this
# fixed view at 256 px, or it is legible and the outcome question does not recruit it. This probe
# asks the geometry alone, scored against the recorded eef/cube positions, exactly as the
# side/lateral probes did for the lateral axis.
#
# Scored on a margin, not a single cut: <=0.025 m counts as LEVEL, >=0.06 m as ABOVE, and the band
# between is left unscored, so a mid-descent frame cannot be graded on an arbitrary boundary. The
# nominal episodes alone split 17 LEVEL / 18 ABOVE, which means height perception can be measured
# on frames whose outcome label is constant -- the reading cannot be contaminated by the outcome.
HEIGHT_VERSION = "height_v1"
HEIGHT_CHOICES: list[str] = ["LEVEL", "ABOVE"]
HEIGHT_LEVEL_MAX = 0.025  # m, eef_z - cube_z
HEIGHT_ABOVE_MIN = 0.06
HEIGHT_PROMPT = ("Are the gripper's fingertips level with the red cube, or clearly above it?")


def height_prompt_variants() -> list[tuple[str, str]]:
    return [
        ("canonical", HEIGHT_PROMPT),
        ("swapped", "Are the gripper's fingertips clearly above the red cube, or level with it?"),
        ("paraphrase", "Look at the gripper's fingertips and the top of the red cube. Is there a "
                       "clear vertical gap between them, or are the fingertips down beside the cube?"),
    ]


# --------------------------------------------------------------------------- outcome + stated fact
# v2. v1 ended every item with "Does the attempt end with the cube held above the table?", which
# invites a Yes/No answer while the labels are SUCCESS/FAILURE. Run TEXT (2026-09-18) answered
# SUCCESS on all 12 item-conditions, and the thinking shows why it was not a physics gap: on
# `too_high_fail` it reasoned "since the gripper never touched the cube, the cube remains on the
# table ... No." and the constrained turn still returned SUCCESS. The reasoning was right and the
# label mapping was wrong -- the same failure this pilot already recorded for the gripper probe
# ("only phrasings whose wording matches the label vocabulary discriminate"). The question now uses
# succeed/fail so the answer vocabulary matches the choice list.
#
# Third arm of the decomposition. If the perception probes above read the state correctly but the
# outcome question still misses the fault, the gap is recruitment; if they read it wrongly, the gap
# is perception. Either way one more thing has to be ruled out: that the model simply does not know
# a grasp 8 cm too high fails. These items state the geometry in words with no image at all, so a
# wrong answer here is a physics gap and not a vision gap.
OUTCOME_VERSION = "outcome_v1"
OUTCOME_CHOICES: list[str] = ["SUCCESS", "FAILURE"]
OUTCOME_SYSTEM = ("You answer with exactly one label and nothing else.")

# (id, expected, text). Each fault has a matched control that differs in exactly one clause.
TEXT_ITEMS: list[tuple[str, str, str]] = [
    ("too_high_fail", "FAILURE",
     "A parallel-jaw gripper is centred directly over a 4 cm cube resting on a table. Its "
     "fingertips are 8 cm above the top of the cube. The fingers now close, and then the arm "
     "lifts 25 cm. Will this attempt succeed or fail?"),
    ("too_high_ok", "SUCCESS",
     "A parallel-jaw gripper is centred directly over a 4 cm cube resting on a table. Its "
     "fingertips are level with the cube, one finger on each side of it. The fingers now close, "
     "and then the arm lifts 25 cm. Will this attempt succeed or fail?"),
    ("closed_fail", "FAILURE",
     "A parallel-jaw gripper descends onto a 4 cm cube resting on a table. Its two fingers are "
     "already fully closed, pressed together, throughout the descent. The arm then lifts 25 cm. "
     "Will this attempt succeed or fail?"),
    ("closed_ok", "SUCCESS",
     "A parallel-jaw gripper descends onto a 4 cm cube resting on a table. Its two fingers are "
     "open, one passing down each side of the cube. The fingers then close, and the arm lifts "
     "25 cm. Will this attempt succeed or fail?"),
    ("lateral_fail", "FAILURE",
     "A parallel-jaw gripper is 8 cm to the side of a 4 cm cube resting on a table, with its "
     "fingertips level with the cube and nothing between them. The fingers now close, and then "
     "the arm lifts 25 cm. Will this attempt succeed or fail?"),
    ("lateral_ok", "SUCCESS",
     "A parallel-jaw gripper is directly over a 4 cm cube resting on a table, with its fingertips "
     "level with the cube and one finger on each side. The fingers now close, and then the arm "
     "lifts 25 cm. Will this attempt succeed or fail?"),
]

# Fourth arm: the same outcome question as the lead-time run, but with one true sentence about the
# state prepended. The sentence is derived from the recorded eef/cube positions and the gripper
# command, never from the label, and it never names the outcome. If this recovers the two blind
# faults, the prescription is the one the lateral decomposition already pointed at -- ask about the
# state first, then the outcome -- rather than a better single question.
FACT_OUTCOME_PROMPT = (
    "This image is a single frame from a Franka Panda robot attempting to grasp the small cube on "
    "the table and lift it off the table surface.\n\n"
    "Viewpoint: a fixed camera in front of the table, looking down at it.\n\n"
    "Measured state of the gripper in this frame: {fact}\n\n"
    "Question: will this grasp attempt succeed or fail?"
)


FINGERS_PRESSED_MAX = 0.01  # m between the two finger joints; fully open is ~0.079


def state_fact(height_m: float, finger_gap_m: float) -> str:
    """One true sentence about the frame, built from the recorded state. No outcome, no advice.

    The finger clause comes from the measured gap, never from the gripper command: a successful
    grasp is commanded closed while the cube holds the fingers apart, so a command-based sentence
    would have stated the opposite of what the frame shows on every such frame.
    """
    vert = f"its fingertips are {height_m * 100:.1f} cm above the top of the cube"
    if height_m <= HEIGHT_LEVEL_MAX:
        vert = f"its fingertips are level with the cube ({height_m * 100:.1f} cm above its top)"
    fingers = (f"the two fingers are pressed together, {finger_gap_m * 100:.1f} cm apart"
               if finger_gap_m < FINGERS_PRESSED_MAX
               else f"the two fingers are {finger_gap_m * 100:.1f} cm apart")
    return f"{vert}, and {fingers}."
