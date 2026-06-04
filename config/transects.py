"""
Transect and segment definitions for all disturbance types.

All values are absolute seconds elapsed since the start of the named video
file (30 FPS dataset stored in annotated_video/GP1/ and annotated_video/diver/).

For cross-video transects two entries appear in videos/starts/ends:
    starts = [start_in_first_video, 0]
    ends   = [end_of_first_video,   end_in_second_video]

Diver transects use the annotated_video/diver/ directory.
All other transects use annotated_video/GP1/.
"""

# ── Turtle-pass transects (17 total, 3 cross-video) ──────────────────────────
TURTLE_TRANSECTS = [
    {"id": "T1",  "videos": ["GX100534"],             "starts": [420],      "ends": [470]},
    {"id": "T2",  "videos": ["GX100534"],             "starts": [530],      "ends": [675]},
    {"id": "T3",  "videos": ["GX100534", "GX110534"], "starts": [690, 0],   "ends": [708, 120]},
    {"id": "T4",  "videos": ["GX110534"],             "starts": [210],      "ends": [270]},
    {"id": "T5",  "videos": ["GX110534"],             "starts": [350],      "ends": [420]},
    {"id": "T6",  "videos": ["GX110534"],             "starts": [515],      "ends": [572]},
    {"id": "T7",  "videos": ["GX110534"],             "starts": [615],      "ends": [675]},
    {"id": "T8",  "videos": ["GX110534", "GX120534"], "starts": [705, 0],   "ends": [708, 85]},
    {"id": "T9",  "videos": ["GX120534"],             "starts": [105],      "ends": [190]},
    {"id": "T10", "videos": ["GX120534"],             "starts": [210],      "ends": [255]},
    {"id": "T11", "videos": ["GX120534"],             "starts": [345],      "ends": [495]},
    {"id": "T12", "videos": ["GX120534"],             "starts": [540],      "ends": [610]},
    {"id": "T13", "videos": ["GX120534", "GX130534"], "starts": [645, 0],   "ends": [708, 20]},
    {"id": "T14", "videos": ["GX130534"],             "starts": [60],       "ends": [140]},
    {"id": "T15", "videos": ["GX130534"],             "starts": [168],      "ends": [250]},
    {"id": "T16", "videos": ["GX130534"],             "starts": [360],      "ends": [411]},
    {"id": "T17", "videos": ["GX130534"],             "starts": [450],      "ends": [505]},
]

# ── AUV transects (18 total, all single-video) ────────────────────────────────
AUV_TRANSECTS = [
    {"id": "A1",  "videos": ["GX180534"], "starts": [255], "ends": [280]},
    {"id": "A2",  "videos": ["GX180534"], "starts": [346], "ends": [374]},
    {"id": "A3",  "videos": ["GX180534"], "starts": [453], "ends": [470]},
    {"id": "A4",  "videos": ["GX180534"], "starts": [543], "ends": [568]},
    {"id": "A5",  "videos": ["GX180534"], "starts": [641], "ends": [662]},
    {"id": "A6",  "videos": ["GX190534"], "starts": [50],  "ends": [74]},
    {"id": "A7",  "videos": ["GX190534"], "starts": [146], "ends": [172]},
    {"id": "A8",  "videos": ["GX190534"], "starts": [230], "ends": [258]},
    {"id": "A9",  "videos": ["GX190534"], "starts": [341], "ends": [367]},
    {"id": "A10", "videos": ["GX190534"], "starts": [415], "ends": [438]},
    {"id": "A11", "videos": ["GX190534"], "starts": [546], "ends": [570]},
    {"id": "A12", "videos": ["GX190534"], "starts": [630], "ends": [672]},
    {"id": "A13", "videos": ["GX200534"], "starts": [8],   "ends": [38],  "prev_video": "GX190534"},
    {"id": "A14", "videos": ["GX200534"], "starts": [90],  "ends": [120]},
    {"id": "A15", "videos": ["GX200534"], "starts": [188], "ends": [218]},
    {"id": "A16", "videos": ["GX200534"], "starts": [291], "ends": [322]},
    {"id": "A17", "videos": ["GX200534"], "starts": [428], "ends": [458]},
    {"id": "A18", "videos": ["GX200534"], "starts": [488], "ends": [518]},
]

# ── Control segments (20 total) ───────────────────────────────────────────────
# Ambient (no-robot) clips from the gap between the turtle session
# (GX100534–GX130534) and the AUV session (GX180534–GX200534).
# 5 evenly-spaced 60-second clips per video across GX140534–GX170534.
CONTROL_SEGMENTS = [
    # GX140534 — first gap video
    {"id": "C1",  "videos": ["GX140534"], "starts": [110], "ends": [170]},
    {"id": "C2",  "videos": ["GX140534"], "starts": [230], "ends": [290]},
    {"id": "C3",  "videos": ["GX140534"], "starts": [350], "ends": [410]},
    {"id": "C4",  "videos": ["GX140534"], "starts": [470], "ends": [530]},
    {"id": "C5",  "videos": ["GX140534"], "starts": [590], "ends": [650]},
    # GX150534 — second gap video
    {"id": "C6",  "videos": ["GX150534"], "starts": [110], "ends": [170]},
    {"id": "C7",  "videos": ["GX150534"], "starts": [230], "ends": [290]},
    {"id": "C8",  "videos": ["GX150534"], "starts": [350], "ends": [410]},
    {"id": "C9",  "videos": ["GX150534"], "starts": [470], "ends": [530]},
    {"id": "C10", "videos": ["GX150534"], "starts": [590], "ends": [650]},
    # GX160534 — third gap video
    {"id": "C11", "videos": ["GX160534"], "starts": [110], "ends": [170]},
    {"id": "C12", "videos": ["GX160534"], "starts": [230], "ends": [290]},
    {"id": "C13", "videos": ["GX160534"], "starts": [350], "ends": [410]},
    {"id": "C14", "videos": ["GX160534"], "starts": [470], "ends": [530]},
    {"id": "C15", "videos": ["GX160534"], "starts": [590], "ends": [650]},
    # GX170534 — fourth gap video
    {"id": "C16", "videos": ["GX170534"], "starts": [110], "ends": [170]},
    {"id": "C17", "videos": ["GX170534"], "starts": [230], "ends": [290]},
    {"id": "C18", "videos": ["GX170534"], "starts": [350], "ends": [410]},
    {"id": "C19", "videos": ["GX170534"], "starts": [470], "ends": [530]},
    {"id": "C20", "videos": ["GX170534"], "starts": [590], "ends": [650]},
]

# ── Baseline segments (4 total) ───────────────────────────────────────────────
# Long ambient windows used to compute per-metric baselines.
BASELINE_SEGMENTS = [
    {"id": "B1", "videos": ["GX100534"],              "starts": [0],       "ends": [419]},
    {"id": "B2", "videos": ["GX130534", "GX140534"],  "starts": [505, 0],  "ends": [708, 708]},
    {"id": "B3", "videos": ["GX170534", "GX180534"],  "starts": [0, 0],    "ends": [708, 254]},
    {"id": "B4", "videos": ["GX200534"],              "starts": [518],     "ends": [708]},
]

# ── Diver transects (4 total) ─────────────────────────────────────────────────
# 2024-11-14 experiment, annotated_video/diver/ at 30 FPS.
DIVER_TRANSECTS = [
    {
        "id": "diver_1",
        "videos": ["GX010533"],
        "starts": [595],   # 9:55
        "ends": [606],     # 10:06
    },
    {
        "id": "diver_2",
        "videos": ["GX010533", "GX020533"],
        "starts": [706, 0],  # 11:46 in GX010533
        "ends": [709, 11],   # 00:11 in GX020533
    },
    {
        "id": "diver_3",
        "videos": ["GX020533"],
        "starts": [157],   # 2:37
        "ends": [171],     # 2:51
    },
    {
        "id": "diver_4",
        "videos": ["GX020533"],
        "starts": [240],   # 4:00
        "ends": [250],     # 4:10
    },
]

# ── Shark transects ──────────────────────────────────────────────────────────
# Loaded at 30 FPS from annotated_video/GP1/.
SHARK_TRANSECTS = [
    {"id": "shark_1", "videos": ["GX160534"], "starts": [607], "ends": [614]},
]
