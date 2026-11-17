"""Sciezki, fakty o meczu i parametry metody. Wszystkie progi w jednym miejscu."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR") or ROOT.parent / "dane_od_organizatorow")
REPO_DIR = Path(os.environ.get("MTAH_REPO") or ROOT.parent / "more-than-a-hack-2027")
METRICA_DIR = Path(os.environ.get("METRICA_DIR") or ROOT.parent / "metrica_sample_data")
OUT_DIR = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "cache"
LOG_DIR = OUT_DIR / "logs"
MATCH_DIR = OUT_DIR / "match"
TEAM_NAME = "the_boys_agh"

STEM = "BAR-ATH_tactical"
FPS = 25
N_FRAMES = 199319
PITCH = (105.3, 68.0)
HOME, AWAY, DEAD, ABSTAIN = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
NOBODY = -1
HALVES = ((11185, 11185, 90652), (110808, 113742, 190198))

BALL_MAX_SPEED = 45.0
BALL_JUMP_TOL = 1.0
BALL_MIN_RUN = 3
BALL_SMOOTH = 2
BALL_GAP_FILL = 10

CONTROL_RADIUS = 2.0
CONTROL_MARGIN = 0.2
TOUCH_PAIR = 75
TOUCH_MIN_DV = 3.0
TOUCH_LAG = 12
TOUCH_RADIUS = 3.5
TOUCH_MARGIN = 0.5
TURNOVER_FRAMES = 13

CONTROL_BALL_SPEED = 12.0
MIN_SPELL = 50

BALL_OUT_MARGIN = 0.5
EXIT_HORIZON = 25
EXIT_MIN_GAP = 50
EXIT_MIN_SPEED = 3.0
EXIT_NEAR = 6.0
DEAD_SPEED = 1.1
DEAD_SMOOTH = 25
DEAD_MIN = 75
DEAD_MERGE = 25
LIVE_MIN = 50
DEAD_EXTEND = 1.7
BALL_LIVE_SPEED = 2.0
BALL_LIVE_SMOOTH = 12

BASELINE_VOTE = 50
