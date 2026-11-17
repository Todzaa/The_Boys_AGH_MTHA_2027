# The Boys AGH – Ball Possession (More Than A Hack 2027)

Deterministic, rule-based ball-possession labelling for the More Than A Hack 2027 Initial Challenge
(FC Barcelona – Athletic Club). Every frame of the tactical video gets one label:

| Label | Meaning |
|---|---|
| `H` | FC Barcelona (home) in possession, ball in play |
| `A` | Athletic Club (away) in possession, ball in play |
| `D` | dead ball (throw-in, corner, foul, goal, other stoppage) |
| `X` | outside the graded periods only (never on a graded frame) |

Only the organisers' automatic tracking data (ATD) is used. No machine learning, no video, no GPU: NumPy and pandas,
under a minute on a laptop CPU. The full method, validation and limitations are described in the technical report
(`report/the_boys_agh.tex`).

## Reproduce the submission

Requirements: Python 3.12.

```bash
pip install -r requirements.txt
python -m unittest          # unit tests on synthetic data (no competition data needed)
python run_match.py         # -> outputs/match/the_boys_agh.csv
```

`run_match.py` computes the labels twice and stops with an error if the two SHA-256 hashes differ. It then validates the
file with the organisers' `validate_submission` and prints a sanity check on the public five-minute window.

### Data layout

The repository contains code only. Competition data and outputs are never committed (`.gitignore`). By default the
code expects the folders next to the repository; each path can be overridden with an environment variable.

| What | Default path | Variable |
|---|---|---|
| organisers' data (`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`, directly or in `tactical/`) | `../dane_od_organizatorow` | `MTAH_DATA_DIR` |
| organisers' repository (scorer `evaluation/possession_metrics.py`, public GT in `evaluation/gt/`) | `../more-than-a-hack-2027` | `MTAH_REPO` |
| Metrica sample data (validation only) | `../metrica_sample_data` | `METRICA_DIR` |

Example (PowerShell):

```powershell
$env:MTAH_DATA_DIR = "C:\path\to\dane_od_organizatorow"
$env:MTAH_REPO     = "C:\path\to\more-than-a-hack-2027"
python run_match.py
```

Without the organisers' repository the labels are still written; only validation and the sanity check are skipped.

### Outputs (not committed)

| File | Content |
|---|---|
| `outputs/match/the_boys_agh.csv` | submission: `frame,label`, 199 319 rows, frames 0–199 318 |
| `outputs/logs/match_summary.json` | aggregate statistics of the run |
| `outputs/cache/atd.npz` | parsed ATD cache, rebuilt automatically when the source files change |

## Method

All thresholds live in `geo/config.py`.

1. **Ball filtering** (`geo/possession.py`). Short stretches of the ball track cut off by a jump faster than 45 m/s
   are skipped. The remaining track is smoothed with a ±2-frame rolling median and gaps of up to 10 frames are
   interpolated.
2. **Control.** A team controls the ball when its nearest player is within 2 m, the nearest rival is at least 0.2 m
   farther, and the ball is slower than 12 m/s. Otherwise nobody controls it (loose ball, ball in flight, duel, no data).
3. **Dead ball** (`geo/dead_ball.py`). `D` when the ball is more than 0.5 m beyond a line (until it is back on the
   pitch), when a disappearing ball is projected to cross a line, or when the median player speed stays below 1.1 m/s
   for 3 s (extended while below 1.7 m/s). A rolling ball means play. Live fragments shorter than 2 s between two
   stoppages are `D`.
4. **Touches.** A change of ball velocity of at least 3 m/s (0.48 s before vs after) with one team's player within
   3.5 m and the rival at least 0.5 m farther.
5. **Possession tracking.** Within each live stretch, the first team in control opens possession. The rival takes over
   after 0.5 s of net control advantage (decaying counter) or after two of its touches within 3 s; the change is dated
   back to the start of the advantage. A loose ball stays with the last team. A spell shorter than 2 s reverts to the
   previous team.
6. **Composition** (`geo/submission.py`). `X` only outside the halves; graded frames never get `X`.

## Validation

No labels of the competition match were used for training, and the private ground truth was never used. Thresholds
were chosen on the two public Metrica Sports sample games (the control-margin sweep also printed the public-window
sanity check, see the report), with a H/A/D reference built mechanically from their event data
(`geo/metrica.py`), scored with the organisers' `possession_metrics.py`:

```bash
python validate_metrica.py                   # both games, our pipeline vs the organisers' baseline
python validate_metrica.py --atd-like        # Metrica ball degraded to look like ATD (smoothed, offset), 4 seeds
python validate_metrica.py --sweep --atd-like  # parameter grid used to choose the touch thresholds
```

Clone the Metrica data first: `git clone https://github.com/metrica-sports/sample-data.git ../metrica_sample_data`.
The public five-minute window of the competition match (17 spells) is printed by `run_match.py` as a correctness
check only.

## Repository layout

```
run_match.py          whole match: labels, determinism check, validation, sanity check
validate_metrica.py   validation and parameter sweep on Metrica
geo/config.py         paths, match constants and every threshold
geo/atd.py            ATD reader (vectorised parser, cache, frame alignment check); python -m geo.atd prints coverage
geo/possession.py     ball filtering, distances, control, touches, possession tracking, baseline
geo/dead_ball.py      ball out of play, projected exit, low collective activity
geo/window.py         graded and live frame masks
geo/submission.py     label composition, CSV writing, spells, organisers' scorer loading
geo/metrica.py        Metrica loader, ATD-like degradation, event-based reference
tests/                unit tests on synthetic data
```

## Team

The Boys AGH – AGH University of Krakow: Miłosz Nowak, Michał Mróz, Tymon Szyler, Filip Różański.
