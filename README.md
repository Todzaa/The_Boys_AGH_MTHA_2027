# The Boys AGH - Ball Possession

Deterministic football ball-possession estimator for the More Than A Hack 2027
Initial Challenge.

The algorithm predicts one label for every tactical-video frame:

- `H` — Barcelona possession;
- `A` — Athletic Club possession;
- `D` — dead ball;
- `X` — outside the graded periods.

## Method

The solution uses:

- organizer-provided automatic tracking data;
- geometric distances between the ball and players;
- temporal possession rules;
- deterministic dead-ball detection;
- NumPy and pandas.

