"""Martwa pilka (D): pilka poza boiskiem albo wszyscy gracze ida (port the_boys dead_ball.py na m/s z ATD).

  1. pilka poza boiskiem: od obserwacji pilki dalej niz BALL_OUT_MARGIN za linia az do obserwacji z powrotem na boisku
     (wyrzut z autu, rozny, wykop od bramki); luka w obserwacjach przy pilce za linia to tez D
  2. niska aktywnosc: mediana predkosci graczy z ATD (m/s) wygladzona +/- DEAD_SMOOTH ponizej DEAD_SPEED przez
     >= DEAD_MIN klatek, rozszerzona w obie strony, poki ponizej DEAD_EXTEND; klatki, w ktorych zaufana pilka toczy sie
     szybciej niz BALL_LIVE_SPEED, to gra (wolna rozgrywka, nie przerwa); krotkie dziury miedzy przerwami sklejone
Liczone osobno w kazdej polowie (od wykopu), wygladzanie nie przechodzi przez przerwe.
"""
import warnings

import numpy as np
import pandas as pd

from . import config
from .possession import ball_speed, runs

MIN_TRACKS = 6


def outside(xy, margin=config.BALL_OUT_MARGIN, pitch=config.PITCH):
    """Punkt dalej niz margin poza prostokatem boiska (NaN -> False)."""
    x, y = xy[..., 0], xy[..., 1]
    return (x < -margin) | (x > pitch[0] + margin) | (y < -margin) | (y > pitch[1] + margin)


def ball_out(ball_xy, trusted):
    """Klatki od wyjscia pilki za linie do jej powrotu na boisko (wg ostatniej zaufanej obserwacji)."""
    out = outside(ball_xy) & trusted
    last = np.maximum.accumulate(np.where(trusted, np.arange(len(trusted)), -1))
    return (last >= 0) & out[np.maximum(last, 0)]


def projected_exit(ball_xy, known, horizon=config.EXIT_HORIZON, min_gap=config.EXIT_MIN_GAP,
                   min_speed=config.EXIT_MIN_SPEED, near=config.EXIT_NEAR, pitch=config.PITCH):
    """Klatki od przewidzianego przeciecia linii do nastepnej obserwacji pilki, ktora zniknela w drodze za linie.

    Luka >= min_gap klatek; predkosc z ostatnich 0.25 s przed zniknieciem >= min_speed; przy stalej predkosci pilka
    przecina linie najpozniej horizon klatek po ostatniej obserwacji; potwierdzenie: nastepna obserwacja <= near od tej
    linii albo poza boiskiem (aut, rozny, rzut od bramki). Bez potwierdzenia nic, bo pilka przyjeta przy linii to gra.
    """
    out = np.zeros(len(known), dtype=bool)
    if not horizon:
        return out
    back = int(round(0.25 * config.FPS))
    for a, b in runs(~np.asarray(known, dtype=bool)):
        if a == 0 or b == len(known) or b - a < min_gap:
            continue
        lo = max(a - 1 - back, 0)
        j = lo + int(np.flatnonzero(known[lo:a])[0])
        if a - 1 - j < 2:
            continue
        p, q = ball_xy[a - 1], ball_xy[b]
        v = (p - ball_xy[j]) * config.FPS / (a - 1 - j)
        if np.hypot(*v) < min_speed:
            continue
        hits = [(max((limit - p[axis]) / v[axis], 0.0), axis, limit)
                for axis, limit in ((0, 0.0), (0, pitch[0]), (1, 0.0), (1, pitch[1]))
                if (v[axis] < 0 if limit == 0.0 else v[axis] > 0)]
        if not hits:
            continue
        t, axis, limit = min(hits)
        frames = int(np.ceil(t * config.FPS - 1e-6))
        if frames > horizon or not (abs(q[axis] - limit) <= near or outside(q)):
            continue
        out[max(a - 1 + frames, a):b] = True
    return out


def activity(speed, ok, min_tracks=MIN_TRACKS):
    """Mediana predkosci graczy (m/s) w klatce; NaN przy mniej niz min_tracks graczach z predkoscia."""
    v = np.where(ok, speed, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, axis=1)
    return np.where(np.isfinite(v).sum(axis=1) >= min_tracks, med, np.nan)


def ball_moving(ball_xy, trusted, speed=config.BALL_LIVE_SPEED, half=config.BALL_LIVE_SMOOTH):
    """Klatki, w ktorych zaufana pilka sie toczy: mediana predkosci z sasiednich obserwacji +/- half > speed."""
    smooth = ball_speed(ball_xy, trusted, half)
    return np.isfinite(smooth) & (smooth > speed)


def low_activity(act, moving=None, speed=config.DEAD_SPEED, half=config.DEAD_SMOOTH, min_len=config.DEAD_MIN,
                 merge=config.DEAD_MERGE, extend=config.DEAD_EXTEND):
    """D z aktywnosci: wygladzona < speed przez >= min_len klatek, rozszerzona, poki < extend, potem sklejenie dziur
    < merge. NaN = nie D; moving (pilka sie toczy) wyklucza klatke z przerwy."""
    smooth = pd.Series(act).rolling(2 * half + 1, center=True, min_periods=1).median().to_numpy()
    calm = ~moving if moving is not None else np.ones(len(act), dtype=bool)
    low = np.isfinite(smooth) & (smooth < speed) & calm
    wide = np.isfinite(smooth) & (smooth < max(extend, speed)) & calm
    out = np.zeros(len(act), dtype=bool)
    for a, b in runs(wide):
        core = runs(low[a:b])
        if any(d - c >= min_len for c, d in core):
            out[a:b] = True
    found = runs(out)
    for (_a, b), (c, _d) in zip(found[:-1], found[1:]):
        if c - b < merge:
            out[b:c] = True
    return out


def close_gaps(mask, merge=config.DEAD_MERGE):
    """Dziury krotsze niz merge klatek miedzy przerwami wypelnione (skrawek gry miedzy dwiema przerwami to nie gra)."""
    out = mask.copy()
    found = runs(out)
    for (_a, b), (c, _d) in zip(found[:-1], found[1:]):
        if c - b < merge:
            out[b:c] = True
    return out


def dead_mask(ball_xy, trusted, speed, player_ok, halves=config.HALVES, live_min=config.LIVE_MIN,
              exit_horizon=config.EXIT_HORIZON, exit_gap=config.EXIT_MIN_GAP):
    """(D na klatke, D z pilki za linia, D z aktywnosci); poza polowami False (tam X). Pilka za linia obejmuje tez
    przewidziane wyjscie w luce (projected_exit). Krotkie dziury sklejane na sumie obu zrodel, bo miedzy D z autu a D
    z aktywnosci tez moze zostac skrawek gry."""
    n = len(trusted)
    by_out = np.zeros(n, dtype=bool)
    by_act = np.zeros(n, dtype=bool)
    dead = np.zeros(n, dtype=bool)
    act = activity(speed, player_ok)
    for _graded, kickoff, end in halves:
        by_out[kickoff:end] = (ball_out(ball_xy[kickoff:end], trusted[kickoff:end])
                               | projected_exit(ball_xy[kickoff:end], trusted[kickoff:end], exit_horizon, exit_gap))
        moving = ball_moving(ball_xy[kickoff:end], trusted[kickoff:end])
        by_act[kickoff:end] = low_activity(act[kickoff:end], moving)
        dead[kickoff:end] = close_gaps(by_out[kickoff:end] | by_act[kickoff:end], live_min)
    return dead, by_out, by_act
