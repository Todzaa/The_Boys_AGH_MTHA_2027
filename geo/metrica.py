"""Publiczne mecze Metrica (github.com/metrica-sports/sample-data) jako zbior walidacyjny spoza meczu konkursowego.

Sample_Game_1 i Sample_Game_2: tracking w CSV (gospodarze i goscie osobno, 25 fps, 0-1 x 105 x 68 m) i zdarzenia
(RawEventsData.csv) zsynchronizowane z trackingiem. Tracking wczytujemy do tej samej struktury co ATD (atd.Tracking),
wiec pipeline z run_match.label_match dziala bez zmian. Predkosc graczy (ATD ma ja w pliku) liczymy z pozycji.

Wzorzec H/A/D skladamy mechanicznie ze zdarzen (to przyblizenie, nie GT organizatorow):
  - posiadanie: druzyna ostatniego zdarzenia z pilka (PASS, RECOVERY, SET PIECE, SHOT, BALL LOST, BALL OUT); pojedynki
    (CHALLENGE), faule i kartki niczego nie zmieniaja, wiec pilka luzem zostaje przy ostatniej druzynie; przejecie
    tylko kontrolowane (controlled_teams): samo dotkniecie albo przechwyt z natychmiastowa strata go nie zmienia,
  - D: od konca ostatniego zdarzenia przed wznowieniem (pilka za linia, faul, gol, spalony) do wznowienia (SET PIECE
    albo PASS z GOAL KICK),
  - X poza polowami: przed pierwszym i po ostatnim zdarzeniu kazdej polowy.
Klatka zdarzenia (Start Frame, od 1) - 1 = indeks klatki trackingu.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .atd import Tracking

GAMES = (1, 2)
PITCH = (105.0, 68.0)
SPEED_SMOOTH = 5
ON_BALL = ("PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST", "BALL OUT")
IGNORED = ("CARD",)
TEAMS = {"Home": config.HOME, "Away": config.AWAY}
CONTROL_HOLD = 3 * config.FPS


def game_dir(game, root=None):
    d = Path(root or config.METRICA_DIR) / "data" / "Sample_Game_{}".format(game)
    if not d.is_dir():
        raise SystemExit("brak {}; sklonuj github.com/metrica-sports/sample-data albo ustaw METRICA_DIR".format(d))
    return d


def _team_xy(path):
    """(okresy, pozycje (n, gracze, 2) w 0-1, pilka (n, 2) w 0-1) z pliku RawTrackingData jednej druzyny."""
    df = pd.read_csv(path, skiprows=2, low_memory=False)
    values = df.iloc[:, 3:].to_numpy(dtype=np.float64)
    pairs = values.reshape(len(df), -1, 2)
    return df["Period"].to_numpy(), pairs[:, :-1], pairs[:, -1]


def speed(xy, fps=config.FPS, smooth=SPEED_SMOOTH):
    """(n, gracze) m/s z pozycji w metrach: roznica centralna, srednia ruchoma; NaN tam, gdzie brak pozycji."""
    v = np.full(xy.shape[:2], np.nan)
    step = np.hypot(*(xy[2:] - xy[:-2]).transpose(2, 0, 1)) * fps / 2
    v[1:-1] = step
    v = pd.DataFrame(v).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    return np.where(np.isfinite(xy[:, :, 0]), v, np.nan)


def load_tracking(game, root=None):
    """(atd.Tracking w metrach, okres na klatke) dla meczu Metrica."""
    d = game_dir(game, root)
    period, home, ball = _team_xy(d / "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game))
    _p, away, _b = _team_xy(d / "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game))
    scale = np.array(PITCH)
    xy = np.concatenate([home, away], axis=1) * scale
    players = np.concatenate([xy, speed(xy)[:, :, None]], axis=2).astype(np.float32)
    ball3 = np.concatenate([ball * scale, np.full((len(ball), 1), np.nan)], axis=1).astype(np.float32)
    slot_team = np.array([config.HOME] * home.shape[1] + [config.AWAY] * away.shape[1], dtype=np.int8)
    frame_id = np.arange(1, len(ball) + 1, dtype=np.int64)
    return Tracking(players, ball3, slot_team, ["x", "y", "speed"], ["x", "y", "z"], frame_id), period


ATD_SMOOTH = 25
ATD_NOISE = 1.0
ATD_SLOW = 250
ATD_SEEDS = (0, 1, 2, 3)


def atd_like(trk, seed=0, smooth=ATD_SMOOTH, noise=ATD_NOISE, slow=ATD_SLOW):
    """Kopia trackingu z pilka popsuta jak w ATD (wygladzony tor + wolnozmienny szum); obserwacje brakujace zostaja
    brakujace (Metrica ma ich juz ok. 40%, jak ATD). Deterministyczna dla danego seed."""
    ball = trk.ball.copy()
    ok = np.isfinite(ball[:, 0])
    xy = pd.DataFrame(ball[:, :2]).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    walk = pd.DataFrame(np.random.default_rng(seed).normal(size=(len(ball), 2)))
    walk = walk.rolling(slow, center=True, min_periods=1).mean().rolling(slow, center=True, min_periods=1).mean()
    walk = walk.to_numpy() * noise / walk.to_numpy().std()
    ball[:, :2] = np.where(ok[:, None], xy + walk, np.nan)
    return Tracking(trk.players, ball.astype(np.float32), trk.slot_team, trk.player_channels, trk.ball_channels,
                    trk.frame_id)


def load_events(game, root=None):
    d = game_dir(game, root)
    return pd.read_csv(d / "Sample_Game_{}_RawEventsData.csv".format(game))


def _frames(ev):
    """(poczatek, koniec) zdarzen jako indeksy klatek; koniec 0 w pliku = zdarzenie chwilowe."""
    start = ev["Start Frame"].to_numpy(dtype=np.int64) - 1
    end = np.maximum(ev["End Frame"].to_numpy(dtype=np.int64) - 1, start)
    return start, end


def halves(ev):
    """Polowy w formacie config.HALVES: (pierwsza oceniana, wykop, ostatnia + 1), od pierwszego do konca ostatniego
    zdarzenia polowy (oceniamy tylko tam, gdzie zdarzenia mowia, co sie dzieje)."""
    start, end = _frames(ev)
    out = []
    for p in sorted(ev["Period"].unique()):
        sel = (ev["Period"] == p).to_numpy()
        a, b = int(start[sel].min()), int(end[sel].max()) + 1
        out.append((a, a, b))
    return tuple(out)


def is_restart(types, subtypes):
    sub = subtypes.fillna("").str.upper()
    return ((types == "SET PIECE") | ((types == "PASS") & sub.str.contains("GOAL KICK"))).to_numpy()


def controlled_teams(types, teams, start, restart, hold=CONTROL_HOLD):
    """Druzyna w posiadaniu po kazdym zdarzeniu z pilka. Seria zdarzen jednej druzyny przejmuje posiadanie tylko, gdy
    jest kontrolowana (CHALLENGE.md): wznowienie, celne podanie (po podaniu kolejne zdarzenie tej samej druzyny), strzal
    albo pilka przy druzynie >= hold klatek do zdarzenia rywala. Inaczej (dotkniecie, przechwyt i od razu strata)
    posiadanie zostaje przy poprzedniej druzynie."""
    out = list(teams)
    bounds = [0] + [i for i in range(1, len(teams)) if teams[i] != teams[i - 1] or restart[i]] + [len(teams)]
    current = None
    for a, b in zip(bounds[:-1], bounds[1:]):
        until = start[b] if b < len(teams) else start[b - 1]
        passed = any(types[k] == "PASS" for k in range(a, b - 1))
        shot = any(types[k] == "SHOT" for k in range(a, b))
        if current is None or restart[a] or passed or shot or until - start[a] >= hold:
            current = teams[a]
        out[a:b] = [current] * (b - a)
    return out


def ground_truth(ev, n_frames):
    """Kody H/A/D/X na kazda klatke trackingu ze zdarzen (zasady w opisie modulu)."""
    ev = ev[~ev["Type"].isin(IGNORED)].reset_index(drop=True)
    start, end = _frames(ev)
    order = np.argsort(start, kind="stable")
    ev, start, end = ev.iloc[order].reset_index(drop=True), start[order], end[order]
    restart = is_restart(ev["Type"], ev["Subtype"])
    team = np.full(n_frames, config.NOBODY, dtype=np.int8)
    on_ball = np.flatnonzero(ev["Type"].isin(ON_BALL).to_numpy())
    owner = controlled_teams(ev["Type"].to_numpy()[on_ball], ev["Team"].to_numpy()[on_ball], start[on_ball],
                             restart[on_ball])
    for i, who in zip(on_ball, owner):
        team[start[i]:] = TEAMS[who]
    dead = np.zeros(n_frames, dtype=bool)
    period = ev["Period"].to_numpy()
    for i in np.flatnonzero(restart):
        before = np.flatnonzero((np.arange(len(ev)) < i) & (period == period[i]) & (start < start[i]) & ~restart)
        if len(before):
            dead[int(end[before].max()):start[i]] = True
    out = np.full(n_frames, config.ABSTAIN, dtype=np.int8)
    for a, _kickoff, b in halves(ev):
        part = np.where(dead[a:b], config.DEAD, team[a:b])
        out[a:b] = np.where(part == config.NOBODY, config.DEAD, part)
    return out
