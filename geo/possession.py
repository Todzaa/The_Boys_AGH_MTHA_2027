"""Posiadanie z geometrii ATD: odsiew pilki, najblizszy gracz kazdej druzyny, kontrola i zmiana posiadania.

Pozycje graczy z ATD nie sa zmieniane. Pilka, ktorej nie ufamy (krotki odcinek toru odciety nierealnym skokiem), jest
pomijana; zaufana pilka przechodzi filtr medianowy, a krotkie luki (<= 0.4 s) sa interpolowane (clean_ball).
Zmiana posiadania wymaga przewagi kontroli rywala przez TURNOVER_FRAMES klatek (licznik z zanikaniem z wisla
possession.py): dotkniecie, rykoszet czy krotki pojedynek nie zmieniaja posiadania, pilka luzem zostaje przy ostatniej
druzynie (CHALLENGE.md). Zmiana jest datowana od poczatku przewagi rywala, bo granice posiadan licza sie w spellF1.
"""
import numpy as np
import pandas as pd

from . import config


def runs(mask):
    """[(od, do)] ciagow True, do wylacznie."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def trusted_ball(ball_xy, ball_ok, max_speed=config.BALL_MAX_SPEED, tol=config.BALL_JUMP_TOL,
                 min_run=config.BALL_MIN_RUN):
    """Maska obserwacji pilki, ktorym ufamy. Odcinek toru odciety skokiem szybszym niz max_speed (+ tol) i krotszy niz
    min_run klatek wypada; drugi przebieg, bo po usunieciu pojedynczego skoku sasiedzi znow tworza jeden odcinek."""
    ok = np.asarray(ball_ok, dtype=bool).copy()
    for _ in range(2):
        idx = np.flatnonzero(ok)
        if len(idx) < 2:
            break
        step = np.hypot(*(ball_xy[idx[1:]] - ball_xy[idx[:-1]]).T)
        jump = step > max_speed * np.diff(idx) / config.FPS + tol
        ids = np.concatenate(([0], np.cumsum(jump)))
        ok[idx[np.bincount(ids)[ids] < min_run]] = False
    return ok


def clean_ball(ball_xy, trusted, smooth=config.BALL_SMOOTH, max_gap=config.BALL_GAP_FILL):
    """(pozycje pilki, maska znanej pilki): mediana ruchoma +/- smooth klatek z samych zaufanych obserwacji, potem luki
    <= max_gap klatek miedzy zaufanymi obserwacjami wypelnione liniowo. Dluzsze luki i brzegi zostaja puste."""
    trusted = np.asarray(trusted, dtype=bool)
    xy = np.where(trusted[:, None], ball_xy, np.nan).astype(np.float64)
    if smooth:
        xy = pd.DataFrame(xy).rolling(2 * smooth + 1, center=True, min_periods=1).median().to_numpy()
        xy[~trusted] = np.nan
    ok = trusted.copy()
    idx = np.flatnonzero(trusted)
    if max_gap and len(idx) >= 2:
        for a, b in runs(~trusted):
            if a > 0 and b < len(ok) and b - a <= max_gap:
                ok[a:b] = True
        fill = np.flatnonzero(ok & ~trusted)
        for k in range(2):
            xy[fill, k] = np.interp(fill, idx, xy[idx, k])
    return xy, ok


def team_distances(players_xy, player_ok, slot_team, ball_xy, ball_ok):
    """(n, 2): odleglosc najblizszego gracza H i A od pilki w metrach; inf bez pilki albo bez graczy druzyny."""
    d = np.hypot(players_xy[:, :, 0] - ball_xy[:, None, 0], players_xy[:, :, 1] - ball_xy[:, None, 1])
    d = np.where(player_ok & np.asarray(ball_ok)[:, None], d, np.inf)
    out = np.full((len(d), 2), np.inf, dtype=np.float32)
    for team in (config.HOME, config.AWAY):
        if (slot_team == team).any():
            out[:, team] = d[:, slot_team == team].min(axis=1)
    return out


def ball_speed(ball_xy, trusted, half=2):
    """Predkosc pilki (m/s) z sasiednich zaufanych obserwacji, mediana ruchoma +/- half; NaN = nieznana."""
    v = np.full(len(trusted), np.nan)
    both = trusted[1:] & trusted[:-1]
    v[1:][both] = np.hypot(*(ball_xy[1:] - ball_xy[:-1])[both].T) * config.FPS
    return pd.Series(v).rolling(2 * half + 1, center=True, min_periods=1).median().to_numpy()


def control(dist, radius=config.CONTROL_RADIUS, speed=None, max_speed=config.CONTROL_BALL_SPEED,
            margin=config.CONTROL_MARGIN):
    """Druzyna kontrolujaca pilke w klatce: najblizszy gracz w promieniu radius, najblizszy rywal co najmniej margin
    dalej, a pilka nie przelatuje obok szybciej niz max_speed (NOBODY: pilka luzem, sporna, w locie albo brak danych)."""
    best = np.argmin(dist, axis=1)
    near = dist.min(axis=1) <= radius
    with np.errstate(invalid="ignore"):
        near &= np.abs(dist[:, 0] - dist[:, 1]) >= margin
    if speed is not None:
        near &= ~(speed > max_speed)
    return np.where(near, best, config.NOBODY).astype(np.int8)


def touches(ball_xy, known, dist, min_dv=config.TOUCH_MIN_DV, lag=config.TOUCH_LAG, radius=config.TOUCH_RADIUS,
            margin=config.TOUCH_MARGIN):
    """Druzyna dotykajaca pilki w klatce (NOBODY: brak dotkniecia albo niczyje).

    Dotkniecie: zmiana wektora predkosci pilki (lag klatek przed vs po) >= min_dv m/s, w lokalnym maksimum, przy
    znanej pilce w trzech klatkach; najblizszy gracz <= radius, rywal co najmniej margin dalej. Maksimum musi miec
    zmierzone obie sasiednie klatki: na brzegu luki prawdziwy szczyt moze lezec w niezmierzonej czesci, wiec brzeg nie
    jest dotknieciem. Kolejne dotkniecie tej samej druzyny blizej niz 2 * lag to ten sam kontakt; dotkniecie rywala
    i kandydaci odrzuceni (nikt blisko, pojedynek) niczego nie blokuja.
    """
    n = len(known)
    xy = np.where(np.asarray(known, dtype=bool)[:, None], ball_xy, np.nan)
    dv = np.full(n, np.nan)
    if n > 2 * lag:
        before = xy[lag:n - lag] - xy[:n - 2 * lag]
        after = xy[2 * lag:] - xy[lag:n - lag]
        dv[lag:n - lag] = np.hypot(*(after - before).T) * config.FPS / lag
    peak = pd.Series(dv).rolling(2 * lag + 1, center=True, min_periods=1).max().to_numpy()
    inner = np.zeros(n, dtype=bool)
    inner[1:-1] = np.isfinite(dv[:-2]) & np.isfinite(dv[2:])
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero((dv >= min_dv) & (dv == peak) & inner)
        own, gap = dist.min(axis=1), dist.max(axis=1) - dist.min(axis=1)
    best = np.argmin(dist, axis=1)
    out = np.full(n, config.NOBODY, dtype=np.int8)
    last, last_team = -n, config.NOBODY
    for t in cand.tolist():
        if not (own[t] <= radius and gap[t] >= margin):
            continue
        if best[t] == last_team and t - last <= 2 * lag:
            continue
        out[t] = best[t]
        last, last_team = t, best[t]
    return out


def nearest_team(dist):
    """Baseline organizatorow: druzyna najblizszego gracza, bez promienia (NOBODY bez pilki)."""
    return np.where(np.isfinite(dist.min(axis=1)), np.argmin(dist, axis=1), config.NOBODY).astype(np.int8)


def majority_vote(team, half_width):
    """Glosowanie wiekszosciowe +/- half_width klatek (baseline organizatorow): D bez glosow, remis dla H."""
    home = np.concatenate(([0], np.cumsum(team == config.HOME)))
    away = np.concatenate(([0], np.cumsum(team == config.AWAY)))
    index = np.arange(len(team))
    low, high = np.maximum(0, index - half_width), np.minimum(len(team), index + half_width + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    return np.where(h + a == 0, config.DEAD, np.where(h >= a, config.HOME, config.AWAY)).astype(np.int8)


def track_possession(ctrl, live, turnover_frames=config.TURNOVER_FRAMES, touch=None, pair_gap=config.TOUCH_PAIR):
    """Druzyna w posiadaniu na kazda klatke (0 / 1, NOBODY poza gra albo bez zadnego dowodu).

    Kazdy odcinek zywej gry zaczyna od nowa: posiadanie ma pierwsza druzyna z kontrola (dopelnione wstecz do restartu).
    Rywal przejmuje, gdy jego licznik kontroli (+1 za klatke rywala, -1 za klatke posiadacza) dojdzie do turnover_frames;
    zmiana od klatki, w ktorej licznik ruszyl od zera. Odcinek bez kontroli dostaje druzyne z poprzedniego odcinka.
    touch (z touches): dwa kolejne dotkniecia rywala w odstepie <= pair_gap, bez dotkniecia posiadacza pomiedzy, tez
    zmieniaja posiadanie, od pierwszego z nich; pair_gap = 0 wylacza.
    Cofniecie przejecia: gdy rywal przejal pilke licznikiem po swoim dotknieciu, a dawny posiadacz dotknie pilki w
    ciagu pair_gap od tamtego dotkniecia, zanim nowy posiadacz dotknie jej znowu, przejecia nie bylo (rywal nie
    utrzymal pilki, CHALLENGE.md) i posiadanie wraca do dawnego posiadacza od tamtego dotkniecia.
    """
    out = np.full(len(ctrl), config.NOBODY, dtype=np.int8)
    carry = config.NOBODY
    use_touch = touch is not None and pair_gap > 0
    for a, b in runs(live):
        state, counter, start, pending = config.NOBODY, 0, a, -1
        for i in range(a, b):
            t = touch[i] if use_touch else config.NOBODY
            if state != config.NOBODY and t != config.NOBODY:
                if t == state:
                    pending = -1
                elif pending >= 0 and i - pending <= pair_gap:
                    out[pending:i] = t
                    state, counter, pending = t, 0, -1
                else:
                    pending = i
            c = ctrl[i]
            if state == config.NOBODY:
                if c != config.NOBODY:
                    state = c
                    out[a:i] = c
            elif c == state:
                counter = max(counter - 1, 0)
            elif c != config.NOBODY:
                if counter == 0:
                    start = i
                counter += 1
                if counter >= turnover_frames:
                    state, counter = c, 0
                    out[start:i] = c
            out[i] = state
        if state == config.NOBODY:
            out[a:b] = carry
        else:
            carry = state
    return out


def merge_short(team, live, min_len=config.MIN_SPELL):
    """Posiadanie krotsze niz min_len klatek w srodku odcinka gry dostaje druzyne poprzedniego posiadania (rywal nie
    utrzymal pilki, wiec przejecia nie bylo). Krotkie pierwsze posiadanie po wznowieniu dostaje druzyne nastepnego:
    track_possession dopelnia je wstecz do wznowienia, wiec jedna klatka rywala blizej pilki przy wznowieniu dawalaby
    mikro-posiadanie."""
    out = team.copy()
    for a, b in runs(live):
        part = out[a:b]
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(part)) + 1, [len(part)]))
        for i, (s, t) in enumerate(zip(bounds[:-1], bounds[1:])):
            if i > 0 and t - s < min_len and part[s - 1] != config.NOBODY:
                part[s:t] = part[s - 1]
        first = np.flatnonzero(np.diff(part)) + 1
        if len(first) and first[0] < min_len and part[first[0]] != config.NOBODY:
            part[:first[0]] = part[first[0]]
    return out
