"""Syntetyczne pliki w formacie ATD (FIFA/EPTS) do testow. Zadnych danych konkursowych."""
from pathlib import Path

import numpy as np

PLAYER_CHANNELS = ("x", "y", "speed")
BALL_CHANNELS = ("ball_x", "ball_y", "ball_z")


def write_xml(path, slot_team):
    """Metadane: 2 druzyny (pierwsza = gospodarze), Player na kazdy slot, specyfikacja formatu linii."""
    teams = '<Team id="T0"><Name>Home</Name></Team><Team id="T1"><Name>Away</Name></Team>'
    players = "".join('<Player id="P{0}" teamId="T{1}"><Name>Track {0}</Name></Player>'.format(i + 1, t)
                      for i, t in enumerate(slot_team))
    groups = "".join('<SplitRegister separator=",">' + "".join(
        '<PlayerChannelRef playerChannelId="player{}_{}"/>'.format(i + 1, c) for c in PLAYER_CHANNELS) + "</SplitRegister>"
        for i in range(len(slot_team)))
    ball = '<SplitRegister separator=",">' + "".join('<BallChannelRef channelId="{}"/>'.format(c) for c in BALL_CHANNELS) \
        + "</SplitRegister>"
    spec = ('<DataFormatSpecifications><DataFormatSpecification><SplitRegister separator=":">'
            '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>{}'
            '</SplitRegister></DataFormatSpecification></DataFormatSpecifications>').format(groups, ball)
    xml = '<main><Metadata><Teams>{}</Teams><Players>{}</Players></Metadata>{}</main>'.format(teams, players, spec)
    Path(path).write_text(xml, encoding="utf-8")


def _fmt(values, short_nan):
    if short_nan and not np.isfinite(values[0]):
        return "NaN"
    return ",".join("NaN" if not np.isfinite(v) else "{:.6f}".format(v) for v in values)


def write_txt(path, players, ball, first_id=1, short_nan=False):
    """players (n, sloty, 3) i ball (n, 3) w jednostkach pliku (0-1). short_nan: brak obiektu jako samo 'NaN'."""
    lines = []
    for i in range(len(ball)):
        groups = ";".join(_fmt(players[i, s], short_nan) for s in range(players.shape[1]))
        lines.append("{}:{}:{}".format(first_id + i, groups, _fmt(ball[i], short_nan)))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def random_match(n, slot_team, seed=0, missing=0.2):
    """(gracze, pilka) w 0-1 z czesciowo brakujacymi obserwacjami."""
    rng = np.random.RandomState(seed)
    players = rng.uniform(0, 1, (n, len(slot_team), 3))
    players[:, :, 2] *= 8.0
    players[rng.uniform(size=(n, len(slot_team))) < missing] = np.nan
    ball = rng.uniform(0, 1, (n, 3))
    ball[rng.uniform(size=n) < missing] = np.nan
    return players, ball
