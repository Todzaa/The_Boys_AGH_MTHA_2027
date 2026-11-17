"""Tracking organizatorow (ATD, FIFA/EPTS): wektorowe wczytanie do tablic w metrach, cache i kontrola wyrownania.

    python -m geo.atd               # wczytaj (albo cache), wypisz liczby zbiorcze
    python -m geo.atd --no-cache    # wczytaj od nowa z pliku .txt

Linia .txt: "<frame_id>:<46 grup 'x,y,v' rozdzielonych ;>:<pilka 'a,b,c'>", NaN = brak obiektu, video_frame = frame_id - 1.
Wspolrzedne 0-1 mnozymy przez PITCH (105.3 x 68 m). Druzyna slotu z .xml: pierwsza druzyna w pliku = gospodarze = H.
Nazwy kanalow (np. pol pilki) czytamy ze specyfikacji formatu w .xml, nie zakladamy ich.
"""
import argparse
import json
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .window import graded_mask

CACHE_VERSION = 2
CHUNK_LINES = 20000


@dataclass
class Tracking:
    """Caly mecz po klatkach wideo (indeks 0 = klatka 0)."""
    players: np.ndarray
    ball: np.ndarray
    slot_team: np.ndarray
    player_channels: list
    ball_channels: list
    frame_id: np.ndarray

    @property
    def n(self):
        return len(self.ball)

    @property
    def player_ok(self):
        return np.isfinite(self.players[:, :, 0])

    @property
    def ball_ok(self):
        return np.isfinite(self.ball[:, 0])


def find_file(name, data_dir=None):
    """Plik danych z podfolderu tactical/ albo bezposrednio z folderu danych."""
    base = Path(data_dir) if data_dir else config.DATA_DIR
    candidates = [base / "tactical" / name, base / name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("brak {} (sprawdzone: {}); ustaw MTAH_DATA_DIR".format(name, ", ".join(map(str, candidates))))


def _tag(el):
    return el.tag.rsplit("}", 1)[-1]


def read_meta(xml_path):
    """(druzyna slotu 0/1, kanaly gracza, kanaly pilki) z metadanych ATD."""
    root = ET.parse(str(xml_path)).getroot()
    team_ids = [el.get("id") for el in root.iter() if _tag(el) == "Team"]
    if len(team_ids) != 2:
        raise ValueError("{}: oczekiwano 2 druzyn, jest {}".format(xml_path, len(team_ids)))
    slots = [el.get("teamId") for el in root.iter() if _tag(el) == "Player"]
    if not slots:
        raise ValueError("{}: brak wpisow Player".format(xml_path))
    code = {team_ids[0]: config.HOME, team_ids[1]: config.AWAY}
    slot_team = np.array([code.get(t, config.NOBODY) for t in slots], dtype=np.int8)
    return (slot_team, *_format_channels(root))


def _format_channels(root):
    """Nazwy kanalow pierwszej grupy gracza i grupy pilki z DataFormatSpecification (puste listy, gdy nie rozpoznano)."""
    spec = next((el for el in root.iter() if _tag(el) == "DataFormatSpecification"), None)
    if spec is None:
        return [], []

    def first_group(ref_tag):
        for el in spec.iter():
            if _tag(el) == "SplitRegister" and any(_tag(k) == ref_tag for k in el):
                return [k.get("channelId") or k.get("playerChannelId") or "?" for k in el if _tag(k).endswith("ChannelRef")]
        return []

    return first_group("PlayerChannelRef"), first_group("BallChannelRef")


def _triplet(text):
    out = [np.nan] * 3
    for k, value in enumerate(text.split(",")[:3]):
        try:
            out[k] = float(value)
        except ValueError:
            pass
    return out


def _parse_line(line, n_slots):
    """Wolna sciezka dla linii o nietypowym ksztalcie (np. brakujaca grupa zapisana jako samo 'NaN')."""
    row = np.full(1 + 3 * n_slots + 3, np.nan)
    parts = line.split(":")
    row[0] = float(parts[0])
    if len(parts) == 3:
        for slot, group in enumerate(parts[1].split(";")[:n_slots]):
            row[1 + 3 * slot:4 + 3 * slot] = _triplet(group)
        row[-3:] = _triplet(parts[2])
    return row


def _parse_chunk(lines, n_slots):
    """(tablica (linie, 1 + 3*sloty + 3), czy uzyto wolnej sciezki). Szybko: jeden np.fromstring na caly kawalek."""
    width = 1 + 3 * n_slots + 3
    text = ",".join(lines).replace(":", ",").replace(";", ",")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = np.fromstring(text, dtype=np.float64, sep=",")
    if values.size == len(lines) * width:
        return values.reshape(len(lines), width), False
    return np.stack([_parse_line(line, n_slots) for line in lines]), True


def parse_raw(txt_path, n_slots):
    """(frame_id, gracze (n, sloty, 3) 0-1, pilka (n, 3) 0-1, liczba kawalkow z wolna sciezka)."""
    blocks, slow = [], 0
    with open(str(txt_path), encoding="utf-8") as handle:
        batch = []
        for line in handle:
            line = line.strip()
            if line:
                batch.append(line)
            if len(batch) == CHUNK_LINES:
                block, was_slow = _parse_chunk(batch, n_slots)
                blocks.append(block)
                slow += was_slow
                batch = []
        if batch:
            block, was_slow = _parse_chunk(batch, n_slots)
            blocks.append(block)
            slow += was_slow
    table = np.vstack(blocks) if blocks else np.zeros((0, 1 + 3 * n_slots + 3))
    frame_id = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3).astype(np.float32)
    ball = table[:, -3:].astype(np.float32)
    return frame_id, players, ball, slow


def alignment(frame_id, expected=config.N_FRAMES):
    """Czy plik ma jeden rekord na klatke wideo, numerowany od 1 (DATA.md)."""
    contiguous = len(frame_id) > 0 and frame_id[0] == 1 and bool(np.all(np.diff(frame_id) == 1))
    return dict(n_records=int(len(frame_id)), first=int(frame_id[0]) if len(frame_id) else None,
                last=int(frame_id[-1]) if len(frame_id) else None, contiguous=contiguous,
                agrees=contiguous and len(frame_id) == expected)


def source_stamp(data_dir=None):
    """Nazwa, rozmiar i czas modyfikacji plikow ATD: zmiana ktoregokolwiek uniewaznia cache."""
    paths = [find_file(config.STEM + suffix, data_dir) for suffix in ("_atd.xml", "_atd.txt")]
    return [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths]


def load(data_dir=None, use_cache=True, cache_dir=None, expected=config.N_FRAMES):
    """Tracking calego meczu w metrach. Cache .npz jest wazny, dopoki pliki zrodlowe sie nie zmienia."""
    xml_path = find_file(config.STEM + "_atd.xml", data_dir)
    txt_path = find_file(config.STEM + "_atd.txt", data_dir)
    cache = Path(cache_dir or config.CACHE_DIR) / "atd.npz"
    meta = dict(version=CACHE_VERSION, pitch=list(config.PITCH), files=source_stamp(data_dir))
    if use_cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            if json.loads(str(z["meta"])) == meta:
                return Tracking(z["players"], z["ball"], z["slot_team"], list(z["player_channels"]),
                                list(z["ball_channels"]), z["frame_id"])

    slot_team, player_channels, ball_channels = read_meta(xml_path)
    frame_id, players, ball, slow = parse_raw(txt_path, len(slot_team))
    check = alignment(frame_id, expected)
    if not check["agrees"]:
        raise ValueError("ATD nie jest jednym rekordem na klatke od 1: {}".format(check))
    scale = np.array(config.PITCH, dtype=np.float32)
    players[:, :, :2] *= scale
    ball[:, :2] *= scale
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, players=players, ball=ball, slot_team=slot_team, frame_id=frame_id,
             player_channels=np.array(player_channels, dtype=str), ball_channels=np.array(ball_channels, dtype=str),
             meta=json.dumps(meta))
    if slow:
        print("uwaga: {} kawalkow pliku mialo nietypowy ksztalt linii (wolna sciezka)".format(slow))
    return Tracking(players, ball, slot_team, player_channels, ball_channels, frame_id)


def summary(trk):
    """Wylacznie liczby zbiorcze o pokryciu trackingu (oceniane klatki)."""
    graded = graded_mask(trk.n)
    halves = []
    for first, _kickoff, end in config.HALVES:
        m = np.zeros(trk.n, dtype=bool)
        m[first:min(end, trk.n)] = True
        halves.append(m)
    ok = trk.player_ok[graded]
    per_team = {t: ok[:, trk.slot_team == t].sum(axis=1) for t in (config.HOME, config.AWAY)}
    speed_nan = trk.player_ok & ~np.isfinite(trk.players[:, :, 2])
    return dict(
        records=trk.n,
        slots_per_team=[int((trk.slot_team == t).sum()) for t in (config.HOME, config.AWAY)],
        player_channels=trk.player_channels, ball_channels=trk.ball_channels,
        ball_pct_graded=100.0 * trk.ball_ok[graded].mean(),
        ball_pct_half=[100.0 * trk.ball_ok[m].mean() for m in halves],
        objects_per_frame=float(ok.sum(axis=1).mean()),
        objects_per_team=[float(per_team[t].mean()) for t in (config.HOME, config.AWAY)],
        over_11_pct=[100.0 * float((per_team[t] > 11).mean()) for t in (config.HOME, config.AWAY)],
        speed_nan_pct=100.0 * speed_nan.sum() / max(trk.player_ok.sum(), 1),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-cache", action="store_true", help="wczytaj od nowa z pliku .txt")
    args = ap.parse_args()
    t0 = time.time()
    trk = load(use_cache=not args.no_cache)
    s = summary(trk)
    print("ATD: {} rekordow, jeden na klatke od frame_id 1 (wczytanie {:.0f} s)".format(s["records"], time.time() - t0))
    print("sloty H / A: {} / {}".format(*s["slots_per_team"]))
    print("kanaly gracza: {}".format(", ".join(s["player_channels"]) or "nie rozpoznano"))
    print("kanaly pilki:  {}".format(", ".join(s["ball_channels"]) or "nie rozpoznano"))
    print("pilka na ocenianych klatkach {:.1f}%   (1. polowa {:.1f}%, 2. polowa {:.1f}%)".format(
        s["ball_pct_graded"], *s["ball_pct_half"]))
    print("obiekty na ocenianej klatce: {:.1f}   H {:.1f}   A {:.1f}".format(s["objects_per_frame"], *s["objects_per_team"]))
    print("klatki z ponad 11 obiektami w druzynie: H {:.1f}%   A {:.1f}%".format(*s["over_11_pct"]))
    print("pozycja bez predkosci (v = NaN): {:.2f}% obserwacji".format(s["speed_nan_pct"]))
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR / "atd_summary.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    print("zapisano {}".format(config.LOG_DIR / "atd_summary.json"))


if __name__ == "__main__":
    main()
