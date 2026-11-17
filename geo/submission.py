"""Plik zgloszeniowy: skladanie etykiet na caly mecz, zapis, walidacja i kontrola na publicznym GT scorerem organizatorow."""
import hashlib
import sys

import numpy as np

from . import config


def compose(team, dead, halves=config.HALVES, n_frames=config.N_FRAMES):
    """Kody H/A/D/X na caly mecz. team: 0 / 1 / NOBODY po klatkach, dead: bool.

    Poza polowami X. Na ocenianych klatkach nigdy X (scorer liczy X jako blad w macroF1): przed wykopem 2. polowy D,
    martwa pilka D, klatka bez zadnego dowodu posiadania D.
    """
    out = np.full(n_frames, config.ABSTAIN, dtype=np.int8)
    for graded, kickoff, end in halves:
        out[graded:kickoff] = config.DEAD
        part = np.where(dead[kickoff:end], config.DEAD, team[kickoff:end])
        out[kickoff:end] = np.where(part == config.NOBODY, config.DEAD, part)
    return out


def write(path, codes):
    """frame,label; dokladnie N_FRAMES wierszy, koniec linii \\n (SUBMISSION.md). Zwraca sha256 pliku."""
    if len(codes) != config.N_FRAMES:
        raise ValueError("{} etykiet, plik zgloszeniowy wymaga {}".format(len(codes), config.N_FRAMES))
    text = "frame,label\n" + "".join("{},{}\n".format(i, config.LABELS[c]) for i, c in enumerate(codes))
    data = text.encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def spells(codes, halves=config.HALVES):
    """Dlugosci (klatki) posiadan = maksymalnych serii H albo A w ocenianych fragmentach."""
    out = []
    for graded, _kickoff, end in halves:
        part = codes[graded:end]
        edges = np.flatnonzero(np.diff(part)) + 1
        bounds = np.concatenate(([0], edges, [len(part)]))
        out += [int(b - a) for a, b in zip(bounds[:-1], bounds[1:]) if part[a] in (config.HOME, config.AWAY)]
    return out


def load_scorer(repo=config.REPO_DIR):
    """(modul possession_metrics organizatorow, sciezka publicznego GT) albo (None, powod)."""
    evaluation = repo / "evaluation"
    gt = sorted((evaluation / "gt").glob("*_possession_gt_public.csv"))
    if not (evaluation / "possession_metrics.py").is_file() or len(gt) != 1:
        return None, "brak scorera albo publicznego GT w {} (ustaw MTAH_REPO)".format(evaluation)
    if str(evaluation) not in sys.path:
        sys.path.insert(0, str(evaluation))
    import possession_metrics as pm
    return pm, gt[0]
