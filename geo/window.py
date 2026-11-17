"""Oceniane fragmenty meczu."""
import numpy as np

from .config import HALVES, N_FRAMES


def graded_mask(n_frames=N_FRAMES, halves=HALVES):
    """True na klatkach ocenianych (obie polowy, razem z D przed wykopem 2. polowy)."""
    mask = np.zeros(n_frames, dtype=bool)
    for graded, _kickoff, end in halves:
        mask[graded:min(end, n_frames)] = True
    return mask


def play_mask(n_frames=N_FRAMES, halves=HALVES):
    """True od wykopu do konca kazdej polowy (gra moze trwac)."""
    mask = np.zeros(n_frames, dtype=bool)
    for _graded, kickoff, end in halves:
        mask[kickoff:min(end, n_frames)] = True
    return mask
