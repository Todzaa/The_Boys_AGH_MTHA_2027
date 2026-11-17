"""Skladanie i zapis pliku zgloszeniowego."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geo import config, submission

H, A, D, X, NB = config.HOME, config.AWAY, config.DEAD, config.ABSTAIN, config.NOBODY
HALVES = ((2, 4, 8), (10, 12, 15))


class TestCompose(unittest.TestCase):
    def test_x_outside_d_before_kickoff_no_x_on_graded(self):
        team = np.array([H] * 16, np.int8)
        team[13] = NB
        dead = np.zeros(16, bool)
        dead[6] = True
        out = submission.compose(team, dead, HALVES, 16).tolist()
        self.assertEqual(out[:2], [X, X])
        self.assertEqual(out[2:4], [D, D])
        self.assertEqual(out[4:8], [H, H, D, H])
        self.assertEqual(out[8:10], [X, X])
        self.assertEqual(out[10:12], [D, D])
        self.assertEqual(out[12:15], [H, D, H])
        self.assertEqual(out[15], X)

    def test_spells_count_inside_graded_only(self):
        codes = np.array([X, X, D, D, H, H, A, A, X, X, D, D, H, A, A, X], np.int8)
        self.assertEqual(submission.spells(codes, HALVES), [2, 2, 1, 2])


class TestWrite(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_contract_and_hash_is_stable(self):
        codes = np.full(config.N_FRAMES, X, np.int8)
        codes[100:200] = H
        sha1 = submission.write(self.dir / "a.csv", codes)
        sha2 = submission.write(self.dir / "b.csv", codes.copy())
        self.assertEqual(sha1, sha2)
        lines = (self.dir / "a.csv").read_text(encoding="ascii").splitlines()
        self.assertEqual(lines[0], "frame,label")
        self.assertEqual(len(lines), config.N_FRAMES + 1)
        self.assertEqual(lines[101], "100,H")
        self.assertEqual(lines[-1], "{},X".format(config.N_FRAMES - 1))

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            submission.write(self.dir / "c.csv", np.zeros(10, np.int8))


if __name__ == "__main__":
    unittest.main()
