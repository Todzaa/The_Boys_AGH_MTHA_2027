"""Loader ATD na syntetycznych plikach."""
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from geo import atd, config
from tests import synthetic

SLOTS = np.array([0, 0, 0, 1, 1, 0, 1], dtype=np.int8)
STEM = config.STEM + "_atd"


class TestLoad(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.players, self.ball = synthetic.random_match(60, SLOTS)
        synthetic.write_xml(self.dir / (STEM + ".xml"), SLOTS)
        synthetic.write_txt(self.dir / (STEM + ".txt"), self.players, self.ball)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def load(self, **kw):
        return atd.load(self.dir, cache_dir=self.dir / "cache", expected=60, **kw)

    def assert_matches(self, trk):
        scale = np.array(config.PITCH)
        np.testing.assert_allclose(trk.players[:, :, :2], self.players[:, :, :2] * scale, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(trk.players[:, :, 2], self.players[:, :, 2], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(trk.ball[:, :2], self.ball[:, :2] * scale, rtol=1e-5, atol=1e-3)
        np.testing.assert_array_equal(trk.player_ok, np.isfinite(self.players[:, :, 0]))

    def test_values_in_metres_teams_and_channels(self):
        trk = self.load(use_cache=False)
        self.assert_matches(trk)
        np.testing.assert_array_equal(trk.slot_team, SLOTS)
        self.assertEqual(trk.ball_channels, list(synthetic.BALL_CHANNELS))
        self.assertEqual(trk.player_channels, ["player1_" + c for c in synthetic.PLAYER_CHANNELS])

    def test_short_nan_groups_take_slow_path_with_same_result(self):
        synthetic.write_txt(self.dir / (STEM + ".txt"), self.players, self.ball, short_nan=True)
        self.assert_matches(self.load(use_cache=False))

    def test_misaligned_frame_ids_raise(self):
        synthetic.write_txt(self.dir / (STEM + ".txt"), self.players, self.ball, first_id=2)
        with self.assertRaises(ValueError):
            self.load(use_cache=False)

    def test_cache_is_reused_and_invalidated(self):
        first = self.load()
        self.assertTrue((self.dir / "cache" / "atd.npz").is_file())
        cached = self.load()
        np.testing.assert_array_equal(first.players, cached.players)
        self.assertEqual(cached.ball_channels, first.ball_channels)
        time.sleep(0.01)
        self.players[:, :, :2] = 0.5
        synthetic.write_txt(self.dir / (STEM + ".txt"), self.players, self.ball)
        os.utime(self.dir / (STEM + ".txt"), None)
        self.assert_matches(self.load())

    def test_files_found_in_tactical_subfolder(self):
        (self.dir / "tactical").mkdir()
        for suffix in (".xml", ".txt"):
            shutil.move(str(self.dir / (STEM + suffix)), str(self.dir / "tactical" / (STEM + suffix)))
        self.assert_matches(self.load(use_cache=False))


class TestReferee(unittest.TestCase):
    def test_slot_outside_both_teams_is_nobody(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        synthetic.write_xml(d / "m.xml", [0, 1, 2, 0])                  # teamId T2 nie jest druzyna
        slot_team, _p, _b = atd.read_meta(d / "m.xml")
        self.assertEqual(slot_team.tolist(), [config.HOME, config.AWAY, config.NOBODY, config.HOME])


class TestSummary(unittest.TestCase):
    def test_counts_on_graded_frames(self):
        n, slots = config.N_FRAMES, np.array([0] * 12 + [1] * 2, dtype=np.int8)
        players = np.full((n, len(slots), 3), np.nan, dtype=np.float32)
        ball = np.full((n, 3), np.nan, dtype=np.float32)
        first_half = slice(config.HALVES[0][0], config.HALVES[0][2])
        players[first_half, :12] = 1.0                              # H: 12 obiektow w 1. polowie (duplikat)
        players[first_half, 12:] = 1.0                              # A: 2 obiekty
        ball[first_half] = 1.0                                      # pilka przez cala 1. polowe
        trk = atd.Tracking(players, ball, slots, [], [], np.arange(1, n + 1))
        s = atd.summary(trk)
        self.assertAlmostEqual(s["ball_pct_half"][0], 100.0)
        self.assertAlmostEqual(s["ball_pct_half"][1], 0.0)
        share_first = (config.HALVES[0][2] - config.HALVES[0][0]) / float(sum(e - g for g, _k, e in config.HALVES))
        self.assertAlmostEqual(s["ball_pct_graded"], 100.0 * share_first, places=4)
        self.assertAlmostEqual(s["over_11_pct"][0], 100.0 * share_first, places=4)
        self.assertEqual(s["over_11_pct"][1], 0.0)
        self.assertEqual(s["slots_per_team"], [12, 2])


class TestAlignment(unittest.TestCase):
    def test_gap_is_not_contiguous(self):
        self.assertFalse(atd.alignment(np.array([1, 2, 4]), expected=3)["agrees"])
        self.assertTrue(atd.alignment(np.array([1, 2, 3]), expected=3)["agrees"])


if __name__ == "__main__":
    unittest.main()
