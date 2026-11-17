"""Wzorzec H/A/D ze zdarzen Metrica i predkosc z pozycji, na syntetycznych zdarzeniach."""
import unittest

import numpy as np
import pandas as pd

from geo import config, metrica

H, A, D, X = config.HOME, config.AWAY, config.DEAD, config.ABSTAIN
COLS = ("Team", "Type", "Subtype", "Period", "Start Frame", "End Frame")


def events(rows):
    return pd.DataFrame(rows, columns=COLS)


class TestGroundTruth(unittest.TestCase):
    def test_turnover_ball_out_and_restart(self):
        ev = events([
            ("Away", "SET PIECE", "KICK OFF", 1, 1, 0),
            ("Away", "PASS", None, 1, 1, 10),
            ("Away", "BALL LOST", "INTERCEPTION", 1, 20, 30),
            ("Home", "CHALLENGE", "AERIAL-WON", 1, 25, 0),          # pojedynek nie zmienia posiadania
            ("Home", "RECOVERY", "INTERCEPTION", 1, 31, 0),
            ("Home", "PASS", None, 1, 33, 38),                       # celne podanie = przejecie kontrolowane
            ("Home", "BALL OUT", None, 1, 40, 50),
            ("Away", "CARD", "YELLOW", 1, 55, 0),                   # kartka w przerwie nie konczy D
            ("Away", "SET PIECE", "THROW IN", 1, 60, 0),
            ("Away", "PASS", None, 1, 60, 70),
        ])
        gt = metrica.ground_truth(ev, 80)
        self.assertTrue((gt[:30] == A).all())
        self.assertTrue((gt[30:49] == H).all())
        self.assertTrue((gt[49:59] == D).all())                     # od konca BALL OUT do wyrzutu z autu
        self.assertTrue((gt[59:70] == A).all())
        self.assertTrue((gt[70:] == X).all())                       # po ostatnim zdarzeniu polowy

    def test_touch_without_control_keeps_possession(self):
        ev = events([
            ("Home", "SET PIECE", "KICK OFF", 1, 1, 0),
            ("Home", "PASS", None, 1, 1, 10),
            ("Home", "BALL LOST", "INTERCEPTION", 1, 20, 30),
            ("Away", "RECOVERY", "INTERCEPTION", 1, 30, 0),          # przechwyt i od razu strata: bez kontroli
            ("Away", "BALL LOST", "INTERCEPTION", 1, 30, 40),
            ("Home", "RECOVERY", None, 1, 41, 0),
            ("Home", "PASS", None, 1, 45, 60),
            ("Home", "PASS", None, 1, 60, 70),
            ("Away", "RECOVERY", None, 1, 71, 0),                    # przejecie utrzymane >= 3 s
            ("Away", "BALL LOST", None, 1, 160, 170),
            ("Home", "RECOVERY", None, 1, 171, 0),
            ("Home", "PASS", None, 1, 171, 180),
        ])
        gt = metrica.ground_truth(ev, 200)
        self.assertTrue((gt[:70] == H).all())
        self.assertTrue((gt[70:170] == A).all())

    def test_goal_kick_pass_is_restart_and_halves_separate(self):
        ev = events([
            ("Home", "SET PIECE", "KICK OFF", 1, 1, 0),
            ("Home", "SHOT", "OFF TARGET-OUT", 1, 5, 10),
            ("Away", "PASS", "GOAL KICK", 1, 20, 30),
            ("Away", "SET PIECE", "KICK OFF", 2, 101, 0),
            ("Away", "PASS", None, 2, 101, 110),
        ])
        gt = metrica.ground_truth(ev, 120)
        self.assertEqual(metrica.halves(ev), ((0, 0, 30), (100, 100, 110)))
        self.assertTrue((gt[9:19] == D).all())
        self.assertTrue((gt[19:30] == A).all())
        self.assertTrue((gt[30:100] == X).all())                    # przerwa miedzy polowami
        self.assertTrue((gt[100:110] == A).all())                   # wykop 2. polowy bez D przed nim


class TestSpeed(unittest.TestCase):
    def test_constant_run_and_missing(self):
        xy = np.zeros((20, 2, 2))
        xy[:, 0, 0] = 0.2 * np.arange(20)                          # 5 m/s przy 25 fps
        xy[5, 1] = np.nan
        v = metrica.speed(xy)
        np.testing.assert_allclose(v[3:17, 0], 5.0, atol=1e-6)
        self.assertTrue(np.isnan(v[5, 1]))


if __name__ == "__main__":
    unittest.main()
