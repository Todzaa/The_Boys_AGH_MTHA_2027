"""Martwa pilka na syntetycznych danych."""
import unittest

import numpy as np

from geo import config, dead_ball


class TestBallOut(unittest.TestCase):
    def test_out_until_back_in_play_including_gap(self):
        xy = np.full((10, 2), 30.0)
        xy[3:5, 1] = 70.0                                   # 2 m za linia boczna
        trusted = np.ones(10, bool)
        trusted[5:8] = False                                # pilka niewidoczna przy wyrzucie
        out = dead_ball.ball_out(xy, trusted).tolist()
        self.assertEqual(out, [False] * 3 + [True] * 5 + [False] * 2)

    def test_margin(self):
        self.assertFalse(dead_ball.outside(np.array([[50.0, 68.5]]))[0])          # 0.5 m za linia: w marginesie
        self.assertTrue(dead_ball.outside(np.array([[-1.5, 30.0]]))[0])


class TestProjectedExit(unittest.TestCase):
    def track(self, speed=10.0, gap=(10, 110), after_y=67.0):
        """Pilka leci w strone linii bocznej (y = 68) z y = 60, znika na luce, potem widziana na y = after_y."""
        n = 140
        xy = np.stack([np.full(n, 50.0), 60.0 + speed / config.FPS * np.arange(n)], axis=1)
        known = np.ones(n, bool)
        known[gap[0]:gap[1]] = False
        xy[gap[1]:, 1] = after_y
        return dead_ball.projected_exit(xy, known, horizon=25, min_gap=50)

    def test_lost_on_way_out_then_seen_at_line_is_dead_from_crossing(self):
        out = self.track()                                  # ostatnia obserwacja y = 63.6, linia po 0.44 s = 11 klatek
        self.assertFalse(out[:20].any())
        self.assertTrue(out[20:110].all())
        self.assertFalse(out[110:].any())

    def test_no_confirmation_short_gap_slow_or_far_ball_is_not_dead(self):
        self.assertFalse(self.track(after_y=40.0).any())    # nastepna obserwacja w srodku boiska: pilka przyjeta
        self.assertFalse(self.track(gap=(10, 40)).any())    # luka 1.2 s
        self.assertFalse(self.track(speed=1.0).any())       # pilka sie toczy
        self.assertFalse(self.track(speed=-10.0).any())     # leci od linii (do y = 0 za daleko)

    def test_zero_horizon_disables(self):
        known = np.ones(50, bool)
        known[10:45] = False
        self.assertFalse(dead_ball.projected_exit(np.zeros((50, 2)), known, horizon=0).any())


class TestActivity(unittest.TestCase):
    def test_median_and_too_few_players(self):
        speed = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0] * 3 + [np.nan] * 4])
        act = dead_ball.activity(speed, np.ones(speed.shape, bool))
        self.assertEqual(act[0], 4.0)
        self.assertTrue(np.isnan(act[1]))

    def test_long_slow_stretch_is_dead_short_one_not(self):
        act = np.full(400, 4.0)
        act[50:150] = 0.5                                   # 4 s chodzenia
        act[250:290] = 0.5                                  # 1.6 s
        dead = dead_ball.low_activity(act)
        self.assertTrue(dead[70:130].all())
        self.assertFalse(dead[255:285].any())

    def test_extended_while_below_extend(self):
        act = np.full(400, 4.0)
        act[100:200] = 0.5
        act[60:100] = 1.3                                   # gracze zwalniaja przed przerwa
        dead = dead_ball.low_activity(act, speed=1.1, extend=1.5, half=0)
        self.assertTrue(dead[60:200].all())
        self.assertFalse(dead[:60].any() or dead[200:].any())
        self.assertFalse(dead_ball.low_activity(act, speed=1.1, extend=1.1, half=0)[60:100].any())

    def test_rolling_ball_is_not_dead(self):
        xy = np.stack([np.arange(200) * 0.2, np.full(200, 30.0)], axis=1)      # 5 m/s
        xy[:100] = (10.0, 30.0)                                               # lezy
        moving = dead_ball.ball_moving(xy, np.ones(200, bool), speed=2.0, half=2)
        self.assertFalse(moving[:95].any())
        self.assertTrue(moving[105:].all())
        act = np.full(200, 0.5)
        dead = dead_ball.low_activity(act, moving, half=0)
        self.assertTrue(dead[:95].all())
        self.assertFalse(dead[105:].any())

    def test_short_gap_between_out_and_activity_closed(self):
        mask = np.zeros(200, bool)
        mask[20:60] = True                                  # np. D z pilki za linia
        mask[70:120] = True                                 # D z aktywnosci, 10 klatek pozniej
        mask[160:180] = True                                # 40 klatek pozniej: osobna przerwa
        out = dead_ball.close_gaps(mask, merge=25)
        self.assertTrue(out[20:120].all())
        self.assertFalse(out[120:160].any())

    def test_halves_only(self):
        n = config.N_FRAMES
        speed = np.full((n, 8), 0.2)
        mask, by_out, by_act = dead_ball.dead_mask(np.full((n, 2), 30.0), np.ones(n, bool), speed, np.ones((n, 8), bool))
        self.assertFalse(mask[: config.HALVES[0][1]].any())
        self.assertTrue(mask[config.HALVES[0][1]:config.HALVES[0][2]].all())
        self.assertFalse(by_out.any())


if __name__ == "__main__":
    unittest.main()
