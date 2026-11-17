"""Posiadanie z geometrii na syntetycznych danych (metry, 25 fps)."""
import unittest

import numpy as np

from geo import config, possession

H, A, NB = config.HOME, config.AWAY, config.NOBODY


class TestTrustedBall(unittest.TestCase):
    def test_single_jump_dropped_rest_kept(self):
        xy = np.stack([10.0 + 0.2 * np.arange(60), np.full(60, 30.0)], axis=1)
        xy[30] = (95.0, 5.0)
        ok = possession.trusted_ball(xy, np.ones(60, bool))
        self.assertFalse(ok[30])
        self.assertEqual(int(ok.sum()), 59)

    def test_fast_kick_kept(self):
        xy = np.stack([10.0 + 1.2 * np.arange(40), np.full(40, 30.0)], axis=1)   # 30 m/s
        self.assertTrue(possession.trusted_ball(xy, np.ones(40, bool)).all())


class TestCleanBall(unittest.TestCase):
    def test_short_gap_filled_long_gap_and_edges_kept_empty(self):
        n = 80
        xy = np.stack([np.arange(n, dtype=float), np.full(n, 30.0)], axis=1)
        ok = np.ones(n, bool)
        ok[:3] = ok[20:25] = ok[40:60] = False                                      # brzeg, 5 klatek, 20 klatek
        out, known = possession.clean_ball(xy, ok, smooth=0, max_gap=10)
        self.assertTrue(known[20:25].all())
        np.testing.assert_allclose(out[20:25, 0], np.arange(20, 25))               # interpolacja liniowa
        self.assertFalse(known[:3].any() or known[40:60].any())
        self.assertTrue(np.isnan(out[40:60]).all())

    def test_median_removes_single_wobble(self):
        xy = np.stack([np.full(30, 10.0), np.full(30, 30.0)], axis=1)
        xy[15, 0] = 10.8
        out, known = possession.clean_ball(xy, np.ones(30, bool), smooth=2, max_gap=0)
        self.assertTrue(known.all())
        np.testing.assert_allclose(out[:, 0], 10.0)

    def test_zero_disables(self):
        xy = np.random.default_rng(0).uniform(0, 50, (40, 2))
        ok = np.ones(40, bool)
        ok[10:13] = False
        out, known = possession.clean_ball(xy, ok, smooth=0, max_gap=0)
        np.testing.assert_array_equal(known, ok)
        np.testing.assert_allclose(out[ok], xy[ok])


class TestDistancesAndControl(unittest.TestCase):
    def test_contested_ball_is_nobody(self):
        d = np.array([[1.0, 1.3], [1.0, 1.6], [1.0, np.inf], [np.inf, np.inf]])
        self.assertEqual(possession.control(d, margin=0.5).tolist(), [NB, H, H, NB])
        self.assertEqual(possession.control(d, margin=0.0).tolist(), [H, H, H, NB])

    def test_nearest_per_team_and_radius(self):
        players = np.array([[[10.0, 10.0], [50.0, 50.0], [11.5, 10.0]]])          # H, H, A
        ok = np.ones((1, 3), bool)
        ball = np.array([[10.5, 10.0]])
        d = possession.team_distances(players, ok, np.array([H, H, A], np.int8), ball, np.array([True]))
        np.testing.assert_allclose(d[0], [0.5, 1.0])
        self.assertEqual(int(possession.control(d)[0]), H)
        self.assertEqual(int(possession.control(d + 5.0)[0]), NB)                   # nikt w promieniu 2 m

    def test_slot_without_team_is_ignored(self):
        players = np.array([[[10.2, 10.0], [30.0, 30.0]]])                          # sedzia przy pilce, gracz A daleko
        d = possession.team_distances(players, np.ones((1, 2), bool), np.array([NB, A], np.int8),
                                      np.array([[10.0, 10.0]]), np.array([True]))
        self.assertTrue(np.isinf(d[0, H]))
        self.assertEqual(int(possession.control(d)[0]), NB)

    def test_no_ball_is_inf(self):
        d = possession.team_distances(np.zeros((1, 2, 2)), np.ones((1, 2), bool), np.array([H, A], np.int8),
                                      np.zeros((1, 2)), np.array([False]))
        self.assertTrue(np.isinf(d).all())
        self.assertEqual(int(possession.nearest_team(d)[0]), NB)


class TestTrackPossession(unittest.TestCase):
    def run_(self, ctrl, live=None, k=5):
        ctrl = np.array(ctrl, np.int8)
        live = np.ones(len(ctrl), bool) if live is None else np.array(live, bool)
        return possession.track_possession(ctrl, live, k).tolist()

    def test_short_touch_does_not_change(self):
        self.assertEqual(self.run_([H] * 5 + [A] * 3 + [H] * 5), [H] * 13)

    def test_sustained_control_changes_from_its_start(self):
        out = self.run_([H] * 5 + [NB] * 2 + [A] * 6)
        self.assertEqual(out, [H] * 7 + [A] * 6)

    def test_loose_ball_keeps_last_team_and_start_is_backfilled(self):
        self.assertEqual(self.run_([NB] * 3 + [A] * 2 + [NB] * 4), [A] * 9)

    def test_scramble_decays(self):
        self.assertEqual(self.run_([H, A, H, A, H, A, H, A, H, A, A, A]), [H] * 12)

    def test_new_live_segment_restarts_and_empty_one_carries(self):
        ctrl = [H] * 4 + [NB] * 2 + [A] * 3 + [NB] * 2 + [NB] * 3
        live = [1] * 4 + [0] * 2 + [1] * 3 + [0] * 2 + [1] * 3
        out = self.run_(ctrl, live)
        self.assertEqual(out[:4], [H] * 4)
        self.assertEqual(out[4:6], [NB] * 2)
        self.assertEqual(out[6:9], [A] * 3)                                         # restart: pierwsza kontrola
        self.assertEqual(out[11:], [A] * 3)                                         # bez kontroli: poprzednia druzyna


class TestBallSpeedAndShortSpells(unittest.TestCase):
    def test_fast_ball_passing_player_is_not_control(self):
        xy = np.stack([np.arange(20) * 0.8, np.full(20, 30.0)], axis=1)          # 20 m/s
        speed = possession.ball_speed(xy, np.ones(20, bool))
        np.testing.assert_allclose(speed[1:], 20.0, rtol=1e-5)
        d = np.array([[0.5, np.inf]] * 20)
        self.assertTrue((possession.control(d, speed=speed, max_speed=12.0)[1:] == NB).all())
        self.assertTrue((possession.control(d, speed=speed / 4, max_speed=12.0) == H).all())

    def test_short_spell_returns_to_previous_team(self):
        team = np.array([H] * 100 + [A] * 30 + [H] * 100 + [A] * 80, np.int8)
        live = np.ones(len(team), bool)
        out = possession.merge_short(team, live, min_len=50)
        self.assertTrue((out[:230] == H).all())
        self.assertTrue((out[230:] == A).all())                                    # 3.2 s: prawdziwe przejecie

    def test_short_first_spell_after_restart_goes_to_next_team(self):
        team = np.array([A] * 20 + [H] * 100, np.int8)
        out = possession.merge_short(team, np.ones(120, bool), min_len=50)
        self.assertTrue((out == H).all())

    def test_long_first_spell_after_restart_kept(self):
        team = np.array([A] * 60 + [H] * 100, np.int8)
        out = possession.merge_short(team, np.ones(160, bool), min_len=50)
        np.testing.assert_array_equal(out, team)


class TestTouches(unittest.TestCase):
    def ball(self):
        """Pilka 10 m/s w prawo, w klatce 50 odbita z powrotem 10 m/s w lewo."""
        t = np.arange(100)
        x = np.where(t <= 50, 10 + 0.4 * t, 30 - 0.4 * (t - 50))
        return np.stack([x, np.full(100, 30.0)], axis=1), np.ones(100, bool)

    def test_single_touch_at_reversal_credited_to_near_team(self):
        xy, known = self.ball()
        dist = np.tile([1.0, 5.0], (100, 1))
        out = possession.touches(xy, known, dist, min_dv=5.0, lag=3)
        self.assertEqual(np.flatnonzero(out != NB).tolist(), [50])
        self.assertEqual(int(out[50]), H)

    def test_straight_ball_contested_or_far_is_no_touch(self):
        t = np.arange(100)
        straight = np.stack([10 + 0.4 * t, np.full(100, 30.0)], axis=1)
        self.assertTrue((possession.touches(straight, np.ones(100, bool), np.tile([1.0, 5.0], (100, 1))) == NB).all())
        xy, known = self.ball()
        self.assertTrue((possession.touches(xy, known, np.tile([1.0, 1.2], (100, 1))) == NB).all())   # pojedynek
        self.assertTrue((possession.touches(xy, known, np.tile([4.0, 9.0], (100, 1))) == NB).all())   # nikt blisko


def zigzag(turns, n=200, speed=10.0):
    """Pilka po osi x z predkoscia +/- speed, zmiana kierunku w klatkach turns."""
    v = np.full(n, speed / config.FPS)
    for t in turns:
        v[t:] *= -1
    x = 50.0 + np.concatenate(([0.0], np.cumsum(v[:-1])))
    return np.stack([x, np.full(n, 30.0)], axis=1)


class TestTouchesProductionDefaults(unittest.TestCase):
    """Scenariusze z code review na domyslnych parametrach (okno 12 klatek, 3 m/s, 3.5 m)."""

    def test_rival_touch_then_possessor_touch_within_a_second_both_kept(self):
        dist = np.where(np.arange(200)[:, None] < 109, [5.0, 1.0], [1.0, 5.0])     # A przy pilce, potem H
        out = possession.touches(zigzag([100, 118]), np.ones(200, bool), dist)
        self.assertEqual([(t, int(out[t])) for t in np.flatnonzero(out != NB)], [(100, A), (118, H)])

    def test_rejected_candidate_does_not_block_next_touch(self):
        dist = np.where(np.arange(200)[:, None] < 109, [9.0, 9.0], [5.0, 1.0])     # najpierw nikt blisko, potem A
        out = possession.touches(zigzag([100, 118]), np.ones(200, bool), dist)
        self.assertEqual([(t, int(out[t])) for t in np.flatnonzero(out != NB)], [(118, A)])

    def test_kick_right_after_ball_reappears_is_not_shifted(self):
        n = 200
        x = np.where(np.arange(n) < 103, 10.0, 10.0 + 0.6 * (np.arange(n) - 103))  # kopniecie 3 klatki po powrocie
        known = np.arange(n) >= 100
        out = possession.touches(np.stack([x, np.full(n, 30.0)], axis=1), known, np.tile([5.0, 1.0], (n, 1)))
        self.assertTrue((out == NB).all())                                         # nie przypisujemy w locie


class TestTouchTurnover(unittest.TestCase):
    def counter_turnover(self, touches):
        ctrl = np.full(200, NB, np.int8)
        ctrl[:20] = H
        ctrl[40:60] = A                                                            # A przejmuje licznikiem w 52
        touch = np.full(200, NB, np.int8)
        for frame, team in touches:
            touch[frame] = team
        return possession.track_possession(ctrl, np.ones(200, bool), 13, touch, 75)

    def test_old_possessor_touch_soon_after_counter_turnover_reverts_it(self):
        out = self.counter_turnover([(40, A), (90, H)])                            # A nie utrzymal pilki
        self.assertTrue((out == H).all())

    def test_counter_turnover_stays_without_touch_or_after_new_possessor_touch(self):
        self.assertTrue((self.counter_turnover([])[40:] == A).all())
        self.assertTrue((self.counter_turnover([(40, A), (70, A), (90, H)])[40:] == A).all())
        self.assertTrue((self.counter_turnover([(40, A), (130, H)])[40:] == A).all())   # H za pozno

    def run_(self, touch_at, gap=75):
        ctrl = np.full(200, NB, np.int8)
        ctrl[:5] = H                                                               # posiadanie H od poczatku
        touch = np.full(200, NB, np.int8)
        for frame, team in touch_at:
            touch[frame] = team
        return possession.track_possession(ctrl, np.ones(200, bool), 13, touch, gap)

    def test_two_rival_touches_change_from_first(self):
        out = self.run_([(40, A), (70, A)])
        self.assertTrue((out[:40] == H).all())
        self.assertTrue((out[40:] == A).all())

    def test_own_touch_between_single_touch_far_pair_or_disabled_keep(self):
        self.assertTrue((self.run_([(40, A), (55, H), (70, A)]) == H).all())       # rykoszet, posiadacz gra dalej
        self.assertTrue((self.run_([(40, A)]) == H).all())                         # samo dotkniecie
        self.assertTrue((self.run_([(40, A), (150, A)]) == H).all())               # za daleko od siebie
        self.assertTrue((self.run_([(40, A), (70, A)], gap=0) == H).all())         # wylaczone


class TestVote(unittest.TestCase):
    def test_vote_removes_flicker_and_marks_empty_dead(self):
        team = np.array([H, H, A, H, H, NB, NB, NB, NB, NB, NB], np.int8)
        out = possession.majority_vote(team, 2).tolist()
        self.assertEqual(out[:5], [H] * 5)
        self.assertEqual(out[-1], config.DEAD)


if __name__ == "__main__":
    unittest.main()
