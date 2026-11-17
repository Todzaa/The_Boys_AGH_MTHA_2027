"""Whole match in one command: ATD -> outputs/match/the_boys_agh.csv, validation, public-window sanity check.

    python run_match.py

Labels are computed twice and both SHA-256 hashes must match (determinism). The console shows aggregate numbers only;
the full summary goes to outputs/logs/match_summary.json. The public five-minute window (17 spells) is a correctness
check, never used to choose variants or thresholds.

PIPELINE (label_match, all positions from ATD in metres, thresholds in geo/config.py)
  1. trusted_ball    : skip short ball-track stretches cut off by a jump faster than BALL_MAX_SPEED.
     clean_ball      : rolling median +/- BALL_SMOOTH frames, gaps <= BALL_GAP_FILL frames interpolated.
  2. team_distances  : distance from the ball to the nearest H and nearest A player (inf without ball or players).
  3. control         : team of the nearest player within CONTROL_RADIUS, if the rival is at least CONTROL_MARGIN farther
                       and the ball is not faster than CONTROL_BALL_SPEED; otherwise NOBODY (loose, contested, in flight).
  4. dead_mask       : D = ball beyond a line (until back on the pitch), or projected to cross a line while missing,
                       or median player speed < DEAD_SPEED for >= DEAD_MIN frames (extended while < DEAD_EXTEND);
                       a rolling ball (> BALL_LIVE_SPEED) is play; live fragments shorter than LIVE_MIN are D.
  5. touches         : change of ball velocity >= TOUCH_MIN_DV near one team's player (TOUCH_RADIUS, TOUCH_MARGIN).
     track_possession: first team in control opens each live stretch; the rival takes over when its control counter
                       (+1 rival, -1 holder, NOBODY leaves it) reaches TURNOVER_FRAMES, or after two rival touches
                       within TOUCH_PAIR frames; the change is dated back to the start of the advantage.
     merge_short     : a spell shorter than MIN_SPELL inside play reverts to the previous team.
  6. compose         : X only outside the halves, never on graded frames; no evidence at all = D.
"""
import json
import sys
import time

import numpy as np

from geo import atd, config, dead_ball, possession, submission
from geo.window import graded_mask, play_mask


TUNABLE = dict(margin=config.CONTROL_MARGIN, smooth=config.BALL_SMOOTH, gap=config.BALL_GAP_FILL,
               live_min=config.LIVE_MIN, min_spell=config.MIN_SPELL, exit=config.EXIT_HORIZON,
               exit_gap=config.EXIT_MIN_GAP, touch_pair=config.TOUCH_PAIR, touch_dv=config.TOUCH_MIN_DV,
               touch_lag=config.TOUCH_LAG, touch_radius=config.TOUCH_RADIUS, radius=config.CONTROL_RADIUS)


def label_match(trk, halves=config.HALVES, **params):
    """(H/A/D/X codes for the whole match, organisers' baseline codes, intermediate masks for statistics).

    halves: (first graded, kick-off, last + 1) per half; competition match by default, others for Metrica.
    params: overrides of TUNABLE (Metrica sweep only).
    """
    assert set(params) <= set(TUNABLE), "unknown parameters: {}".format(sorted(set(params) - set(TUNABLE)))
    p = {**TUNABLE, **params}
    ball_xy = trk.ball[:, :2]
    trusted = possession.trusted_ball(ball_xy, trk.ball_ok)
    ball_xy, known = possession.clean_ball(ball_xy, trusted, p["smooth"], p["gap"])
    dist = possession.team_distances(trk.players[:, :, :2], trk.player_ok, trk.slot_team, ball_xy, known)
    speed = possession.ball_speed(ball_xy, known)
    ctrl = possession.control(dist, p["radius"], speed=speed, margin=p["margin"])
    dead, by_out, by_act = dead_ball.dead_mask(ball_xy, known, trk.players[:, :, 2], trk.player_ok,
                                               halves, p["live_min"], p["exit"], p["exit_gap"])
    live = play_mask(trk.n, halves) & ~dead
    touch = possession.touches(ball_xy, known, dist, p["touch_dv"], p["touch_lag"], p["touch_radius"])
    team = possession.track_possession(ctrl, live, touch=touch, pair_gap=p["touch_pair"])
    team = possession.merge_short(team, live, p["min_spell"])
    codes = submission.compose(team, dead, halves, trk.n)

    raw = possession.team_distances(trk.players[:, :, :2], trk.player_ok, trk.slot_team, trk.ball[:, :2], trk.ball_ok)
    baseline = possession.majority_vote(possession.nearest_team(raw), config.BASELINE_VOTE)
    return codes, baseline, dict(trusted=known, ctrl=ctrl, touch=touch, dead=dead, by_out=by_out, by_act=by_act,
                                 live=live, team=team)


def stats(trk, codes, m):
    graded = graded_mask(trk.n)
    play = play_mask(trk.n) & graded

    def pct(mask, within):
        return 100.0 * float((mask & within).sum()) / max(float(within.sum()), 1.0)

    lengths = submission.spells(codes)
    stops = sum(len(possession.runs(m["dead"][k:e])) for _g, k, e in config.HALVES)
    return dict(
        ball_raw_pct=pct(trk.ball_ok, graded), ball_trusted_pct=pct(m["trusted"], graded),
        ball_dropped=int((trk.ball_ok & ~m["trusted"]).sum()),
        control_pct=pct(m["ctrl"] != config.NOBODY, play),
        touches=int((m["touch"][play & m["live"]] != config.NOBODY).sum()),
        dead_pct=pct(m["dead"], play), dead_out_pct=pct(m["by_out"], play), dead_activity_pct=pct(m["by_act"], play),
        stoppages=stops,
        longest_out_s=max((b - a for a, b in possession.runs(m["by_out"])), default=0) / config.FPS,
        no_evidence_pct=pct(m["live"] & (m["team"] == config.NOBODY), graded),
        labels_pct={config.LABELS[c]: pct(codes == c, graded) for c in range(4)},
        spells=len(lengths), mean_spell_s=float(np.mean(lengths)) / config.FPS if lengths else 0.0,
    )


def public_check(pm, gt_path, variants):
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    out = {}
    for name, codes in variants.items():
        r = pm.evaluate(gt, codes, taus=pm.TAUS, bootstrap_n=0)
        out[name] = {k: r[k] for k in ("macro_f1", "spell_f1_mean", "n_pred_spells", "n_gt_spells",
                                        "mean_spell_duration_s", "gt_mean_spell_duration_s")}
    return out


def main():
    t0 = time.time()
    trk = atd.load()
    t_load = time.time() - t0
    codes, baseline, masks = label_match(trk)
    again, _b, _m = label_match(trk)
    path = config.MATCH_DIR / "{}.csv".format(config.TEAM_NAME)
    sha = submission.write(path, codes)
    sha_again = submission.write(config.MATCH_DIR / "rerun_check.csv", again)
    (config.MATCH_DIR / "rerun_check.csv").unlink()
    s = stats(trk, codes, masks)
    s.update(sha256=sha, deterministic=sha == sha_again, seconds=round(time.time() - t0, 1))

    print("ATD loaded in {:.0f} s, labels in {:.0f} s".format(t_load, time.time() - t0 - t_load))
    print("\n== ball and control (graded frames) ==")
    print("ball in ATD {:.1f}%   after filtering and gap filling {:.1f}%   (dropped observations {})".format(
        s["ball_raw_pct"], s["ball_trusted_pct"], s["ball_dropped"]))
    print("control (player <= {:.1f} m from the ball) in {:.1f}% of play   touches assigned to a team {}".format(
        config.CONTROL_RADIUS, s["control_pct"], s["touches"]))
    print("\n== dead ball (frames from kick-off) ==")
    print("D {:.1f}%   (ball out {:.1f}%, low activity {:.1f}%)   stoppages {}   longest 'ball out' {:.0f} s".format(
        s["dead_pct"], s["dead_out_pct"], s["dead_activity_pct"], s["stoppages"], s["longest_out_s"]))
    print("\n== submission file ==")
    print("graded frames: {}".format("  ".join("{} {:.1f}%".format(k, v) for k, v in s["labels_pct"].items())))
    print("spells {}   mean {:.1f} s   no evidence at all (D) {:.1f}% of graded frames".format(
        s["spells"], s["mean_spell_s"], s["no_evidence_pct"]))
    print("deterministic: {}   sha256 {}".format("YES" if s["deterministic"] else "NO", sha[:16]))

    pm, gt_path = submission.load_scorer()
    if pm is None:
        print("\nvalidation and sanity check skipped: {}".format(gt_path))
    else:
        reason = pm.validate_submission(pm.read_label_csv(path), config.N_FRAMES)
        s["validation"] = reason or "OK"
        print("validation (organisers' validate_submission): {}".format(s["validation"]))
        s["public_check"] = public_check(pm, gt_path, {"ours": codes, "baseline_smooth50": baseline})
        print("\n== public five-minute window (sanity check, not for choosing variants) ==")
        print("{:<18} {:>8} {:>8} {:>6} {:>8}".format("variant", "macroF1", "spellF1", "spells", "dur(s)"))
        for name, r in s["public_check"].items():
            print("{:<18} {:>8.4f} {:>8.4f} {:>6} {:>8.1f}".format(
                name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["mean_spell_duration_s"]))
        r = next(iter(s["public_check"].values()))
        print("GT: {} spells, mean {:.1f} s".format(r["n_gt_spells"], r["gt_mean_spell_duration_s"]))

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR / "match_summary.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    print("\nwritten {}  ({:.0f} s)".format(path, time.time() - t0))
    return 0 if s["deterministic"] and s.get("validation", "OK") == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
