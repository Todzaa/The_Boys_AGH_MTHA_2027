"""Walidacja pipeline'u na publicznych meczach Metrica (inne mecze niz konkursowy), scorerem organizatorow.

    python validate_metrica.py                 # oba mecze, nasz pipeline i baseline kontra wzorzec ze zdarzen
    python validate_metrica.py --game 1
    python validate_metrica.py --sweep         # przeglad siatki GRID (kilka minut), ranking po sredniej spellF1
    python validate_metrica.py --atd-like      # to samo na Metrice z pilka popsuta jak w ATD (metrica.atd_like,
                                               # kazdy mecz w ATD_SEEDS losowaniach szumu); dziala tez z --sweep

Wzorzec H/A/D jest skladany ze zdarzen Metrica (geo/metrica.py), wiec to przyblizenie GT, nie GT organizatorow.
Dane: METRICA_DIR, domyslnie ../metrica_sample_data (git clone https://github.com/metrica-sports/sample-data.git).
Pelne wyniki w outputs/logs/metrica.json.
"""
import argparse
import itertools
import json
import sys
import time

import numpy as np

from geo import config, metrica, submission
from run_match import TUNABLE, label_match

# siatka przegladu: obecne wartosci z config.py musza w niej byc (punkt odniesienia)
GRID = dict(touch_pair=(0, 75), touch_lag=(3, 6, 12), touch_radius=(2.5, 3.5), touch_dv=(3.0, 5.0))
KEYS = ("macro_f1", "spell_f1_mean", "alive_dead_f1", "team_acc_both_alive", "n_pred_spells", "n_gt_spells",
        "mean_spell_duration_s", "gt_mean_spell_duration_s")


def score(pm, gt, pred):
    r = pm.evaluate(gt, pred, taus=pm.TAUS, bootstrap_n=0)
    out = {k: r[k] for k in KEYS}
    out["transition_f1"] = r["transitions"][pm.TAUS[-1]]["f1"]
    return out


def load_games(games, atd_like=False):
    """[(nazwa, tracking, polowy, wzorzec)]; z atd_like kazdy mecz w ATD_SEEDS wersjach z pilka popsuta jak w ATD."""
    out = []
    for game in games:
        trk, _period = metrica.load_tracking(game)
        ev = metrica.load_events(game)
        halves, gt = metrica.halves(ev), metrica.ground_truth(ev, trk.n)
        if atd_like:
            out += [("{}/atd{}".format(game, s), metrica.atd_like(trk, s), halves, gt) for s in metrica.ATD_SEEDS]
        else:
            out.append((str(game), trk, halves, gt))
    return out


def run_game(pm, trk, halves, gt):
    t0 = time.time()
    codes, baseline, _masks = label_match(trk, halves)
    graded = gt != config.ABSTAIN
    share = {config.LABELS[c]: float(np.mean(gt[graded] == c) * 100) for c in range(3)}
    return dict(frames=int(trk.n), graded=int(graded.sum()), ball_pct=float(trk.ball_ok[graded].mean() * 100),
                gt_pct=share, nasz=score(pm, gt, codes), baseline_smooth50=score(pm, gt, baseline),
                seconds=round(time.time() - t0, 1))


def sweep(pm, data, top=10):
    """Wszystkie kombinacje GRID na wszystkich meczach; ranking po sredniej spellF1, obecne ustawienie oznaczone *."""
    games = [name for name, _t, _h, _g in data]
    data = [(trk, halves, gt) for _n, trk, halves, gt in data]
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        rows.append((params, [score(pm, gt, label_match(trk, halves, **params)[0]) for trk, halves, gt in data]))
    mean = lambda s, k: float(np.mean([x[k] for x in s]))
    rows.sort(key=lambda r: (-mean(r[1], "spell_f1_mean"), -mean(r[1], "macro_f1")))
    current = {k: TUNABLE[k] for k in GRID}
    print("\n== przeglad {} ustawien na meczach {} (spellF1 mecz po meczu, srednia) ==".format(len(rows), games))
    print("   {:<58} {:>8} {:>8} {:>8}  {}".format("ustawienie", "spellF1", "macroF1", "poss", "spellF1 meczow"))
    for rank, (params, s) in enumerate(rows):
        if rank < top or params == current:
            print("{} {:<58} {:>8.4f} {:>8.4f} {:>8.0f}  {}".format(
                "*" if params == current else " ", " ".join("{}={}".format(k, v) for k, v in params.items()),
                mean(s, "spell_f1_mean"), mean(s, "macro_f1"), mean(s, "n_pred_spells"),
                " ".join("{:.4f}".format(x["spell_f1_mean"]) for x in s)))
    print("* = obecny config.py; roznice < ok. 0.01 sredniej traktuj jako szum")
    return [dict(params=p, scores=s) for p, s in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", type=int, choices=metrica.GAMES, action="append")
    ap.add_argument("--sweep", action="store_true", help="przeglad siatki GRID zamiast pojedynczego przebiegu")
    ap.add_argument("--atd-like", action="store_true", help="pilka popsuta jak w ATD (metrica.atd_like)")
    args = ap.parse_args()
    pm, reason = submission.load_scorer()
    if pm is None:
        raise SystemExit(reason)
    data = load_games(args.game or metrica.GAMES, args.atd_like)
    suffix = "_atd_like" if args.atd_like else ""
    if args.sweep:
        t0 = time.time()
        rows = sweep(pm, data)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = config.LOG_DIR / "metrica_sweep{}.json".format(suffix)
        log.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
        print("({:.0f} s, pelne wyniki w {})".format(time.time() - t0, log))
        return 0
    results = {}
    for game, trk, halves, gt in data:
        r = results["game_{}".format(game)] = run_game(pm, trk, halves, gt)
        print("\n== Sample_Game_{}: {} klatek ocenianych, pilka {:.1f}%, wzorzec {} ({:.0f} s) ==".format(
            game, r["graded"], r["ball_pct"], "  ".join("{} {:.1f}%".format(k, v) for k, v in r["gt_pct"].items()),
            r["seconds"]))
        print("{:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>7} {:>7}".format(
            "wariant", "spellF1", "macroF1", "zywa/D", "druzyna", "przejsc", "poss", "GT", "dur(s)", "GT(s)"))
        for name in ("nasz", "baseline_smooth50"):
            s = r[name]
            print("{:<18} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>6} {:>6} {:>7.1f} {:>7.1f}".format(
                name, s["spell_f1_mean"], s["macro_f1"], s["alive_dead_f1"], s["team_acc_both_alive"],
                s["transition_f1"], s["n_pred_spells"], s["n_gt_spells"], s["mean_spell_duration_s"],
                s["gt_mean_spell_duration_s"]))
    if len(results) > 1:
        print("\nsrednio spellF1: " + "  ".join("{} {:.4f}".format(
            name, np.mean([r[name]["spell_f1_mean"] for r in results.values()])) for name in ("nasz", "baseline_smooth50")))
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR / "metrica{}.json".format(suffix)).write_text(json.dumps(results, indent=2, default=float),
                                                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
