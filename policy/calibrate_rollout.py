#!/usr/bin/env python3
"""Offline calibration of rollout.py's advance thresholds against the real, trained
best_v3_tactile.pt checkpoint on held-out (val-split) episodes -- no Isaac Sim needed
(docs/research/design_v3_stage_memory_tactile.md §5 item 4 / rollout.py's own TODOs).

Two passes, run as two separate CLI invocations because pass 2 must run AFTER rollout.py's
TAU_HI/TAU_LO/TAU_PROG/TactileReleaseDetector constants have been hand-updated with pass 1's
recommendation -- pass 2 needs to exercise the real, edited `replay_episode()`, not a copy:

  pass1  Oracle-fed diagnostic (no compounding error): batched forward over the CALIB split
         of held-out episodes with the TRUE auto-labeled fold_id fed every frame (like
         eval_ckpt.py), at rollout.py's 10 fps operating rate. Records completion_logit
         (sigmoid) + progress_pred + raw tactile-force energy at every frame. Then, purely in
         postprocessing (no more GPU calls), grid-searches TAU_HI/TAU_LO/TAU_PROG (isolating
         the completion/progress heads from the tactile backstop) and separately the release
         detector's hi/lo, by simulating rollout.py's exact `_advance()` OR-logic per TRUE
         fold-segment and scoring predicted-transition-time vs the true boundary (+/- a
         tolerance window).
  pass2  Actual closed-loop `replay_episode()` (model-driven fold_id, compounding) on the
         held-out TEST split, using rollout.py's CURRENT constants -- so edit rollout.py with
         pass 1's recommendation BEFORE running this.

held-out val episodes (53, from outputs/norm_stats.json's split.val -- never seen in training)
are further split CALIB (~2/3, threshold selection) / TEST (~1/3, final closed-loop numbers)
so pass 2's reported numbers aren't threshold-selection-fit to the same episodes (seed=42,
fixed, printed below for reproducibility).

Usage:
  python policy/calibrate_rollout.py pass1
  python policy/calibrate_rollout.py pass2
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "data"))
sys.path.insert(0, str(HERE / "model"))
from fast_folding_dataset import FastFoldingDataset               # noqa: E402
from autolabel_folds import TACT                                  # noqa: E402
from folding_act import FoldingACT                                # noqa: E402
import rollout                                                    # noqa: E402  (reuse compute_fold_budgets/TactileReleaseDetector)

CKPT = ROOT / "outputs" / "ckpt" / "best_v3_tactile.pt"
OUT_DIR = ROOT / "outputs" / "experiments" / "rollout_calibration"
SPLIT_SEED = 42
CALIB_FRAC = 0.66
FPS_OP = 10                     # operating rate, matches rollout.py's default
N_FOLDS = 6


def load_model(dev="cuda:0", ckpt_path=None):
    """Build FoldingACT from the checkpoint's OWN saved construction kwargs (not hardcoded
    flags) -- catches drift like rollout.py's _smoke() omitting encoder='dinov3' (defaults
    to resnet18 there, which would silently build the WRONG architecture for this checkpoint;
    harmless in _smoke() since it never loads a state_dict, but a real trap if copy-pasted).
    v8: ckpt_path selects the checkpoint (default = the module-level v3 path, unchanged);
    n_tempos/n_cams are honored so v5+/v6/v7 ckpts load too. Returns (model, ck_meta)."""
    ckpt_path = Path(ckpt_path) if ckpt_path else CKPT
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = FoldingACT(encoder=ck["encoder"], tactile=ck["tactile"], n_decoder_layers=ck["n_decoder_layers"],
                       aux_heads=ck["aux_heads"], decoder_tactile_attn=ck["decoder_tactile_attn"],
                       pre_encoder_tactile_attn=ck["pre_encoder_tactile_attn"],
                       contact_gate=ck["contact_gate"], detach_aux=ck["detach_aux"],
                       n_tempos=ck.get("n_tempos", 0), n_cams=ck.get("n_cams", 4)).to(dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    print(f"loaded {ckpt_path.name}: step={ck['step']} saved_val_l1={ck['val_l1']:.4f} "
          f"encoder={ck['encoder']} n_tempos={ck.get('n_tempos', 0)} n_cams={ck.get('n_cams', 4)} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M (strict load)")
    meta = {"tag": ckpt_path.stem.removeprefix("best_"),
            "tempos": list(ck.get("tempos") or []), "n_cams": ck.get("n_cams", 4)}
    return model, meta


def make_ds(meta):
    """Dataset matched to the checkpoint: tempo-conditioned ckpts need `tempo_id` in every
    item (fixed to the 1.0x safety tempo -- an identity warp), 3-cam ckpts need three_cam."""
    tempos = meta["tempos"] or None
    return FastFoldingDataset(split="val", tactile=True, tempos=tempos,
                              fixed_tempo_id=0 if tempos else None,
                              three_cam=meta["n_cams"] == 3)


def calib_test_split(ds):
    eps = list(range(len(ds.dur)))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(eps)
    n_calib = round(len(eps) * CALIB_FRAC)
    return sorted(eps[:n_calib]), sorted(eps[n_calib:])


TRACE_CACHE = OUT_DIR / "traces_cache.npz"


def load_or_compute_traces(model, ds, episode_ids, dev="cuda:0", cache=None):
    """The oracle-fed forward pass (~15-17 min for 35 calib episodes on one RTX 5070 Ti) is
    by far the expensive step; the threshold grid search that follows is pure numpy
    postprocessing. Cache the raw per-frame traces to disk so grid/tolerance choices can be
    iterated on without re-running the model.
    v8: `cache` selects a TAG-KEYED cache file (traces_cache_<ckpt_tag>.npz) -- the legacy
    single-file cache was keyed only by episode-id set, so a different checkpoint would
    silently reuse stale traces. Default (None) keeps the legacy v3 path untouched."""
    TRACE_CACHE = cache if cache is not None else (OUT_DIR / "traces_cache.npz")
    if TRACE_CACHE.exists():
        d = np.load(TRACE_CACHE, allow_pickle=True)
        cached_ids = d["episode_ids"].tolist()
        if set(cached_ids) == set(episode_ids):
            print(f"loaded cached traces from {TRACE_CACHE} ({len(cached_ids)} episodes)")
            traces = {}
            for ei in episode_ids:
                traces[ei] = {k: d[f"{ei}_{k}"] for k in ("t_s", "fid", "p_complete", "p_prog", "energy")}
            return traces
    traces = oracle_forward_pass(model, ds, episode_ids, dev)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"episode_ids": np.array(episode_ids)}
    for ei, tr in traces.items():
        for k, v in tr.items():
            save_kwargs[f"{ei}_{k}"] = v
    np.savez(TRACE_CACHE, **save_kwargs)
    print(f"cached traces -> {TRACE_CACHE}")
    return traces


@torch.no_grad()
def oracle_forward_pass(model, ds, episode_ids, dev="cuda:0", batch_size=128):
    """Feed the TRUE (oracle) fold_id every frame at the 10fps operating stride, batched (no
    sequential dependency needed here -- unlike replay_episode, oracle fold_id doesn't depend
    on the model's own output). Returns per-episode dict of aligned numpy traces."""
    stride = max(1, round(ds.fps / FPS_OP))
    traces = {}
    for ei in episode_ids:
        T = ds.dur[ei]
        base = sum(ds.dur[:ei])
        ts = list(range(0, T, stride))
        items = [ds[base + t] for t in ts]
        n_done = 0
        pc_all, pp_all = [], []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            batch = {k: torch.stack([it[k] for it in chunk]).to(dev) for k in chunk[0]}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            pc_all.append(torch.sigmoid(out.completion_logit)[:, 0].float().cpu().numpy())
            pp_all.append(out.progress_pred[:, 0].float().cpu().numpy())
            n_done += len(chunk)
        tac_raw = ds.low[ei]["observation.tactile"]
        energy = np.abs(tac_raw[:, TACT]).sum(1)[ts]
        fid = np.array([fold_id_of(ds, ei, t) for t in ts])
        traces[ei] = {
            "t_s": np.array(ts) / ds.fps,
            "fid": fid,
            "p_complete": np.concatenate(pc_all),
            "p_prog": np.concatenate(pp_all),
            "energy": energy,
        }
    return traces


def fold_id_of(ds, ei, t):
    season, ep = ds._ep_keys[ei]
    lab = ds.labels[f"{season}/ep{ep}"]
    from fold_utils import fold_features
    fid, _ = fold_features(t / ds.fps, lab["boundaries_s"], ds.dur[ei] / ds.fps)
    return fid


def boundaries_of(ds, ei):
    season, ep = ds._ep_keys[ei]
    return ds.labels[f"{season}/ep{ep}"]["boundaries_s"]


# ---------------- Step A: TAU_HI/TAU_LO/TAU_PROG grid, heads-only (backstop disabled) ----------------

def search_tau(traces, ds, episode_ids, tau_hi_grid, tau_lo_grid, tau_prog_grid, tol_s=1.0):
    """For each (tau_hi, tau_lo, tau_prog): per true segment k=0..4 in each episode, find the
    first frame where (p_complete>tau_hi) or (p_prog>tau_prog and p_complete>tau_lo) -- the
    exact primary+co-signal half of rollout._advance(), budget/backstop excluded so this
    isolates the completion/progress heads' own timing quality. Score = match rate within
    +/-tol_s of the true boundary + mean |offset| among matches."""
    results = []
    for tau_hi in tau_hi_grid:
        for tau_lo in tau_lo_grid:
            for tau_prog in tau_prog_grid:
                if tau_lo >= tau_hi:
                    continue
                n_seg, n_match, offsets, n_primary, n_cosignal = 0, 0, [], 0, 0
                for ei in episode_ids:
                    tr = traces[ei]
                    bnds = boundaries_of(ds, ei)
                    for k in range(N_FOLDS - 1):
                        sel = tr["fid"] == k
                        if not sel.any():
                            continue
                        pc, pp, ts = tr["p_complete"][sel], tr["p_prog"][sel], tr["t_s"][sel]
                        primary = pc > tau_hi
                        cosig = (pp > tau_prog) & (pc > tau_lo)
                        fire = primary | cosig
                        n_seg += 1
                        if fire.any():
                            idx = int(np.argmax(fire))
                            offset = ts[idx] - bnds[k]
                            if abs(offset) <= tol_s:
                                n_match += 1
                                offsets.append(offset)
                            if primary[idx]:
                                n_primary += 1
                            else:
                                n_cosignal += 1
                results.append({
                    "tau_hi": tau_hi, "tau_lo": tau_lo, "tau_prog": tau_prog,
                    "n_segments": n_seg, "n_matched": n_match,
                    "match_rate": n_match / max(n_seg, 1),
                    "mean_abs_offset": float(np.mean(np.abs(offsets))) if offsets else None,
                    "median_abs_offset": float(np.median(np.abs(offsets))) if offsets else None,
                    "frac_primary": n_primary / max(n_primary + n_cosignal, 1),
                })
    return sorted(results, key=lambda r: (-r["match_rate"], r["mean_abs_offset"] or 1e9))


def _advance_param(p_complete, p_prog, step_in_fold, budget_k, released, tau_hi, tau_lo, tau_prog):
    """Parametrized copy of rollout._advance()'s exact OR-logic. NOT calling rollout._advance()
    directly here on purpose: that function reads the MODULE-LEVEL TAU_HI/TAU_LO/TAU_PROG
    globals (still the original 0.9/0.5/0.9 placeholders until rollout.py is hand-edited with
    this script's recommendation), so calling it during a grid search would silently ignore
    every candidate value being tested (caught during this task: first run of search_release
    used rollout._advance and returned near-zero matches because of exactly this)."""
    return (p_complete > tau_hi) or (p_prog > tau_prog and p_complete > tau_lo) or \
           (step_in_fold > budget_k and released)


# ---------------- Step B: release-detector hi/lo, backstop-only segments ----------------

def search_release(traces, ds, episode_ids, tau_hi, tau_lo, tau_prog, budgets, hi_grid, lo_grid,
                    win=5, tol_s=1.0):
    """Fixing tau_hi/tau_lo/tau_prog at Step A's pick, grid-search the release detector's
    hi/lo using the FULL 3-way _advance() (primary+co-signal+budget-backstop), scored only on
    how well the backstop rescues the segments Step A's heads-only search left unmatched."""
    results = []
    for hi in hi_grid:
        for lo in lo_grid:
            if lo >= hi:
                continue
            n_seg, n_match, offsets, n_backstop = 0, 0, [], 0
            for ei in episode_ids:
                tr = traces[ei]
                bnds = boundaries_of(ds, ei)
                det = rollout.TactileReleaseDetector(hi=hi, lo=lo, win=win)
                # rebuild a fake 60-D raw tactile frame whose TACT-indexed abs-sum equals the
                # precomputed energy e (det.update() indexes tactile_raw[TACT], TACT has indices
                # up to 38 -- a length-len(TACT) dummy array indexes out of bounds)
                released = []
                for e in tr["energy"]:
                    fake = np.zeros(60, np.float32)
                    fake[TACT] = e / len(TACT)
                    released.append(det.update(fake))
                released = np.array(released)
                for k in range(N_FOLDS - 1):
                    sel = tr["fid"] == k
                    if not sel.any():
                        continue
                    pc, pp, ts, rel = tr["p_complete"][sel], tr["p_prog"][sel], tr["t_s"][sel], released[sel]
                    n_seg += 1
                    step_in_fold, fired_at, branch = 0, None, None
                    for i in range(len(pc)):
                        step_in_fold += 1
                        if _advance_param(pc[i], pp[i], step_in_fold, budgets[k], rel[i], tau_hi, tau_lo, tau_prog):
                            fired_at = ts[i]
                            branch = "backstop" if not (pc[i] > tau_hi or (pp[i] > tau_prog and pc[i] > tau_lo)) else "head"
                            break
                    if fired_at is not None:
                        offset = fired_at - bnds[k]
                        if abs(offset) <= tol_s:
                            n_match += 1
                            offsets.append(offset)
                        if branch == "backstop":
                            n_backstop += 1
            results.append({
                "hi": hi, "lo": lo, "n_segments": n_seg, "n_matched": n_match,
                "match_rate": n_match / max(n_seg, 1),
                "mean_abs_offset": float(np.mean(np.abs(offsets))) if offsets else None,
                "n_backstop_fired": n_backstop,
            })
    return sorted(results, key=lambda r: (-r["match_rate"], r["mean_abs_offset"] or 1e9))


def pass1(args):
    dev = "cuda:0"
    model, meta = load_model(dev)
    ds = make_ds(meta)
    calib_ids, test_ids = calib_test_split(ds)
    print(f"held-out val episodes: {len(ds.dur)} total -> calib={len(calib_ids)} test={len(test_ids)} (seed={SPLIT_SEED})")
    print(f"calib episode indices: {calib_ids}")
    print(f"test  episode indices: {test_ids}")

    print(f"\nrunning (or loading cached) oracle-fed diagnostic forward pass over {len(calib_ids)} calib episodes @ {FPS_OP}fps...")
    traces = load_or_compute_traces(model, ds, calib_ids, dev)
    n_frames = sum(len(t["t_s"]) for t in traces.values())
    print(f"collected {n_frames} frames")

    # sanity: raw p_complete/p_prog behavior mid-fold vs near-boundary
    all_pc, all_pp, all_near = [], [], []
    for ei in calib_ids:
        tr = traces[ei]
        bnds = boundaries_of(ds, ei)
        for k in range(N_FOLDS - 1):
            sel = tr["fid"] == k
            if not sel.any():
                continue
            near = np.abs(tr["t_s"][sel] - bnds[k]) <= 1.0
            all_pc.append(tr["p_complete"][sel]); all_pp.append(tr["p_prog"][sel]); all_near.append(near)
    all_pc, all_pp, all_near = np.concatenate(all_pc), np.concatenate(all_pp), np.concatenate(all_near)
    print(f"\np_complete: near-boundary(+/-1s) mean={all_pc[all_near].mean():.3f} std={all_pc[all_near].std():.3f}  "
          f"mid-fold mean={all_pc[~all_near].mean():.3f} std={all_pc[~all_near].std():.3f}")
    print(f"p_prog:     near-boundary(+/-1s) mean={all_pp[all_near].mean():.3f} std={all_pp[all_near].std():.3f}  "
          f"mid-fold mean={all_pp[~all_near].mean():.3f} std={all_pp[~all_near].std():.3f}")

    print("\n--- Step A: TAU_HI/TAU_LO/TAU_PROG grid (heads-only, backstop disabled), tol=+/-1s ---")
    # widened down into the observed operating range after the first run showed p_complete
    # never exceeds ~0.35-0.4 even near a true boundary (mean 0.167, std 0.089) -- the original
    # [0.5, 0.95] grid never fired the primary branch at all (frac_primary=0.00 everywhere)
    tau_hi_grid = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tau_lo_grid = [0.05, 0.1, 0.15, 0.2, 0.3]
    tau_prog_grid = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    tau_results = search_tau(traces, ds, calib_ids, tau_hi_grid, tau_lo_grid, tau_prog_grid, tol_s=1.0)
    for r in tau_results[:15]:
        print(f"  tau_hi={r['tau_hi']:.2f} tau_lo={r['tau_lo']:.2f} tau_prog={r['tau_prog']:.2f}  "
              f"match={r['n_matched']}/{r['n_segments']} ({r['match_rate']*100:.1f}%)  "
              f"mean|off|={r['mean_abs_offset']}  frac_primary={r['frac_primary']:.2f}")
    # also report at a looser (3s) tolerance -- matches the known DTW label-noise scale (2.6-2.8s)
    tau_results_loose = search_tau(traces, ds, calib_ids, tau_hi_grid, tau_lo_grid, tau_prog_grid, tol_s=3.0)
    print("\n  (same grid @ +/-3s tolerance, matching known auto-label boundary noise)")
    for r in tau_results_loose[:5]:
        print(f"  tau_hi={r['tau_hi']:.2f} tau_lo={r['tau_lo']:.2f} tau_prog={r['tau_prog']:.2f}  "
              f"match={r['n_matched']}/{r['n_segments']} ({r['match_rate']*100:.1f}%)  mean|off|={r['mean_abs_offset']}")

    best = tau_results[0]
    print(f"\n>>> Step A pick (best @ +/-1s): tau_hi={best['tau_hi']} tau_lo={best['tau_lo']} tau_prog={best['tau_prog']}")

    print("\n--- Step B: TactileReleaseDetector hi/lo grid (full 3-way _advance, backstop rescue) ---")
    budgets = rollout.compute_fold_budgets(fps=FPS_OP)
    print(f"fold step budgets (p90 @10fps): {budgets}")
    hi_grid = [15, 25, 35, 50, 70]
    lo_grid = [3, 5, 8, 12]
    rel_results = search_release(traces, ds, calib_ids, best["tau_hi"], best["tau_lo"], best["tau_prog"],
                                 budgets, hi_grid, lo_grid)
    for r in rel_results[:15]:
        print(f"  hi={r['hi']:.0f} lo={r['lo']:.0f}  match={r['n_matched']}/{r['n_segments']} "
              f"({r['match_rate']*100:.1f}%)  mean|off|={r['mean_abs_offset']}  n_backstop_fired={r['n_backstop_fired']}")
    best_rel = rel_results[0]
    print(f"\n>>> Step B pick: hi={best_rel['hi']} lo={best_rel['lo']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "calib_ids": calib_ids, "test_ids": test_ids, "split_seed": SPLIT_SEED,
        "n_calib_frames": n_frames,
        "p_complete_near_boundary_mean": float(all_pc[all_near].mean()),
        "p_complete_mid_fold_mean": float(all_pc[~all_near].mean()),
        "p_prog_near_boundary_mean": float(all_pp[all_near].mean()),
        "p_prog_mid_fold_mean": float(all_pp[~all_near].mean()),
        "tau_search_top10": tau_results[:10],
        "tau_search_loose_tol3s_top5": tau_results_loose[:5],
        "chosen_tau": {"tau_hi": best["tau_hi"], "tau_lo": best["tau_lo"], "tau_prog": best["tau_prog"]},
        "release_search_top10": rel_results[:10],
        "chosen_release": {"hi": best_rel["hi"], "lo": best_rel["lo"]},
        "budgets_steps_10fps": budgets,
    }
    out_path = OUT_DIR / "pass1_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


def cascade_stats(pred_ts):
    """Cascade metrics over one episode's predicted advance times (sorted). Definitions match
    docs/experiments/rollout_calibration_v3.md §4.1: legacy cascade = >=3 consecutive advances
    <=0.2s apart; sub-second pair = any two consecutive advances <1s apart (the Phase-0
    hardening target: min-dwell makes these structurally impossible)."""
    gaps = np.diff(pred_ts) if len(pred_ts) > 1 else np.array([])
    run, max_run = 1, 1
    for g in gaps:
        run = run + 1 if g <= 0.2 else 1
        max_run = max(max_run, run)
    return {"n_subsecond_pairs": int((gaps < 1.0).sum()), "legacy_cascade": bool(max_run >= 3)}


def nn_match(pred_ts, gt_bnds, tol_s=3.0):
    """Nearest-neighbor event matching (the report §4's metric, not naive index pairing):
    globally greedy on |offset| <= tol, each predicted advance / true boundary used once."""
    pairs = sorted(((abs(p - b), i, j, p - b) for i, p in enumerate(pred_ts)
                    for j, b in enumerate(gt_bnds) if abs(p - b) <= tol_s))
    used_p, used_g, offs = set(), set(), []
    for _, i, j, off in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); offs.append(off)
    return len(offs), offs


def pass2(args):
    dev = "cuda:0"
    prev = json.loads((OUT_DIR / "pass1_results.json").read_text())
    test_ids = prev["test_ids"]
    print(f"pass1 chosen thresholds (for reference): {prev['chosen_tau']}  release={prev['chosen_release']}")
    print(f"rollout.py CURRENT constants: TAU_HI={rollout.TAU_HI} TAU_LO={rollout.TAU_LO} TAU_PROG={rollout.TAU_PROG}")
    print(f"hardened FoldTracker: budgets_s={rollout.compute_fold_budgets_s()} "
          f"min_dwell_s={rollout.FOLD_MIN_DWELL_S} sustain_s={rollout.SUSTAIN_S}")
    import inspect
    sig = inspect.signature(rollout.TactileReleaseDetector.__init__)
    print(f"TactileReleaseDetector defaults: hi={sig.parameters['hi'].default} lo={sig.parameters['lo'].default}")

    model, meta = load_model(dev, ckpt_path=getattr(args, "ckpt", None))
    ds = make_ds(meta)
    stage_runner = None
    if getattr(args, "stage_net", None):
        from stage_net import StageNetRunner
        stage_runner = StageNetRunner(args.stage_net, device=dev)
        assert stage_runner.tau is not None, \
            "stage ckpt has no tau -- run pass1v8 first, then train_stage_net.py --set-tau"
        print(f"stage co-signal ON: {args.stage_net} (tau={stage_runner.tau})")
    print(f"\nrunning closed-loop replay_episode() on {len(test_ids)} TEST episodes...")
    results = []
    for ei in test_ids:
        res = rollout.replay_episode(model, ds, ei, dev=dev, fps=FPS_OP, max_minutes=10.0,
                                     stage_runner=stage_runner)
        bnds = res["gt_boundaries_s"]
        pred_ts = [e["t_s"] for e in res["advance_log"]]
        reasons = [e.get("reason", "?") for e in res["advance_log"]]
        n_matched, offs = 0, []
        for i, b in enumerate(bnds):
            if i < len(pred_ts):
                off = pred_ts[i] - b
                offs.append(off)
                if abs(off) <= 3.0:
                    n_matched += 1
        n_nn, nn_offs = nn_match(pred_ts, bnds)
        results.append({
            "episode": res["episode"], "gt_duration_s": res["gt_duration_s"],
            "completed_count": res["completed_count"], "n_steps": res["n_steps"],
            "within_cap": res["n_steps"] < int(10 * 60 * FPS_OP),
            "gt_boundaries_s": bnds, "pred_advance_t_s": pred_ts, "reasons": reasons,
            "n_matched_3s": n_matched, "offsets_3s": offs,
            "nn_matched_3s": n_nn, "nn_offsets_3s": nn_offs,
            **cascade_stats(pred_ts),
        })
        print(f"  [{res['episode']}] completed={res['completed_count']}/6  steps={res['n_steps']}  "
              f"gt_bnds={bnds}  pred_t={[round(t,1) for t in pred_ts]}  reasons={reasons}")

    completed = [r["completed_count"] for r in results]
    print(f"\nmean completed_count: {np.mean(completed):.2f} advances of 5 boundaries  (episodes: {len(results)})")
    within_cap = sum(r["within_cap"] for r in results)
    print(f"episodes terminating within the 10-min cap: {within_cap}/{len(results)}")
    all_offs = [o for r in results for o in r["offsets_3s"]]
    if all_offs:
        print(f"matched-transition timing offset (n={len(all_offs)}): mean={np.mean(all_offs):.2f}s "
              f"median={np.median(all_offs):.2f}s |mean|={np.mean(np.abs(all_offs)):.2f}s")
    # Phase-0 gate metrics (hardening targets: 0 sub-second cascades; NN match rate >= old 21%)
    n_sub = sum(r["n_subsecond_pairs"] for r in results)
    n_casc_eps = sum(r["legacy_cascade"] for r in results)
    n_gt = sum(len(r["gt_boundaries_s"]) for r in results)
    n_nn_total = sum(r["nn_matched_3s"] for r in results)
    n_budget = sum(r["reasons"].count("budget") for r in results)
    n_signal = sum(r["reasons"].count("signal") for r in results)
    print(f"cascades: sub-second advance pairs={n_sub}  legacy(>=3 within 0.2s) episodes={n_casc_eps}/{len(results)}")
    print(f"NN event match @ +/-3s: {n_nn_total}/{n_gt} ({n_nn_total/max(n_gt,1)*100:.1f}%)   "
          f"advance reasons: signal={n_signal} budget={n_budget}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / (f"pass2_results_{args.tag}.json" if args.tag else "pass2_results.json")
    out_path.write_text(json.dumps({"results": results,
                                    "mean_completed": float(np.mean(completed)),
                                    "within_cap": within_cap, "n_episodes": len(results),
                                    "n_subsecond_pairs": n_sub, "n_legacy_cascade_episodes": n_casc_eps,
                                    "nn_matched_3s": n_nn_total, "n_gt_boundaries": n_gt,
                                    "nn_match_rate": n_nn_total / max(n_gt, 1),
                                    "advance_reasons": {"signal": n_signal, "budget": n_budget}}, indent=2))
    print(f"\nwrote {out_path}")


def simulate_tracker(trace, bnds, stage_pi=None, tau=None):
    """Replay one cached trace through the FULL hardened FoldTracker (dwell + sustain +
    budget, wall-clock) with an optional StageNet co-signal. Zero GPU. Returns the
    advance list [(t_s, reason, from_fold)]."""
    tracker = rollout.FoldTracker()
    tracker.reset(float(trace["t_s"][0]))
    advances = []
    for i, t_now in enumerate(trace["t_s"]):
        co = None
        if stage_pi is not None and tau is not None and tracker.fold_id < N_FOLDS - 1:
            co = bool(stage_pi[i, tracker.fold_id + 1] >= tau)
        fold_before = tracker.fold_id
        reason = tracker.update(float(trace["p_complete"][i]), float(trace["p_prog"][i]),
                                float(t_now), co_signal=co)
        if reason is not None:
            advances.append((float(t_now), reason, fold_before))
    return advances


def _advance_metrics(advances, bnds):
    """Per-episode metrics for a simulated advance list vs GT boundaries."""
    pred_ts = [a[0] for a in advances]
    n_nn, nn_offs = nn_match(pred_ts, list(bnds))
    lateness = 0.0
    n_early = n_deep_early = 0
    per_fold_off = []
    for t_adv, reason, k in advances:
        if k < len(bnds):
            off = t_adv - bnds[k]
            per_fold_off.append(off)
            lateness += max(0.0, off)
            if off < -3.0:
                n_early += 1
            if off < -8.0:                 # DEEP early: beyond ~2x the label-noise band --
                n_deep_early += 1          # genuinely wrong conditioning, not ramp ambiguity
    return {"n_advances": len(advances),
            "n_budget": sum(1 for a in advances if a[1] == "budget"),
            "nn_matched_3s": n_nn, "nn_offsets": nn_offs,
            "lateness_s": lateness, "n_early_wrong": n_early,
            "n_deep_early": n_deep_early,
            "per_fold_offsets": per_fold_off, **cascade_stats(pred_ts)}


def pass1v8(args):
    """v8 tau calibration (plan Phase E): oracle-fed traces for the POLICY ckpt (heads) +
    StageNet posterior traces on the SAME 10 Hz timeline, calib split only; then a
    zero-GPU tau grid over the FULL hardened tracker, baseline (no co-signal) included.
    tau* = max lateness reduction s.t. early-wrong advances <= baseline and 0 cascades."""
    dev = "cuda:0"
    assert args.stage_net, "pass1v8 needs --stage-net"
    model, meta = load_model(dev, ckpt_path=args.ckpt)
    ds = make_ds(meta)
    calib_ids, test_ids = calib_test_split(ds)
    print(f"calib={len(calib_ids)} test={len(test_ids)} (seed={SPLIT_SEED}); "
          f"tau grid on CALIB only -- TEST stays honest for pass2")

    cache = OUT_DIR / f"traces_cache_{meta['tag']}.npz"
    traces = load_or_compute_traces(model, ds, calib_ids, dev, cache=cache)

    # StageNet posterior traces on the identical timeline (features fast path)
    from stage_net import StageNetRunner
    runner = StageNetRunner(args.stage_net, device=dev)
    stage_tag = Path(args.stage_net).stem
    scache = OUT_DIR / f"stage_traces_{stage_tag}_{meta['tag']}.npz"
    if scache.exists():
        d = np.load(scache)
        if set(d["episode_ids"].tolist()) == set(calib_ids):
            stage_tr = {ei: d[f"{ei}_pi"] for ei in calib_ids}
            print(f"loaded cached stage traces from {scache}")
        else:
            stage_tr = None
    else:
        stage_tr = None
    if stage_tr is None:
        feat_dir = ROOT / "outputs" / "stage_features" / runner.encoder_tag
        stage_tr = {}
        for ei in calib_ids:
            season, ep = ds._ep_keys[ei]
            feats = np.load(feat_dir / season / f"ep{ep}.npy", mmap_mode="r")
            st = ds.low[ei]["observation.state"]
            tc = ds.low[ei]["observation.tactile"]
            runner.reset()
            pis = []
            for t_s in traces[ei]["t_s"]:
                t = int(round(t_s * ds.fps))
                pis.append(runner.update(float(t_s), None, st[t], tc[t],
                                         vis_feat=np.asarray(feats[t], np.float32)))
            stage_tr[ei] = np.stack(pis)
            print(f"  stage trace {season}/ep{ep}: {len(pis)} steps", flush=True)
        np.savez(scache, episode_ids=np.array(calib_ids),
                 **{f"{ei}_pi": v for ei, v in stage_tr.items()})
        print(f"cached stage traces -> {scache}")

    # grid: baseline + taus, full tracker simulation (pure numpy/python)
    taus = [round(0.5 + 0.05 * i, 2) for i in range(9)]
    rows = []
    for name, tau in [("baseline", None)] + [(f"tau={t}", t) for t in taus]:
        agg = {"n_advances": 0, "n_budget": 0, "nn_matched_3s": 0, "lateness_s": 0.0,
               "n_early_wrong": 0, "n_deep_early": 0, "n_subsecond_pairs": 0,
               "n_casc_eps": 0, "offs": []}
        n_gt = 0
        for ei in calib_ids:
            bnds = boundaries_of(ds, ei)
            adv = simulate_tracker(traces[ei], bnds,
                                   stage_pi=None if tau is None else stage_tr[ei], tau=tau)
            m = _advance_metrics(adv, bnds)
            n_gt += len(bnds)
            for k in ("n_advances", "n_budget", "nn_matched_3s", "lateness_s", "n_early_wrong",
                      "n_deep_early", "n_subsecond_pairs"):
                agg[k] += m[k]
            agg["n_casc_eps"] += int(m["legacy_cascade"])
            agg["offs"] += m["nn_offsets"]
        row = {"config": name, "tau": tau,
               "nn_match_rate": round(agg["nn_matched_3s"] / max(n_gt, 1), 4),
               "budget_advances": agg["n_budget"], "lateness_s": round(agg["lateness_s"], 1),
               "early_wrong": agg["n_early_wrong"], "deep_early": agg["n_deep_early"],
               "subsec_pairs": agg["n_subsecond_pairs"],
               "cascade_eps": agg["n_casc_eps"],
               "mean_offset_s": round(float(np.mean(agg["offs"])), 2) if agg["offs"] else None}
        rows.append(row)
        print(f"  {name:12s} match3s={row['nn_match_rate']:.3f}  budget={row['budget_advances']}"
              f"  lateness={row['lateness_s']}s  early-wrong={row['early_wrong']}"
              f"  deep-early={row['deep_early']}"
              f"  subsec={row['subsec_pairs']}  casc_eps={row['cascade_eps']}"
              f"  mean_off={row['mean_offset_s']}", flush=True)

    base = rows[0]
    # Early-advance constraint is on the co-signal's MARGINAL contribution, not the
    # absolute count: the baseline itself fires early ~100/175 times (shared heads-OR
    # + the budget backstop cutting slower-than-p90 folds early in BOTH arms), so an
    # absolute <= comparison rejects trades like -80% lateness for +3% early. Allow
    # up to max(3, 5% of GT segments) marginal early advances; cascades stay at zero.
    n_seg = sum(len(boundaries_of(ds, ei)) for ei in calib_ids)
    early_tol = max(3, round(0.05 * n_seg))
    # DEEP-early (< -8 s, beyond ~2x label noise) is the true risk dimension -- the first
    # TEST run at tau=0.65 surfaced a doubled deep-early tail that the original selection
    # never measured (see pass2_results_v8_stagenet.json). Constraint added
    # 2026-08-05 and tau re-selected on CALIB only; the TEST rerun at the final tau is the
    # documented second (and last) test-arm evaluation.
    ok = [r for r in rows[1:]
          if r["early_wrong"] <= base["early_wrong"] + early_tol
          and r["deep_early"] <= base["deep_early"] + 2
          and r["subsec_pairs"] == 0 and r["cascade_eps"] == 0]
    # among admissible taus, primary = max match rate; tie-break = min lateness
    star = max(ok, key=lambda r: (r["nn_match_rate"], -r["lateness_s"])) if ok else None
    if ok:
        print(f"(tolerances: early {base['early_wrong']}+{early_tol}, "
              f"deep-early {base['deep_early']}+2)")
    print(f"\nbaseline: match {base['nn_match_rate']:.3f}, lateness {base['lateness_s']}s, "
          f"budget {base['budget_advances']}")
    if star:
        print(f"tau* = {star['tau']}  (lateness {base['lateness_s']} -> {star['lateness_s']}s, "
              f"match {star['nn_match_rate']:.3f}, budget {star['budget_advances']})")
        print(f"write it into the stage ckpt:  python policy/train_stage_net.py "
              f"--set-tau {star['tau']} --ckpt {args.stage_net}")
    else:
        print("NO tau satisfies the constraints -- co-signal not viable at this quality")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"pass1v8_results_{stage_tag}_{meta['tag']}.json"
    out.write_text(json.dumps({"calib_ids": calib_ids, "test_ids": test_ids,
                               "policy_ckpt": str(args.ckpt or CKPT),
                               "stage_net": args.stage_net, "rows": rows,
                               "baseline": base, "tau_star": star}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pass1", "pass2", "pass1v8"])
    ap.add_argument("--tag", default="", help="pass2 only: suffix for the output json "
                    "(pass2_results_<tag>.json) so baseline results are never overwritten")
    ap.add_argument("--ckpt", default=None,
                    help="policy checkpoint (default: the module-level v3 path); v8 runs "
                         "pass this as outputs/ckpt/best_v7_fullobs.pt")
    ap.add_argument("--stage-net", default=None,
                    help="StageNet ckpt: required by pass1v8; optional in pass2 (enables "
                         "the co-signal in the closed-loop replay)")
    args = ap.parse_args()
    {"pass1": pass1, "pass2": pass2, "pass1v8": pass1v8}[args.mode](args)
