#!/usr/bin/env python3
"""Minimal closed-loop rollout harness (design_v3_stage_memory_tactile.md §3.2).

STATUS (2026-07-25): thresholds below are CALIBRATED against the real trained
`outputs/ckpt/best_v3_tactile.pt` on held-out (val-split) episodes -- offline, no Isaac Sim
required (`policy/calibrate_rollout.py`, methodology + numbers in
`docs/experiments/rollout_calibration_v3.md`). Read that report before trusting these values
on hardware/sim: the short version is the completion/progress heads are WEAK boundary
detectors (best calibrated operating point matches only ~7-8% of the 5 true transitions per
episode within a strict +/-1s window, ~29-30% within +/-3s, against 35 held-out calib
episodes) -- calibration found the best available operating point, it did not make the heads
good. `TAU_HI` in particular had to be lowered from the original placeholder 0.9 all the way
to 0.25 because `completion_logit`'s sigmoid essentially never exceeds ~0.35 even right next
to a true boundary (measured near-boundary mean 0.167 vs mid-fold mean 0.080 -- a real but
small lift, not a confident spike). The per-fold step `budget[k]` deterministic backstop
(p90 fold duration, by design a rare safety net) fired only 3/175 times on the calib set, so
it is NOT compensating for the heads' weakness at the tested tolerances. See the report's
"honest assessment" section for what this implies before autonomous deployment.

`replay_episode()` feeds a recorded held-out episode's REAL observations frame-by-frame
through the model (via FastFoldingDataset), overriding the dataset's oracle `fold_id`
with the rollout loop's own running `k` before every forward call -- the auto-labeled
`boundaries_s` are used ONLY afterward, to compare the predicted-advance sequence against
the true one. IMPORTANT CAVEAT: this is a REPLAY, not a true closed-loop test -- the fed
observations (images/tactile/state) at step t are whatever the teleoperator actually did at
time t in the recorded demo, regardless of what the model's own `fold_id`/actions were at
steps < t. The robot never actually re-observes the consequences of its own decisions here;
a real closed loop (Isaac Sim or hardware) is still required to see how the policy behaves
once its own actions start diverging from the recorded demo. Treat `replay_episode()`'s
"completed X/6" and timing numbers as an optimistic offline proxy, not a sim/hardware result.

KNOWN LIMITATION (flagged, not fixed here -- out of this task's scope): the CVAE
encoder branch inside `FoldingACT.forward()` unconditionally consumes `batch["action"]`
to produce the latent `z` (no `use_vae and self.training`-style eval-mode gate like
lerobot's ACT). `replay_episode()` therefore still feeds the RECORDED action chunk at
each replayed frame for that branch -- the same simplification `eval_ckpt.py` already
makes for its open-loop eval. True action-free inference (deployment on the real
rig / Isaac Sim, where no recorded future action chunk exists) needs a `z=0`-when-
no-action-given path added to `FoldingACT.forward()` first. `fold_id` (this doc's actual
subject) IS fully action-free / model-driven here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "data"))
sys.path.insert(0, str(HERE / "model"))
from fold_utils import load_labels                                    # noqa: E402
from autolabel_folds import ACTIVE_FINGERS, FORCE_AXES, TACT           # noqa: E402  (reuse, don't reinvent)
from folding_act import FoldingACT                                    # noqa: E402

BUDGET_CACHE = ROOT / "outputs" / "fold_step_budget.json"
N_FOLDS = 6
STAGES = ["first_corner_nose", "second_corner_nose", "body_fold", "wing1_fold", "wing2_fold", "body_refine"]

# CALIBRATED (2026-07-25) against best_v3_tactile.pt on 35 held-out calib episodes
# (policy/calibrate_rollout.py pass1, docs/experiments/rollout_calibration_v3.md). Chosen as
# the best-match-rate point on a grid search of the primary+co-signal OR-branches, simulated
# per true fold-segment against the auto-labeled boundary +/-1s. NOTE the absolute match rate
# at this best point is still low (~7-8% of transitions within +/-1s, ~29-30% within +/-3s) --
# these are the best AVAILABLE thresholds, not evidence the heads are reliable. Original
# placeholders (0.9/0.5/0.9) never fired at all: completion_logit's sigmoid essentially never
# exceeds ~0.35 even at a true boundary (see report).
TAU_HI = 0.25     # completion_logit sigmoid: primary advance signal
TAU_LO = 0.2      # completion co-signal threshold (paired with TAU_PROG)
TAU_PROG = 0.7    # progress_pred threshold for the smoother co-signal

# Phase-0 hardening (2026-07-29): minimum wall-clock dwell per fold before a SIGNAL-based
# advance is allowed (the budget backstop bypasses this -- it must always be able to fire).
# Derivation: 0.5 x p10 fold duration across the 582 train episodes' auto-labeled
# boundaries_s in outputs/fold_labels_auto.json (p10 durations, seconds:
# fold0 19.4, fold1 21.0, fold2 25.4, fold3 25.6, fold4 31.0, fold5 15.8 -- computed
# exactly as compute_fold_budgets() computes its p90; see docs/experiments/
# rollout_calibration_v3.md §4.1 for the sub-second cascade mode this floors out).
FOLD_MIN_DWELL_S = {0: 9.7, 1: 10.5, 2: 12.7, 3: 12.8, 4: 15.5, 5: 7.9}
# A signal-based advance additionally requires the signal to hold on EVERY call across a
# >=0.3s wall-clock span (~10 consecutive calls at 30Hz, ~3-4 at 10Hz) -- one marginal
# single-call blip can no longer advance a fold.
SUSTAIN_S = 0.3


def compute_fold_budgets(fps=10, pctl=90, cache=BUDGET_CACHE, force=False):
    """Per-fold step budget = p90 fold DURATION (from fold_labels_auto.json boundaries_s
    deltas across the 582 train episodes) * fps -- a free offline safety ceiling (§3.2:
    "if the completion head is stuck ... the rollout cannot strand itself in fold k").
    Cached to `outputs/fold_step_budget.json`, recomputed only if fps/pctl change."""
    if cache.exists() and not force:
        d = json.loads(cache.read_text())
        if d.get("fps") == fps and d.get("pctl") == pctl:
            return {int(k): v for k, v in d["budget_steps"].items()}
    labels = load_labels()
    durs = [[] for _ in range(N_FOLDS)]
    for v in labels.values():
        edges = [0.0] + list(v["boundaries_s"]) + [v["n_frames"] / 30]
        for i in range(N_FOLDS):
            durs[i].append(edges[i + 1] - edges[i])
    budget_s = {i: float(np.percentile(durs[i], pctl)) for i in range(N_FOLDS)}
    budget_steps = {i: max(1, round(budget_s[i] * fps)) for i in range(N_FOLDS)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"fps": fps, "pctl": pctl, "budget_s": budget_s,
                                  "budget_steps": budget_steps}, indent=2))
    return budget_steps


def compute_fold_budgets_s(pctl=90, cache=BUDGET_CACHE):
    """Per-fold budget in WALL-CLOCK SECONDS (p90 fold duration) -- the call-rate-independent
    form the hardened FoldTracker uses (Phase 0, 2026-07-29: the organizers' runtime may call
    infer() at any rate, so step counts assuming a fixed fps are wrong by construction; the
    cached budget_s field never depended on fps in the first place). Reads the existing cache
    without rewriting it (unlike compute_fold_budgets, which clobbers the cache's fps field)."""
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("pctl") == pctl and "budget_s" in d:
            return {int(k): float(v) for k, v in d["budget_s"].items()}
    compute_fold_budgets(fps=30, pctl=pctl, cache=cache, force=True)   # regenerates budget_s (fps only affects budget_steps)
    return {int(k): float(v) for k, v in json.loads(cache.read_text())["budget_s"].items()}


class FoldTracker:
    """The ONE shared fold-phase advance implementation (policy_server.py imports this;
    replay_episode() below uses it with simulated time) -- Phase-0 hardened (2026-07-29):

      signal advance:  (p_complete > tau_hi) OR (p_prog > tau_prog AND p_complete > tau_lo),
                       held on every call over a >= sustain_s wall-clock span, and only after
                       min_dwell_s[fold] of wall time in the fold (kills the measured 78%
                       sub-second cascade mode, docs/experiments/rollout_calibration_v3.md §4.1);
      budget backstop: elapsed wall time in fold > budgets_s[fold] (p90 fold duration).
                       Bypasses dwell + sustain -- it must always be able to fire, so the
                       rollout can never strand itself in a fold past the p90 duration.

    All timing is wall-clock via the caller-supplied ``t_now`` (pass time.monotonic() when
    serving live; pass frame_index/fps when replaying recorded data), so behavior is
    independent of the infer() call rate. ``calls_in_fold`` is kept purely for telemetry.
    NOTE vs the legacy _advance(): the backstop no longer requires a tactile-release
    co-signal -- serving has no release detector wired in, and calibration must measure the
    exact logic that deploys (the release requirement rescued only 3/175 segments anyway)."""

    def __init__(self, budgets_s=None, min_dwell_s=None, sustain_s=SUSTAIN_S,
                 tau_hi=None, tau_lo=None, tau_prog=None, n_folds=N_FOLDS,
                 co_min_frac=0.75, budget_min_p=0.0, budget_hard_mult=2.0):
        self.budgets_s = budgets_s if budgets_s is not None else compute_fold_budgets_s()
        self.min_dwell_s = dict(FOLD_MIN_DWELL_S if min_dwell_s is None else min_dwell_s)
        self.sustain_s = sustain_s
        self.tau_hi = TAU_HI if tau_hi is None else tau_hi
        self.tau_lo = TAU_LO if tau_lo is None else tau_lo
        self.tau_prog = TAU_PROG if tau_prog is None else tau_prog
        self.n_folds = n_folds
        # v9.3 (2026-09-29). The budget backstop is a WALL-CLOCK timeout that bypasses dwell
        # and sustain. On hardware (2026-09-29) it fired in 4/5 episodes at ~35s with
        # p_complete = 0.03-0.06 -- i.e. the completion head was confident the fold was NOT
        # done and the phase token advanced anyway, so the policy ran fold 1's motion on a
        # still-flat sheet. In training the fold label is ground truth, so this can only bite
        # at deployment. budget_min_p > 0 requires that much completion confidence before the
        # timeout may fire; budget_hard_mult x budget always fires, so progress is still
        # guaranteed. budget_min_p = 0.0 (default) is bit-identical to the pre-fix tracker.
        self.budget_min_p = float(budget_min_p)
        self.budget_hard_mult = float(budget_hard_mult)
        # v8: the co-signal is only HONORED once the fold has consumed co_min_frac of its
        # own p90 budget (wall-clock). Rationale (2026-08-05, first TEST run at tau=0.65):
        # StageNet's rare misfires are HIGH-confidence (tau-independent) and land early in
        # a fold -> deep-early advances; the legitimate co-signal wins (recovering p90
        # budget lateness) fire late in the fold. Gating by elapsed/budget removes the
        # former surgically and also breaks early-advance chains. Head-based signals and
        # the budget backstop are NOT affected by this gate. Default 0.75 chosen on the
        # calib split via the conditioning-error-seconds grid (early+late wrongness vs
        # demo boundaries): the only region beating the no-co baseline on TOTAL wrongness
        # while keeping the match/lateness/budget gains -- see
        # outputs/experiments/rollout_calibration/co_min_frac_grid.txt.
        self.co_min_frac = co_min_frac
        self.reset(0.0)

    def reset(self, t_now=0.0):
        self.fold_id = 0
        self.fold_entry_t = t_now
        self.signal_since = None      # t_now when the signal last became continuously true
        self.calls_in_fold = 0        # telemetry only -- no advance logic reads this

    def elapsed(self, t_now):
        return t_now - self.fold_entry_t

    def update(self, p_complete, p_prog, t_now, co_signal=None):
        """One infer() call. Returns "signal" | "budget" if the fold advanced, else None.

        co_signal (v8, optional): an EXTERNAL boolean advance vote (StageNet's
        next-stage posterior crossing its calibrated threshold -- see
        policy/model/stage_net.py). It is OR-ed into the head-based signal and
        inherits ALL the hardening: min-dwell, the >=0.3 s sustain, and the budget
        backstop are unchanged. co_signal=None (default) is bit-identical to the
        pre-v8 behavior; all existing callers pass 3 positional args."""
        self.calls_in_fold += 1
        elapsed = t_now - self.fold_entry_t
        co_ok = (bool(co_signal)
                 and elapsed >= self.co_min_frac * self.budgets_s.get(self.fold_id, 0.0))
        signal = ((p_complete > self.tau_hi)
                  or (p_prog > self.tau_prog and p_complete > self.tau_lo)
                  or co_ok)
        if signal and self.signal_since is None:
            self.signal_since = t_now
        elif not signal:
            self.signal_since = None
        reason = None
        budget = self.budgets_s.get(self.fold_id, float("inf"))
        if elapsed > budget and (p_complete >= self.budget_min_p
                                 or elapsed > self.budget_hard_mult * budget):
            reason = "budget"
        elif (signal and elapsed >= self.min_dwell_s.get(self.fold_id, 0.0)
              and t_now - self.signal_since >= self.sustain_s - 1e-6):  # eps: fp-safe at exact 30Hz spans
            reason = "signal"
        if reason is not None and self.fold_id < self.n_folds - 1:
            self.fold_id += 1
            self.fold_entry_t = t_now
            self.signal_since = None
            self.calls_in_fold = 0
            return reason
        return None


class TactileReleaseDetector:
    """Online version of design_structure.md §1.2's force-derivative release heuristic,
    reusing autolabel_folds.py's ACTIVE_FINGERS/FORCE_AXES contact-energy signal (not
    reinventing it): a release fires when a sustained press (energy above `hi`) is
    followed by energy dropping below `lo` (hysteresis avoids chattering on noisy dips).
    CALIBRATED (2026-07-25) on the held-out calib episodes' raw abs-force-sum trace
    (policy/calibrate_rollout.py pass1 Step B): the RAW scale is much larger than the
    original hi=1.0/lo=0.3 placeholders assumed -- measured percentiles across 285k held-out
    frames are p10=1.1, p50=12.0, p90=53.7, p99=115.1, i.e. the old hi=1.0 was already below
    the p10 baseline and would have been permanently "armed". hi=15/lo=3 (~p50-55 / ~p20-25)
    was the best grid point for backstop-rescue match rate, but note it fires rarely in
    practice (3/175 true-transition segments in the calib grid search) since budget[k] is a
    p90-duration safety net by design -- this detector is NOT doing most of the work in
    `_advance()`; see docs/experiments/rollout_calibration_v3.md."""
    def __init__(self, hi=15.0, lo=3.0, win=5):
        self.hi, self.lo, self.win = hi, lo, win
        self.hist = []
        self.armed = False

    def update(self, tactile_raw):        # tactile_raw: (60,) raw observation.tactile frame
        e = float(np.abs(tactile_raw[TACT]).sum())
        self.hist.append(e)
        self.hist = self.hist[-self.win:]
        smoothed = float(np.mean(self.hist))
        released = False
        if smoothed > self.hi:
            self.armed = True
        elif self.armed and smoothed < self.lo:
            released = True
            self.armed = False
        return released


def replay_episode(model, ds, ei, dev="cuda:0", fps=10, max_minutes=10.0,
                   tracker=None, stage_runner=None):
    """Structural-only replay of episode `ei` of `ds` (a FastFoldingDataset). Returns a
    dict with the model-driven advance log + the GT boundaries for post-hoc comparison.
    Phase 0 (2026-07-29): now drives the shared hardened FoldTracker (wall-clock budgets +
    min-dwell + sustained signal) with SIMULATED time t/ds.fps, so this offline replay
    exercises the exact advance logic policy_server.py deploys. The TactileReleaseDetector
    is still run, but only logged per advance -- see FoldTracker's docstring NOTE.

    v8 (additive, defaults = pre-v8 behavior bit-for-bit):
      tracker: injectable FoldTracker (A/B testing); None -> a fresh default one.
      stage_runner: a stage_net.StageNetRunner; when given, it is fed each replayed
        frame (head_left = item["images"][0] in both 3- and 4-cam modes, raw state,
        raw tactile) and its co_signal(fold_id) is passed to tracker.update()."""
    season, ep = ds._ep_keys[ei]
    lab = ds.labels[f"{season}/ep{ep}"]
    T = ds.dur[ei]
    base = sum(ds.dur[:ei])                             # flat-index offset for this episode
    tactile_raw = ds.low[ei]["observation.tactile"]      # (T,60) raw, for the release detector
    state_raw = ds.low[ei]["observation.state"]          # (T,65) raw, for the stage runner
    stride = max(1, round(ds.fps / fps))
    max_steps = int(max_minutes * 60 * fps)

    completed = 0
    if tracker is None:
        tracker = FoldTracker()
    tracker.reset(0.0)
    if stage_runner is not None:
        stage_runner.reset()
    release_det = TactileReleaseDetector()
    advance_log = []
    model.eval()
    n_steps = 0
    with torch.no_grad():
        for step, t in enumerate(range(0, T, stride)):
            if completed >= N_FOLDS or step >= max_steps:
                break
            n_steps = step + 1
            t_now = t / ds.fps                           # simulated wall-clock (episode seconds)
            item = ds[base + t]
            item["fold_id"] = torch.tensor(tracker.fold_id, dtype=torch.long)  # override the oracle -- model-driven
            batch = {kk: v[None].to(dev) for kk, v in item.items()}
            out = model(batch)
            p_complete = torch.sigmoid(out.completion_logit)[0, 0].item() if out.completion_logit is not None else 0.0
            p_prog = out.progress_pred[0, 0].item() if out.progress_pred is not None else 0.0
            released = release_det.update(tactile_raw[t])

            co = None
            if stage_runner is not None:
                head_left = item["images"][0].numpy()          # (3,224,224) normalized; slot 0
                pi = stage_runner.update(t_now, head_left, state_raw[t], tactile_raw[t])
                co = stage_runner.co_signal(tracker.fold_id, pi)

            reason = tracker.update(p_complete, p_prog, t_now, co_signal=co)
            if reason is not None:
                rec = {"step": step, "t_s": round(t_now, 2), "from_fold": tracker.fold_id - 1,
                       "reason": reason, "released": bool(released),
                       "p_complete": round(p_complete, 3), "p_prog": round(p_prog, 3)}
                if co is not None:
                    rec["co_signal"] = bool(co)
                advance_log.append(rec)
                completed += 1

    k = tracker.fold_id

    return {"episode": f"{season}/ep{ep}", "n_steps": n_steps, "completed_count": completed,
            "final_fold_id": k, "advance_log": advance_log, "gt_boundaries_s": lab["boundaries_s"],
            "gt_duration_s": T / ds.fps}


def _advance(p_complete, p_prog, step_in_fold, budget_k, released):
    """LEGACY (pre-Phase-0) 3-way OR, kept only for its unit test + external callers --
    replay_episode() and policy_server.py now share FoldTracker above instead."""
    return (p_complete > TAU_HI) or (p_prog > TAU_PROG and p_complete > TAU_LO) or \
           (step_in_fold > budget_k and released)


def _unit_test_fold_tracker():
    """Direct checks of every FoldTracker rule with synthetic budgets/dwells (no model)."""
    mk = lambda: FoldTracker(budgets_s={i: 30.0 for i in range(6)},
                             min_dwell_s={i: 5.0 for i in range(6)}, sustain_s=0.3)
    # 1) sustained strong signal AFTER dwell advances with reason "signal" (at the call
    #    where the sustain span reaches 0.3s, i.e. the 10th call at 30Hz)
    tr = mk()
    rs = [tr.update(0.9, 0.0, 6.0 + i / 30) for i in range(12)]
    assert tr.fold_id == 1 and rs[9] == "signal", (tr.fold_id, rs)
    # 2) the same signal BEFORE min dwell must NOT advance
    tr = mk()
    for i in range(12):
        assert tr.update(0.9, 0.0, 1.0 + i / 30) is None
    assert tr.fold_id == 0
    # 3) a single-call blip (signal not sustained >=0.3s) must NOT advance
    tr = mk()
    assert tr.update(0.9, 0.0, 6.0) is None
    assert tr.update(0.0, 0.0, 6.033) is None              # signal drops -> sustain resets
    assert tr.update(0.9, 0.0, 6.066) is None              # fresh start, span < 0.3s
    assert tr.fold_id == 0
    # 4) budget backstop fires on wall time alone, bypassing dwell/sustain, at ANY call rate
    tr = mk()
    assert tr.update(0.0, 0.0, 10.0) is None               # 10s < 30s budget
    assert tr.update(0.0, 0.0, 31.0) == "budget" and tr.fold_id == 1
    # 5) sustain is wall-clock, not call-count: 3 calls spanning 0.4s at ~10Hz advance
    tr = mk()
    for i, t in enumerate([6.0, 6.2, 6.4]):
        r = tr.update(0.9, 0.0, t)
    assert r == "signal" and tr.fold_id == 1
    # 6) co-signal branch respects the same dwell+sustain
    tr = mk()
    rs = [tr.update(0.22, 0.9, 6.0 + i / 30) for i in range(12)]
    assert rs[9] == "signal" and tr.fold_id == 1
    # 7) never advances past the last fold
    tr = mk()
    tr.fold_id, tr.fold_entry_t = 5, 0.0
    assert tr.update(0.9, 0.9, 100.0) is None and tr.fold_id == 5
    # 8) v8 co_signal: advances via the co-signal alone (heads quiet), but ONLY after
    #    dwell + sustain + the co-budget gate (elapsed >= co_min_frac*budget = 22.5s here)
    tr = mk()
    rs = [tr.update(0.0, 0.0, 23.0 + i / 30, co_signal=True) for i in range(12)]
    assert tr.fold_id == 1 and rs[9] == "signal", (tr.fold_id, rs)
    tr = mk()
    for i in range(12):                                   # co before dwell: no advance
        assert tr.update(0.0, 0.0, 1.0 + i / 30, co_signal=True) is None
    assert tr.fold_id == 0
    tr = mk()
    for i in range(12):                                   # co after dwell but BEFORE the
        assert tr.update(0.0, 0.0, 6.0 + i / 30, co_signal=True) is None  # co-budget gate
    assert tr.fold_id == 0                                # (elapsed 6s < 0.75*30s): blocked
    tr = mk()
    assert tr.update(0.0, 0.0, 23.0, co_signal=True) is None
    assert tr.update(0.0, 0.0, 23.033, co_signal=False) is None  # sustain resets on drop
    assert tr.update(0.0, 0.0, 23.066, co_signal=True) is None
    assert tr.fold_id == 0
    tr = mk()
    tr.fold_id, tr.fold_entry_t = 5, 0.0                  # co_signal can't pass the last fold
    assert tr.update(0.0, 0.0, 100.0, co_signal=True) is None and tr.fold_id == 5
    tr_a, tr_b = mk(), mk()                                # None == 3-arg behavior
    for i in range(12):
        ra = tr_a.update(0.9, 0.0, 6.0 + i / 30)
        rb = tr_b.update(0.9, 0.0, 6.0 + i / 30, co_signal=None)
        assert ra == rb
    assert tr_a.fold_id == tr_b.fold_id == 1
    # 9) heads-signal is NOT affected by the co-budget gate (fires at 6s as always)
    tr = mk()
    rs = [tr.update(0.9, 0.0, 6.0 + i / 30, co_signal=False) for i in range(12)]
    assert tr.fold_id == 1 and rs[9] == "signal"
    print("FoldTracker unit test: dwell/sustain/budget/wall-clock/last-fold/co-signal rules verified")


def _unit_test_advance_logic():
    """The 60s replay window in _smoke() happened not to trigger any of the 3 advance
    branches with a random-init model + real (low, early-episode) tactile data -- expected,
    but it means that code path needs its own direct check. Exercise each branch here with
    synthetic inputs, plus the release detector's press->release hysteresis in isolation."""
    assert _advance(0.95, 0.0, 0, 999, False), "primary completion-signal branch failed"
    assert _advance(0.0, 0.95, 0, 999, False) is False, "co-signal needs BOTH prog AND p_complete>TAU_LO"
    assert _advance(0.6, 0.95, 0, 999, False), "co-signal branch failed"
    assert _advance(0.0, 0.0, 100, 50, True), "budget+release backstop branch failed"
    assert _advance(0.0, 0.0, 100, 50, False) is False, "backstop must require BOTH budget overrun AND release"
    print("advance-logic unit test: all 3 OR-branches independently verified")

    det = TactileReleaseDetector(hi=1.0, lo=0.3, win=1)
    events = []
    for e in [0.1, 0.1, 1.5, 1.5, 1.5, 0.1, 0.1, 0.1]:      # rest -> press -> release
        events.append(det.update(np.array([e / len(TACT)] * 60)))
    assert not any(events[:3]), "should not fire before a press"
    assert any(events[3:]), "should fire once energy drops after a press"
    print(f"release-detector unit test: fired at index {events.index(True)} of {events} (rest->press->release)")
    _unit_test_fold_tracker()


def _smoke():
    """Plumbing check ONLY (see module docstring) -- random-init model, small held-out
    episode. Confirms: loop terminates, fold_id is model-driven (0..6 count bounded),
    shapes/dtypes correct, no crash. Does NOT validate advance-timing accuracy."""
    _unit_test_advance_logic()
    sys.path.insert(0, str(HERE / "data"))
    from fast_folding_dataset import FastFoldingDataset

    dev = "cuda:0"
    seasons = ["season_POC22061_2026_07_02_14_23_26_train"]
    ds = FastFoldingDataset(split="train", seasons=seasons, tactile=True)
    print(f"replay dataset: {len(ds.dur)} episode(s), {len(ds)} frames total")

    budgets_s = compute_fold_budgets_s()
    print(f"fold wall-clock budgets_s (p90, cached -> {BUDGET_CACHE}): {budgets_s}")
    print(f"min dwell_s (0.5 x p10): {FOLD_MIN_DWELL_S}")

    torch.manual_seed(0)
    # random-init only (no state_dict loaded) -- fine for a plumbing smoke test. NOTE for
    # anyone copying this line to load a REAL checkpoint (as policy/calibrate_rollout.py
    # does): best_v3_tactile.pt also needs encoder="dinov3" (this constructor's default is
    # "resnet18") -- a strict state_dict load fails loudly if it's omitted, but don't rely on
    # guessing; the checkpoint dict itself stores its exact construction kwargs (ck["encoder"],
    # ck["tactile"], etc.) -- load from those instead of hardcoding flags here.
    model = FoldingACT(tactile=True, n_decoder_layers=3, aux_heads=True,
                       decoder_tactile_attn=True, pre_encoder_tactile_attn=True,
                       contact_gate=True).to(dev)

    for ei in range(min(2, len(ds.dur))):
        # smoke test only replays 2 episodes x ~1 sim-minute (~600 unbatched fwd passes
        # @10fps each) -- plenty to exercise every code path (loop, budgets, release
        # detector, advance logic, termination) without a multi-minute smoke test.
        res = replay_episode(model, ds, ei, dev=dev, fps=10, max_minutes=1.0)
        print(f"\n[{res['episode']}] n_steps={res['n_steps']}  completed_count={res['completed_count']}  "
              f"final_fold_id={res['final_fold_id']}")
        print(f"  GT boundaries_s: {res['gt_boundaries_s']}  (duration {res['gt_duration_s']:.1f}s)")
        print(f"  predicted advances ({len(res['advance_log'])}):")
        for e in res["advance_log"][:8]:
            print(f"    step {e['step']:5d}  t={e['t_s']:6.1f}s  fold {e['from_fold']}->{e['from_fold']+1}  "
                  f"p_complete={e['p_complete']}  p_prog={e['p_prog']}")
        assert 0 <= res["completed_count"] <= N_FOLDS
        assert 0 <= res["final_fold_id"] <= N_FOLDS - 1
        assert res["n_steps"] > 0

    print("\nROLLOUT SMOKE OK (plumbing verified; advance-timing is meaningless with an "
          "untrained/random-init model -- see docstring)")


if __name__ == "__main__":
    _smoke()
