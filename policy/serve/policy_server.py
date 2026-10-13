#!/usr/bin/env python3
"""OpenPI-style websocket policy server that serves our trained FoldingACT checkpoint
(v3 or v4) against the inf_code_v1/origami-inference-kit-main robot I/O contract
(inf_code_v1/origami-inference-kit-main/docs/robot_io_spec.md).

This wires our own model into the vendored ``openpi.serving.WebsocketPolicyServer`` /
``openpi_client`` framework WITHOUT installing anything into the vendored
``inf_code_v1/origami-inference-kit-main/`` tree (no pip install -e, no edits there) -- it adds the
two relevant ``src`` dirs to ``sys.path`` at import time and otherwise only touches
``policy/``.

CONTRACT GAP -- what this server approximates and why (see docs/experiments/
isaac_sim_deployment_setup.md for the full writeup):

1. No ``head_right`` stream. The contract only provides head_left/wrist_left/wrist_right
   (docs/robot_io_spec.md), but FoldingACT was trained on 4 camera views in the fixed
   order [head_left, head_right, wrist_left, wrist_right] (policy/data/fast_folding_dataset.py
   CAMS). We duplicate head_left into the head_right slot. This is a documented
   approximation, not a fix: the model loses the real stereo head signal it trained on.
   v6 UPDATE: 3-cam-native checkpoints (trained with ``train_full.py --three-cam``, which
   stores ``n_cams: 3`` in the checkpoint dict) have NO approximation here at all -- the
   model consumes exactly the contract's three streams and this server feeds them
   directly, no duplication. 4-cam checkpoints keep the duplication behavior above.

2. TACTILE (updated 2026-07-27 for the v1 contract, which DOES document tactile --
   robot_io_spec.md section 5). The server now consumes tactile with graceful degradation,
   auto-detected per call and logged on first inference / mode change:
     mode "full"      : ``observation/tactile`` (60,) + ``observation/image/tactile_raw``
                        (480x1600 2x5 grid) present -> real 60-D force/torque into
                        obs_lowdim AND real (10,6,112,112) Sparsh tactile_img rebuilt with
                        the exact training geometry (predecode_tactile.py: uniform resize
                        1600x480 -> 560x224, split into 10 tiles of 112x112, row-major =
                        [left thumb..little ; right thumb..little] -- VERIFIED to match the
                        contract section-5 finger order by correlating per-tile image
                        activity against the 60-D force blocks on a real episode, corr
                        0.82-0.99 on every finger with usable force signal). Temporal pair
                        [I_t ; I_{t-167ms}] via a TIMESTAMPED rolling mosaic buffer
                        (training spacing 5 frames @30fps = 167ms, selected by wall-clock
                        age so it is correct at any call rate -- PHASE-0 EXTRAS below);
                        at episode start the buffer only holds the current mosaic, so
                        prev = cur, exactly like FastFoldingDataset (prev = max(0, t-5)).
     mode "vec_only"  : ``observation/tactile`` present, no usable tactile image -> real
                        60-D into obs_lowdim (FoAR contact gate + tactile slice real),
                        tactile_img zeroed.
     mode "none"      : no tactile fields -> previous behavior, exactly eval_ckpt.py's
                        ``--zero-tactile`` ablation: zero raw tactile before normalizing,
                        THEN re-zero the normalized last-30 slice (a raw zero does not
                        normalize to 0, and the FoAR gate reads the normalized slice) and
                        zero tactile_img. The re-zeroing is ONLY applied in this mode --
                        with real tactile the normalized slice must pass through untouched.
   ``observation/image/tactile_deform`` (480x1200, 240x240 cells) is deliberately NOT
   used as a tactile_img substitute when tactile_raw is absent: training's predecode
   (scripts/predecode_tactile.py) used tactile_raw ONLY (320x240 cells, different
   appearance domain than what the fine-tuned Sparsh encoder saw), so silently feeding
   deform tiles would be out-of-distribution input dressed up as real signal. vec_only
   is the honest fallback ("tactile_raw is provided on request" per the spec -- request it).
   Joint torque remains absent from the contract: raw torque is fed as zeros(65) and
   normalized as-is (unchanged v0 behavior; training always had real torque, so there is
   no in-distribution value to substitute).

3. No pre-computed future action chunk to feed the CVAE encoder branch. Real deployment
   (this server) uses the new ``FoldingACT.forward(batch, use_vae=False)`` path (added
   alongside this server) which sets the CVAE latent z to the prior mean (zeros) instead of
   encoding a (nonexistent) future action chunk -- standard ACT eval-time behaviour.

4. fold_id (the CVAE-conditioning "which of the 6 folds are we in" token) is not part of the
   contract's observation either. This server tracks it itself across calls via the SHARED
   ``rollout.FoldTracker`` (Phase-0 hardening, 2026-07-29 -- one implementation for offline
   calibration and serving): calibrated completion/progress thresholds
   (docs/experiments/rollout_calibration_v3.md) + per-fold minimum WALL-CLOCK dwell +
   sustained-signal requirement (>=0.3s of consecutive true calls), with a wall-clock p90
   fold-duration budget backstop that bypasses both (fires on time.monotonic() elapsed, so
   it is correct at ANY infer() call rate -- the previous call-count budget assumed 30Hz
   calls and would have expired 3x late at the organizers' 10Hz async default). The heads
   remain weak boundary detectors (see rollout.py); the hardening bounds the damage
   (no sub-second cascades, worst-case schedule ~205s to reach the last fold).

PHASE-0 SERVING EXTRAS (2026-07-29):
   - Tactile temporal pair is TIME-BASED: every mosaic is buffered with time.monotonic();
     the pair partner is the buffered mosaic whose age is closest to 167ms (training
     spacing: 5 frames @30Hz), independent of call rate. ~1.2s of buffer depth; at episode
     start (or pathologically slow call rates) selection degrades to prev=cur, which is
     exactly the trained episode-start input mode.
   - --telemetry-file: per-call jsonl (timestamps, measured dt, fold trace, head sigmoids,
     tactile mode + pair age, chunk norm stats, advance events with reason). Off by
     default; a concise stdout summary line is always logged every ~2s.
   - Optional serving stabilizers, BOTH DEFAULT OFF (they change behavior; gate separately):
     --blend-chunks (exponentially blend the overlap with the previously returned chunk,
     favoring the new chunk) and --continuity-ramp (if |actions[0] - state| exceeds the
     per-joint train-set p99 step size, outputs/action_step_p99.json, ramp the excess out
     over the first 10 steps).

JOINT CLAMP (2026-08-14, --clamp-limits, DEFAULT OFF):
   The 2026-08-14 Shadow run came back "Compatible: NO" with 308 violations. Diagnosis
   (docs/experiments/shadow_compatibility_v8.md): 281 were finger joints commanded a few
   degrees below the URDF floor -- the training targets are hard-clipped at exactly 0.000,
   so an unclamped regressor straddles the floor -- and the other 27 were all at chunk
   boundaries (steps 0/45/90), which --continuity-ramp already exists to absorb.
   --clamp-limits closes the first class by clamping every returned action into
   outputs/action_clamp_bounds.json (scripts/build_clamp_bounds.py: the URDF <limit>
   reconciled with the 4.88M-frame demo envelope, because this URDF's hand ceiling is
   demonstrably too low -- the organizers' own demos exceed it in ~80% of frames).
   Applied AFTER the stabilizers, since they can themselves push a joint past a bound.
   Offline replay through the kit's own validator: 308 -> 0 violations, verdict YES.

v3 (outputs/ckpt/best_v3_tactile.pt) vs v4 (outputs/ckpt/best_v4_tactile.pt): both load with
identical constructor kwargs (this server reads them from the checkpoint dict, see
FoldingACTPolicy.__init__) so either serves via ``--ckpt``. Default is v3 (the v1
contract DOES deliver real tactile, which removed the original v0-era rationale for v4),
but the measured replay comparison under the real contract
(docs/experiments/replay_test_contract.md, 2026-07-27) found v4 equal-or-better in ALL
four ckpt x tactile-mode conditions (v4 0.0254 vs v3 0.0260 rad L1 with full tactile;
0.0252 vs 0.0302 without) and recommends submitting v4 -- pass ``--ckpt`` accordingly.

Run:
    conda run -n origami_infer python policy/serve/policy_server.py \\
        --ckpt outputs/ckpt/best_v3_tactile.pt --port 8000

Protocol check (separate shell, same or different env -- only needs numpy/PIL/msgpack/websockets):
    cd inf_code_v1/origami-inference-kit-main/sharpa_north_ces_lite_sdk-main
    conda run -n origami_infer python examples/check_policy_server.py --policy-port 8000

PATH NOTE (2026-07-27): the user's inference-kit folders now live under
``inf_code_v0/`` (the original snapshot this server was built against) and
``inf_code_v1/`` (a repo update -- same file layout, but the *docs* changed
meaningfully: v1's contract now documents tactile fields that v0 did not).
This server targets ``inf_code_v1`` and, as of 2026-07-27, actually consumes
the v1 tactile fields (CONTRACT GAP point 2 above).

TESTING EXTRA (non-contract): an observation containing a truthy ``"reset"`` key
triggers ``policy.reset()`` (fold tracker + tactile buffer) before that inference.
The organizers' runtime never sends it; policy/serve/replay_test.py uses it to
replay multiple episodes over one connection.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# This server always strict-loads a full checkpoint over the freshly built model, so the
# pretrained trunk downloads (timm DINOv3, torchvision ResNet, facebook/sparsh) are pure
# startup waste -- and a hard failure on an offline eval machine. Skip them.
os.environ.setdefault("FOLDING_ACT_SKIP_PRETRAINED", "1")

ROOT = Path(__file__).resolve().parents[2]
# inf_code_v1 = the current inference-kit snapshot (repo was updated after inf_code_v0;
# v1's contract now documents tactile fields -- see docs/robot_io_spec.md).
KIT = ROOT / "inf_code_v1" / "origami-inference-kit-main"
sys.path.insert(0, str(KIT / "openpi-base-main" / "src"))
sys.path.insert(0, str(KIT / "openpi-base-main" / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(ROOT / "policy" / "model"))
sys.path.insert(0, str(ROOT / "policy" / "data"))
sys.path.insert(0, str(ROOT / "policy"))

from openpi_client import base_policy as _base_policy   # noqa: E402
from openpi.serving import websocket_policy_server        # noqa: E402

from folding_act import FoldingACT                         # noqa: E402
from fold_utils import Norm                                 # noqa: E402
import rollout as _rollout                                  # noqa: E402  (calibrated budgets/thresholds, no tactile use)

logger = logging.getLogger("folding_act_policy_server")

ACTION_DIM = 65
N_TORQUE_RAW = 65             # raw joint_torque dim; Norm.obs_lowdim applies keep_idx (58 of 65) itself
N_TACTILE_RAW = 60
N_FOLDS = 6
IMN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMS = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
# tactile_raw mosaic geometry -- must match scripts/predecode_tactile.py + fast_folding_dataset.py
TAC_GRID_HW = (480, 1600)     # contract section-5 tactile_raw grid (2x5 cells of 240x320)
TAC_RESIZE = (560, 224)       # PIL (W,H): uniform scale -> each cell becomes exactly 112x112
TAC_ROWS, TAC_COLS, TAC_SZ = 2, 5, 112
FPS = 30                      # the chunk's step cadence (training data rate; contract section 2)
TAC_PAIR_S = 5 / FPS          # temporal-pair target age in SECONDS (training: 5 frames @30fps = 167ms)
TAC_BUF_S = 3.5               # mosaic buffer depth. Was 1.2, which SILENTLY BROKE the pair in sync
                              # mode (2026-09-29): the Gateway calls infer every ~1.6s, so the only
                              # other entry was evicted before the next call and the buffer held just
                              # the current mosaic -> prev = cur -> ZERO temporal delta on 518/518
                              # calls. 3.5s keeps a partner at every observed call rate.
TAC_BUF_N = 64                # hard cap on buffered mosaics (~1.5MB each) so a fast caller cannot
                              # grow the buffer without bound.
STDOUT_EVERY_S = 2.0          # cadence of the always-on concise telemetry line
BLEND_W0, BLEND_TAU = 0.5, 8.0   # --blend-chunks: old-chunk weight w0*exp(-i/tau), always favors new
RAMP_STEPS = 10               # --continuity-ramp: decay the first-step excess over this many steps
RAMP_FLOOR = 0.01             # rad; min per-joint threshold (~measured executed state noise) so
                              # near-frozen joints (train p99 step ~0) don't trigger the ramp on noise


def _prep_image(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB (contract delivers 480x480) -> (3,224,224) float32, ImageNet-normalized.
    Matches training preprocessing exactly (scripts/predecode_frames.py `scale=224:224` +
    fast_folding_dataset.py's IMN/IMS normalize). Contract images are square, so a plain
    resize introduces no aspect distortion beyond what training already had."""
    im = Image.fromarray(np.asarray(img)).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
    return (arr - IMN) / IMS


def _prep_tactile_mosaic(grid: np.ndarray) -> np.ndarray:
    """480x1600x3 uint8 tactile_raw grid -> (10,3,112,112) float32, ImageNet-normalized.
    Replicates training geometry (scripts/predecode_tactile.py ffmpeg ``scale=560:224`` then
    fast_folding_dataset.py's row-major 112px tile split + IMN/IMS normalize). BICUBIC chosen
    to match ffmpeg's default scale filter: verified against the training LMDB tiles on a
    real episode (mean |diff| 2.7/255 bicubic vs 5.5 bilinear; residual is the LMDB's JPEG
    q3 compression, which we intentionally do NOT reproduce -- it is an artifact, not signal)."""
    grid = np.asarray(grid)
    if grid.shape[:2] != TAC_GRID_HW:
        raise ValueError(f"tactile_raw grid must be {TAC_GRID_HW}+(3,), got {grid.shape}")
    m = np.asarray(Image.fromarray(grid).convert("RGB").resize(TAC_RESIZE, Image.BICUBIC),
                   np.float32) / 255.0                                    # (224,560,3)
    tiles = [m[r * TAC_SZ:(r + 1) * TAC_SZ, c * TAC_SZ:(c + 1) * TAC_SZ].transpose(2, 0, 1)
             for r in range(TAC_ROWS) for c in range(TAC_COLS)]
    return ((np.stack(tiles) - IMN) / IMS).astype(np.float32)             # (10,3,112,112)


class FoldingACTPolicy(_base_policy.BasePolicy):
    """Wraps a FoldingACT checkpoint behind the OpenPI ``infer(obs) -> {"actions": ...}`` API."""

    def __init__(self, ckpt_path: str, device: str = "cuda:0", telemetry_file: str | None = None,
                 blend_chunks: bool = False, continuity_ramp: bool = False, serve_tempo: float = 1.0,
                 tactile_pair_interp: bool = False, fold_budget_min_p: float = 0.0,
                 stage_net: str | None = None, clamp_limits: str | None = None,
                 clock=time.monotonic, exec_horizon: int = 0):
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            logger.warning("CUDA not available; falling back to CPU (will be slow)")
        # injectable clock: everything time-based (fold budgets/dwell/sustain, tactile pairing)
        # reads THIS, so an offline closed-loop rollout can advance simulated time instead of
        # wall-clock. Default time.monotonic == the deployed behavior, unchanged.
        self._clock = clock
        # v9.3: serve only the first exec_horizon steps of the predicted chunk (0 = full chunk).
        # The organizer Gateway executes an ENTIRE chunk open-loop before re-querying, so a
        # shorter chunk == a proportionally faster feedback loop. See
        # docs/experiments/live_test_v9_2_analysis.md.
        self._exec_horizon = int(exec_horizon)
        ck = torch.load(ckpt_path, map_location=self.device)
        # v9: use the stats the ckpt was TRAINED with (embedded under ck["norm_stats"] since v9;
        # v1-v8 ckpts have none -> outputs/norm_stats.json as before). A finetuned model decoded
        # with the wrong stats is off by up to 0.25 rad on some joints.
        self.norm = Norm.from_ckpt(ck)
        logger.info("norm stats: %s", "embedded in ckpt (v9)" if ck.get("norm_stats") else "outputs/norm_stats.json (default)")
        self.kwargs = dict(
            encoder=ck["encoder"], tactile=ck["tactile"], n_decoder_layers=ck["n_decoder_layers"],
            aux_heads=ck["aux_heads"], decoder_tactile_attn=ck["decoder_tactile_attn"],
            pre_encoder_tactile_attn=ck["pre_encoder_tactile_attn"], contact_gate=ck["contact_gate"],
            n_tempos=ck.get("n_tempos", 0),   # v5 tempo token; absent in v1-v4 ckpts -> 0
            n_cams=ck.get("n_cams", 4),       # v6 3-cam-native ckpts; absent in v1-v5 -> 4
        )
        # v7 (2026-08-03): which deployment contract the ckpt was serve-match-trained for.
        # "v1" (absent in v1-v6 ckpts) = the old websocket kit: no torque, no head_right ->
        # this server zeroes torque and (4-cam) duplicates head_left. "v2" = the Zenoh
        # participant_v2 kit: the observation MAY carry observation/state/joint_torque and
        # observation/image/head_right -> feed them to the model for real (with graceful
        # zero/duplicate fallback if a caller omits them, matching the 10% torque-dropout
        # the v2 recipe trains). Behavior for v1-contract ckpts is UNCHANGED.
        self.serve_contract = ck.get("serve_contract", "v1")
        # v5: which tempo to SERVE at (Phase-3 speed knob). Default = the 1.0x safety tempo;
        # `serve_tempo` selects the closest trained tempo from the checkpoint's tempo list.
        self.tempos = list(ck.get("tempos") or [])
        self.tempo_id = 0
        if self.kwargs["n_tempos"]:
            serve_tempo = float(serve_tempo)
            self.tempo_id = int(np.argmin([abs(t - serve_tempo) for t in self.tempos]))
            logger.info("tempo conditioning: trained tempos=%s -> serving tempo %s (id %d)",
                        self.tempos, self.tempos[self.tempo_id], self.tempo_id)
        self.model = FoldingACT(**self.kwargs).to(self.device)
        self.model.load_state_dict(ck["model"], strict=True)
        self.model.eval()
        self.chunk = self.model.chunk
        if not 0 <= self._exec_horizon <= self.chunk:
            raise ValueError(f"exec_horizon must be in [0, {self.chunk}], got {self._exec_horizon}")
        # what actually crosses the wire (and therefore what metadata must advertise)
        self.served_horizon = self._exec_horizon or self.chunk
        if self._exec_horizon:
            logger.info("exec horizon: serving the first %d of %d predicted steps "
                        "(%.0f ms of motion per query)", self._exec_horizon, self.chunk,
                        self._exec_horizon / 30.0 * 1000.0)
        self.ckpt_path = str(ckpt_path)
        self.step = ck.get("step")
        logger.info("loaded %s (step=%s val_l1=%s) kwargs=%s device=%s",
                    ckpt_path, self.step, ck.get("val_l1"), self.kwargs, self.device)

        # Shared hardened fold-phase tracker (rollout.FoldTracker, module docstring §4):
        # wall-clock p90 budget_s backstop + min-dwell + sustained signal, correct at any
        # infer() call rate (time.monotonic() driven).
        self.tracker = _rollout.FoldTracker(budget_min_p=fold_budget_min_p)
        logger.info("fold tracker: budgets_s=%s min_dwell_s=%s sustain_s=%s",
                    self.tracker.budgets_s, self.tracker.min_dwell_s, self.tracker.sustain_s)
        if self.tracker.budget_min_p > 0:
            logger.info("fold budget gate ON: the wall-clock backstop needs p_complete >= %.2f "
                        "(hard backstop at %.1fx budget)",
                        self.tracker.budget_min_p, self.tracker.budget_hard_mult)

        # v8 (default OFF -- flag unset = pre-v8 behavior bit-for-bit): StageNet stage
        # recognizer as a CONSERVATIVE FoldTracker co-signal (policy/model/stage_net.py).
        # Local-testing feature; the shipped submission does not enable it.
        self.stage_runner = None
        if stage_net:
            from stage_net import StageNetRunner
            self.stage_runner = StageNetRunner(stage_net, device=self.device)
            assert self.stage_runner.tau is not None, \
                "stage ckpt has no tau -- calibrate first (calibrate_rollout.py pass1v8 " \
                "+ train_stage_net.py --set-tau)"
            logger.info("stage co-signal ON: %s (tau=%s, encoder=%s)",
                        stage_net, self.stage_runner.tau, self.stage_runner.encoder_tag)

        # telemetry (module docstring "PHASE-0 SERVING EXTRAS"): jsonl file off by default
        self._tele_f = open(telemetry_file, "a", buffering=1) if telemetry_file else None
        if telemetry_file:
            logger.info("telemetry jsonl -> %s", telemetry_file)

        # serving stabilizers -- BOTH DEFAULT OFF
        self._blend = blend_chunks
        # v9.3 (2026-09-29): reconstruct the 167ms tactile pair partner by interpolation
        # instead of snapping to the nearest buffered mosaic. Default OFF = pre-fix behaviour.
        self._tac_interp = bool(tactile_pair_interp)
        if self._tac_interp:
            logger.info("tactile pair interpolation ON: target age %.0f ms, buffer %.1fs/%d",
                        TAC_PAIR_S * 1000.0, TAC_BUF_S, TAC_BUF_N)
        self._ramp: np.ndarray | None = None
        if continuity_ramp:
            p99_path = ROOT / "outputs" / "action_step_p99.json"
            p99 = np.asarray(json.loads(p99_path.read_text())["p99_rad"], np.float32)
            self._ramp = np.maximum(p99, RAMP_FLOOR)
            logger.info("continuity ramp ON: per-joint threshold %.3f..%.3f rad (floor %.2f)",
                        self._ramp.min(), self._ramp.max(), RAMP_FLOOR)
        if blend_chunks:
            logger.info("chunk overlap blending ON: w_old = %.2f*exp(-i/%.0f)", BLEND_W0, BLEND_TAU)

        # joint-limit clamp (2026-08-14): last step before the actions leave the process.
        # Bounds are precomputed (scripts/build_clamp_bounds.py) because neither authority
        # exists at runtime -- the URDF is not in the image and the dataset certainly is not.
        self._clamp: tuple[np.ndarray, np.ndarray] | None = None
        if clamp_limits:
            spec = json.loads(Path(clamp_limits).read_text())
            lo = np.asarray(spec["lower_rad"], np.float32)
            hi = np.asarray(spec["upper_rad"], np.float32)
            if lo.shape != (ACTION_DIM,) or hi.shape != (ACTION_DIM,):
                raise ValueError(f"clamp bounds must be [{ACTION_DIM}], got {lo.shape}/{hi.shape}")
            if not np.all(lo < hi):
                raise ValueError("clamp bounds must satisfy lower < upper for every joint")
            self._clamp = (lo, hi)
            logger.info("joint clamp ON: %s (lower_policy=%s, %d joints widened above the "
                        "URDF ceiling, %d clamped below the demo floor)",
                        Path(clamp_limits).name, spec.get("lower_policy"),
                        len(spec.get("widened_upper", [])), len(spec.get("tightened_lower", [])))
        self.reset()

    def reset(self) -> None:
        now = self._clock()
        self.tracker.reset(now)
        if self.stage_runner is not None:
            self.stage_runner.reset()
        self._tac_hist: list[tuple[float, np.ndarray]] = []  # (t_mono, mosaic) buffer for the time-based pair
        self._tac_mode: str | None = None       # last logged tactile mode ("full"/"vec_only"/"none")
        self._tac_pair_age_ms = 0.0
        self._tac_pair_interp_used = False
        self._last_call_t: float | None = None
        self._last_stdout_t = 0.0
        self._last_chunk: tuple[float, np.ndarray] | None = None   # (t_mono, returned (chunk,65) rad)
        self._n_calls = 0

    def _tactile_inputs(self, obs: dict, now: float) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """-> (tactile_vec (60,) or None, tactile_img (10,6,112,112) or None, mode).
        Behavior matrix in the module docstring section 2. tactile_deform is intentionally
        never used as a tactile_img source (wrong appearance domain for Sparsh -- see there).
        The temporal pair partner is TIME-BASED (closest buffered mosaic to 167ms old);
        at episode start the buffer holds only the current mosaic -> prev = cur, exactly
        training's max(0, t-5) frame-pair semantics at t=0."""
        vec = obs.get("observation/tactile")
        raw = obs.get("observation/image/tactile_raw")
        if vec is None:
            return None, None, "none"
        vec = np.asarray(vec, np.float32).reshape(N_TACTILE_RAW)
        if raw is None:
            return vec, None, "vec_only"
        cur = _prep_tactile_mosaic(raw)                      # (10,3,112,112)
        self._tac_hist.append((now, cur))
        while len(self._tac_hist) > 1 and (now - self._tac_hist[0][0] > TAC_BUF_S
                                           or len(self._tac_hist) > TAC_BUF_N):
            self._tac_hist.pop(0)
        prev = self._pair_partner(now)
        return vec, np.concatenate([cur, prev], axis=1), "full"   # (10,6,112,112)

    def _pair_partner(self, now: float) -> np.ndarray:
        """The 167ms-old half of the tactile pair.

        The buffer only receives one mosaic per infer() call, so the ages actually available
        are multiples of the Gateway's inference interval -- {0, 110, 220, ...} in async,
        {0, 1600, ...} in sync. Picking the nearest (pre-2026-09-29 behaviour) therefore
        returns the CURRENT mosaic in sync, collapsing the temporal delta to zero.

        With --tactile-pair-interp we linearly interpolate between the two buffered mosaics
        that bracket TAC_PAIR_S, reconstructing I_{t-167ms} instead of snapping to a
        neighbour. At episode start the buffer holds only the current mosaic and prev = cur,
        which is exactly training's max(0, t-5) semantics at t=0."""
        ages = [now - t for t, _ in self._tac_hist]
        if not self._tac_interp or len(self._tac_hist) == 1:
            i = int(np.argmin([abs(a - TAC_PAIR_S) for a in ages]))
            self._tac_pair_age_ms = ages[i] * 1000.0
            self._tac_pair_interp_used = False
            return self._tac_hist[i][1]
        # buffer is append-ordered -> ages descend; find the bracketing pair
        older = [i for i, a in enumerate(ages) if a >= TAC_PAIR_S]
        newer = [i for i, a in enumerate(ages) if a < TAC_PAIR_S]
        if not older or not newer:                     # target outside the buffer span
            i = int(np.argmin([abs(a - TAC_PAIR_S) for a in ages]))
            self._tac_pair_age_ms = ages[i] * 1000.0
            self._tac_pair_interp_used = False
            return self._tac_hist[i][1]
        io_, in_ = older[-1], newer[0]                 # closest older / closest newer
        a_o, a_n = ages[io_], ages[in_]
        w = 0.0 if a_o <= a_n else (TAC_PAIR_S - a_n) / (a_o - a_n)   # 0 -> newer, 1 -> older
        self._tac_pair_age_ms = TAC_PAIR_S * 1000.0
        self._tac_pair_interp_used = True
        return ((1.0 - w) * self._tac_hist[in_][1] + w * self._tac_hist[io_][1]).astype(np.float32)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        if obs.get("reset"):                                 # non-contract testing hook (module docstring)
            logger.info("reset requested via observation key")
            self.reset()
        now = self._clock()                                  # ONE timestamp per call: tactile pair, fold
        dt_ms = None if self._last_call_t is None else (now - self._last_call_t) * 1000.0  # tracker + telemetry all use it
        self._last_call_t = now
        self._n_calls += 1
        state = np.asarray(obs["observation/state"], dtype=np.float32).reshape(ACTION_DIM)
        head_left = np.asarray(obs["observation/image/head_left"])
        wrist_left = np.asarray(obs["observation/image/wrist_left"])
        wrist_right = np.asarray(obs["observation/image/wrist_right"])
        if self.kwargs["n_cams"] == 3:
            # v6 3-cam-native ckpt: the contract's three streams feed the model directly --
            # NO head_right approximation (module docstring §1 v6 update)
            images = np.stack([
                _prep_image(head_left), _prep_image(wrist_left), _prep_image(wrist_right),
            ], axis=0)[None]  # (1,3,3,224,224)
        else:
            head_right = head_left  # v1-contract approximation: no head_right stream (docstring §1)
            if self.serve_contract == "v2":
                hr = obs.get("observation/image/head_right")
                if hr is not None:
                    head_right = np.asarray(hr)   # v2 contract provides the real right eye
            images = np.stack([
                _prep_image(head_left), _prep_image(head_right),
                _prep_image(wrist_left), _prep_image(wrist_right),
            ], axis=0)[None]  # (1,4,3,224,224)

        tac_vec, tac_img, mode = self._tactile_inputs(obs, now)
        if mode != self._tac_mode:
            logger.info("tactile mode: %s (was %s)", mode, self._tac_mode)
            self._tac_mode = mode

        # torque: v1 contract has none -> raw zeros (docstring §2). v2-contract ckpts consume
        # the observation's real joint_torque when present (zeros fallback = the trained
        # torque-dropout mode, so an omitted field stays in-distribution).
        torque_raw = np.zeros(N_TORQUE_RAW, np.float32)
        if self.serve_contract == "v2":
            tq = obs.get("observation/state/joint_torque")
            if tq is not None:
                torque_raw = np.asarray(tq, np.float32).reshape(N_TORQUE_RAW)
        obs_lowdim = self.norm.obs_lowdim(
            state, torque_raw,
            tac_vec if tac_vec is not None else np.zeros(N_TACTILE_RAW, np.float32))
        obs_lowdim = obs_lowdim.astype(np.float32)
        if tac_vec is None:
            obs_lowdim[-30:] = 0.0  # no-tactile mode ONLY: re-zero the normalized slice (FoAR gate reads it)

        fold_used = self.tracker.fold_id
        batch = {
            "obs_lowdim": torch.from_numpy(obs_lowdim).float()[None].to(self.device),
            "fold_id": torch.tensor([fold_used], dtype=torch.long, device=self.device),
            "images": torch.from_numpy(images).float().to(self.device),
        }
        if self.kwargs["tactile"]:
            batch["tactile_img"] = (torch.from_numpy(tac_img).float()[None].to(self.device)
                                    if tac_img is not None
                                    else torch.zeros(1, 10, 6, 112, 112, device=self.device))
        if self.kwargs["n_tempos"]:
            batch["tempo_id"] = torch.tensor([self.tempo_id], dtype=torch.long, device=self.device)

        out = self.model(batch, use_vae=False)
        pred = out.pred[0].float().cpu().numpy()                     # (chunk, 65) normalized
        actions = self.norm.unnormalize_action(pred).astype(np.float32)  # (chunk, 65) radians

        # fold-phase tracking (shared hardened logic -- module docstring §4)
        p_complete = torch.sigmoid(out.completion_logit)[0, 0].item() if out.completion_logit is not None else 0.0
        p_prog = out.progress_pred[0, 0].item() if out.progress_pred is not None else 0.0
        fold_elapsed = self.tracker.elapsed(now)
        calls_in_fold = self.tracker.calls_in_fold + 1
        co = None
        if self.stage_runner is not None:                        # v8 co-signal (default off)
            pi = self.stage_runner.update(
                now, images[0, 0],                               # the prepped head_left, reused
                state, tac_vec if tac_vec is not None else np.zeros(N_TACTILE_RAW, np.float32))
            co = self.stage_runner.co_signal(fold_used, pi)
            self._stage_pi = pi
        reason = self.tracker.update(p_complete, p_prog, now, co_signal=co)
        if reason is not None:
            logger.info("fold advance %d->%d reason=%s (elapsed %.1fs, %d calls, p_complete=%.3f p_prog=%.3f)",
                        fold_used, self.tracker.fold_id, reason, fold_elapsed, calls_in_fold, p_complete, p_prog)

        actions, blended, ramped = self._stabilize(actions, state, now)

        # clamp LAST: the stabilizers above can themselves push a joint past a bound, and
        # nothing downstream of here can put it back
        clamped = 0
        if self._clamp is not None:
            lo, hi = self._clamp
            out = np.clip(actions, lo, hi)
            clamped = int(np.count_nonzero(out != actions))
            actions = out

        self._telemetry(now, dt_ms, fold_used, fold_elapsed, calls_in_fold, p_complete, p_prog,
                        mode, reason, actions, blended, ramped, clamped)
        # truncate LAST so the stabilizers/clamp still see the full predicted chunk and the
        # telemetry keeps reporting the model's real chunk statistics
        if self._exec_horizon:
            actions = np.ascontiguousarray(actions[:self._exec_horizon])
        return {"actions": actions}

    def _stabilize(self, actions: np.ndarray, state: np.ndarray, now: float):
        """Optional serving stabilizers (BOTH default off -- module docstring extras).
        Order: overlap blending first (temporal consistency across chunks), then the
        first-step continuity ramp (guarantee on the executed gap from the current state)."""
        blended = ramped = False
        if self._blend:
            if self._last_chunk is not None:
                t0, prev = self._last_chunk
                k = int(round((now - t0) * FPS))         # prev-chunk steps presumed executed since t0
                if 0 <= k < len(prev):
                    n = min(len(actions), len(prev) - k)
                    w_old = (BLEND_W0 * np.exp(-np.arange(n) / BLEND_TAU)).astype(np.float32)[:, None]
                    actions[:n] = (1.0 - w_old) * actions[:n] + w_old * prev[k:k + n]
                    blended = True
        if self._ramp is not None:
            gap = actions[0] - state
            excess = gap - np.clip(gap, -self._ramp, self._ramp)   # per-joint beyond-p99 part only
            if np.any(excess != 0.0):
                decay = (1.0 - np.arange(RAMP_STEPS) / RAMP_STEPS).astype(np.float32)[:, None]
                actions[:RAMP_STEPS] -= excess[None] * decay
                ramped = True
        if self._blend:
            self._last_chunk = (now, actions.copy())
        return actions, blended, ramped

    def _telemetry(self, now, dt_ms, fold_used, fold_elapsed, calls_in_fold, p_complete, p_prog,
                   tac_mode, advance, actions, blended, ramped, clamped=0) -> None:
        """Per-call jsonl (only when --telemetry-file) + concise stdout line every ~2s.
        Chunk stats are only computed when one of the two sinks will actually consume them."""
        stdout_due = now - self._last_stdout_t >= STDOUT_EVERY_S
        if self._tele_f is None and not stdout_due:
            return
        steps = np.abs(np.diff(actions, axis=0))
        rec = {
            "t": round(now, 4), "call": self._n_calls,
            "dt_ms": None if dt_ms is None else round(dt_ms, 1),
            "fold_id": fold_used, "fold_elapsed_s": round(fold_elapsed, 2),
            "calls_in_fold": calls_in_fold,
            "p_complete": round(p_complete, 4), "p_prog": round(p_prog, 4),
            "tac_mode": tac_mode, "tac_pair_age_ms": round(self._tac_pair_age_ms, 1),
            "tac_pair_interp": self._tac_pair_interp_used,
            "advance": advance,
            "a0_l2": round(float(np.linalg.norm(actions[0])), 4),
            "step_mean": round(float(steps.mean()), 5), "step_max": round(float(steps.max()), 5),
            "blended": blended, "ramped": ramped, "clamped": clamped,
        }
        if self.stage_runner is not None and getattr(self, "_stage_pi", None) is not None:
            pi = self._stage_pi
            rec["stage_argmax"] = int(pi.argmax())
            rec["stage_pi_k"] = round(float(pi[fold_used]), 4)
            if fold_used < N_FOLDS - 1:
                rec["stage_pi_k1"] = round(float(pi[fold_used + 1]), 4)
        if self._tele_f is not None:
            self._tele_f.write(json.dumps(rec) + "\n")
        if stdout_due:
            self._last_stdout_t = now
            logger.info("tele call=%d dt=%sms fold=%d(%.0fs,%d calls) p_c=%.2f p_p=%.2f tac=%s "
                        "pair=%.0fms |a0|=%.2f step_mean=%.4f%s",
                        rec["call"], rec["dt_ms"], fold_used, fold_elapsed, calls_in_fold,
                        p_complete, p_prog, tac_mode, self._tac_pair_age_ms, rec["a0_l2"],
                        rec["step_mean"], f" ADVANCE({advance})" if advance else "")


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ckpt", default=str(ROOT / "outputs" / "ckpt" / "best_v3_tactile.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--telemetry-file", default=None,
                    help="per-call jsonl diagnostics (default off; stdout summary is always on)")
    ap.add_argument("--blend-chunks", action="store_true",
                    help="stabilizer, DEFAULT OFF: blend the overlap with the previous chunk")
    ap.add_argument("--continuity-ramp", action="store_true",
                    help="stabilizer, DEFAULT OFF: ramp out first-step gaps beyond the train p99 step size")
    ap.add_argument("--tempo", type=float, default=1.0,
                    help="v5 tempo-conditioned ckpts only: serve at this trained tempo (default 1.0x safety)")
    ap.add_argument("--stage-net", default=None,
                    help="v8, DEFAULT OFF (local testing only): StageNet ckpt path -- adds the "
                         "stage-recognition co-signal to the fold tracker (policy/model/stage_net.py)")
    ap.add_argument("--clamp-limits", default=None,
                    help="DEFAULT OFF: path to outputs/action_clamp_bounds.json -- clamp every "
                         "returned action into the per-joint reachable envelope")
    args = ap.parse_args()

    policy = FoldingACTPolicy(args.ckpt, device=args.device, telemetry_file=args.telemetry_file,
                              blend_chunks=args.blend_chunks, continuity_ramp=args.continuity_ramp,
                              serve_tempo=args.tempo, stage_net=args.stage_net,
                              clamp_limits=args.clamp_limits)
    # version string = checkpoint tag + hardened-server suffix, derived from the ckpt path
    # so a rebuilt bundle always announces what it actually ships (a v4 bundle once shipped
    # with a stale hardcoded string -- organizers see this metadata).
    _tag = Path(args.ckpt).stem.removeprefix("best_")
    metadata = {
        "policy": f"foldingact-{_tag} (hardened server r2026-07-29)",
        "ckpt": args.ckpt,
        "ckpt_step": policy.step,
        "action_dim": ACTION_DIM,
        "action_horizon": policy.chunk,
        "tactile_in_ckpt": policy.kwargs["tactile"],
        "tactile_modes": "auto per call: full (60-D vec + tactile_raw grid) / vec_only / none",
        "tactile_temporal_pair": "time-based, target 167ms (call-rate independent)",
        "fold_tracker": "wall-clock budgets(p90) + min-dwell + 0.3s sustained signal (rollout.FoldTracker)",
        "tactile_deform_used": False,
        "serve_contract": policy.serve_contract,
        "head_right_approximation": ("none (3-cam-native checkpoint)"
                                     if policy.kwargs["n_cams"] == 3
                                     else ("none (v2-contract ckpt consumes the real head_right)"
                                           if policy.serve_contract == "v2"
                                           else "duplicated from head_left")),
        "stabilizers": {"blend_chunks": args.blend_chunks, "continuity_ramp": args.continuity_ramp},
        "serve_tempo": args.tempo,
        "stage_net": args.stage_net,   # v8 co-signal ckpt or None (off = pre-v8 behavior)
    }
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    logger.info("serving FoldingACT (%s) on %s:%d (action_horizon=%d)", args.ckpt, args.host, args.port, policy.chunk)
    server.serve_forever()


if __name__ == "__main__":
    main()
