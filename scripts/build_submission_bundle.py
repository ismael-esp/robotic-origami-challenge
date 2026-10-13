#!/usr/bin/env python3
"""Assemble the self-contained remote-eval submission bundle.

The organizers run our policy server on THEIR machine (RTX 4090) against the
inf_code_v1 robot I/O contract -- we upload a bundle containing the server, its
model code, normalization/label assets, ONE checkpoint, the two vendored openpi
serving/client modules, requirements, and a README. The bundle mirrors the repo's
relative layout (policy/..., outputs/...) so every internal `parents[N]`-style path
in the copied sources resolves unchanged; only the server's sys.path block pointing
at the vendored kit is rewritten (verified below).

Run:
  conda run -n origami_infer python scripts/build_submission_bundle.py            # v6 (default)
  conda run -n origami_infer python scripts/build_submission_bundle.py --ckpt best_v4_tactile.pt

Verify on a genuinely fresh env (see bundle README.md, "Fresh-machine check").
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "inf_code_v1" / "origami-inference-kit-main"
KIT_V2 = ROOT / "inf_code_v1" / "origami-inference-kit-participant_v2"
# async kit (2026-08-26): current kit generation. Wire protocol/validator unchanged vs
# participant_v2; adds the execution_mode metadata field + EXECUTION_MODE build-arg +
# source-kit OCI label (see docs/participant_zenoh_submission.md diff, ANNOUNCEMENT.md).
KIT_V3 = ROOT / "inf_code_v2" / "1787717851349_origami-inference-kit-async-20260826"
BUNDLE = ROOT / "outputs" / "submission_bundle"

# repo-relative source -> bundle-relative destination (same path = layout mirror)
CODE = [
    "policy/serve/policy_server.py",
    "policy/serve/zenoh_policy_service.py",   # v2 kit (2026-08-03): the Zenoh transport;
                                              # imports FoldingACTPolicy from policy_server
    "policy/model/folding_act.py",
    "policy/model/act_decoder_v3.py",
    "policy/model/sparsh_encoder.py",
    "policy/model/stage_net.py",              # v8: StageNet recognizer + StageNetRunner
                                              # (fold-tracker co-signal; enabled via the
                                              # entrypoint's FOLDING_ACT_STAGE_NET)
    "policy/data/fold_utils.py",
    "policy/data/autolabel_folds.py",
    "policy/rollout.py",
]
ASSETS = [
    "outputs/norm_stats.json",
    "outputs/fold_labels_auto.json",
    "outputs/fold_step_budget.json",
    "outputs/action_step_p99.json",   # per-joint p99 step sizes (--continuity-ramp, ON since v8.1)
    "outputs/action_clamp_bounds.json",  # v8.2: per-joint reachable envelope, URDF ceiling (--clamp-limits, ON)
]
VENDOR = {  # kit source -> bundle vendor path (import roots: vendor/)
    "openpi-base-main/src/openpi/serving/websocket_policy_server.py":
        "vendor/openpi/serving/websocket_policy_server.py",
    "openpi-base-main/packages/openpi-client/src/openpi_client/__init__.py":
        "vendor/openpi_client/__init__.py",
    "openpi-base-main/packages/openpi-client/src/openpi_client/base_policy.py":
        "vendor/openpi_client/base_policy.py",
    "openpi-base-main/packages/openpi-client/src/openpi_client/msgpack_numpy.py":
        "vendor/openpi_client/msgpack_numpy.py",
    # protocol check, runnable as-is from inside the bundle (self-contained pair)
    "sharpa_north_ces_lite_sdk-main/examples/check_policy_server.py": "check/check_policy_server.py",
    "sharpa_north_ces_lite_sdk-main/examples/openpi_origami_io.py": "check/openpi_origami_io.py",
}
VENDOR_V3 = {  # from the async kit (2026-08-26; validator byte-identical to participant_v2's)
    # the organizers' own black-box Zenoh validator -- self-contained (msgpack/numpy/zenoh only)
    "sharpa_north_ces_lite_sdk-main/examples/check_zenoh_policy.py": "check/check_zenoh_policy.py",
}

# the ONLY source rewrite: point the server at vendor/ instead of the kit tree
OLD_PATH_BLOCK = (
    'KIT = ROOT / "inf_code_v1" / "origami-inference-kit-main"\n'
    'sys.path.insert(0, str(KIT / "openpi-base-main" / "src"))\n'
    'sys.path.insert(0, str(KIT / "openpi-base-main" / "packages" / "openpi-client" / "src"))\n'
)
NEW_PATH_BLOCK = 'sys.path.insert(0, str(ROOT / "vendor"))   # vendored openpi serving + openpi_client\n'

REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.10.0+cu128
torchvision==0.25.0+cu128
lerobot==0.4.4
timm==1.0.28
numpy==2.2.6
pillow==12.2.0
msgpack==1.2.1
websockets==16.1.1
pandas==3.0.5
eclipse-zenoh==1.9.0
"""

# --- v2 kit (2026-08-03): the bundle is now ALSO a Docker build context -------------
DOCKERFILE = """\
# FoldingACT origami-zenoh-v1 submission image (async kit, 2026-08-26).
# Build FROM THE BUNDLE DIR:  docker build -t <team>/origami-policy:submission .
#   --build-arg EXECUTION_MODE=sync|async selects the organizer Gateway execution
#   strategy advertised in metadata. OUR default is async since v9.3: the 2026-09-01
#   live test ran sync, and sync means the Gateway executes an ENTIRE 45-step chunk
#   (1.5 s) open-loop before re-querying -- a 0.63 Hz feedback loop. Measured on the
#   closed-loop instrument (policy/eval_closed_loop.py, using the kit's own
#   TemporalEnsembler): async cuts arm drift 8.7 % AND removes the ~100 ms inter-chunk
#   stall. sync stays one build-arg away.
# torch cu128 pip wheels bundle the CUDA runtime; the host driver comes in via the
# NVIDIA container runtime (works on RTX 4090 / sm_89 and 5070 Ti / sm_120).
# Base pinned by digest (guide section 10.1: avoid build-day drift) = python:3.11-slim.
FROM python@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    HOME=/tmp/team-home \\
    XDG_CACHE_HOME=/tmp/team-cache \\
    TORCH_HOME=/tmp/team-cache/torch \\
    HF_HOME=/tmp/team-cache/hf \\
    TRITON_CACHE_DIR=/tmp/team-cache/triton \\
    MPLCONFIGDIR=/tmp/team-cache/mpl \\
    FOLDING_ACT_SKIP_PRETRAINED=1

RUN apt-get update \\
    && apt-get install -y --no-install-recommends ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# gcc/headers exist only for this layer: lerobot pulls evdev, an sdist that compiles
# against linux/input.h (no wheel on PyPI); purge the toolchain in the same RUN so it
# never ships in the final layer
COPY requirements.txt /app/requirements.txt
RUN apt-get update \\
    && apt-get install -y --no-install-recommends gcc python3-dev libc6-dev linux-libc-dev \\
    && python3 -m pip install --no-cache-dir -r /app/requirements.txt \\
    && apt-get purge -y gcc python3-dev libc6-dev linux-libc-dev \\
    && apt-get autoremove -y \\
    && rm -rf /var/lib/apt/lists/*

COPY policy/ /app/policy/
COPY outputs/ /app/outputs/
COPY vendor/ /app/vendor/
COPY licenses/ /licenses/
COPY entrypoint.sh /app/entrypoint.sh
COPY README.md SUBMISSION_NOTES.md /app/

RUN chmod 0755 /app/entrypoint.sh \\
    && chmod -R a+rX /app /licenses

# async kit (2026-08-26): ARG/LABEL/ENV deliberately LATE (after the pip layer) so the
# expensive cached layers are reused; the ARG only feeds the ENV below it.
ARG EXECUTION_MODE=async
LABEL org.opencontainers.image.source-kit="origami-inference-kit-async"

# UID 65532 has no passwd entry in slim images; torch-inductor's cache-dir helper
# calls getpass.getuser() -> pwd.getpwuid() and crashes without these (getpass
# checks LOGNAME/USER env first). Late ENV layer so the pip layer cache is reused.
ENV USER=team \\
    LOGNAME=team \\
    TORCHINDUCTOR_CACHE_DIR=/tmp/team-cache/inductor \\
    EXECUTION_MODE=${EXECUTION_MODE}

USER 65532:65532

HEALTHCHECK NONE
ENTRYPOINT ["/app/entrypoint.sh"]
"""

ENTRYPOINT = """\
#!/usr/bin/env bash
set -euo pipefail

: "${ORIGAMI_ZENOH_ENDPOINT:?ORIGAMI_ZENOH_ENDPOINT is required}"
: "${ORIGAMI_SESSION_ID:?ORIGAMI_SESSION_ID is required}"

export HOME="${HOME:-/tmp/team-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/team-cache}"
mkdir -p "$HOME" "$XDG_CACHE_HOME"

# async kit (2026-08-26): Gateway execution strategy, baked at build time via
# ARG EXECUTION_MODE (our default: async since v9.3 -- see the Dockerfile header).
export EXECUTION_MODE="${EXECUTION_MODE:-async}"
case "${EXECUTION_MODE}" in
  sync|async) ;;
  *)
    echo "[submission][ERROR] EXECUTION_MODE must be sync or async, got: ${EXECUTION_MODE}" >&2
    exit 2
    ;;
esac
echo "[submission] transport=origami-zenoh-v1 execution_mode=${EXECUTION_MODE}" >&2
{STAGE_NET_LINE}
# v8.2 (2026-08-14) Shadow-compatibility pair -- see docs/experiments/shadow_compatibility_v8.md.
# Together they took the local Shadow verdict from 308 violations to 0.
export FOLDING_ACT_CLAMP_LIMITS="${FOLDING_ACT_CLAMP_LIMITS-/app/outputs/action_clamp_bounds.json}"
export FOLDING_ACT_CONTINUITY_RAMP="${FOLDING_ACT_CONTINUITY_RAMP-1}"

# v9.3: serve only the first N of the 45 predicted steps (0/unset = full chunk). Operator
# valve for a sync Gateway that executes whole chunks open-loop. Keep N >= 10: below that
# every served step lands inside the 10-step continuity ramp and the trajectory distorts.
export FOLDING_ACT_EXEC_HORIZON="${FOLDING_ACT_EXEC_HORIZON-0}"

# v9.3 (2026-09-29) deployment-correctness pair, from the second live test
# (docs/experiments/live_test_v9_2_async_20260929.md). Both default-OFF in code; ON here.
#  1. The tactile temporal pair is buffered one mosaic per infer() call, so the ages
#     available are multiples of the Gateway's call interval. At the sync rate (~1.6 s) the
#     nearest candidate to the 167 ms training spacing is the CURRENT mosaic, so the pair
#     collapsed to [I_t ; I_t] and the contact delta was zero on 518/518 live calls.
#     Interpolating between the bracketing mosaics restores a true 167 ms partner.
#  2. The fold-phase token's wall-clock backstop fired in 4/5 live episodes at ~35 s with
#     p_complete 0.03-0.06 -- advancing the phase while the completion head said the fold
#     was not done. Require some confidence first; a hard backstop at 2x budget still fires.
export FOLDING_ACT_TACTILE_PAIR_INTERP="${FOLDING_ACT_TACTILE_PAIR_INTERP-1}"
export FOLDING_ACT_FOLD_BUDGET_MIN_P="${FOLDING_ACT_FOLD_BUDGET_MIN_P-0.15}"
exec python3 /app/policy/serve/zenoh_policy_service.py
"""

# v8: the stage-recognition co-signal ships ENABLED (uses `-` not `:-` so an operator can
# disable it by exporting FOLDING_ACT_STAGE_NET="" -- never required, purely a safety valve)
STAGE_NET_LINE = 'export FOLDING_ACT_STAGE_NET="${FOLDING_ACT_STAGE_NET-/app/outputs/ckpt/%s}"'

DOCKERIGNORE = """\
check/
Dockerfile
.dockerignore
"""

NOTICES = """\
# Third-party notices — FoldingACT origami submission image

This image contains the team's own policy code (FoldingACT) plus the following
third-party software and model weights. Full license texts ship with the
respective packages installed under site-packages in this image; the vendored
inference-kit code's licenses are included in this directory.

## Python packages (pip-installed, pinned in requirements.txt)
- PyTorch, torchvision — BSD-3-Clause (Meta Platforms / PyTorch contributors)
- lerobot — Apache-2.0 (Hugging Face; FoldingACT reuses its ACT encoder/decoder modules)
- timm — Apache-2.0 (Ross Wightman / Hugging Face)
- numpy — BSD-3-Clause
- Pillow — MIT-CMU (HPND)
- msgpack — Apache-2.0
- websockets — BSD-3-Clause
- pandas — BSD-3-Clause
- eclipse-zenoh — EPL-2.0 OR Apache-2.0 (Eclipse Foundation / ZettaScale)

## Vendored code
- vendor/openpi/, vendor/openpi_client/ — from the organizers' inference kit
  (openpi, Physical Intelligence, Apache-2.0). See LICENSE.inference-kit and
  THIRD_PARTY_LICENSES.inference-kit.md in this directory.
- policy/serve/zenoh_policy_service.py embeds the kit's policy_server_template.py
  server boundary (same kit licensing).

## Model weights (inside outputs/ckpt/*.pt — fine-tuned by the team)
- Vision encoders were initialized from DINOv3 ViT-S (Meta Platforms; released
  under the DINOv3 License) and fine-tuned on the competition training data.
- The tactile encoder was initialized from Sparsh (Meta Platforms / FAIR;
  released under CC-BY-NC-4.0) and fine-tuned on the competition training data.
  Used here for non-commercial research/competition purposes.
- The StageNet stage-recognizer checkpoint (stage_net_*.pt) embeds a FROZEN
  (not fine-tuned) DINOv3 ViT-S trunk (Meta Platforms, DINOv3 License) plus a
  small temporal head trained from scratch by the team on the competition data.
- All remaining weights were trained from scratch by the team on the official
  competition dataset (SharpaIT/Robotic_Origami_Challenge, gated).
"""


def build(ckpt_name: str, stage_net_name: str = "stage_net_v8.pt") -> None:
    ckpt_src = ROOT / "outputs" / "ckpt" / ckpt_name
    assert ckpt_src.exists(), f"checkpoint not found: {ckpt_src}"
    stage_src = None
    if stage_net_name and stage_net_name.lower() != "none":
        stage_src = ROOT / "outputs" / "ckpt" / stage_net_name
        assert stage_src.exists(), f"stage-net checkpoint not found: {stage_src}"
        # the shipped stage ckpt must carry its calibrated tau (co-signal threshold)
        import torch as _torch
        _tau = _torch.load(stage_src, map_location="cpu", weights_only=False).get("tau")
        assert _tau is not None, f"{stage_src} has no tau -- calibrate first (pass1v8 + --set-tau)"

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    for rel in CODE + ASSETS:
        src = ROOT / rel
        assert src.exists(), f"missing {src}"
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for src_rel, dst_rel in VENDOR.items():
        src = KIT / src_rel
        assert src.exists(), f"missing {src}"
        dst = BUNDLE / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for src_rel, dst_rel in VENDOR_V3.items():
        src = KIT_V3 / src_rel
        assert src.exists(), f"missing {src}"
        dst = BUNDLE / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for init in ["vendor/openpi/__init__.py", "vendor/openpi/serving/__init__.py"]:
        (BUNDLE / init).parent.mkdir(parents=True, exist_ok=True)
        (BUNDLE / init).write_text("")

    server = BUNDLE / "policy/serve/policy_server.py"
    text = server.read_text()
    assert OLD_PATH_BLOCK in text, "server sys.path block changed -- update OLD_PATH_BLOCK"
    text = text.replace(OLD_PATH_BLOCK, NEW_PATH_BLOCK)
    # make the bundled server's DEFAULT --ckpt the one checkpoint actually shipped (found
    # 2026-07-29: a v4 bundle whose default still pointed at best_v3_tactile.pt crashed the
    # README's no-args run line -- a preventable remote-slot failure)
    default_line = 'ap.add_argument("--ckpt", default=str(ROOT / "outputs" / "ckpt" / "best_v3_tactile.pt"))'
    assert default_line in text, "server --ckpt default changed -- update default_line"
    text = text.replace(default_line, default_line.replace("best_v3_tactile.pt", ckpt_name))
    server.write_text(text)

    (BUNDLE / "outputs/ckpt").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_src, BUNDLE / "outputs/ckpt" / ckpt_name)
    if stage_src is not None:
        shutil.copy2(stage_src, BUNDLE / "outputs/ckpt" / stage_net_name)
    (BUNDLE / "requirements.txt").write_text(REQUIREMENTS)
    # v2 kit: third-party license notices are REQUIRED image contents
    # (container_submission.md sections 1 + 8)
    lic = BUNDLE / "licenses"
    lic.mkdir(parents=True, exist_ok=True)
    (lic / "NOTICES.md").write_text(NOTICES)
    shutil.copy2(KIT_V3 / "LICENSE", lic / "LICENSE.inference-kit")
    shutil.copy2(KIT_V3 / "THIRD_PARTY_LICENSES.md", lic / "THIRD_PARTY_LICENSES.inference-kit.md")
    # v2 kit: the bundle doubles as the Docker build context for the submission image
    (BUNDLE / "Dockerfile").write_text(DOCKERFILE)
    stage_line = (STAGE_NET_LINE % stage_net_name) if stage_src is not None else ""
    (BUNDLE / "entrypoint.sh").write_text(ENTRYPOINT.replace("{STAGE_NET_LINE}", stage_line))
    (BUNDLE / "entrypoint.sh").chmod(0o755)
    (BUNDLE / ".dockerignore").write_text(DOCKERIGNORE)
    # the zenoh service defaults its --ckpt to the shipped checkpoint too
    zsvc = BUNDLE / "policy/serve/zenoh_policy_service.py"
    ztext = zsvc.read_text()
    z_default = 'str(ROOT / "outputs" / "ckpt" / "best_v6_3cam.pt")'
    assert z_default in ztext, "zenoh service --ckpt default changed -- update z_default"
    zsvc.write_text(ztext.replace(z_default, z_default.replace("best_v6_3cam.pt", ckpt_name)))
    # the notes quote the server's expected startup log -- read the real step/val_l1
    # from the shipped checkpoint so the quoted line matches whatever ckpt is bundled
    import torch
    ck = torch.load(ckpt_src, map_location="cpu", weights_only=False)
    step, val_l1 = int(ck["step"]), float(ck["val_l1"])
    init_ckpt = ck.get("init_ckpt")          # v9: finetune provenance (None for v1-v8 ckpts)
    del ck
    with_stage = stage_src is not None
    (BUNDLE / "README.md").write_text(readme(ckpt_name, with_stage))
    (BUNDLE / "SUBMISSION_NOTES.md").write_text(submission_notes(ckpt_name, step, val_l1, with_stage,
                                                                  init_ckpt=init_ckpt))

    budget = json.loads((BUNDLE / "outputs/fold_step_budget.json").read_text())
    assert budget.get("pctl") == 90 and "budget_s" in budget, (
        f"shipped fold_step_budget.json lacks p90 budget_s ({list(budget)}); regenerate "
        "(python -c 'import rollout; rollout.compute_fold_budgets(fps=30, force=True)') so the "
        "bundled server's wall-clock FoldTracker (rollout.compute_fold_budgets_s) never recomputes")

    n = sum(1 for _ in BUNDLE.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file()) / 2**20
    print(f"bundle: {BUNDLE}  ({n} files, {size:.0f} MiB, ckpt={ckpt_name})")


def readme(ckpt_name: str, with_stage: bool = True) -> str:
    stage_code = ("\n  (incl. `policy/model/stage_net.py`, the stage-recognition co-signal -- see\n"
                  "  SUBMISSION_NOTES section 3; disable via env `FOLDING_ACT_STAGE_NET=\"\"` if ever needed)"
                  if with_stage else "")
    stage_ckpt = " (+ the StageNet ckpt)" if with_stage else ""
    return f"""\
# FoldingACT policy service -- Robotic Origami Challenge submission bundle

Self-contained policy service for the **`origami-zenoh-v1`** contract
(async kit, 2026-08-26): this directory is the **Docker build context** for the
submission image, and also runs directly on a host for development. The older
websocket server (v1 kit contract) is retained alongside as a fallback.

**Execution mode** (async kit): the image advertises
`metadata["execution_mode"]` and the organizer Gateway picks the matching
strategy. Select it at build time with `--build-arg EXECUTION_MODE=sync|async`;
**this bundle's default is `async`** — under sync the Gateway executes each full
45-step chunk (1.5 s) open-loop before re-querying, which measured worse for us
(our per-step prediction error grows ~2.3x from the first to the last step of a
chunk). The policy is safe under either mode: fold-phase tracking is wall-clock
based (call-rate independent) and the output stabilizers anchor on the current
observed state.

> **Organizers: please read `SUBMISSION_NOTES.md`** — exact run commands, runtime
> notes, and our questions, in one page. This README is the reference detail
> behind it; the two are consistent.

## Contents
- `Dockerfile` + `entrypoint.sh` -- build/run the submission image from THIS directory
- `SUBMISSION_NOTES.md` -- one-page organizer handoff
- `policy/serve/zenoh_policy_service.py` -- the origami-zenoh-v1 service (image entry point)
- `policy/serve/policy_server.py` -- the v1-kit websocket server (fallback; same model class)
- `policy/model/`, `policy/data/`, `policy/rollout.py` -- model + preprocessing code{stage_code}
- `outputs/ckpt/{ckpt_name}` -- the checkpoint served by default{stage_ckpt}
- `outputs/*.json` -- normalization stats, fold labels, fold step budgets
- `vendor/` -- vendored openpi serving + openpi_client modules (websocket fallback only)
- `check/` -- protocol validators: `check_zenoh_policy.py` (async kit) + the v1 websocket pair
- `requirements.txt` -- pinned deps (torch cu128 wheels; run on RTX 4090 / sm_89 and sm_120)

## Build + run the submission image (the v2 evaluation path)
    docker build -t team/origami-policy:submission .            # EXECUTION_MODE=async (default)
    # docker build --build-arg EXECUTION_MODE=sync -t ... .     # sync Gateway variant
    # runtime (the organizer injects the two env vars):
    docker run --gpus all --read-only --cap-drop ALL --security-opt no-new-privileges=true \\
      --user 65532:65532 --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \\
      --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \\
      -e ORIGAMI_ZENOH_ENDPOINT=tcp/<router>:7447 -e ORIGAMI_SESSION_ID=<session> \\
      team/origami-policy:submission

The service declares the three `origami-zenoh-v1/{{metadata,reset,infer}}` queryables,
warms the model BEFORE going READY (metadata is the readiness probe), and answers
`infer` with finite float32[45, 65] absolute-radian chunks. `reset` clears all episode
state (fold-phase tracker + tactile history). No network, no downloads, no listener:
`FOLDING_ACT_SKIP_PRETRAINED=1` is set and the full checkpoint is strict-loaded.

## Validate the image (no robot needed)
    # router + hardened image + the kit's own validator -- see SUBMISSION_NOTES.md section 2
    python check/check_zenoh_policy.py --endpoint tcp/127.0.0.1:17447 \\
        --session-id <session> --timeout 180 --requests 3 --expected-horizon 45
    # expect: "PASS: policy is compatible with origami-zenoh-v1"

## Fallback: run the v1 websocket server directly on a host
    pip install -r requirements.txt
    python policy/serve/policy_server.py --port 8000     # v1-kit websocket contract
    python check/check_policy_server.py --policy-host 127.0.0.1 --policy-port 8000

Both transports wrap the SAME policy class -- verified bit-identical actions on real
recorded frames (policy/serve/transport_equivalence_test.py in the source repo).
Metadata declares `action_horizon` = 45 (30Hz absolute joint-angle targets, 65-D,
contract order). ~2.5 GB VRAM, ~35-40 ms/inference on an RTX 5070 Ti.

## Tactile input (contract section 5)
Auto-detected per call, logged on change:
- `observation/tactile` (60,) + `observation/image/tactile_raw` (480x1600x3) -> full
  tactile (force vector + per-fingertip Sparsh image branch). **Please provide
  `tactile_raw`** -- the spec marks it "provided on request"; this policy uses it.
- vector only -> force vector consumed, image branch zeroed (graceful degradation).
- no tactile fields -> fully zeroed tactile (still contract-compatible).
`observation/image/tactile_deform` is not used (the model was trained on tactile_raw
tiles; deform is a different appearance domain).

## Notes for the eval runtime
- Please run inference at 20-30 Hz if configurable (the model takes ~40 ms/call on an
  RTX 5070 Ti; faster on a 4090). The server is CORRECT at any call rate -- fold-phase
  budgets are wall-clock and the tactile temporal pair is selected by wall-clock age
  (167 ms training spacing) -- but higher call rates improve reactivity.
- Optional diagnostics: `--telemetry-file /path/tele.jsonl` logs one json line per call
  (timestamps, measured inter-call dt, fold-phase trace, tactile mode, chunk stats).
  We would greatly appreciate this file back with the results.
- A truthy `"reset"` key in an observation resets internal episode state (fold-phase
  tracker + tactile history). Optional; only relevant if one server process is reused
  across multiple episodes.
- The contract has no joint_torque field; that model input is zeroed (documented,
  trained-around limitation).
"""


def submission_notes(ckpt_name: str, step: int, val_l1: float, with_stage: bool = True,
                     init_ckpt: str | None = None) -> str:
    """Organizer-facing one-pager. Written for the v5_tempo_b submission (tempo
    conditioning + the runtime requests below) and reused for v6_3cam (same recipe,
    natively 3-camera); review the text before shipping a bundle built around a
    different checkpoint. step/val_l1 come from the shipped ckpt (build())."""
    tag = Path(ckpt_name).stem.removeprefix("best_")
    finetune_note = ("" if not init_ckpt else
                     f" This checkpoint was **finetuned from our previous submission model "
                     f"(`{Path(init_ckpt).name}`) on the organizers' `competition-paper-set` "
                     f"dataset** (white paper with printed guide lines, lighting randomization), "
                     f"with a replay slice of the earlier recordings so the original behaviour is kept.")
    stage_note = ("""- **Fold-stage tracking**: the policy self-tracks which of the 6 folds it is
  executing. This image additionally runs a small stage-recognition model
  ("StageNet", ~5 ms every 100 ms on the head camera) whose output feeds the
  tracker as a CONSERVATIVE extra advance signal -- it is bounded by the same
  minimum-dwell / sustained-signal / wall-clock-budget safeguards as the policy's
  own heads and only acts late in a fold's time budget. Offline it converts most
  timeout-driven fold advances into recognition-driven ones (17 -> 5 over 18
  held-out episodes) with zero cascade regressions. It cannot request robot I/O;
  it only reads the same observation the policy receives.""" if with_stage else
"""- **Fold-stage tracking**: the policy self-tracks which of the 6 folds it is
  executing from its own completion/progress heads, bounded by minimum-dwell /
  sustained-signal / wall-clock-budget safeguards. No auxiliary stage-recognition
  model runs in this image.""")
    return f"""\
# Submission notes -- FoldingACT {tag} (Robotic Origami Challenge)

One page with everything needed to run this policy. The bundle `README.md` has the
reference detail; nothing there contradicts this page.

## 1. What this is

**FoldingACT {tag}**: an ACT-style CVAE transformer policy with action chunking,
served behind the **`origami-zenoh-v1`** contract (async kit, 2026-08-26). Per `infer`
query it consumes one full contract observation and returns **45 absolute
joint-angle targets (45 x 65, radians, contract joint order) at the 30 Hz training
cadence** -- `metadata["action_horizon"] = 45`, receding-horizon execution;
consuming only a prefix of the chunk is fine. Inference is action-free (the CVAE
latent is fixed to the prior mean). `reset` clears all episode state (fold-phase
tracker + tactile history) and never runs inference.{finetune_note} The checkpoint is
tempo-conditioned (trained speed variants); **this image serves the default 1.0x
tempo**. Footprint: ~2.5 GB VRAM, ~35-40 ms/call on an RTX 5070 Ti (faster on a
4090); model warm-up runs BEFORE the metadata queryable goes READY. No network
access is needed at runtime.

## 2. Run + validate the image (the v2 evaluation path)

Build from this directory, then run under the standard sandbox:

    docker build -t team/origami-policy:submission .
    docker run -d --gpus all --read-only --cap-drop ALL \\
      --security-opt no-new-privileges=true --user 65532:65532 \\
      --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \\
      --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \\
      -e ORIGAMI_ZENOH_ENDPOINT=tcp/<router>:7447 -e ORIGAMI_SESSION_ID=<session> \\
      team/origami-policy:submission

Expected container log (INFO):

    tempo conditioning: trained tempos=[1.0, 1.25, 1.5] -> serving tempo 1.0 (id 0)
    loaded .../outputs/ckpt/{ckpt_name} (step={step} val_l1={val_l1:.4f}...) kwargs=...
    fold tracker: budgets_s={{0: 34.8, ...}} min_dwell_s={{0: 9.7, ...}} sustain_s=0.3
    warm-up inference done in ... ms (device=cuda:0)
    READY transport=origami-zenoh-v1 endpoint=... horizon=45 execution_mode=sync

Validate with the kit's own black-box checker:

    python check/check_zenoh_policy.py --endpoint tcp/127.0.0.1:17447 \\
        --session-id <session> --timeout 180 --requests 3 --expected-horizon 45
    # expect: "PASS: policy is compatible with origami-zenoh-v1"

## 3. Runtime notes

{stage_note}
- **Tactile**: the policy consumes `observation/tactile` and (when present)
  `observation/image/tactile_raw`; it degrades gracefully when the optional raw
  grid is absent (validated by the kit checker's last request). **If enabling
  `tactile_raw` is possible for our runs, please do** -- full tactile is the trained
  configuration. `tactile_deform` is accepted but unused (different appearance
  domain from the tiles the tactile encoder was fine-tuned on).
- **Execution mode**: this image declares `metadata["execution_mode"] = "async"`.
  In our 2026-09-01 test run the sync Gateway executed each full 45-step chunk
  (1.5 s) before re-querying, i.e. the policy received a fresh observation only
  0.63 times per second and was committed open-loop in between. Our prediction
  error grows ~2.3x from the first to the last step of a chunk, so that costs real
  accuracy at the moment of a grasp. The async path re-plans at ~10 Hz and
  temporally aggregates overlapping chunks, which measured better for us and also
  removes the inter-chunk stall. A sync build of the same bundle is available
  (`--build-arg EXECUTION_MODE=sync`) if you need it.
- **Chunk prefix**: if the runtime consumes only a prefix of each chunk, earlier
  steps are more accurate than later ones -- consuming ~10-15 of the 45 steps per
  query is the sweet spot we measured. The image can also be built to serve a
  shorter chunk directly (`FOLDING_ACT_EXEC_HORIZON`).
- **Call rate**: the service is correct at ANY query rate -- fold-phase budgets are
  wall-clock and the tactile temporal pair is selected by wall-clock age (167 ms
  target). Higher rates only improve reactivity; chunks are produced at the 30 Hz
  training cadence.
- **Timing**: cold start (model load + warm-up) ~30-60 s on GPU before READY;
  steady-state infer ~35-40 ms on a 5070 Ti. The metadata query is the readiness
  probe, per the kit contract.

## 4. Questions for the organizers

1. Does the evaluation paper have printed guide-lines (as in all post-2026-07-01
   training demonstrations), or is it plain paper?
2. Will `observation/image/tactile_raw` be enabled during our evaluation runs?
3. What receding-horizon prefix (steps consumed per chunk) and query rate does the
   evaluation runtime use, and may teams request values?

## 5. Package integrity

The `.tar.zst` archive's SHA-256 is recorded in the submission manifest (an archive
cannot contain its own hash). Reload-verify: `zstd -dc <archive> | docker load`,
then repeat the section-2 validation.

## 6. Fallback: v1 websocket server

`policy/serve/policy_server.py` still implements the previous kit's websocket
contract (same model class; bit-identical actions verified on real frames). Run
`python policy/serve/policy_server.py --port 8000` +
`python check/check_policy_server.py --policy-port 8000` if the v1 flow is ever
needed.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="best_v7_fullobs.pt",
                    help="checkpoint filename under outputs/ckpt/ to include (exactly one)")
    ap.add_argument("--stage-net", default="stage_net_v8.pt",
                    help="StageNet ckpt under outputs/ckpt/ to ship + enable in the "
                         "entrypoint (v8 fold-tracker co-signal); 'none' = v7 behavior")
    args = ap.parse_args()
    build(args.ckpt, stage_net_name=args.stage_net)
