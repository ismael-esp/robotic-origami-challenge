#!/usr/bin/env python3
"""origami-zenoh-v1 policy service for FoldingACT (async kit, 2026-08-26).

The organizers' v2 inference kit replaced the websocket contract with Zenoh
query/reply + a Docker-image submission (see
outputs/reports/kit_v2_vs_v1_comparison.html and
inf_code_v1/origami-inference-kit-participant_v2/docs/participant_zenoh_submission.md).
This file is the kit's ``examples/policy_server_template.py`` with ``TeamPolicy``
replaced by an adapter around our existing, verified ``FoldingACTPolicy``
(policy/serve/policy_server.py) -- ALL model-facing behavior (preprocessing,
tactile trichotomy, FoldTracker fold-phase tracking, tempo serving, telemetry)
lives there, in ONE implementation shared by both transports. The server boundary
below (codec, envelope, queryables, validation, errors, lock, SIGTERM) is kept
VERBATIM from the template per its own instruction: "The public server boundary
must remain unchanged."

What the adapter maps:
  - kit observation keys are a superset of the old websocket ones with the SAME
    names (head_left/wrist_left/wrist_right/tactile/tactile_raw/state), so the
    inner policy consumes them directly; images arrive pre-squashed at 224x224
    (organizer-side 1920x1536 -> 224 stretch) and _prep_image's 224->224 resize
    is a verified bit-exact no-op.
  - v2-only fields (observation/image/head_right, observation/state/joint_torque)
    are consumed by FoldingACTPolicy iff the checkpoint records
    serve_contract == "v2" (v7+ recipes); v1-contract ckpts (v1-v6) ignore them
    -- identical behavior to the websocket path, verified byte-identical by
    policy/serve/transport_equivalence_test.py.
  - ``reset`` (first-class in the v2 protocol, called before every episode) maps
    to FoldingACTPolicy.reset() = FoldTracker + tactile-history clear.
  - action_horizon is the checkpoint's chunk length (45) and is validated against
    --action-horizon/ORIGAMI_ACTION_HORIZON if one is supplied.
  - execution_mode (async kit, 2026-08-26): advertised in metadata; the organizer
    Gateway picks the matching execution strategy. "sync" = the original
    sequential-chunk path (all our closed-loop evidence was gathered here);
    "async" = the Gateway calls infer at ~10 Hz and temporally ensembles
    overlapping chunks. Policy-side both are safe: FoldTracker/StageNetRunner are
    wall-clock based (call-rate independent) and the clamp/continuity-ramp anchor
    on the CURRENT observation state, not on the previous chunk's tail. The value
    comes from --execution-mode / env EXECUTION_MODE; the submission image bakes
    its choice via the Dockerfile ARG (see build_submission_bundle.py).

Run (local dev against a router):
    conda run -n origami_infer python policy/serve/zenoh_policy_service.py \
        --endpoint tcp/127.0.0.1:17447 --session-id local-contract-test \
        --ckpt outputs/ckpt/best_v6_3cam.pt
In the submission image the entrypoint execs this file with no args; endpoint and
session come from ORIGAMI_ZENOH_ENDPOINT / ORIGAMI_SESSION_ID (the only two env
vars the organizer injects).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import zenoh

# --- import our policy (same sys.path bootstrap as policy_server.py) -----------------
ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "policy" / "serve"))
from policy_server import FoldingACTPolicy  # noqa: E402  (does its own sys.path setup)

logger = logging.getLogger("foldingact_zenoh_service")

# ===================================================================================
# EVERYTHING from here to the TeamPolicy replacement is VERBATIM from the kit's
# examples/policy_server_template.py (async kit, 2026-08-24) -- do not edit.
# ===================================================================================
TRANSPORT_VERSION = "origami-zenoh-v1"
SEMANTIC_VERSION = "origami-v1"
# from the kit's openpi reference server (serve_policy_zenoh.py): identifies the kit
# generation this image was built against; the README pairs it with the OCI
# org.opencontainers.image.source-kit label (extra metadata fields are allowed).
INFERENCE_KIT = "origami-inference-kit-async"
ACTION_DIM = 65
IMAGE_SHAPE = (224, 224, 3)
REQUIRED_IMAGE_SPECS = {
    "observation/image/head_left": IMAGE_SHAPE,
    "observation/image/head_right": IMAGE_SHAPE,
    "observation/image/wrist_left": IMAGE_SHAPE,
    "observation/image/wrist_right": IMAGE_SHAPE,
    "observation/image/tactile_deform": (480, 1200, 3),
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": (480, 1600, 3),
}
VECTOR_SPECS = {
    "observation/state": (ACTION_DIM,),
    "observation/state/joint_torque": (ACTION_DIM,),
    "observation/tactile": (60,),
}
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE",
        f"{side}_thumb_CMC_AA",
        f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA",
        f"{side}_thumb_IP",
        f"{side}_index_MCP_FE",
        f"{side}_index_MCP_AA",
        f"{side}_index_PIP",
        f"{side}_index_DIP",
        f"{side}_middle_MCP_FE",
        f"{side}_middle_MCP_AA",
        f"{side}_middle_PIP",
        f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE",
        f"{side}_ring_MCP_AA",
        f"{side}_ring_PIP",
        f"{side}_ring_DIP",
        f"{side}_pinky_CMC",
        f"{side}_pinky_MCP_FE",
        f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP",
        f"{side}_pinky_DIP",
    )


JOINT_NAMES = (
    tuple(f"left_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("right")
    + (
        "lower_body_joint_1",
        "lower_body_joint_2",
        "lower_body_joint_3",
        "lower_body_joint_4",
        "lower_body_joint_5",
        "neck_joint_1",
        "neck_joint_2",
    )
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("joint contract must contain 65 unique names")


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise ValueError(f"unsupported numpy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _mapping_value(value: Mapping[Any, Any], key: str) -> Any:
    return value[key] if key in value else value.get(key.encode())


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if _mapping_value(value, "__ndarray__") is True:
        data = _mapping_value(value, "data")
        shape = _mapping_value(value, "shape")
        dtype = np.dtype(_mapping_value(value, "dtype"))
        if (
            not isinstance(data, bytes)
            or not isinstance(shape, (list, tuple))
            or len(shape) > 8
            or dtype.kind in {"O", "V", "c"}
            or dtype.hasobject
        ):
            raise ValueError("invalid numpy array payload")
        normalized_shape = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in normalized_shape):
            raise ValueError("invalid numpy array shape")
        expected_size = math.prod(normalized_shape) * dtype.itemsize
        if expected_size > MAX_PAYLOAD_BYTES or len(data) != expected_size:
            raise ValueError("numpy array payload size does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(normalized_shape)
    if _mapping_value(value, "__npgeneric__") is True:
        return np.dtype(_mapping_value(value, "dtype")).type(
            _mapping_value(value, "data")
        )
    return value


def pack_payload(value: Any) -> bytes:
    payload = msgpack.packb(value, default=_pack_numpy, use_bin_type=True)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("response exceeds 64 MiB")
    return payload


def unpack_payload(value: Any) -> Any:
    payload = value.to_bytes() if hasattr(value, "to_bytes") else bytes(value)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("request exceeds 64 MiB")
    return msgpack.unpackb(
        payload,
        object_hook=_unpack_numpy,
        raw=False,
        strict_map_key=False,
        max_bin_len=MAX_PAYLOAD_BYTES,
        max_array_len=1_000_000,
        max_map_len=10_000,
        max_str_len=1_000_000,
    )


# ===================================================================================
# TeamPolicy replacement: the FoldingACT adapter (the ONLY participant-edited part).
# ===================================================================================
class FoldingACTTeamPolicy:
    """Kit-facing adapter: TeamPolicy ABI (reset / infer -> float32[T,65]) around the
    shared FoldingACTPolicy. The inner policy returns {"actions": (chunk,65) rad}; the
    kit boundary wants the bare array. Episode state (FoldTracker + tactile history +
    stabilizer buffers) is cleared by reset()."""

    def __init__(self, ckpt_path: str, device: str = "cuda:0", serve_tempo: float = 1.0,
                 telemetry_file: str | None = None, stage_net: str | None = None,
                 continuity_ramp: bool = False, clamp_limits: str | None = None,
                 exec_horizon: int = 0, tactile_pair_interp: bool = False,
                 fold_budget_min_p: float = 0.0) -> None:
        self.inner = FoldingACTPolicy(ckpt_path, device=device,
                                      telemetry_file=telemetry_file,
                                      serve_tempo=serve_tempo, stage_net=stage_net,
                                      continuity_ramp=continuity_ramp,
                                      clamp_limits=clamp_limits,
                                      exec_horizon=exec_horizon,
                                      tactile_pair_interp=tactile_pair_interp,
                                      fold_budget_min_p=fold_budget_min_p)
        # what crosses the wire: the model's chunk (45), or the shorter exec horizon if set.
        # The contract requires metadata's action_horizon to equal the returned T exactly.
        self.action_horizon = int(self.inner.served_horizon)

    def reset(self) -> None:
        self.inner.reset()

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        result = self.inner.infer(observation)
        return np.ascontiguousarray(np.asarray(result["actions"], dtype=np.float32))


# ===================================================================================
# OrigamiZenohServer: VERBATIM from the kit template (boundary unchanged).
# ===================================================================================
class OrigamiZenohServer:
    def __init__(
        self,
        policy: FoldingACTTeamPolicy,
        *,
        endpoint: str,
        session_id: str,
        action_horizon: int,
        execution_mode: str = "async",
    ) -> None:
        if action_horizon < 1 or action_horizon > 1024:
            raise ValueError("action_horizon must be in [1, 1024]")
        if execution_mode not in {"sync", "async"}:
            raise ValueError("execution_mode must be 'sync' or 'async'")
        self.policy = policy
        self.endpoint = endpoint
        self.session_id = session_id
        self.action_horizon = action_horizon
        self.execution_mode = execution_mode
        self._policy_lock = threading.Lock()
        self._stop = threading.Event()
        self._session: Any | None = None
        self._queryables: list[Any] = []
        self.metadata = {
            "protocol_version": SEMANTIC_VERSION,
            "action_dim": ACTION_DIM,
            "action_horizon": action_horizon,
            "action_type": "absolute_joint_position",
            "action_units": "radians",
            "joint_names": JOINT_NAMES,
            "execution_mode": execution_mode,
            "inference_kit": INFERENCE_KIT,
        }

    def serve_forever(self) -> None:
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([self.endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self._session = zenoh.open(config)
        self._queryables = [
            self._session.declare_queryable(
                f"{TRANSPORT_VERSION}/{operation}",
                self._handle_query,
                complete=True,
            )
            for operation in ("metadata", "reset", "infer")
        ]
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        logging.info(
            "READY transport=%s endpoint=%s horizon=%d execution_mode=%s",
            TRANSPORT_VERSION,
            self.endpoint,
            self.action_horizon,
            self.execution_mode,
        )
        self._stop.wait()
        for queryable in self._queryables:
            queryable.undeclare()
        self._session.close()

    def _handle_query(self, query: Any) -> None:
        operation = str(query.key_expr).rsplit("/", 1)[-1]
        request: Any = None
        try:
            request = unpack_payload(query.payload)
            response = self.process(operation, request)
        except Exception as exc:  # noqa: BLE001 - sanitized protocol error
            error_id = uuid.uuid4().hex
            logging.error(
                "request failed operation=%s error_id=%s type=%s",
                operation,
                error_id,
                type(exc).__name__,
            )
            public_message = f"request failed; error_id={error_id}"
            if not isinstance(request, Mapping):
                query.reply_err(
                    pack_payload(
                        {
                            "error": {
                                "code": "INVALID_REQUEST",
                                "message": public_message,
                                "retryable": False,
                            }
                        }
                    ),
                    encoding="application/msgpack",
                )
                return
            response = self._envelope(operation, request)
            response["error"] = {
                "code": (
                    "INFERENCE_FAILED" if operation == "infer" else "INVALID_REQUEST"
                ),
                "message": public_message,
                "retryable": False,
            }
        query.reply(
            str(query.key_expr),
            pack_payload(response),
            encoding="application/msgpack",
        )

    def process(self, operation: str, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be a MessagePack map")
        response = self._envelope(operation, request)
        if request.get("protocol_version") != TRANSPORT_VERSION:
            raise ValueError("invalid protocol_version")
        if request.get("operation") != operation:
            raise ValueError("operation does not match queryable key")
        if request.get("session_id") != self.session_id:
            raise ValueError("session_id does not match assigned session")
        if not isinstance(request.get("request_id"), str) or not request["request_id"]:
            raise ValueError("request_id must be a non-empty string")

        if operation == "metadata":
            response["metadata"] = self.metadata
            return response
        if operation == "reset":
            with self._policy_lock:
                self.policy.reset()
            response["ok"] = True
            return response
        if operation != "infer":
            raise ValueError(f"unsupported operation: {operation}")

        observation = request.get("observation")
        self._validate_observation(observation)
        started = time.monotonic()
        with self._policy_lock:
            actions = self.policy.infer(dict(observation))
        actions = np.asarray(actions)
        expected_shape = (self.action_horizon, ACTION_DIM)
        if actions.dtype != np.float32 or actions.shape != expected_shape:
            raise ValueError(
                f"policy actions must be float32{expected_shape}, "
                f"got {actions.dtype}{actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        response["actions"] = np.ascontiguousarray(actions)
        response["server_timing"] = {
            "infer_ms": (time.monotonic() - started) * 1000.0
        }
        return response

    def _envelope(
        self,
        operation: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "protocol_version": TRANSPORT_VERSION,
            "operation": operation,
            "request_id": request.get("request_id"),
            "session_id": self.session_id,
        }

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, Mapping):
            raise ValueError("infer request must contain an observation map")
        required = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS, "prompt"}
        allowed = required | set(OPTIONAL_IMAGE_SPECS)
        if not required.issubset(observation) or not set(observation).issubset(allowed):
            raise ValueError("observation keys do not match the public full schema")
        for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items():
            if key not in observation:
                continue
            image = observation.get(key)
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.shape != shape
            ):
                raise ValueError(f"{key} must be uint8{shape}")
        for key, shape in VECTOR_SPECS.items():
            vector = observation.get(key)
            if (
                not isinstance(vector, np.ndarray)
                or vector.dtype != np.float32
                or vector.shape != shape
                or not np.isfinite(vector).all()
            ):
                raise ValueError(f"{key} must be finite float32{shape}")
        if not isinstance(observation.get("prompt"), str):
            raise ValueError("prompt must be a string")


# ===================================================================================
# main: CLI matches the template's env-var behavior + our ckpt/device/tempo flags.
# ===================================================================================
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"))
    parser.add_argument("--session-id", default=os.environ.get("ORIGAMI_SESSION_ID"))
    parser.add_argument("--action-horizon", type=int,
                        default=int(os.environ.get("ORIGAMI_ACTION_HORIZON", "0")),
                        help="0 (default) = use the checkpoint's chunk length (45); a nonzero "
                             "value is VALIDATED against the checkpoint and must match")
    parser.add_argument("--ckpt",
                        default=os.environ.get("FOLDING_ACT_CKPT",
                                               str(ROOT / "outputs" / "ckpt" / "best_v6_3cam.pt")))
    parser.add_argument("--device", default=os.environ.get("FOLDING_ACT_DEVICE", "cuda:0"))
    parser.add_argument("--tempo", type=float,
                        default=float(os.environ.get("FOLDING_ACT_TEMPO", "1.0")),
                        help="tempo-conditioned ckpts only: serve at this trained tempo "
                             "(default 1.0x safety)")
    parser.add_argument("--telemetry-file",
                        default=os.environ.get("FOLDING_ACT_TELEMETRY_FILE") or None)
    parser.add_argument("--stage-net",
                        default=os.environ.get("FOLDING_ACT_STAGE_NET") or None,
                        help="v8: StageNet ckpt path -- enables the stage-recognition "
                             "co-signal in the fold tracker (policy/model/stage_net.py). "
                             "Empty/unset = tracker runs on the policy heads alone.")
    parser.add_argument("--clamp-limits",
                        default=os.environ.get("FOLDING_ACT_CLAMP_LIMITS") or None,
                        help="v8.1: action_clamp_bounds.json path -- clamp every returned "
                             "action into the per-joint reachable envelope. Empty/unset = off.")
    parser.add_argument("--continuity-ramp",
                        action="store_true",
                        default=os.environ.get("FOLDING_ACT_CONTINUITY_RAMP", "") not in ("", "0"),
                        help="v8.1: ramp out first-step gaps beyond the train-set p99 step "
                             "size over 10 steps (FOLDING_ACT_CONTINUITY_RAMP=0 disables).")
    parser.add_argument("--exec-horizon", type=int,
                        default=int(os.environ.get("FOLDING_ACT_EXEC_HORIZON", "0")),
                        help="serve only the first N of the model's 45 predicted steps "
                             "(0 = full chunk). A sync Gateway executes a whole chunk "
                             "open-loop, so a shorter chunk = a faster feedback loop; "
                             "keep N >= 10 or every served step lands inside the "
                             "continuity ramp (measured: it distorts the trajectory).")
    parser.add_argument("--tactile-pair-interp", action="store_true",
                        default=os.environ.get("FOLDING_ACT_TACTILE_PAIR_INTERP", "") not in ("", "0"),
                        help="v9.3: reconstruct the 167ms tactile pair partner by interpolating "
                             "between buffered mosaics. Without it the pair collapses to zero "
                             "delta at the sync Gateway call rate (measured live 2026-09-29).")
    parser.add_argument("--fold-budget-min-p", type=float,
                        default=float(os.environ.get("FOLDING_ACT_FOLD_BUDGET_MIN_P", "0")),
                        help="v9.3: minimum p_complete before the fold-phase WALL-CLOCK backstop "
                             "may advance the token. 0 = pre-fix behaviour. A hard backstop at "
                             "2x budget always fires, so progress is still guaranteed.")
    parser.add_argument("--execution-mode",
                        choices=("sync", "async"),
                        default=os.environ.get("EXECUTION_MODE", "async"),
                        help="async kit (2026-08-26): Gateway execution strategy advertised "
                             "in metadata. Omitted everywhere = the kit's async default; the "
                             "submission image bakes its choice via the Dockerfile "
                             "ARG EXECUTION_MODE.")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if not args.endpoint:
        raise SystemExit("--endpoint or ORIGAMI_ZENOH_ENDPOINT is required")
    if not args.session_id:
        raise SystemExit("--session-id or ORIGAMI_SESSION_ID is required")
    # Load the model BEFORE opening the Zenoh session: the metadata queryable is the
    # organizer's readiness probe, so it must not answer until infer can actually run
    # ("Load and warm the model during startup" -- participant_zenoh_submission.md §6).
    policy = FoldingACTTeamPolicy(args.ckpt, device=args.device, serve_tempo=args.tempo,
                                  telemetry_file=args.telemetry_file,
                                  stage_net=args.stage_net,
                                  continuity_ramp=args.continuity_ramp,
                                  clamp_limits=args.clamp_limits,
                                  exec_horizon=args.exec_horizon,
                                  tactile_pair_interp=args.tactile_pair_interp,
                                  fold_budget_min_p=args.fold_budget_min_p)
    if args.action_horizon and args.action_horizon != policy.action_horizon:
        raise SystemExit(f"--action-horizon {args.action_horizon} != checkpoint chunk "
                         f"{policy.action_horizon}; the horizon is fixed by the model")
    # warm-up inference (synthetic, full schema) so the first organizer query does not
    # pay cudnn/autotune costs; also proves infer works end-to-end before READY.
    warm = {
        **{k: np.zeros(s, np.uint8) for k, s in REQUIRED_IMAGE_SPECS.items()},
        **{k: np.zeros(s, np.float32) for k, s in VECTOR_SPECS.items()},
        "prompt": "warmup",
    }
    t0 = time.monotonic()
    policy.infer(dict(warm))
    policy.reset()                                   # warm-up must not leak episode state
    logging.info("warm-up inference done in %.0f ms (device=%s)",
                 (time.monotonic() - t0) * 1000.0, policy.inner.device)
    server = OrigamiZenohServer(
        policy,
        endpoint=args.endpoint,
        session_id=args.session_id,
        action_horizon=policy.action_horizon,
        execution_mode=args.execution_mode,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
