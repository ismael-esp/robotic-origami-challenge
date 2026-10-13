#!/usr/bin/env python3
"""Minimal origami-zenoh-v1 client for OUR OWN offline tooling (2026-08-03).

Speaks the participant_v2 kit's Zenoh query/reply protocol to a policy service
(ours or any contract-compliant one): metadata / reset / infer with the 4-field
correlation envelope, msgpack-numpy codec, exactly-one-reply semantics. The query
pattern mirrors the kit's own examples/check_zenoh_policy.py `query_once` (single
enveloped get, ConsolidationMode.NONE, envelope echo validated).

Used by policy/serve/transport_equivalence_test.py and
policy/serve/zenoh_replay_test.py. NOT part of the submission image (the image is
a server; the organizer owns the client side).

Usage:
    from zenoh_client import ZenohPolicyClient
    c = ZenohPolicyClient("tcp/127.0.0.1:17447", session_id="local-contract-test")
    print(c.metadata()["action_horizon"])
    c.reset()
    reply = c.infer(observation_dict)      # -> full reply map (actions, server_timing)
    c.close()
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import zenoh

# reuse the service's codec so client and server can never drift apart
from zenoh_policy_service import TRANSPORT_VERSION, pack_payload, unpack_payload

OPERATION_KEYS = {op: f"{TRANSPORT_VERSION}/{op}" for op in ("metadata", "reset", "infer")}


class ZenohPolicyClient:
    def __init__(self, endpoint: str, session_id: str, timeout_s: float = 30.0):
        self.session_id = session_id
        self.timeout_s = timeout_s
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self._session = zenoh.open(config)

    def _query(self, operation: str, **body: Any) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        request = {
            "protocol_version": TRANSPORT_VERSION,
            "operation": operation,
            "request_id": request_id,
            "session_id": self.session_id,
            **body,
        }
        replies = self._session.get(
            OPERATION_KEYS[operation],
            payload=pack_payload(request),
            timeout=self.timeout_s,
            consolidation=zenoh.ConsolidationMode.NONE,
        )
        successful, errors = [], []
        for reply in replies:
            sample = reply.ok
            if sample is not None:
                successful.append(unpack_payload(sample.payload.to_bytes()))
                continue
            err = reply.err
            if err is not None:
                try:
                    errors.append(err.payload.to_string())
                except Exception:
                    errors.append("<non-text Zenoh error>")
        if errors:
            raise RuntimeError(f"{operation}: Zenoh error reply: {'; '.join(errors)}")
        if len(successful) != 1:
            raise RuntimeError(f"{operation}: expected exactly one reply, got {len(successful)}")
        rep = successful[0]
        for field, want in (("protocol_version", TRANSPORT_VERSION), ("operation", operation),
                            ("request_id", request_id), ("session_id", self.session_id)):
            if rep.get(field) != want:
                raise RuntimeError(f"{operation}: envelope field {field!r} mismatch: "
                                   f"{rep.get(field)!r} != {want!r}")
        if "error" in rep:
            e = rep["error"]
            raise RuntimeError(f"{operation}: {e.get('code')}: {e.get('message')}")
        return rep

    def metadata(self) -> dict[str, Any]:
        return self._query("metadata")["metadata"]

    def reset(self) -> None:
        rep = self._query("reset")
        assert rep.get("ok") is True, f"reset did not ack: {rep}"

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._query("infer", observation=observation)

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
