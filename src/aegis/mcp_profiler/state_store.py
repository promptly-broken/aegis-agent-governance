"""Stateful MCP session persistence layer: Memory & Redis-backed state stores."""

from __future__ import annotations

import abc
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class StateStoreMetrics(BaseModel):
    """Latency and payload telemetry for an MCP state operation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    pointer: str
    payload_bytes: int = Field(..., ge=0)
    write_latency_ms: float = Field(default=0.0, ge=0.0)
    read_latency_ms: float = Field(default=0.0, ge=0.0)
    is_cached: bool = False


class BaseMCPStateStore(abc.ABC):
    """Abstract interface for stateful MCP session storage."""

    @staticmethod
    def generate_pointer(session_id: str, payload_bytes: bytes) -> str:
        """Derive deterministic session pointer URN from state hash."""
        digest = hashlib.sha256(payload_bytes).hexdigest()[:16]
        return f"urn:mcp:state:{session_id}:{digest}"

    @abc.abstractmethod
    def save_state(self, session_id: str, state: Dict[str, Any]) -> Tuple[str, StateStoreMetrics]:
        """Persist state and return session pointer and metrics."""
        pass

    @abc.abstractmethod
    def load_state(self, session_pointer: str) -> Optional[Dict[str, Any]]:
        """Retrieve state from session pointer."""
        pass

    @abc.abstractmethod
    def update_delta(
        self, session_id: str, delta: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], StateStoreMetrics]:
        """Merge delta into existing session state and return new pointer."""
        pass


class MemoryMCPStateStore(BaseMCPStateStore):
    """In-memory reference implementation for local testing and zero-dependency profiling."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._pointer_to_session: Dict[str, str] = {}

    def save_state(self, session_id: str, state: Dict[str, Any]) -> Tuple[str, StateStoreMetrics]:
        t0 = time.time()
        payload_bytes = json.dumps(state, sort_keys=True).encode("utf-8")
        pointer = self.generate_pointer(session_id, payload_bytes)

        self._sessions[session_id] = state
        self._pointer_to_session[pointer] = session_id
        duration_ms = (time.time() - t0) * 1000.0

        metrics = StateStoreMetrics(
            session_id=session_id,
            pointer=pointer,
            payload_bytes=len(payload_bytes),
            write_latency_ms=round(duration_ms, 3),
            is_cached=False,
        )
        return pointer, metrics

    def load_state(self, session_pointer: str) -> Optional[Dict[str, Any]]:
        session_id = self._pointer_to_session.get(session_pointer)
        if not session_id or session_id not in self._sessions:
            return None
        return self._sessions[session_id]

    def update_delta(
        self, session_id: str, delta: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], StateStoreMetrics]:
        existing = self._sessions.get(session_id, {})
        merged = {**existing, **delta}
        pointer, metrics = self.save_state(session_id, merged)
        return pointer, merged, metrics


class RedisMCPStateStore(BaseMCPStateStore):
    """Redis-backed state store with graceful in-memory fallback."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = 3600) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.is_connected = False
        self._redis_client: Any = None
        self._fallback_memory = MemoryMCPStateStore()

        self._try_connect()

    def _try_connect(self) -> None:
        try:
            import redis
            client = redis.Redis.from_url(self.redis_url, decode_responses=True, socket_timeout=0.2)
            client.ping()
            self._redis_client = client
            self.is_connected = True
        except Exception:
            self.is_connected = False
            self._redis_client = None

    def save_state(self, session_id: str, state: Dict[str, Any]) -> Tuple[str, StateStoreMetrics]:
        if not self.is_connected or self._redis_client is None:
            return self._fallback_memory.save_state(session_id, state)

        t0 = time.time()
        payload_str = json.dumps(state, sort_keys=True)
        payload_bytes = payload_str.encode("utf-8")
        pointer = self.generate_pointer(session_id, payload_bytes)

        key = f"mcp:session:{session_id}"
        pointer_key = f"mcp:ptr:{pointer}"

        pipe = self._redis_client.pipeline()
        pipe.set(key, payload_str, ex=self.ttl_seconds)
        pipe.set(pointer_key, session_id, ex=self.ttl_seconds)
        pipe.execute()
        duration_ms = (time.time() - t0) * 1000.0

        metrics = StateStoreMetrics(
            session_id=session_id,
            pointer=pointer,
            payload_bytes=len(payload_bytes),
            write_latency_ms=round(duration_ms, 3),
            is_cached=False,
        )
        return pointer, metrics

    def load_state(self, session_pointer: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected or self._redis_client is None:
            return self._fallback_memory.load_state(session_pointer)

        pointer_key = f"mcp:ptr:{session_pointer}"
        session_id = self._redis_client.get(pointer_key)
        if not session_id:
            return None

        key = f"mcp:session:{session_id}"
        raw = self._redis_client.get(key)
        if not raw:
            return None
        return json.loads(raw)

    def update_delta(
        self, session_id: str, delta: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], StateStoreMetrics]:
        if not self.is_connected or self._redis_client is None:
            return self._fallback_memory.update_delta(session_id, delta)

        key = f"mcp:session:{session_id}"
        raw = self._redis_client.get(key)
        existing: Dict[str, Any] = json.loads(raw) if raw else {}
        merged = {**existing, **delta}
        pointer, metrics = self.save_state(session_id, merged)
        return pointer, merged, metrics
