"""Tests for MCP state store persistence layers: Memory & Redis with fallback."""

from aegis.mcp_profiler.state_store import (
    BaseMCPStateStore,
    MemoryMCPStateStore,
    RedisMCPStateStore,
    StateStoreMetrics,
)


def test_base_pointer_generation():
    session_id = "session_xyz"
    payload = b'{"status": "active", "user_id": 100}'
    pointer = BaseMCPStateStore.generate_pointer(session_id, payload)

    assert pointer.startswith("urn:mcp:state:session_xyz:")
    assert len(pointer.split(":")[-1]) == 16


def test_memory_mcp_state_store():
    store = MemoryMCPStateStore()
    session_id = "sess_001"
    initial_state = {"order_id": "ORD-123", "status": "pending"}

    # Save
    ptr1, metrics = store.save_state(session_id, initial_state)
    assert ptr1.startswith("urn:mcp:state:sess_001:")
    assert metrics.payload_bytes > 0
    assert metrics.session_id == session_id

    # Load
    loaded = store.load_state(ptr1)
    assert loaded == initial_state

    # Non-existent pointer
    assert store.load_state("urn:mcp:state:unknown:deadbeef") is None

    # Delta update
    delta = {"status": "shipped", "tracking": "TRK-999"}
    ptr2, merged, update_metrics = store.update_delta(session_id, delta)
    assert ptr2 != ptr1
    assert merged["status"] == "shipped"
    assert merged["order_id"] == "ORD-123"
    assert merged["tracking"] == "TRK-999"

    # Verify latest state loaded with new pointer
    loaded_updated = store.load_state(ptr2)
    assert loaded_updated == merged


def test_redis_mcp_state_store_fallback():
    # Attempt connection to a non-existent port to test fallback resilience
    store = RedisMCPStateStore(redis_url="redis://127.0.0.1:59999/0")
    assert store.is_connected is False

    session_id = "fallback_sess"
    state = {"key": "value", "count": 42}

    # Should transparently use memory fallback
    pointer, metrics = store.save_state(session_id, state)
    assert pointer.startswith("urn:mcp:state:fallback_sess:")
    assert metrics.payload_bytes > 0

    loaded = store.load_state(pointer)
    assert loaded == state

    # Delta update on fallback
    ptr2, merged, m2 = store.update_delta(session_id, {"count": 43})
    assert merged["count"] == 43
    assert store.load_state(ptr2) == merged


class _FakeRedisPipeline:
    def __init__(self, backing_dict: dict):
        self._backing = backing_dict
        self._ops = []

    def set(self, key: str, value: str, ex: int | None = None):
        self._ops.append((key, value))
        return self

    def execute(self):
        for k, v in self._ops:
            self._backing[k] = v
        self._ops.clear()
        return True


class _FakeRedisClient:
    def __init__(self):
        self.data = {}

    def pipeline(self):
        return _FakeRedisPipeline(self.data)

    def get(self, key: str):
        return self.data.get(key)


def test_redis_mcp_state_store_connected():
    store = RedisMCPStateStore(redis_url="redis://127.0.0.1:59999/0")
    fake_client = _FakeRedisClient()
    store._redis_client = fake_client
    store.is_connected = True

    session_id = "live_redis_sess"
    initial_state = {"dispute_id": 101, "stage": "intake"}

    ptr1, m1 = store.save_state(session_id, initial_state)
    assert ptr1.startswith(f"urn:mcp:state:{session_id}:")
    assert m1.payload_bytes > 0
    assert store.load_state(ptr1) == initial_state
    assert store.load_state("urn:mcp:state:missing:0000") is None

    # Verify update_delta reads mcp:session:{session_id} and merges state
    ptr2, merged, m2 = store.update_delta(session_id, {"stage": "resolved", "refund": 50.0})
    assert ptr2 != ptr1
    assert merged == {"dispute_id": 101, "stage": "resolved", "refund": 50.0}
    assert store.load_state(ptr2) == merged
