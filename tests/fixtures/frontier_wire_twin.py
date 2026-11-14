"""Local HTTP Wire-Contract Twin for OpenAI, Anthropic, and Gemini Frontier APIs.

Spins up a real loopback ThreadingHTTPServer speaking the exact wire protocols,
authentication headers, parameter constraints, tool-call payloads, and prompt-caching
usage schemas of OpenAI (/v1/chat/completions), Anthropic (/v1/messages), and
Google Gemini (/v1beta/models/...:generateContent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.parse


@dataclass
class RecordedWireRequest:
    """Captured HTTP request received by the Frontier Wire Twin."""

    method: str
    path: str
    headers: Dict[str, str]
    body: Dict[str, Any]


@dataclass
class TwinState:
    """Mutable behavior configuration and request log for the Wire Twin server."""

    requests: List[RecordedWireRequest] = field(default_factory=list)
    force_status_code: Optional[int] = None
    response_delay_seconds: float = 0.0
    simulate_cached_tokens: bool = False
    simulate_tool_only_response: bool = False


class _FrontierWireHandler(BaseHTTPRequestHandler):
    """HTTP handler validating and emulating OpenAI, Anthropic, and Gemini wire contracts."""

    server_version = "FrontierWireTwin/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Suppress console noise during pytest runs
        return

    @property
    def twin_state(self) -> TwinState:
        return getattr(self.server, "twin_state")

    def _send_json(self, status_code: int, payload: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        content_len = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            body = json.loads(raw_body)
        except Exception:
            self._send_json(400, {"error": {"message": "Malformed JSON body"}})
            return

        headers_dict = {k.lower(): v for k, v in self.headers.items()}
        self.twin_state.requests.append(
            RecordedWireRequest(
                method="POST",
                path=self.path,
                headers=headers_dict,
                body=body,
            )
        )

        if self.twin_state.response_delay_seconds > 0:
            time.sleep(self.twin_state.response_delay_seconds)

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.endswith("/chat/completions"):
            self._handle_openai(headers_dict, body)
        elif path.endswith("/messages"):
            self._handle_anthropic(headers_dict, body)
        elif ":generateContent" in path:
            self._handle_gemini(parsed_url.query, headers_dict, body)
        else:
            self._send_json(404, {"error": f"Unknown endpoint: {path}"})

    def _handle_openai(self, headers: Dict[str, str], body: Dict[str, Any]) -> None:
        auth = headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth == "Bearer invalid-key":
            self._send_json(
                401,
                {
                    "error": {
                        "message": "Incorrect API key provided.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
            return

        if self.twin_state.force_status_code == 429:
            self._send_json(
                429,
                {
                    "error": {
                        "message": "Rate limit reached for model in organization.",
                        "type": "tokens",
                        "code": "rate_limit_exceeded",
                    }
                },
                extra_headers={"Retry-After": "5"},
            )
            return

        model = str(body.get("model", ""))
        messages = body.get("messages")
        if not model or not isinstance(messages, list) or not messages:
            self._send_json(
                400,
                {
                    "error": {
                        "message": "Missing required parameter: 'model' or 'messages'.",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        # Enforce OpenAI reasoning model parameter contract (o1/o3/o4/gpt-5 reject max_tokens)
        is_reasoning = bool(re.match(r"^(o1|o3|o4|gpt-5)(\b|-)", model.lower()))
        if is_reasoning and "max_tokens" in body:
            self._send_json(
                400,
                {
                    "error": {
                        "message": (
                            "Unsupported parameter: 'max_tokens' is not supported with this model. "
                            "Use 'max_completion_tokens' instead."
                        ),
                        "type": "invalid_request_error",
                        "param": "max_tokens",
                        "code": "unsupported_parameter",
                    }
                },
            )
            return

        tools = body.get("tools")
        usage: Dict[str, Any] = {
            "prompt_tokens": 1000 if self.twin_state.simulate_cached_tokens else 64,
            "completion_tokens": 50 if self.twin_state.simulate_cached_tokens else 24,
            "total_tokens": 1050 if self.twin_state.simulate_cached_tokens else 88,
        }
        if self.twin_state.simulate_cached_tokens:
            usage["prompt_tokens_details"] = {"cached_tokens": 800, "audio_tokens": 0}
        if is_reasoning:
            usage["completion_tokens_details"] = {"reasoning_tokens": 16}

        if isinstance(tools, list) and tools:
            first_tool = tools[0]
            fn_name = (
                first_tool.get("function", {}).get("name")
                or first_tool.get("name")
                or "lookup_dispute_order"
            )
            message_obj: Dict[str, Any] = {
                "role": "assistant",
                "content": None,  # Real OpenAI wire contract sets content to null on tool calls
                "tool_calls": [
                    {
                        "id": "call_wire_oai_01",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps({"order_id": "ORD-99182", "action": "escrow_refund"}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message_obj = {
                "role": "assistant",
                "content": f"OpenAI wire-verified resolution via {model}",
            }
            finish_reason = "stop"

        self._send_json(
            200,
            {
                "id": "chatcmpl-wire-twin-001",
                "object": "chat.completion",
                "created": 1730000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message_obj,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            },
        )

    def _handle_anthropic(self, headers: Dict[str, str], body: Dict[str, Any]) -> None:
        api_key = headers.get("x-api-key", "")
        ant_version = headers.get("anthropic-version", "")
        if not api_key or api_key == "invalid-key":
            self._send_json(
                401,
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "invalid x-api-key",
                    },
                },
            )
            return

        if ant_version != "2023-06-01":
            self._send_json(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "anthropic-version header is required (expected 2023-06-01)",
                    },
                },
            )
            return

        if self.twin_state.force_status_code == 529:
            self._send_json(
                529,
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Anthropic API is temporarily overloaded.",
                    },
                },
            )
            return

        model = str(body.get("model", ""))
        max_tokens = body.get("max_tokens")
        messages = body.get("messages")
        if not model or max_tokens is None or not isinstance(messages, list):
            self._send_json(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "model, max_tokens, and messages are required fields.",
                    },
                },
            )
            return

        tools = body.get("tools")
        if isinstance(tools, list) and tools:
            first_tool = tools[0]
            fn_name = first_tool.get("name") or first_tool.get("function", {}).get("name") or "audit_ledger"
            content_blocks: List[Dict[str, Any]] = []
            if not self.twin_state.simulate_tool_only_response:
                content_blocks.append({"type": "text", "text": "Invoking ledger verification tool."})
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": "toolu_wire_ant_01",
                    "name": fn_name,
                    "input": {"order_id": "ORD-99182", "verify_signature": True},
                }
            )
            stop_reason = "tool_use"
        else:
            content_blocks = [
                {
                    "type": "text",
                    "text": f"Anthropic wire-verified resolution via {model}",
                }
            ]
            stop_reason = "end_turn"

        if self.twin_state.simulate_cached_tokens:
            usage = {
                "input_tokens": 200,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 800,
                "output_tokens": 50,
            }
        else:
            usage = {
                "input_tokens": 60,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 20,
            }

        self._send_json(
            200,
            {
                "id": "msg_wire_twin_001",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "usage": usage,
            },
        )

    def _handle_gemini(self, query: str, headers: Dict[str, str], body: Dict[str, Any]) -> None:
        qs = urllib.parse.parse_qs(query)
        key = (qs.get("key") or [""])[0]
        if not key or key == "invalid-key":
            self._send_json(401, {"error": {"code": 401, "message": "API key not valid."}})
            return

        usage_meta: Dict[str, Any] = {
            "promptTokenCount": 1000 if self.twin_state.simulate_cached_tokens else 48,
            "candidatesTokenCount": 50 if self.twin_state.simulate_cached_tokens else 16,
            "totalTokenCount": 1050 if self.twin_state.simulate_cached_tokens else 64,
        }
        if self.twin_state.simulate_cached_tokens:
            usage_meta["cachedContentTokenCount"] = 800

        self._send_json(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemini wire-verified resolution"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": usage_meta,
            },
        )


class FrontierWireTwinServer:
    """Context manager spinning up a real loopback HTTP server for Frontier API contract tests."""

    def __init__(self) -> None:
        self.state = TwinState()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.base_url: str = ""

    def __enter__(self) -> "FrontierWireTwinServer":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _FrontierWireHandler)
        setattr(self._server, "twin_state", self.state)
        host, port = self._server.server_address
        self.base_url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def cloud_base_urls(self) -> Dict[str, str]:
        """Return cloud_base_urls dictionary suitable for TaskDispatcher."""
        return {
            "openai": f"{self.base_url}/v1",
            "anthropic": f"{self.base_url}/v1",
            "gemini": f"{self.base_url}/v1beta",
        }

