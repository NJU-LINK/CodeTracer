"""Token estimation utilities.

Provides fast token-count approximations used by the compact mechanism
and cost tracker.  Uses ``tiktoken`` when available, otherwise falls
back to a character-based heuristic.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN_HEURISTIC = 3.5

_encoder: Any | None = None
_encoder_checked = False


def _get_encoder() -> Any | None:
    global _encoder, _encoder_checked
    if _encoder_checked:
        return _encoder
    _encoder_checked = True
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.debug("tiktoken not available; using character heuristic")
        _encoder = None
    return _encoder


def estimate_message_tokens(content: str) -> int:
    """Estimate the token count of a single string."""
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(content, disallowed_special=()))
    return max(1, int(len(content) / _CHARS_PER_TOKEN_HEURISTIC))


def estimate_tokens(messages: list[dict[str, Any]], model: str | None = None) -> int:
    """Estimate total token count across a message list.

    Each message is expected to have at least a ``content`` key.
    An additional ~4 tokens per message is added for role/separator
    overhead (matching OpenAI's accounting).
    """
    total = 0
    per_message_overhead = 4
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_message_tokens(content) + per_message_overhead
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_message_tokens(part["text"])
            total += per_message_overhead
    return total
