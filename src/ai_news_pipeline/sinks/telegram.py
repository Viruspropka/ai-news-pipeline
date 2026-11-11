"""Optional Telegram sink (Bot API sendMessage, plain text).

Configured only through the environment, never through files in the repo:
    TELEGRAM_BOT_TOKEN  bot token from BotFather
    TELEGRAM_CHAT_ID    target chat or channel id
Plain text (no parse_mode) means no escaping bugs, and link previews are off.
Long briefs are split at paragraph boundaries under Telegram's 4096-character
limit. The token is checked for its digits:letters shape and never appears in
an error message.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from ..models import Brief
from .base import Sink, SinkError

LIMIT = 4096
_TOKEN = re.compile(r"^\d+:[A-Za-z0-9_-]+$")
Opener = Callable[[urllib.request.Request, float], Any]


def split_message(text: str, limit: int = LIMIT) -> list[str]:
    """Split on blank lines, then on lines, then hard-cut; every chunk <= limit."""
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        piece = para if not current else f"\n\n{para}"
        if len(current) + len(piece) <= limit:
            current += piece
            continue
        if current:
            chunks.append(current)
        current = ""
        while len(para) > limit:
            cut = para.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            chunks.append(para[:cut])
            para = para[cut:].lstrip("\n")
        current = para
    if current:
        chunks.append(current)
    return chunks


def _default_opener(req: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - fixed https host


class TelegramSink(Sink):
    name = "telegram"

    def __init__(self, token: str, chat_id: str, opener: Opener | None = None,
                 api_base: str = "https://api.telegram.org", timeout: float = 20.0) -> None:
        if not token or not chat_id:
            raise SinkError("telegram sink needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment")
        token = token.strip()
        if not _TOKEN.match(token):
            # Also keeps "/" or "?" out of the request URL the token is part of.
            raise SinkError("TELEGRAM_BOT_TOKEN does not look like a bot token (digits:letters)")
        self._token = token
        self.chat_id = chat_id
        self.opener = opener or _default_opener
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "TelegramSink":
        env = os.environ if env is None else env
        return cls(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""))

    def __repr__(self) -> str:  # never print the token
        return f"<TelegramSink chat_id={self.chat_id!r}>"

    def deliver(self, brief: Brief) -> str:
        ids = []
        for chunk in split_message(brief.text):
            body = json.dumps({"chat_id": self.chat_id, "text": chunk,
                               "link_preview_options": {"is_disabled": True}}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.api_base}/bot{self._token}/sendMessage", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with self.opener(req, self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            # The request URL holds the token, so no exception text or chain
            # leaves this method: only the status code or the exception type.
            except urllib.error.HTTPError as exc:
                raise SinkError(f"telegram HTTP {exc.code}") from None
            except Exception as exc:  # noqa: BLE001 - any text could carry the URL
                raise SinkError(f"telegram unreachable: {type(exc).__name__}") from None
            if not payload.get("ok"):
                raise SinkError(f"telegram refused: {payload.get('description', 'unknown')}")
            ids.append(payload["result"]["message_id"])
        return f"message ids {ids}"
