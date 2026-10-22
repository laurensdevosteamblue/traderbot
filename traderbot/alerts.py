from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger(__name__)


class Alerts:
    """Logs every message and forwards it to Telegram if a token and chat id are configured."""

    def __init__(self, token: str | None = None, chat_id: str | None = None, prefix: str = "traderbot"):
        self.token, self.chat_id, self.prefix = token, chat_id, prefix

    def send(self, message: str) -> None:
        log.info("ALERT: %s", message)
        if not (self.token and self.chat_id):
            return
        body = json.dumps({"chat_id": self.chat_id, "text": f"[{self.prefix}] {message}"}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as e:
            log.warning("telegram alert failed: %s", e)
