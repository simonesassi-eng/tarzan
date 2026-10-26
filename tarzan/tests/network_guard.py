"""Test-only rejection of unstubbed network and SMTP transport."""

from __future__ import annotations

from collections import Counter
from threading import Lock


class NetworkGuard:
    """Record channel names only, including attempts caught by product code."""

    def __init__(self):
        self._attempts = Counter()
        self._lock = Lock()

    def denied(self, channel):
        def block(*args, **kwargs):
            # Never include URLs, credentials or request arguments in failures.
            with self._lock:
                self._attempts[channel] += 1
            raise AssertionError(f"Unstubbed external transport: {channel}")

        return block

    def assert_unused(self):
        with self._lock:
            counts = dict(sorted(self._attempts.items()))
        if counts:
            raise AssertionError(f"Unstubbed external transport attempts: {counts}")

    def install(self, monkeypatch):
        import smtplib
        import socket
        import urllib.request

        import requests
        from curl_cffi.requests import Session

        for owner, name, channel in (
            (socket.socket, "connect", "socket.connect"),
            (socket.socket, "connect_ex", "socket.connect_ex"),
            (socket, "create_connection", "socket.create_connection"),
            (socket, "getaddrinfo", "socket.getaddrinfo"),
            (Session, "request", "curl.request"),
            (requests.Session, "request", "requests.request"),
            (urllib.request, "urlopen", "urllib.urlopen"),
            (smtplib, "SMTP", "smtp"),
            (smtplib, "SMTP_SSL", "smtp.ssl"),
        ):
            monkeypatch.setattr(owner, name, self.denied(channel))
