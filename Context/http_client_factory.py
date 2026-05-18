from __future__ import annotations

import httpx


def create_sync_http_client() -> httpx.Client:
    return httpx.Client(trust_env=False)


def create_async_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(trust_env=False)
