from __future__ import annotations

import redis

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)
