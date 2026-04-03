import json
import os
from typing import Optional

# Try to use Redis if available, otherwise fall back to in-memory store
_client = None
_in_memory_store: dict = {}
_use_redis = False

def _init_redis():
    global _client, _use_redis
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        print("No REDIS_URL set — using in-memory session store.")
        return
    try:
        import redis
        _client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=3)
        _client.ping()
        _use_redis = True
        print(f"Connected to Redis at {redis_url[:30]}...")
    except Exception as e:
        print(f"Redis unavailable ({e}) — falling back to in-memory store.")
        _client = None
        _use_redis = False

_init_redis()

SESSION_TTL = 60 * 60 * 4  # 4 hours


def create_session(session_id: str) -> dict:
    session = {
        "id": session_id,
        "transcript": [],
        "summaries": [],
        "action_items": [],
        "decisions": [],
        "open_questions": [],
        "vectors": [],
        "active": True,
    }
    if _use_redis:
        _client.setex(f"session:{session_id}", SESSION_TTL, json.dumps(session))
    else:
        _in_memory_store[session_id] = session
    return session


def get_session(session_id: str) -> Optional[dict]:
    if _use_redis:
        data = _client.get(f"session:{session_id}")
        return json.loads(data) if data else None
    else:
        return _in_memory_store.get(session_id)


def update_session(session_id: str, session: dict):
    if _use_redis:
        _client.setex(f"session:{session_id}", SESSION_TTL, json.dumps(session))
    else:
        _in_memory_store[session_id] = session


def append_transcript(session_id: str, entry: dict):
    session = get_session(session_id)
    if not session:
        return
    session["transcript"].append(entry)
    update_session(session_id, session)


def append_summary(session_id: str, summary: dict):
    session = get_session(session_id)
    if not session:
        return
    session["summaries"].append(summary)
    update_session(session_id, session)


def append_action_item(session_id: str, item: dict):
    session = get_session(session_id)
    if not session:
        return
    session["action_items"].append(item)
    update_session(session_id, session)


def close_session(session_id: str):
    session = get_session(session_id)
    if session:
        session["active"] = False
        update_session(session_id, session)

