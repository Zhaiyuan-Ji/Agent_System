# Context Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first backend refactor slice that upgrades context handling from message-only storage to an event/state/context-block runtime model while preserving the current academic retrieval scenario.

**Architecture:** Introduce explicit runtime data structures under `Context`, store structured events and snapshots in Redis alongside existing message history, and route API/context preview assembly through a selector/assembler layer. Keep current FastAPI, LangChain, Redis, and Milvus stack intact while adding a new structured execution lane.

**Tech Stack:** Python 3.10+, standard library `unittest`, FastAPI, LangChain/LangGraph, Redis.

---

### Task 1: Add runtime model tests

**Files:**
- Create: `tests/context/test_runtime_models.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from Context.runtime_models import (
    AssemblyRecord,
    ContextBlock,
    DerivedState,
    EventRecord,
    StateSnapshot,
)


class RuntimeModelTests(unittest.TestCase):
    def test_event_record_round_trip(self):
        event = EventRecord(
            event_type="tool_called",
            thread_id="thread_1",
            payload={"tool": "hybrid_search"},
            sequence=2,
        )
        rebuilt = EventRecord.from_dict(event.to_dict())

        self.assertEqual(rebuilt.event_type, "tool_called")
        self.assertEqual(rebuilt.payload["tool"], "hybrid_search")
        self.assertEqual(rebuilt.sequence, 2)

    def test_context_block_round_trip(self):
        block = ContextBlock(
            block_id="goal:1",
            block_type="task_goal",
            title="Current goal",
            content="Find related papers",
            source="state",
            priority=90,
            freshness="active",
            estimated_tokens=12,
            selected=True,
            drop_reason=None,
        )
        rebuilt = ContextBlock.from_dict(block.to_dict())

        self.assertEqual(rebuilt.block_type, "task_goal")
        self.assertTrue(rebuilt.selected)
        self.assertEqual(rebuilt.estimated_tokens, 12)

    def test_state_snapshot_round_trip(self):
        state = DerivedState(thread_id="thread_2", current_goal="answer question")
        snapshot = StateSnapshot(thread_id="thread_2", sequence=4, state=state)
        rebuilt = StateSnapshot.from_dict(snapshot.to_dict())

        self.assertEqual(rebuilt.sequence, 4)
        self.assertEqual(rebuilt.state.current_goal, "answer question")

    def test_assembly_record_tracks_selected_and_dropped_blocks(self):
        record = AssemblyRecord(
            thread_id="thread_3",
            sequence=5,
            selected_blocks=["system:agent", "goal:1"],
            dropped_blocks=[{"block_id": "history:1", "reason": "budget"}],
            token_budget=120000,
            estimated_total_tokens=480,
            payload_preview="system + goal + question",
        )
        rebuilt = AssemblyRecord.from_dict(record.to_dict())

        self.assertEqual(rebuilt.selected_blocks, ["system:agent", "goal:1"])
        self.assertEqual(rebuilt.dropped_blocks[0]["reason"], "budget")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_runtime_models -v`
Expected: FAIL with import error because `Context.runtime_models` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create dataclasses for `EventRecord`, `ContextBlock`, `DerivedState`, `StateSnapshot`, and `AssemblyRecord`, each with `to_dict()` and `from_dict()` helpers.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_runtime_models -v`
Expected: PASS

### Task 2: Add Redis runtime storage tests

**Files:**
- Create: `tests/context/test_runtime_store.py`
- Modify: `Context/manager.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from unittest.mock import patch

from Context.manager import RedisContextManager
from Context.runtime_models import AssemblyRecord, DerivedState, EventRecord, StateSnapshot


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def setex(self, key, ttl, value):
        self.values[key] = value

    def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)


class RuntimeStoreTests(unittest.TestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()

    @patch("Context.manager.redis_client")
    def test_append_and_load_events(self, redis_mock):
        redis_mock.get.side_effect = self.fake_redis.get
        redis_mock.set.side_effect = self.fake_redis.set
        redis_mock.delete.side_effect = self.fake_redis.delete
        manager = RedisContextManager()

        event = EventRecord(
            event_type="user_message_received",
            thread_id="thread_1",
            payload={"content": "hello"},
            sequence=1,
        )
        manager.append_event("thread_1", event)
        loaded = manager.load_events("thread_1")

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].payload["content"], "hello")

    @patch("Context.manager.redis_client")
    def test_save_and_load_snapshot(self, redis_mock):
        redis_mock.get.side_effect = self.fake_redis.get
        redis_mock.set.side_effect = self.fake_redis.set
        redis_mock.delete.side_effect = self.fake_redis.delete
        manager = RedisContextManager()

        snapshot = StateSnapshot(
            thread_id="thread_2",
            sequence=3,
            state=DerivedState(thread_id="thread_2", current_goal="goal"),
        )
        manager.save_snapshot("thread_2", snapshot)
        loaded = manager.load_latest_snapshot("thread_2")

        self.assertEqual(loaded.sequence, 3)
        self.assertEqual(loaded.state.current_goal, "goal")

    @patch("Context.manager.redis_client")
    def test_save_and_load_assembly_record(self, redis_mock):
        redis_mock.get.side_effect = self.fake_redis.get
        redis_mock.set.side_effect = self.fake_redis.set
        redis_mock.delete.side_effect = self.fake_redis.delete
        manager = RedisContextManager()

        record = AssemblyRecord(
            thread_id="thread_3",
            sequence=4,
            selected_blocks=["system:agent"],
            dropped_blocks=[],
            token_budget=100,
            estimated_total_tokens=10,
            payload_preview="payload",
        )
        manager.save_assembly_record("thread_3", record)
        loaded = manager.load_latest_assembly_record("thread_3")

        self.assertEqual(loaded.sequence, 4)
        self.assertEqual(loaded.payload_preview, "payload")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_runtime_store -v`
Expected: FAIL because manager methods for runtime storage do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add Redis keys and methods for:
- `append_event()`
- `load_events()`
- `save_snapshot()`
- `load_latest_snapshot()`
- `save_assembly_record()`
- `load_latest_assembly_record()`

Keep current message-history behavior intact.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_runtime_store -v`
Expected: PASS

### Task 3: Add selector and assembler tests

**Files:**
- Create: `tests/context/test_context_assembly.py`
- Create: `Context/runtime_assembly.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from Context.runtime_assembly import assemble_context_blocks
from Context.runtime_models import ContextBlock


class ContextAssemblyTests(unittest.TestCase):
    def test_selector_prefers_high_priority_blocks_within_budget(self):
        blocks = [
            ContextBlock(
                block_id="system:agent",
                block_type="system_rules",
                title="system",
                content="rules",
                source="system",
                priority=100,
                estimated_tokens=30,
            ),
            ContextBlock(
                block_id="goal:1",
                block_type="task_goal",
                title="goal",
                content="goal",
                source="state",
                priority=90,
                estimated_tokens=25,
            ),
            ContextBlock(
                block_id="history:1",
                block_type="historical_summary",
                title="history",
                content="history",
                source="summary",
                priority=10,
                estimated_tokens=80,
            ),
        ]

        selected, dropped = assemble_context_blocks(blocks, token_budget=60)

        self.assertEqual([block.block_id for block in selected], ["system:agent", "goal:1"])
        self.assertEqual(dropped[0]["block_id"], "history:1")
        self.assertEqual(dropped[0]["reason"], "budget")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_context_assembly -v`
Expected: FAIL because selector/assembler module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a selector/assembler helper that sorts by priority descending, keeps blocks within budget, and records dropped blocks with reason `budget`.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_context_assembly -v`
Expected: PASS

### Task 4: Integrate runtime state into context preview and streaming

**Files:**
- Modify: `Context/context_service.py`
- Modify: `Back_end/api_server.py`
- Test: `tests/context/test_context_service.py`

- [ ] **Step 1: Write the failing tests**

Add tests that verify:
- context preview returns structured sections for state, context blocks, and latest assembly record when present
- stream events include structured runtime event types alongside existing text/tool events

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_context_service -v`
Expected: FAIL because preview/stream output does not expose runtime structures yet.

- [ ] **Step 3: Write minimal implementation**

Update context preview to surface:
- latest derived state
- latest assembly record
- latest runtime events summary

Update streaming to emit new structured event types without breaking current text flow.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_context_service -v`
Expected: PASS

### Task 5: Run verification sweep

**Files:**
- Modify: `README.md` only if implementation changes make current behavior descriptions incorrect

- [ ] **Step 1: Run unit tests for new runtime modules**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest tests.context.test_runtime_models tests.context.test_runtime_store tests.context.test_context_assembly tests.context.test_context_service -v`
Expected: PASS

- [ ] **Step 2: Run broader regression sweep**

Run: `D:\Anaconda\envs\jzy\python.exe -m unittest discover -v`
Expected: PASS or a clearly identified pre-existing unrelated failure

- [ ] **Step 3: Manually verify backend health import path**

Run: `D:\Anaconda\envs\jzy\python.exe -c "from Back_end.api_server import app; print(app.title)"`
Expected: prints `Agent System API`
