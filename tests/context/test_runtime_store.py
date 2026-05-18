import unittest
from unittest.mock import patch

from Context.manager import RedisContextManager
from Context.runtime_models import (
    AssemblyRecord,
    DerivedState,
    EventRecord,
    ModelCallRecord,
    StateSnapshot,
    ToolEvidence,
)


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
        snapshots = manager.load_snapshots("thread_2")
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].sequence, 3)

    @patch("Context.manager.redis_client")
    def test_save_snapshot_keeps_history(self, redis_mock):
        redis_mock.get.side_effect = self.fake_redis.get
        redis_mock.set.side_effect = self.fake_redis.set
        redis_mock.delete.side_effect = self.fake_redis.delete
        manager = RedisContextManager()

        manager.save_snapshot(
            "thread_4",
            StateSnapshot(
                thread_id="thread_4",
                sequence=1,
                state=DerivedState(thread_id="thread_4", current_goal="first"),
            ),
        )
        manager.save_snapshot(
            "thread_4",
            StateSnapshot(
                thread_id="thread_4",
                sequence=2,
                state=DerivedState(thread_id="thread_4", current_goal="second"),
            ),
        )

        snapshots = manager.load_snapshots("thread_4")
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0].state.current_goal, "first")
        self.assertEqual(snapshots[1].state.current_goal, "second")
        latest = manager.load_latest_snapshot("thread_4")
        self.assertEqual(latest.sequence, 2)

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

    @patch("Context.manager.redis_client")
    def test_save_assembly_record_keeps_history(self, redis_mock):
        redis_mock.get.side_effect = self.fake_redis.get
        redis_mock.set.side_effect = self.fake_redis.set
        redis_mock.delete.side_effect = self.fake_redis.delete
        manager = RedisContextManager()

        manager.save_assembly_record(
            "thread_5",
            AssemblyRecord(
                thread_id="thread_5",
                sequence=3,
                selected_blocks=["system:agent"],
                dropped_blocks=[],
                token_budget=100,
                estimated_total_tokens=12,
                payload_preview="system",
            ),
        )
        manager.save_assembly_record(
            "thread_5",
            AssemblyRecord(
                thread_id="thread_5",
                sequence=5,
                selected_blocks=["system:agent", "goal:thread_5"],
                dropped_blocks=[{"block_id": "recent:thread_5", "reason": "budget"}],
                token_budget=100,
                estimated_total_tokens=20,
                payload_preview="system | goal",
            ),
        )

        history = manager.load_assembly_records("thread_5")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].sequence, 3)
        self.assertEqual(history[1].sequence, 5)
        self.assertEqual(manager.load_latest_assembly_record("thread_5").sequence, 5)

    @patch("Context.manager.redis_client")
    def test_save_and_load_tool_evidence_keeps_history(self, redis_mock):
        redis_mock.get.side_effect = self.fake_redis.get
        redis_mock.set.side_effect = self.fake_redis.set
        redis_mock.delete.side_effect = self.fake_redis.delete
        manager = RedisContextManager()

        manager.save_tool_evidence(
            "thread_6",
            ToolEvidence(
                evidence_id="tool:1",
                thread_id="thread_6",
                tool_name="hybrid_search",
                content="first result",
                preview="first",
                sequence=4,
            ),
        )
        manager.save_tool_evidence(
            "thread_6",
            ToolEvidence(
                evidence_id="tool:2",
                thread_id="thread_6",
                tool_name="filtered_search",
                content="second result",
                preview="second",
                sequence=8,
            ),
        )

        history = manager.load_tool_evidence("thread_6")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].tool_name, "hybrid_search")
        self.assertEqual(history[1].preview, "second")

    @patch("Context.manager.redis_client")
    def test_save_and_load_model_calls_keeps_history(self, redis_mock):
        redis_mock.get.side_effect = self.fake_redis.get
        redis_mock.set.side_effect = self.fake_redis.set
        redis_mock.delete.side_effect = self.fake_redis.delete
        manager = RedisContextManager()

        manager.save_model_call(
            "thread_7",
            ModelCallRecord(
                thread_id="thread_7",
                call_id="call_1",
                call_index=1,
                sequence=2,
                phase="planning",
                purpose="decide tool",
                input_context=[{"label": "user_request", "value": "search papers"}],
                output={"type": "tool_call", "summary": "called hybrid_search"},
            ),
        )
        manager.save_model_call(
            "thread_7",
            ModelCallRecord(
                thread_id="thread_7",
                call_id="call_2",
                call_index=2,
                sequence=6,
                phase="answering",
                purpose="answer with evidence",
                input_context=[{"label": "tool_message", "value": "results"}],
                raw_think="hidden",
                output={"type": "answer", "summary": "final answer"},
            ),
        )

        calls = manager.load_model_calls("thread_7")

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].call_index, 1)
        self.assertEqual(calls[1].raw_think, "hidden")


if __name__ == "__main__":
    unittest.main()
