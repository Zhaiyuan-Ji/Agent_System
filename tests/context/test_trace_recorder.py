import unittest

from Context.runtime_models import ContextBlock, EventRecord
from Context.trace_recorder import record_context_block_events


class FakeEventStore:
    def __init__(self):
        self.events = []

    def append_event(self, thread_id: str, event: EventRecord):
        stored = EventRecord(
            event_type=event.event_type,
            thread_id=thread_id,
            payload=event.payload,
            sequence=len(self.events) + 1,
            created_at=event.created_at,
        )
        self.events.append(stored)
        return stored


class TraceRecorderTests(unittest.TestCase):
    def test_record_context_block_events_persists_selected_and_dropped_blocks(self):
        store = FakeEventStore()

        events = record_context_block_events(
            context_manager=store,
            thread_id="thread_trace",
            blocks=[
                ContextBlock(
                    block_id="goal:1",
                    block_type="task_goal",
                    title="Goal",
                    content="find papers",
                    source="state",
                    priority=90,
                    estimated_tokens=3,
                    selected=True,
                ),
                ContextBlock(
                    block_id="history:1",
                    block_type="historical_summary",
                    title="History",
                    content="old context",
                    source="summary",
                    priority=20,
                    estimated_tokens=30,
                    selected=False,
                    drop_reason="budget",
                ),
            ],
        )

        self.assertEqual([event.event_type for event in events], ["context_block_selected", "context_block_dropped"])
        self.assertEqual(store.events[0].payload["block_id"], "goal:1")
        self.assertEqual(store.events[1].payload["reason"], "budget")


if __name__ == "__main__":
    unittest.main()
