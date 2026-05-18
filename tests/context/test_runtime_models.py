import unittest

from Context.runtime_models import (
    AssemblyRecord,
    ContextBlock,
    DerivedState,
    EventRecord,
    StateSnapshot,
    ToolEvidence,
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

    def test_tool_evidence_round_trip(self):
        evidence = ToolEvidence(
            evidence_id="tool:1",
            thread_id="thread_4",
            tool_name="hybrid_search",
            content="paper result",
            preview="paper",
            sequence=7,
            metadata={"query": "radar"},
        )
        rebuilt = ToolEvidence.from_dict(evidence.to_dict())

        self.assertEqual(rebuilt.tool_name, "hybrid_search")
        self.assertEqual(rebuilt.preview, "paper")
        self.assertEqual(rebuilt.metadata["query"], "radar")


if __name__ == "__main__":
    unittest.main()
