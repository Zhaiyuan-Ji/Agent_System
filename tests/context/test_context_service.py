import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from Context.context_service import get_prompt_context, refresh_runtime_artifacts
from Context.runtime_models import (
    AssemblyRecord,
    ContextBlock,
    DerivedState,
    EventRecord,
    StateSnapshot,
    ToolEvidence,
)


class FakeContextManager:
    def __init__(self):
        self.saved_snapshot = None
        self.saved_assembly = None
        self.saved_assemblies = []
        self.saved_blocks = None
        self.events = []

    def load_context_messages(self, thread_id: str):
        return [
            HumanMessage(content="please find papers"),
            AIMessage(content="I will search first"),
        ]

    def load_tool_evidence(self, thread_id: str):
        return [
            ToolEvidence(
                evidence_id="tool:1",
                thread_id=thread_id,
                tool_name="hybrid_search",
                content="paper result with citation metadata",
                preview="paper result",
                sequence=5,
            )
        ]

    def load_summary(self, thread_id: str):
        return "Earlier discussion focused on mmWave radar."

    def load_latest_snapshot(self, thread_id: str):
        return StateSnapshot(
            thread_id=thread_id,
            sequence=2,
            state=DerivedState(
                thread_id=thread_id,
                current_goal="please find papers",
                confirmed_facts=["The task is related to academic search"],
                active_constraints=["Need citeable results"],
            ),
        )

    def load_snapshots(self, thread_id: str):
        return [
            StateSnapshot(
                thread_id=thread_id,
                sequence=1,
                state=DerivedState(thread_id=thread_id, current_goal="old goal"),
            ),
            self.load_latest_snapshot(thread_id),
        ]

    def load_latest_assembly_record(self, thread_id: str):
        return AssemblyRecord(
            thread_id=thread_id,
            sequence=2,
            selected_blocks=["system:agent", "goal:1"],
            dropped_blocks=[{"block_id": "history:1", "reason": "budget"}],
            token_budget=120000,
            estimated_total_tokens=360,
            payload_preview="system + goal + request",
        )

    def load_assembly_records(self, thread_id: str):
        if self.saved_assemblies:
            return self.saved_assemblies
        return [
            AssemblyRecord(
                thread_id=thread_id,
                sequence=1,
                selected_blocks=["system:agent"],
                dropped_blocks=[],
                token_budget=120000,
                estimated_total_tokens=240,
                payload_preview="system",
            ),
            self.load_latest_assembly_record(thread_id),
        ]

    def load_context_blocks(self, thread_id: str):
        return [
            ContextBlock(
                block_id="goal:1",
                block_type="task_goal",
                title="Current Goal",
                content="please find papers",
                source="state",
                priority=90,
                estimated_tokens=12,
                selected=True,
            )
        ]

    def load_events(self, thread_id: str):
        if self.events:
            return self.events
        return [
            EventRecord(
                event_type="tool_called",
                thread_id=thread_id,
                payload={"tool": "hybrid_search"},
                sequence=1,
            ),
            EventRecord(
                event_type="assembly_completed",
                thread_id=thread_id,
                payload={"selected_blocks": ["system:agent", "goal:1"]},
                sequence=2,
            ),
        ]

    def load_model_calls(self, thread_id: str):
        return []

    def save_snapshot(self, thread_id: str, snapshot: StateSnapshot):
        self.saved_snapshot = snapshot

    def save_assembly_record(self, thread_id: str, record: AssemblyRecord):
        self.saved_assembly = record
        self.saved_assemblies.append(record)

    def save_context_blocks(self, thread_id: str, blocks):
        self.saved_blocks = blocks

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


class ContextServiceTests(unittest.TestCase):
    @patch("Context.context_service.context_manager", new_callable=lambda: FakeContextManager())
    def test_get_prompt_context_includes_runtime_sections(self, mocked_manager):
        data = get_prompt_context("thread_1", draft_message="please find papers")

        self.assertEqual(data["state"]["current_goal"], "please find papers")
        self.assertEqual(data["assembly"]["selected_blocks"], ["system:agent", "goal:1"])
        self.assertEqual(data["runtime_events"][0]["event_type"], "tool_called")
        self.assertEqual(data["runtime_snapshots"][0]["sequence"], 1)
        self.assertEqual(data["runtime_assemblies"][0]["sequence"], 1)
        self.assertEqual(data["runtime_assemblies"][1]["sequence"], 2)
        self.assertEqual(data["model_calls"], [])

        section_keys = [section["key"] for section in data["sections"]]
        self.assertIn("state", section_keys)
        self.assertIn("context_blocks", section_keys)

    @patch("Context.context_service.context_manager", new_callable=lambda: FakeContextManager())
    def test_refresh_runtime_artifacts_persists_state_blocks_events_and_assembly(self, mocked_manager):
        result = refresh_runtime_artifacts("thread_2", draft_message="please find papers", token_budget=80)

        self.assertEqual(result["state"].current_goal, "please find papers")
        self.assertTrue(result["state"].working_memory)
        self.assertTrue(result["state"].open_loops)
        self.assertTrue(result["state"].active_constraints)
        self.assertTrue(result["state"].active_evidence_refs)
        self.assertIsNotNone(mocked_manager.saved_snapshot)
        self.assertIsNotNone(mocked_manager.saved_assembly)
        self.assertTrue(mocked_manager.saved_blocks)
        block_types = [block.block_type for block in mocked_manager.saved_blocks]
        self.assertIn("tool_evidence", block_types)
        self.assertIn("context_block_selected", [event.event_type for event in result["events"]])
        self.assertEqual(mocked_manager.saved_assembly.selected_blocks[0], "system:agent")
        self.assertEqual(mocked_manager.saved_assemblies[0].sequence, mocked_manager.events[-1].sequence)
        self.assertEqual(mocked_manager.events[-1].event_type, "assembly_completed")


if __name__ == "__main__":
    unittest.main()
