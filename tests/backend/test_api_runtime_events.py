import unittest
from unittest.mock import patch

from Back_end.api_server import (
    build_model_call_payloads,
    build_runtime_completion_payloads,
    build_runtime_error_payloads,
    build_runtime_phase_payload,
    build_runtime_prelude_payloads,
    build_runtime_stream_payload,
    build_tool_called_payloads,
    build_tool_result_payloads,
    ReasoningTextFilter,
)
from Context.runtime_models import AssemblyRecord, ContextBlock, DerivedState, EventRecord, StateSnapshot, ToolEvidence
from Context.runtime_models import ModelCallRecord


class FakeRuntimeContextManager:
    def __init__(self):
        self.events = []
        self.tool_evidence = []
        self.model_calls = []

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

    def save_tool_evidence(self, thread_id: str, evidence: ToolEvidence):
        self.tool_evidence.append(evidence)

    def load_model_calls(self, thread_id: str):
        return self.model_calls


class ApiRuntimeEventTests(unittest.TestCase):
    def test_reasoning_text_filter_removes_streamed_think_blocks(self):
        text_filter = ReasoningTextFilter()

        visible_chunks = [
            text_filter.push("<thi"),
            text_filter.push("nk>hidden reasoning"),
            text_filter.push("</thi"),
            text_filter.push("nk>\n\nfinal answer"),
        ]

        self.assertEqual("".join(visible_chunks), "\n\nfinal answer")

    @patch("Back_end.api_server.context_manager", new_callable=lambda: FakeRuntimeContextManager())
    def test_build_runtime_phase_payload_records_phase_change(self, context_manager_mock):
        payload = build_runtime_phase_payload("thread_0", "awaiting_tool_result")

        self.assertEqual(payload["type"], "runtime_event")
        self.assertEqual(payload["event_type"], "phase_changed")
        self.assertEqual(payload["payload"]["phase"], "awaiting_tool_result")
        self.assertEqual(context_manager_mock.events[0].event_type, "phase_changed")

    @patch("Back_end.api_server.context_manager", new_callable=lambda: FakeRuntimeContextManager())
    @patch("Back_end.api_server.refresh_runtime_artifacts")
    def test_build_runtime_prelude_payloads_emit_runtime_events_state_and_assembly(
        self, refresh_runtime_artifacts_mock, context_manager_mock
    ):
        refresh_runtime_artifacts_mock.return_value = {
            "state": DerivedState(thread_id="thread_1", current_goal="search papers"),
            "snapshot": StateSnapshot(
                thread_id="thread_1",
                sequence=3,
                state=DerivedState(thread_id="thread_1", current_goal="search papers"),
            ),
            "assembly": AssemblyRecord(
                thread_id="thread_1",
                sequence=4,
                selected_blocks=["system:agent", "goal:thread_1"],
                dropped_blocks=[],
                token_budget=120000,
                estimated_total_tokens=48,
                payload_preview="system | goal",
            ),
            "context_blocks": [
                ContextBlock(
                    block_id="system:agent",
                    block_type="system_rules",
                    title="System Rules",
                    content="rules",
                    source="system",
                    priority=100,
                    selected=True,
                    estimated_tokens=12,
                )
            ],
            "events": [
                EventRecord(
                    event_type="state_updated",
                    thread_id="thread_1",
                    payload={"current_goal": "search papers"},
                    sequence=3,
                ),
                EventRecord(
                    event_type="context_block_selected",
                    thread_id="thread_1",
                    payload={"block_id": "system:agent"},
                    sequence=4,
                ),
                EventRecord(
                    event_type="assembly_completed",
                    thread_id="thread_1",
                    payload={"selected_blocks": ["system:agent", "goal:thread_1"]},
                    sequence=5,
                ),
            ],
        }

        payloads = build_runtime_prelude_payloads("thread_1", "search papers")

        self.assertEqual(payloads[0]["type"], "runtime_event")
        self.assertEqual(payloads[0]["event_type"], "phase_changed")
        self.assertEqual(payloads[0]["sequence"], 1)
        self.assertEqual(payloads[1]["type"], "runtime_event")
        self.assertEqual(payloads[1]["event_type"], "state_updated")
        self.assertEqual(payloads[2]["type"], "runtime_event")
        self.assertEqual(payloads[2]["event_type"], "context_block_selected")
        self.assertEqual(payloads[3]["type"], "runtime_event")
        self.assertEqual(payloads[3]["event_type"], "assembly_completed")
        self.assertEqual(payloads[4]["type"], "state_update")
        self.assertEqual(payloads[4]["sequence"], 3)
        self.assertEqual(payloads[5]["type"], "context_assembly")
        self.assertEqual(payloads[5]["sequence"], 4)
        self.assertEqual(context_manager_mock.events[-1].event_type, "phase_changed")

    @patch("Back_end.api_server.context_manager", new_callable=lambda: FakeRuntimeContextManager())
    def test_build_runtime_stream_payload_emits_stream_event(self, context_manager_mock):
        payload = build_runtime_stream_payload("thread_stream", "abc", 3)

        self.assertEqual(payload["type"], "runtime_event")
        self.assertEqual(payload["event_type"], "assistant_answer_streamed")
        self.assertEqual(payload["payload"]["delta_text"], "abc")
        self.assertEqual(payload["payload"]["cumulative_chars"], 3)
        self.assertEqual(context_manager_mock.events[0].event_type, "assistant_answer_streamed")

    @patch("Back_end.api_server.context_manager", new_callable=lambda: FakeRuntimeContextManager())
    def test_build_model_call_payloads_emits_only_new_calls(self, context_manager_mock):
        context_manager_mock.model_calls = [
            ModelCallRecord(
                thread_id="thread_calls",
                call_id="call_1",
                call_index=1,
                sequence=2,
                phase="planning",
                purpose="decide tool",
                input_context=[],
                output={"type": "tool_call", "summary": "called tool"},
            ),
            ModelCallRecord(
                thread_id="thread_calls",
                call_id="call_2",
                call_index=2,
                sequence=6,
                phase="answering",
                purpose="answer",
                input_context=[],
                output={"type": "answer", "summary": "final"},
            ),
        ]
        sent = {"call_1"}

        payloads = build_model_call_payloads("thread_calls", sent)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["type"], "model_call")
        self.assertEqual(payloads[0]["model_call"]["call_id"], "call_2")
        self.assertEqual(sent, {"call_1", "call_2"})

    @patch("Back_end.api_server.context_manager", new_callable=lambda: FakeRuntimeContextManager())
    def test_build_runtime_completion_payloads_emits_completion_phase(self, context_manager_mock):
        payloads = build_runtime_completion_payloads("thread_2", "done answer")

        self.assertEqual(payloads[0]["event_type"], "assistant_answer_completed")
        self.assertEqual(payloads[1]["event_type"], "phase_changed")
        self.assertEqual(payloads[1]["payload"]["phase"], "idle")
        self.assertEqual(context_manager_mock.events[0].event_type, "assistant_answer_completed")
        self.assertEqual(context_manager_mock.events[1].event_type, "phase_changed")

    @patch("Back_end.api_server.context_manager", new_callable=lambda: FakeRuntimeContextManager())
    def test_build_runtime_error_payloads_emits_error_and_idle_phase(self, context_manager_mock):
        payloads = build_runtime_error_payloads("thread_3", "401 unauthorized")

        self.assertEqual(payloads[0]["type"], "runtime_event")
        self.assertEqual(payloads[0]["event_type"], "agent_error")
        self.assertEqual(payloads[0]["payload"]["message"], "401 unauthorized")
        self.assertEqual(payloads[1]["event_type"], "phase_changed")
        self.assertEqual(payloads[1]["payload"]["phase"], "idle")
        self.assertEqual(context_manager_mock.events[0].event_type, "agent_error")
        self.assertEqual(context_manager_mock.events[1].event_type, "phase_changed")

    @patch("Back_end.api_server.context_manager", new_callable=lambda: FakeRuntimeContextManager())
    def test_build_tool_called_payloads_emit_tool_before_phase(self, context_manager_mock):
        payloads = build_tool_called_payloads("thread_4", "hybrid_search", {"query": "mmwave"})

        self.assertEqual(payloads[0]["event_type"], "tool_called")
        self.assertEqual(payloads[0]["sequence"], 1)
        self.assertEqual(payloads[1]["event_type"], "phase_changed")
        self.assertEqual(payloads[1]["sequence"], 2)
        self.assertEqual(payloads[1]["payload"]["phase"], "awaiting_tool_result")

    @patch("Back_end.api_server.context_manager", new_callable=lambda: FakeRuntimeContextManager())
    @patch("Back_end.api_server.refresh_runtime_artifacts")
    def test_build_tool_result_payloads_emit_result_phase_evidence_and_refreshed_artifacts(
        self, refresh_runtime_artifacts_mock, context_manager_mock
    ):
        refresh_runtime_artifacts_mock.return_value = {
            "state": DerivedState(
                thread_id="thread_5",
                current_goal="search papers",
                active_evidence_refs=["hybrid_search:preview text"],
            ),
            "snapshot": StateSnapshot(
                thread_id="thread_5",
                sequence=4,
                state=DerivedState(thread_id="thread_5", current_goal="search papers"),
            ),
            "assembly": AssemblyRecord(
                thread_id="thread_5",
                sequence=6,
                selected_blocks=["system:agent", "evidence:thread_5"],
                dropped_blocks=[],
                token_budget=120000,
                estimated_total_tokens=52,
                payload_preview="system | evidence",
            ),
            "context_blocks": [
                ContextBlock(
                    block_id="evidence:thread_5",
                    block_type="tool_evidence",
                    title="Tool Evidence",
                    content="preview text",
                    source="tools",
                    priority=80,
                    selected=True,
                    estimated_tokens=12,
                )
            ],
            "events": [
                EventRecord(
                    event_type="state_updated",
                    thread_id="thread_5",
                    payload={"active_evidence_refs": 1},
                    sequence=4,
                ),
                EventRecord(
                    event_type="context_block_selected",
                    thread_id="thread_5",
                    payload={"block_id": "evidence:thread_5"},
                    sequence=5,
                ),
                EventRecord(
                    event_type="assembly_completed",
                    thread_id="thread_5",
                    payload={"selected_blocks": ["evidence:thread_5"]},
                    sequence=6,
                ),
            ],
        }

        payloads = build_tool_result_payloads(
            "thread_5",
            "hybrid_search",
            "full result text",
            "preview text",
            current_request="search papers",
        )

        self.assertEqual(payloads[0]["event_type"], "tool_result_received")
        self.assertEqual(payloads[0]["sequence"], 1)
        self.assertEqual(payloads[1]["event_type"], "phase_changed")
        self.assertEqual(payloads[1]["sequence"], 2)
        self.assertEqual(payloads[1]["payload"]["phase"], "processing_tool_result")
        self.assertEqual(payloads[2]["type"], "tool_result")
        self.assertEqual(context_manager_mock.tool_evidence[0].content, "full result text")
        self.assertEqual(payloads[3]["event_type"], "state_updated")
        self.assertEqual(payloads[-2]["type"], "state_update")
        self.assertEqual(payloads[-1]["type"], "context_assembly")
        refresh_runtime_artifacts_mock.assert_called_once_with(
            thread_id="thread_5",
            draft_message="search papers",
        )


if __name__ == "__main__":
    unittest.main()
