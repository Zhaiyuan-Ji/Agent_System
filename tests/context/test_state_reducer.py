import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from Context.runtime_models import ToolEvidence
from Context.state_reducer import build_derived_state


class StateReducerTests(unittest.TestCase):
    def test_build_derived_state_promotes_tool_evidence_to_state_refs(self):
        state = build_derived_state(
            thread_id="thread_state",
            latest_request="find mmWave radar papers",
            context_messages=[
                HumanMessage(content="find mmWave radar papers"),
                ToolMessage(content="retrieved paper A", tool_call_id="call_1", name="hybrid_search"),
                AIMessage(content="I found initial evidence."),
            ],
            summary_text="Earlier discussion focused on radar sensing.",
            tool_evidence=[
                ToolEvidence(
                    evidence_id="evidence:1",
                    thread_id="thread_state",
                    tool_name="hybrid_search",
                    content="retrieved paper B with citation metadata",
                    preview="retrieved paper B",
                    sequence=3,
                )
            ],
        )

        self.assertEqual(state.current_goal, "find mmWave radar papers")
        self.assertIn("Earlier discussion focused on radar sensing.", state.confirmed_facts)
        self.assertTrue(any("hybrid_search" in item for item in state.active_evidence_refs))
        self.assertNotIn(
            "Need collect or verify external evidence before finalizing.",
            state.open_loops,
        )


if __name__ == "__main__":
    unittest.main()
