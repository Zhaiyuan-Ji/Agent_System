import unittest

from langchain_core.messages import HumanMessage

from Context.context_selector import build_context_blocks
from Context.runtime_models import ToolEvidence


class ContextSelectorTests(unittest.TestCase):
    def test_build_context_blocks_exposes_teaching_friendly_block_types(self):
        blocks = build_context_blocks(
            thread_id="thread_selector",
            draft_message="find cited papers",
            context_messages=[HumanMessage(content="older request")],
            summary_text="User is researching mmWave radar.",
            tool_evidence=[
                ToolEvidence(
                    evidence_id="evidence:1",
                    thread_id="thread_selector",
                    tool_name="hybrid_search",
                    content="paper result with citation metadata",
                    preview="paper result",
                    sequence=2,
                )
            ],
        )

        block_types = [block.block_type for block in blocks]

        self.assertIn("system_rules", block_types)
        self.assertIn("task_goal", block_types)
        self.assertIn("current_user_request", block_types)
        self.assertIn("recent_dialogue", block_types)
        self.assertIn("historical_summary", block_types)
        self.assertIn("tool_evidence", block_types)
        self.assertEqual(
            next(block for block in blocks if block.block_type == "task_goal").content,
            "find cited papers",
        )


if __name__ == "__main__":
    unittest.main()
