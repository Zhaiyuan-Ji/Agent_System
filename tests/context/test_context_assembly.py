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


if __name__ == "__main__":
    unittest.main()
