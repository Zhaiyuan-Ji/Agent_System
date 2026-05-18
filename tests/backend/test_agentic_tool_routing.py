import unittest
from unittest.mock import patch

import Back_end.api_server as api_server
from Agent.agent import create_chat_agent


class AgenticToolRoutingTests(unittest.TestCase):
    def test_api_server_does_not_prefetch_hybrid_search(self):
        self.assertFalse(hasattr(api_server, "should_prefetch_hybrid_search"))
        self.assertFalse(hasattr(api_server, "build_prefetch_evidence_message"))
        self.assertFalse(hasattr(api_server, "hybrid_search"))

    @patch("Agent.agent.create_agent")
    @patch("Agent.agent.init_chat_model")
    def test_minimax_model_init_does_not_send_reasoning_split_extra_body(
        self,
        init_chat_model_mock,
        create_agent_mock,
    ):
        create_chat_agent()

        init_kwargs = init_chat_model_mock.call_args.kwargs
        self.assertNotIn("extra_body", init_kwargs)


if __name__ == "__main__":
    unittest.main()
