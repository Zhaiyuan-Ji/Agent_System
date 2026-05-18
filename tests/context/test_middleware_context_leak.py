import unittest
from unittest.mock import patch

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from Context.middleware import ContextInjectMiddleware, inject_context
from Context.runtime_models import DerivedState, EventRecord, StateSnapshot


class EmptyContextManager:
    def load_context_messages(self, thread_id: str):
        return []


class SummaryContextManager:
    def load_context_messages(self, thread_id: str):
        return [
            SystemMessage(id="summary:new_thread", content="historical summary"),
            HumanMessage(content="previous question"),
        ]


class ModelCallContextManager:
    def __init__(self):
        self.events = []
        self.model_calls = []

    def load_model_calls(self, thread_id: str):
        return self.model_calls

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

    def load_latest_snapshot(self, thread_id: str):
        return StateSnapshot(
            thread_id=thread_id,
            sequence=1,
            state=DerivedState(thread_id=thread_id, current_goal="search papers"),
        )

    def save_model_call(self, thread_id: str, record):
        self.model_calls.append(record)


class MiddlewareContextLeakTests(unittest.TestCase):
    def tearDown(self):
        ContextInjectMiddleware.current_thread_id = None
        ContextInjectMiddleware.current_managed_messages = []

    @patch("Context.manager.RedisContextManager", new_callable=lambda: EmptyContextManager)
    def test_inject_context_clears_stale_managed_messages_when_thread_has_no_history(
        self,
        _manager_factory,
    ):
        ContextInjectMiddleware.current_thread_id = "old_thread"
        ContextInjectMiddleware.current_managed_messages = [HumanMessage(content="old message")]

        result = inject_context.before_model({"thread_id": "new_thread"}, runtime=None)

        self.assertIsNone(result)
        self.assertIsNone(ContextInjectMiddleware.current_thread_id)
        self.assertEqual(ContextInjectMiddleware.current_managed_messages, [])

    @patch("Context.manager.RedisContextManager", new_callable=lambda: SummaryContextManager)
    def test_inject_context_converts_summary_system_message_to_non_system_context(
        self,
        _manager_factory,
    ):
        result = inject_context.before_model({"thread_id": "new_thread"}, runtime=None)

        self.assertEqual(result, {"loaded_message_count": 2})
        self.assertEqual(ContextInjectMiddleware.current_thread_id, "new_thread")
        self.assertEqual(
            [getattr(message, "type", "") for message in ContextInjectMiddleware.current_managed_messages],
            ["human", "human"],
        )
        self.assertIn("Historical Summary", ContextInjectMiddleware.current_managed_messages[0].content)

    @patch("Context.manager.RedisContextManager")
    def test_awrap_model_call_records_real_model_call(self, manager_factory):
        manager = ModelCallContextManager()
        manager_factory.return_value = manager

        async def handler(_request):
            return ModelResponse(result=[AIMessage(content="<think>hidden</think>visible answer")])

        request = ModelRequest(
            model=None,
            system_message=SystemMessage(content="system rules"),
            messages=[HumanMessage(content="search papers")],
            state={"thread_id": "thread_model"},
        )

        response = self._run_async(ContextInjectMiddleware().awrap_model_call(request, handler))

        self.assertEqual(response.model_response.result[0].content, "<think>hidden</think>visible answer")
        self.assertEqual(len(manager.model_calls), 1)
        record = manager.model_calls[0]
        self.assertEqual(record.call_index, 1)
        self.assertEqual(record.phase, "planning")
        self.assertEqual(record.raw_think, "hidden")
        self.assertEqual(record.output["answer"], "visible answer")
        self.assertEqual(record.call_id, "call_1")
        self.assertEqual(record.input_context[0]["label"], "Input: Instructions")
        self.assertEqual(record.input_context[0]["sections"][0]["label"], "system_prompt")
        self.assertEqual(record.input_context[1]["sections"][0]["label"], "user_request")
        self.assertEqual(manager.events[0].event_type, "llm_call_started")
        self.assertEqual(manager.events[1].event_type, "llm_call_completed")

    @patch("Context.manager.RedisContextManager")
    def test_awrap_model_call_marks_tool_result_call_as_answering(self, manager_factory):
        manager = ModelCallContextManager()
        manager_factory.return_value = manager

        async def handler(_request):
            return ModelResponse(result=[AIMessage(content="answer from tool result")])

        request = ModelRequest(
            model=None,
            messages=[
                HumanMessage(content="search papers"),
                ToolMessage(content="tool result", tool_call_id="tool_1", name="hybrid_search"),
            ],
            state={"thread_id": "thread_model"},
        )

        self._run_async(ContextInjectMiddleware().awrap_model_call(request, handler))

        record = manager.model_calls[0]
        self.assertEqual(record.phase, "answering")
        self.assertEqual(record.input_context[-1]["label"], "Input: Tool Evidence")
        self.assertEqual(record.input_context[-1]["sections"][0]["label"], "tool_message")

    @staticmethod
    def _run_async(coro):
        import asyncio

        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
