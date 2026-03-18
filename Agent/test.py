import os
import asyncio
from pprint import pprint
import base64

os.environ["OPENAI_BASE_URL"] = "http://localhost:54329/v1"
os.environ["OPENAI_API_KEY"] = "token-abc123"

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage  # 注意：新版是 langchain_core.messages
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():

    client = MultiServerMCPClient(
        {
            "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["D:\AC\Agent_System\MCP\Hybrid_Search.py"],
            }
        }
    )

    # get tools
    tools = await client.get_tools()

    # get prompts
    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content

    agent = create_agent(
        model="gpt-5.1",
        tools=tools,
        system_prompt=prompt
    )
    from langchain.messages import HumanMessage

    config = {"configurable": {"thread_id": "1"}}

    # response = await agent.ainvoke(
    #     {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
    #     config=config
    # )
    # from pprint import pprint
    #
    # pprint(response)

    human_messages_1 = HumanMessage("信号处理方法主要有哪几种？")
    result = agent.astream(
        {"messages": [human_messages_1]},
        stream_mode="values",
        config=config
    )

    async for step in result:
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())