import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Agent'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

os.environ["OPENAI_BASE_URL"] = "http://localhost:54329/v1"
os.environ["OPENAI_API_KEY"] = "token-abc123"

from agent import chat

app = FastAPI(title="Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "1"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = await chat(message=request.message, thread_id=request.thread_id)

        full_response = []
        async for step in result:
            if step.get("messages"):
                last_msg = step["messages"][-1]
                if hasattr(last_msg, "content"):
                    content = last_msg.content
                    if isinstance(content, list):
                        content = str(content)
                    full_response.append(content)

        final_response = "\n".join(full_response) if all(isinstance(x, str) for x in full_response) else str(full_response)
        return {"message": final_response, "done": True}
    except Exception as e:
        return {"message": f"错误: {str(e)}", "done": True, "error": True}

@app.get("/api/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
