from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import your existing support bot setup
from main import support_bot

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

class ChatHistory(BaseModel):
    history: List[dict]

@app.post("/api/chat")
async def chat(chat_message: ChatMessage):
    response = support_bot.query(chat_message.message)
    return {"response": str(response)}

@app.post("/api/reset")
async def reset_chat():
    support_bot.reset_chat()
    return {"status": "success"}

@app.get("/api/history")
async def get_history():
    return {"history": support_bot.chat_history} 