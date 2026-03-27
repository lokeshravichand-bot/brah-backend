from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    user_message: str
    chat_history: list = []

@app.post("/chat")
async def ask_guru(request: ChatRequest):
    # We start with the personality
    messages = [{"role": "system", "content": "You are Brah, a minimalist, posh relationship guru. Your advice is wise and witty."}]

    # We add the old messages so Brah remembers the context
    for msg in request.chat_history:
        messages.append(msg)

    # We add the new message from the user
    messages.append({"role": "user", "content": request.user_message})

    # We ask the brain for the answer
    completion = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages
    )

    return {"reply": completion.choices[0].message.content}
