from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# This is safer: it won't crash if the key is missing on startup
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

class ChatRequest(BaseModel):
    user_message: str
    chat_history: list = []

@app.get("/")
async def health_check():
    # This lets you check if Brah is alive in your browser
    return {"status": "Brah is online", "model": "gpt-5.4-mini"}

@app.post("/chat")
async def ask_guru(request: ChatRequest):
    client = get_client()
    
    messages = [{"role": "system", "content": "You are Brah, a minimalist, posh relationship guru."}]
    
    for msg in request.chat_history:
        messages.append(msg)
        
    messages.append({"role": "user", "content": request.user_message})

    completion = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages
    )
    
    return {"reply": completion.choices[0].message.content}
