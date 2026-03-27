from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import re

app = FastAPI()

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def strip_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r' +', ' ', text)  # collapse multiple spaces into one
    return text.strip()

class ChatRequest(BaseModel):
    user_message: str
    chat_history: list = []

@app.get("/")
async def health_check():
    return {"status": "Brah is online", "model": "gpt-5.4-mini"}

@app.post("/chat")
async def ask_guru(request: ChatRequest):
    client = get_client()
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are Brah, a warm, empathetic relationship communication coach. "
                "You speak like a trusted friend who happens to have deep wisdom about "
                "relationships. You are non-judgmental, honest, and concise. "
                "You never use bullet points or markdown formatting. "
                "You speak in plain, warm, conversational language. "
                "Keep responses under 150 words unless the situation genuinely requires more."
            )
        }
    ]
    
    for msg in request.chat_history:
        messages.append(msg)
        
    messages.append({"role": "user", "content": request.user_message})
    
    completion = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages
    )
    
    reply = completion.choices[0].message.content
    return {"reply": strip_markdown(reply)}
