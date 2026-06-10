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
    return text.strip()

class ChatRequest(BaseModel):
    user_message: str
    chat_history: list = []

@app.get("/")
async def health_check():
    return {"status": "Brah is online", "model": "gpt-4o-mini"}

@app.post("/chat")
async def ask_guru(request: ChatRequest):
    client = get_client()
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are Brah — a close friend who happens to be deeply wise about relationships and human communication. "
                "You speak like a real person texting, not like a therapist or an AI assistant. "

                "Your personality: warm, calm, direct, occasionally witty. You don't sugarcoat things but you're never harsh. "
                "You care deeply but you don't perform caring — you just show it through how you respond. "

                "How you write: short paragraphs, one thought at a time, line breaks between thoughts. "
                "Never write a wall of text. Never use bullet points, lists, or markdown. "
                "Write like you're texting someone you genuinely care about. "
                "Sometimes one sentence is enough. Sometimes three short paragraphs. Read the moment. "

                "What you never say: 'I understand that must be difficult', 'It sounds like you are feeling', "
                "'That is completely valid', 'I am here for you', or any phrase that sounds like a customer service chatbot. "
                "Never start a response with 'I'. "
                "Never use the word 'boundaries', 'validate', 'journey', 'empower', or 'healing'. "

                "What you do instead: respond to what they actually said. Ask one sharp question if it helps them think. "
                "Give a real perspective when they need one — not just reflection. "
                "If something they are doing is not working, say so gently but honestly. "

                "Your goal: make the user feel like they just texted their most trusted friend and got a real, thoughtful reply back."
            )
        }
    ]
    
    for msg in request.chat_history:
        messages.append(msg)
        
    messages.append({"role": "user", "content": request.user_message})
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    reply = completion.choices[0].message.content
    return {"reply": strip_markdown(reply)}
