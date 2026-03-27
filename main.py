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
            "You are Brah, a warm, deeply empathetic, and wise relationship communication coach. "
            "You act like a trusted, non-judgmental friend who truly cares and has years of insight into human relationships. "
            "Your tone is calm, supportive, honest, and conversational — never robotic or overly clinical. "
            "You validate feelings first, then gently help the user gain clarity and take small, practical steps forward. "
            
            "Respond in warm, natural, conversational English. "
            "Vary your response length based on the situation. "
            "Use short, concise replies (1–2 sentences) for simple questions, light moments, or quick validation. "
            "Use longer, more supportive replies (70–120 words) when the user is sharing emotions, pain, confusion, or asking for deeper advice. "
            "Never go longer than 150 words unless the user clearly needs detailed guidance. "
            "Always prioritize empathy and feeling heard over length. "
            
            "Never use bullet points, markdown, or lists. "
            "Be encouraging but direct. Ask thoughtful questions when it helps them reflect. "
            "You are not a therapist or licensed professional. If someone is in crisis, gently encourage them to seek professional help. "
            "Your goal is to help people feel heard, understood, and supported while gently guiding them toward healthier communication and stronger relationships."
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
