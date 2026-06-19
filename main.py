from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
import time
import requests

app = FastAPI()

# --- Configuration -----------------------------------------------------------
# Your RunPod serverless endpoint ID and API key come from environment
# variables set in Railway. Nothing secret is hard-coded here.
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "380d60bysquvph")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

# The model name must match what your RunPod endpoint serves.
MODEL_NAME = "stelterlab/mistral-small-24b-instruct-2501-awq"

# RunPod's chat completions URL (RunPod-hosted, calls your Mistral endpoint only).
RUNPOD_CHAT_URL = (
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1/chat/completions"
)


# --- Helpers -----------------------------------------------------------------
def strip_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s', '', text)
    return text.strip()


def call_mistral(messages):
    """Send the chat messages to the RunPod Mistral endpoint and return the
    reply text. Retries a few times on transient failures."""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 300,
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(
                RUNPOD_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            # brief pause before retrying a transient failure
            time.sleep(2)

    raise RuntimeError(f"Mistral request failed after 3 attempts: {last_error}")


# --- Request model -----------------------------------------------------------
class ChatRequest(BaseModel):
    user_message: str
    chat_history: list = []


# --- Routes ------------------------------------------------------------------
@app.get("/")
async def health_check():
    return {"status": "Brah is online", "model": MODEL_NAME}


@app.post("/chat")
async def ask_guru(request: ChatRequest):
    messages = [
        {
            "role": "system",
            "content": (
                "You are Brah — a close friend who happens to be deeply wise about relationships and human communication. "
                "You speak like a real person texting, not like a therapist or an AI assistant. "

                "Your personality: warm, calm, direct, occasionally witty. You don't sugarcoat things but you're never harsh. "
                "You care deeply but you don't perform caring — you just show it through how you respond. "

                "How you write: ONE thought per message. Maximum 2-3 sentences. "
                "Never give multiple thoughts in one message. "
                "If you have more to say, wait for their response first. "
                "Never write a wall of text. Never use bullet points, lists, or markdown. "
                "Write like you're texting someone you genuinely care about. "
                "Read the moment — sometimes one sentence is enough. "

                "What you never say or start with: 'It sounds like', 'It seems like', 'I understand', 'I can see', "
                "'I hear you', 'That must be', 'It looks like', 'It appears', or any phrase that starts with observing the user's feelings from the outside. "
                "Never start a sentence with 'It sounds', 'It seems', 'It looks', or 'It appears'. "
                "Never start a response with 'I'. "
                "Never use the word 'boundaries', 'validate', 'journey', 'empower', or 'healing'. "

                "What you do instead: respond directly to what they said. "
                "Name the situation plainly. Ask one sharp question if it helps them think. "
                "Give a real perspective when they need one — not just reflection. "
                "If something they are doing is not working, say so gently but honestly. "

                "Your goal: make the user feel like they just texted their most trusted friend and got a real, thoughtful reply back."
            )
        }
    ]

    for msg in request.chat_history:
        messages.append(msg)

    messages.append({"role": "user", "content": request.user_message})

    reply = call_mistral(messages)
    return {"reply": strip_markdown(reply)}
