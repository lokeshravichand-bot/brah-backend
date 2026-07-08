from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
import time
import threading
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

# --- Memory service configuration --------------------------------------------
# The self-hosted memory box, reached through the private Cloudflare tunnel.
# All three come from Railway environment variables.
MEMORY_URL = os.environ.get("MEMORY_URL")                       # https://memory.brah.chat
CF_ACCESS_CLIENT_ID = os.environ.get("CF_ACCESS_CLIENT_ID")     # Cloudflare service token id
CF_ACCESS_CLIENT_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET")  # Cloudflare service token secret

# The two headers Cloudflare Access checks on every request to the memory box.
MEMORY_HEADERS = {
    "CF-Access-Client-Id": CF_ACCESS_CLIENT_ID or "",
    "CF-Access-Client-Secret": CF_ACCESS_CLIENT_SECRET or "",
    "Content-Type": "application/json",
}

# How long to wait on the memory box before giving up and carrying on without it.
SEARCH_TIMEOUT = 2    # seconds - fetching memories must be fast or we skip it
ADD_TIMEOUT = 90      # seconds - storing runs in the background AFTER the reply is
                      # already sent, so a long timeout costs the user nothing. It
                      # needs this room because memory extraction (via Mistral) can
                      # take 20-90s, especially on a cold worker.
DELETE_TIMEOUT = 120  # seconds - full account deletion runs memory (double-sweep) +
                      # Firestore + Auth on the box. It's a rare, user-initiated action
                      # and must complete fully, so we give it generous room.


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


def fetch_memories(user_id, query):
    """Ask the memory box for the most relevant memories about this user.

    Graceful degradation: if the box is unreachable, slow, or misconfigured,
    we log it and return an empty list so Brah still replies (just without
    memory that turn). A memory outage must never break the chat.
    """
    if not MEMORY_URL or not user_id:
        return []

    try:
        response = requests.post(
            f"{MEMORY_URL}/search",
            headers=MEMORY_HEADERS,
            json={"user_id": user_id, "query": query, "limit": 5},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        # The memory service returns {"result": {"results": [{"memory": "...", ...}]}}.
        # We defensively handle a couple of shapes so a small change upstream
        # doesn't silently break retrieval.
        result = data.get("result", {})
        if isinstance(result, dict):
            items = result.get("results", [])
        elif isinstance(result, list):
            items = result
        else:
            items = []

        memories = []
        for item in items:
            if isinstance(item, dict):
                text = item.get("memory") or item.get("text")
                if text:
                    memories.append(text)
            elif isinstance(item, str):
                memories.append(item)

        return memories

    except Exception as e:
        print(f"[memory] fetch degraded - serving without memory: {e}", flush=True)
        return []


def store_memory(user_id, user_message, reply):
    """Store the exchange in the memory box so Brah remembers it next time.

    Runs in a background thread launched from /chat, so storing never delays
    the user's reply. Fails silently on any error - a missed store is
    acceptable; a broken chat is not.
    """
    if not MEMORY_URL or not user_id:
        return

    try:
        print(f"[memory] store starting for user {user_id}", flush=True)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        resp = requests.post(
            f"{MEMORY_URL}/add",
            headers=MEMORY_HEADERS,
            json={"user_id": user_id, "messages": messages},
            timeout=ADD_TIMEOUT,
        )
        print(f"[memory] store finished for user {user_id} - status {resp.status_code}", flush=True)
    except Exception as e:
        print(f"[memory] store FAILED for user {user_id}: {e}", flush=True)


def build_memory_block(memories):
    """Turn a list of memory strings into a system-prompt section. Returns an
    empty string when there are no memories, so nothing is injected."""
    if not memories:
        return ""

    lines = "\n".join(f"- {m}" for m in memories)
    return (
        "\n\nHere's what you remember about this person from past conversations:\n"
        f"{lines}\n"
        "Use this naturally to inform your reply, the way a close friend would. "
        "Do NOT recite these facts back or announce that you remember them."
    )


# --- Request model -----------------------------------------------------------
class ChatRequest(BaseModel):
    user_message: str
    chat_history: list = []
    user_id: str = ""


class DeleteAccountRequest(BaseModel):
    user_id: str


# --- Base system prompt (Brah's voice) ---------------------------------------
BASE_SYSTEM_PROMPT = (
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


# --- Routes ------------------------------------------------------------------
@app.get("/")
async def health_check():
    return {"status": "Brah is online", "model": MODEL_NAME}


@app.post("/chat")
async def ask_guru(request: ChatRequest):
    # 1. Fetch relevant memories first (safe - returns [] if the box is down).
    memories = fetch_memories(request.user_id, request.user_message)

    # 2. Build the system prompt: Brah's voice + a memory block (only if any).
    system_content = BASE_SYSTEM_PROMPT + build_memory_block(memories)

    messages = [{"role": "system", "content": system_content}]

    for msg in request.chat_history:
        messages.append(msg)

    messages.append({"role": "user", "content": request.user_message})

    # 3. Get Brah's reply from Mistral.
    reply = call_mistral(messages)
    clean_reply = strip_markdown(reply)

    # 4. Store this exchange in our OWN background thread, launched here directly.
    #    We do NOT use FastAPI BackgroundTasks - on this host that handoff was
    #    getting dropped, so the store never completed. A daemon thread runs
    #    the store independently and reliably, without delaying the user's reply.
    threading.Thread(
        target=store_memory,
        args=(request.user_id, request.user_message, clean_reply),
        daemon=True,
    ).start()

    return {"reply": clean_reply}


@app.post("/delete-account")
async def delete_account(request: DeleteAccountRequest):
    """Fully delete a user across all surfaces (memory + Firestore + Auth).

    This forwards the request to the memory box's /delete-account endpoint
    through the private Cloudflare tunnel (same secured path as /search and
    /add). The box holds the Firebase key and runs the actual four-surface
    wipe; the backend just triggers it. The app's 'Delete Account' button
    calls THIS endpoint.

    Unlike memory store/fetch, deletion is NOT best-effort: if it fails, we
    return an error so the app can tell the user it didn't complete, rather
    than falsely confirming a deletion that didn't happen.
    """
    if not MEMORY_URL:
        print("[delete-account] MEMORY_URL not configured - cannot delete", flush=True)
        return {"result": "error", "error": "deletion service not configured"}

    if not request.user_id:
        return {"result": "error", "error": "user_id is required"}

    try:
        print(f"[delete-account] forwarding delete for user {request.user_id}", flush=True)
        resp = requests.post(
            f"{MEMORY_URL}/delete-account",
            headers=MEMORY_HEADERS,
            json={"user_id": request.user_id},
            timeout=DELETE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"[delete-account] result for user {request.user_id}: {data.get('result')}", flush=True)
        return data
    except Exception as e:
        print(f"[delete-account] FAILED for user {request.user_id}: {e}", flush=True)
        return {"result": "error", "user_id": request.user_id, "error": str(e)}
