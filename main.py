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
DELETE_TIMEOUT = 30   # seconds - /delete-account now returns IMMEDIATELY (the box
                      # runs the wipe in a background thread), so this only needs to
                      # cover the round trip, not the deletion itself.
STATUS_TIMEOUT = 20   # seconds - the status check reads the four surfaces directly.


# --- Helpers -----------------------------------------------------------------
def strip_markdown(text):
    # Fix tokenizer artifacts: Ġ (U+0120) is the model's marker for a leading
    # space, Ċ (U+010A) for a newline. They sometimes leak through undecoded and
    # show up as literal glyphs in replies. Convert them back to real whitespace.
    text = text.replace("\u0120", " ").replace("\u010A", "\n")
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s', '', text)
    text = re.sub(r' {2,}', ' ', text)   # collapse any double spaces created
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
# --- CRISIS / SELF-HARM DETECTION --------------------------------------------
# Deterministic, runs BEFORE Mistral, cannot be bypassed by the model.
# Philosophy: catch explicit self-referential crisis language. Tuned
# CONSERVATIVE - a false positive (showing the resource when not needed) is
# survivable; a false negative (missing a real crisis) is not. This is a FLOOR,
# not a ceiling: it catches explicit phrasings, not oblique/implicit ones. A
# smarter classifier layer is a planned future hardening step.
#
# Legal basis: CA SB 243 (and similar) require a protocol that refers at-risk
# users to crisis services when they express suicidal ideation / self-harm.
# This must be deterministic - it must NOT depend on the LLM choosing to comply.

# Tier 1: explicit self-referential crisis phrases -> FIRE.
# These are matched as normalized substrings (see _normalize).
CRISIS_PHRASES = [
    # direct intent
    "kill myself", "killing myself", "kill my self", "kms",
    "end my life", "ending my life", "end it all", "ending it all",
    "want to die", "wanna die", "want to be dead", "wanna be dead",
    "dont want to live", "do not want to live",
    "dont want to be alive", "do not want to be alive",
    "dont want to be here anymore", "do not want to be here anymore",
    "dont want to exist", "do not want to exist",
    "take my own life", "taking my own life", "taking my life",
    "commit suicide", "committing suicide", "suicidal", "suicide",
    # hopelessness + self-removal
    "better off without me", "world would be better without me",
    "everyone would be better off without me",
    "everyone would be better off if i",
    "no reason to live", "no reason to go on", "nothing to live for",
    "cant go on anymore", "cannot go on anymore",
    "cant do this anymore", "cannot do this anymore",
    "theres no point anymore", "there is no point anymore",
    # self-harm
    "hurt myself", "want to hurt myself", "harm myself", "want to harm myself",
    "cut myself", "cutting myself",
    # method / plan seeking (highest urgency)
    "how to kill myself", "how to end my life", "ways to die",
    "how to commit suicide", "painless way to die", "best way to kill myself",
]

# Tier 2: idioms that must NOT fire even though they contain trigger words.
# The Tier-1 phrases above are specific enough that most idioms ("killing me",
# "dying to see it") never match them. This list is a documented reference of
# what we intentionally let through.
KNOWN_SAFE_IDIOMS = [
    "killing me", "this is killing me", "back is killing me",
    "dying to", "dying of laughter", "dead tired", "dead serious",
    "drop dead", "kill for a", "kill it", "killed it", "die hard",
    "sudden death", "to die for",
]


def _normalize(text):
    """Lowercase, remove apostrophes (so don't -> dont, can't -> cant), then
    turn remaining punctuation into spaces, collapse whitespace. So
    'Kill  myself!!!', "don't want to live", and 'kill myself' all match."""
    t = text.lower()
    t = t.replace("'", "").replace("\u2019", "")  # apostrophes -> nothing
    t = re.sub(r"[^a-z0-9\s]", " ", t)            # other punctuation -> space
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_crisis_message(text):
    """Return True if the message contains explicit self-referential crisis
    language. Deterministic substring match on the normalized text."""
    if not text:
        return False
    norm = _normalize(text)
    padded = " " + norm + " "
    for phrase in CRISIS_PHRASES:
        # word-boundary-ish: surround with spaces so 'suicide' doesn't match
        # inside another word, but 'kill myself' still matches mid-sentence.
        if (" " + phrase + " ") in padded:
            return True
    return False


# The fixed crisis response. Mistral does NOT write this. It is returned as-is.
# The frontend (FlutterFlow) renders the resource block specially (tappable).
# `country` is an ISO-ish code passed from the app (e.g. "US"). Default: non-US.
CRISIS_MESSAGE_US = (
    "I might be reading this wrong, but the way you said that, I want to stop "
    "and make sure you're okay.\n\n"
    "You don't have to carry this by yourself, and there are people trained for "
    "exactly this moment who you can reach right now:\n\n"
    "Call 988 - Suicide & Crisis Lifeline\n"
    "Text 988\n"
    "Free, confidential, 24/7.\n\n"
    "I'm still right here with you too."
)

CRISIS_MESSAGE_INTL = (
    "I might be reading this wrong, but the way you said that, I want to stop "
    "and make sure you're okay.\n\n"
    "You don't have to carry this by yourself. There are people trained for "
    "exactly this moment, and you can find a free, confidential crisis line for "
    "your country here:\n\n"
    "findahelpline.com\n\n"
    "I'm still right here with you too."
)


def get_crisis_response(country):
    """Return the fixed crisis message for the user's country."""
    if country and country.strip().upper() == "US":
        return CRISIS_MESSAGE_US
    return CRISIS_MESSAGE_INTL


class ChatRequest(BaseModel):
    user_message: str
    chat_history: list = []
    user_id: str = ""
    country: str = ""   # device region code from the app (e.g. "US"); used only
                        # to pick the crisis resource. Optional; defaults to intl.


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
    # 0. CRISIS CHECK - runs FIRST, before memory, before Mistral.
    #    If the message expresses self-harm / suicidal intent, we short-circuit:
    #    return the fixed crisis resource and DO NOT call Mistral at all. This is
    #    deterministic and cannot be overridden by the model. `is_crisis: true`
    #    tells the app to render this message specially (tappable helpline).
    if is_crisis_message(request.user_message):
        print(f"[crisis] triggered for user {request.user_id or 'anon'}", flush=True)
        return {
            "reply": get_crisis_response(request.country),
            "is_crisis": True,
        }

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
    """START a full account deletion. Returns IMMEDIATELY.

    Why immediately: FlutterFlow has a hard ~30-60s request timeout that cannot
    be raised. A real user's deletion (memory + every chat's messages + profile
    + auth) can take well over a minute, so holding the connection open would
    time out on the client even though the deletion is succeeding. Instead the
    box starts the wipe in a background thread and answers right away; the app
    then polls /delete-account-status for live progress.

    Response: {"result": "deletion_started", "user_id": "..."}
    The app must NOT treat this as "deleted" - it means "accepted and running".
    Completion is confirmed only by /delete-account-status returning
    status == "completed".
    """
    if not MEMORY_URL:
        print("[delete-account] MEMORY_URL not configured - cannot delete", flush=True)
        return {"result": "error", "error": "deletion service not configured"}

    if not request.user_id:
        return {"result": "error", "error": "user_id is required"}

    try:
        print(f"[delete-account] starting deletion for user {request.user_id}", flush=True)
        resp = requests.post(
            f"{MEMORY_URL}/delete-account",
            headers=MEMORY_HEADERS,
            json={"user_id": request.user_id},
            timeout=DELETE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"[delete-account] accepted for user {request.user_id}: {data.get('result')}", flush=True)
        return data
    except Exception as e:
        print(f"[delete-account] FAILED to start for user {request.user_id}: {e}", flush=True)
        return {"result": "error", "user_id": request.user_id, "error": str(e)}


@app.post("/delete-account-status")
async def delete_account_status(request: DeleteAccountRequest):
    """Report live progress of an in-flight account deletion.

    The box answers by CHECKING THE ACTUAL FOUR SURFACES (memory vectors,
    chat docs, the user profile doc, the auth record) rather than reading a
    progress flag. That means it cannot report false progress, and it stays
    correct even if a service restarted mid-deletion.

    Response shape:
      {
        "user_id": "...",
        "status": "in_progress" | "completed",
        "stage": "memory" | "chats" | "profile" | "auth" | "done",
        "completed": ["memory", "chats", ...],
        "progress": 0-100,
        "surfaces": {"memory": N, "chats": N, "user_doc": bool, "auth": bool}
      }

    The app should poll this every few seconds and drive its progress UI from
    `progress` / `stage` / `completed`, and only sign the user out and show the
    'account deleted' screen when status == "completed".
    """
    if not MEMORY_URL:
        return {"status": "error", "error": "deletion service not configured"}

    if not request.user_id:
        return {"status": "error", "error": "user_id is required"}

    try:
        resp = requests.post(
            f"{MEMORY_URL}/delete-account-status",
            headers=MEMORY_HEADERS,
            json={"user_id": request.user_id},
            timeout=STATUS_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # A failed status check is NOT a failed deletion - the wipe may still be
        # running. Report the check failure so the app keeps polling rather than
        # falsely telling the user the deletion failed.
        print(f"[delete-account-status] check failed for user {request.user_id}: {e}", flush=True)
        return {"status": "check_failed", "user_id": request.user_id, "error": str(e)}
