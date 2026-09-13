# brain/gpt_llm.py
# Nova's brain: talks to a local LLM (Ollama, OpenAI-compatible) or OpenAI,
# runs the function-calling loop, and keeps conversation memory. Replies stream
# out a sentence at a time so Nova can start speaking before the model is done.

import re
import threading

from openai import OpenAI

from config import (
    ASSISTANT_NAME,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    LOCAL_LLM_MODEL,
    OPENAI_LLM_MODEL,
    OPENAI_API_KEY,
    MEMORY_PATH,
    MEMORY_MAX_TURNS,
    MEMORY_SUMMARIZE_AFTER,
    THINK_FOR_COMMANDS,
    FAST_COMMANDS,
)
from brain.tools import TOOLS, dispatch_tool
from brain.memory import ConversationMemory
from brain.fastpath import route_command

MAX_TOOL_ROUNDS = 5

# qwen3 and other reasoning models emit <think>...</think> blocks. Nova must
# never speak its internal monologue aloud.
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Replies are spoken aloud and the user wants zero emoji/pictographs. Don't trust
# the model to obey the prompt — strip them from the spoken text as well.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoji (incl. 😊 U+1F60A), supplemental
    "\U00002600-\U000026ff"  # miscellaneous symbols
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicator (flag) letters
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U00002190-\U000021ff"  # arrows
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"             # zero-width joiner
    "\U00002022\U000025aa\U000025fe"  # bullets the model sometimes adds
    "]+",
    flags=re.UNICODE,
)


def _speakable(text: str) -> str:
    """Plain spoken text: no emoji, no markdown symbols, single-spaced."""
    text = _EMOJI_RE.sub("", text or "")
    text = re.sub(r"[*#`]+", "", text)
    text = re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)  # list markers
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # "hi 😊." -> "hi ." -> "hi."
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Streaming: turn the model's token stream into speakable sentences the moment
# each one is complete.
# ---------------------------------------------------------------------------
_BOUNDARY_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s)|\n+")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "jr", "sr"}
_MIN_SENTENCE = 30   # short bits ("Hey there!") ride along with the next sentence: smoother, ~50 ms later
_MAX_SENTENCE = 220  # force a break in long run-ons so speech keeps flowing


def _ends_with_abbreviation(before: str) -> bool:
    """True if the '.' after `before` belongs to "Dr." / "p.m." / an initial."""
    m = re.search(r"([A-Za-z.]+)$", before)
    if not m:
        return False
    word = m.group(1).lower().strip(".")
    return word in _ABBREVIATIONS or bool(re.fullmatch(r"(?:[a-z]\.)*[a-z]", word))


def _partial_tag(text: str, tag: str) -> int:
    """Length of the longest prefix of `tag` that `text` ends with."""
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class SentenceStream:
    """Feed raw model text deltas; get back complete, speakable sentences.
    Hides <think> blocks (even when a tag is split across deltas) and strips
    emoji/markdown."""

    def __init__(self):
        self._raw = ""        # undecided text (may end in half a tag)
        self._buf = ""        # visible text not yet emitted
        self._in_think = False

    def feed(self, delta: str) -> list:
        self._raw += delta or ""
        self._hide_thinking()
        return self._drain(final=False)

    def flush(self) -> list:
        if not self._in_think:  # an unclosed <think> is never spoken
            self._buf += self._raw
        self._raw, self._in_think = "", False
        return self._drain(final=True)

    def _hide_thinking(self) -> None:
        while self._raw:
            low = self._raw.lower()
            if self._in_think:
                end = low.find(_THINK_CLOSE)
                if end < 0:  # still thinking: keep only a possible partial "</think>"
                    self._raw = self._raw[-(len(_THINK_CLOSE) - 1):]
                    return
                self._raw = self._raw[end + len(_THINK_CLOSE):]
                self._in_think = False
            else:
                start = low.find(_THINK_OPEN)
                if start >= 0:
                    self._buf += self._raw[:start]
                    self._raw = self._raw[start + len(_THINK_OPEN):]
                    self._in_think = True
                    continue
                hold = _partial_tag(low, _THINK_OPEN)
                self._buf += self._raw[:len(self._raw) - hold]
                self._raw = self._raw[len(self._raw) - hold:]
                return

    def _next_cut(self):
        buf = self._buf
        for m in _BOUNDARY_RE.finditer(buf):
            if m.group(0)[0] == "." and _ends_with_abbreviation(buf[:m.start()]):
                continue
            if len(_speakable(buf[:m.end()])) >= _MIN_SENTENCE:
                return m.end()
        if len(buf) > _MAX_SENTENCE:
            window = buf[:_MAX_SENTENCE]
            for sep in (", ", "; ", ": ", " "):
                i = window.rfind(sep)
                if i > _MIN_SENTENCE:
                    return i + len(sep)
            return _MAX_SENTENCE
        return None

    def _drain(self, final: bool) -> list:
        out = []
        while True:
            cut = self._next_cut()
            if cut is None:
                break
            piece, self._buf = self._buf[:cut], self._buf[cut:]
            if _speakable(piece):
                out.append(_speakable(piece))
        if final:
            if _speakable(self._buf):
                out.append(_speakable(self._buf))
            self._buf = ""
        return out


def _merge_tool_call(calls: list, tc) -> None:
    """Accumulate one streamed tool-call delta. OpenAI sends a call's id + name
    first and its arguments in later fragments (matched by index); Ollama sends
    each call whole in a single delta."""
    index = getattr(tc, "index", None)
    slot = None
    if tc.id:
        slot = next((c for c in calls if c["id"] == tc.id), None)
        if slot is None:
            slot = {"id": tc.id, "index": index, "name": "", "arguments": ""}
            calls.append(slot)
    elif index is not None:
        slot = next((c for c in reversed(calls) if c["index"] == index), None)
    if slot is None:
        if not calls:
            calls.append({"id": "", "index": index, "name": "", "arguments": ""})
        slot = calls[-1]
    fn = getattr(tc, "function", None)
    if fn is not None:
        if fn.name:
            slot["name"] = fn.name
        if fn.arguments:
            slot["arguments"] += fn.arguments


SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME} (Natural-language Oriented Voice Assistant), a voice "
    "assistant for the user's Windows PC with a sharp, witty, playfully sarcastic "
    "personality — a clever friend who teases and roasts but always delivers. Lead with "
    "being genuinely helpful and concise; land a dry quip or a little sass when it fits, "
    "never at the expense of actually answering. Don't be mean-spirited or overdo it.\n"
    "Your replies are spoken aloud, so keep them short and conversational. Write the way "
    "you'd say it out loud: plain sentences only. Never use markdown, bullet points, code "
    "blocks, headings, emoji, or emoticons of any kind.\n"
    "You can control the PC with the provided tools: open websites and applications, "
    "search the web, tell the time, control media, and power actions.\n"
    "CRITICAL — NEVER FAKE AN ACTION: when the user asks you to open an app or website, "
    "search, control media, check stats, or anything a tool can do, you MUST call that "
    "tool. Do NOT reply that you opened or did something unless you actually called the "
    "tool to do it. Saying 'opening Steam' or 'notepad is open' without calling the tool "
    "is a serious failure. If no tool fits, say so — don't pretend.\n"
    "CRITICAL: For the current time or date, you MUST call the get_current_time tool and "
    "use its result exactly — never state a time or date from memory, prior turns, or a "
    "guess, because those are wrong. Likewise, use the web search tool for facts you are "
    "not certain of instead of making them up.\n"
    "Speech recognition can be imperfect. If a request seems garbled or ambiguous, ask a "
    "brief clarifying question instead of guessing."
)

# ---------------------------------------------------------------------------
# Reliability toggle: qwen3 calls tools far more reliably when it thinks first,
# but thinking is slow. So command-like utterances (open X, remind me, play,
# search...) get thinking and plain chat doesn't. Controlled by
# THINK_FOR_COMMANDS in config.
# NOTE: qwen3's "/no_think" soft switch inside the message is IGNORED by current
# Ollama (0.30.9 still streamed ~1000 chars of hidden reasoning with it). The
# switch that actually works is the request option reasoning_effort="none".
# ---------------------------------------------------------------------------
_NO_THINKING = {"reasoning_effort": "none"}
_COMMAND_RE = re.compile(
    r"\b(open|launch|start|run|close|quit|play|pause|resume|skip|next|previous|"
    r"volume|mute|unmute|search|google|look up|remind|reminder|timer|alarm|"
    r"switch (to|model)|use qwen|use gemma|shut ?down|restart|lock|sign out|"
    r"note|notes|type|dictate|roast|vibe|sass|time|date|screenshot)\b",
    re.IGNORECASE,
)


def _is_command(text: str) -> bool:
    return bool(_COMMAND_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Ollama liveness: after a reboot/power-cut the server often isn't running.
# Rather than fail with "Connection error", detect it and start it ourselves.
# ---------------------------------------------------------------------------
def _looks_like_conn_error(err) -> bool:
    s = str(err).lower()
    return any(k in s for k in ("connection", "refused", "max retries",
                                "failed to establish", "connect", "timed out"))


def _ollama_tags_url() -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    return base.rstrip("/") + "/api/tags"


def _ollama_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(_ollama_tags_url(), timeout=2)
        return True
    except Exception:
        return False


def ensure_ollama(verbose: bool = True) -> bool:
    """Make sure the local Ollama server is running; launch it if not (Windows)."""
    import os
    import shutil
    import subprocess
    import time

    if _ollama_up():
        return True
    exe = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    if not os.path.exists(exe):
        exe = shutil.which("ollama")
    if not exe:
        return False
    if verbose:
        print("[brain] Ollama isn't running — starting it...")
    try:
        subprocess.Popen([exe, "serve"],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return False
    for _ in range(15):
        time.sleep(1)
        if _ollama_up():
            return True
    return False


class Brain:
    def __init__(self):
        if LLM_BACKEND == "ollama":
            # Ollama exposes an OpenAI-compatible API; api_key is ignored but required.
            self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            self.model = LOCAL_LLM_MODEL
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_LLM_MODEL

        # Model tags Ollama reports as not supporting tool-calling (e.g. gemma3).
        # For these we fall back to plain chat instead of erroring with a 400.
        self._no_tools = set()

        self._lock = threading.Lock()     # one reply at a time keeps memory in order
        self._cancel = threading.Event()  # set by cancel() when the user barges in

        self.memory = ConversationMemory(
            path=MEMORY_PATH,
            system_prompt=SYSTEM_PROMPT,
            max_turns=MEMORY_MAX_TURNS,
            summarize_after=MEMORY_SUMMARIZE_AFTER,
            summarize_fn=self._summarize,
        )

    def _thinking_args(self, think: bool) -> dict:
        """Request options that switch thinking off. Ollama only: OpenAI's
        non-reasoning models reject reasoning_effort."""
        if think or LLM_BACKEND != "ollama":
            return {}
        return {"extra_body": _NO_THINKING}

    def _summarize(self, messages):
        """Condense old turns into a factual paragraph (no tools, no thinking)."""
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = self.client.chat.completions.create(
            model=self.model,
            **self._thinking_args(think=False),
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this conversation into a short factual paragraph, "
                    "capturing the user's preferences, facts about them, and any context "
                    "worth remembering later. Be concise.",
                },
                {"role": "user", "content": convo},
            ],
        )
        return _THINK_RE.sub("", resp.choices[0].message.content or "").strip()

    def _summarize_in_background(self) -> None:
        """Fold old turns into the summary without holding up Nova's voice."""
        if len(self.memory.messages) <= self.memory.summarize_after:
            return

        def run():
            with self._lock:
                self.memory.maybe_summarize()
                self.memory.save()

        threading.Thread(target=run, name="nova-memory", daemon=True).start()

    def cancel(self) -> None:
        """Stop the reply being generated (barge-in). Safe from any thread."""
        self._cancel.set()

    def ask(self, user_text: str) -> str:
        """The whole reply at once, for callers that don't stream."""
        return " ".join(self.ask_stream(user_text))

    def ask_stream(self, user_text: str):
        """Yield the reply a sentence at a time, as the model writes it. What was
        said (the whole reply, or the part before a barge-in) goes to memory."""
        with self._lock:
            self._cancel.clear()

            # Everyday commands skip the LLM entirely: instant, and always executed.
            fast = route_command(user_text) if FAST_COMMANDS else None
            if fast is not None:
                reply, remember = fast
                if remember:
                    self.memory.add_user(user_text)
                    self.memory.add_assistant(reply)
                    self.memory.save()
                yield reply
                return

            self.memory.add_user(user_text)
            # Think before command-like requests (reliable tool calls); plain chat
            # skips thinking and answers fast.
            think = THINK_FOR_COMMANDS and _is_command(user_text)

            spoken, completed = [], False
            try:
                for attempt in range(2):
                    # Rebuild fresh each attempt (memory only holds the user turn on retry).
                    working = self.memory.build_messages()
                    try:
                        for sentence in self._complete_stream(working, think):
                            spoken.append(sentence)
                            yield sentence
                        completed = not self._cancel.is_set()
                        return
                    except Exception as e:
                        if spoken:  # already talking; can't restart cleanly
                            print(f"[brain] reply broke off: {e}")
                            return
                        # If Ollama simply wasn't running, start it and retry once.
                        if (attempt == 0 and LLM_BACKEND == "ollama"
                                and _looks_like_conn_error(e) and ensure_ollama()):
                            continue
                        if LLM_BACKEND == "ollama":
                            yield (f"(Brain error: {e}) — couldn't reach Ollama. Open the Ollama "
                                   f"app or run 'ollama serve', and make sure '{self.model}' is pulled.")
                        else:
                            yield f"(Brain error: {e})"
                        return
            finally:
                if spoken:
                    self.memory.add_assistant(" ".join(spoken))
                    self.memory.save()
                    if completed:
                        self._summarize_in_background()

    def _complete_stream(self, working, think: bool = True):
        """The tool-call loop, streamed: yields speakable sentences from every
        round. Raises on API errors so ask_stream() can recover."""
        use_tools = self.model not in self._no_tools
        for _ in range(MAX_TOOL_ROUNDS):
            if self._cancel.is_set():
                return
            kwargs = dict(model=self.model, messages=working, temperature=0.3, stream=True,
                          **self._thinking_args(think))
            if use_tools:
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = "auto"
            try:
                stream = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                # Some models (e.g. gemma3) don't support tool-calling — degrade
                # to plain chat rather than crashing the whole reply.
                if use_tools and "does not support tools" in str(e).lower():
                    self._no_tools.add(self.model)
                    use_tools = False
                    print(f"[brain] '{self.model}' can't call tools — replying without "
                          "them (open/search/time commands need a tool-capable model "
                          "like qwen3).")
                    continue
                raise

            sentences = SentenceStream()
            content, calls = [], []
            try:
                for chunk in stream:
                    if self._cancel.is_set():  # barge-in: stop generating right now
                        return
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content.append(delta.content)
                        yield from sentences.feed(delta.content)
                    for tc in delta.tool_calls or []:
                        _merge_tool_call(calls, tc)
            finally:
                stream.close()
            yield from sentences.flush()

            if not calls or self._cancel.is_set():
                return

            # Record the assistant's tool-call request, then run each tool.
            for i, call in enumerate(calls):
                call["id"] = call["id"] or f"call_{i}"
            working.append(
                {
                    "role": "assistant",
                    "content": "".join(content),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"] or "{}",
                            },
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                result = dispatch_tool(call["name"], call["arguments"])
                working.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        # Ran out of tool rounds.
        yield "Sorry, I got a bit tangled up there. Could you say that again?"


# Lazily-built singleton so importing this module doesn't connect to the LLM.
_brain = None


def _get_brain() -> Brain:
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain


def ask_gpt(user_text: str) -> str:
    """Whole reply as one string."""
    return _get_brain().ask(user_text)


def ask_gpt_stream(user_text: str):
    """The reply as a generator of sentences (what the voice loops use)."""
    return _get_brain().ask_stream(user_text)


def cancel_reply() -> None:
    """Barge-in: stop the reply that's being generated (safe from any thread)."""
    if _brain is not None:
        _brain.cancel()


def warm_up_model() -> None:
    """Load the current model into VRAM in the background (Ollama only), so the
    first question after startup, or after Ollama unloaded it while Nova slept,
    doesn't sit through a cold load. An empty prompt loads without generating."""
    if LLM_BACKEND != "ollama":
        return
    import json
    import urllib.request

    model = _brain.model if _brain is not None else LOCAL_LLM_MODEL
    url = _ollama_tags_url()[: -len("/api/tags")] + "/api/generate"

    def load():
        req = urllib.request.Request(
            url, data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=180).read()
        except Exception:
            pass  # a real problem surfaces on the first question

    threading.Thread(target=load, name="ollama-warmup", daemon=True).start()


# ---------------------------------------------------------------------------
# Runtime model switching (so you can say "switch to the fast model" mid-chat)
# ---------------------------------------------------------------------------

# Friendly spoken names -> Ollama tags. Both are tool-capable; gemma3 was
# dropped because it can't call tools.
MODEL_ALIASES = {
    "qwen": "qwen3:8b",
    "qwen 3": "qwen3:8b",
    "qwen three": "qwen3:8b",
    "big": "qwen3:8b",
    "qwen small": "qwen3:4b",
    "small": "qwen3:4b",
    "fast": "qwen3:4b",
}


def set_active_model(name: str) -> str:
    """Switch the live LLM for subsequent replies. Returns the resolved tag."""
    key = (name or "").strip().lower()
    key = re.sub(r"\s+", " ", re.sub(r"\b(?:the|model|brain|one)\b", "", key)).strip() or key
    tag = MODEL_ALIASES.get(key, key)
    _get_brain().model = tag
    return tag


def get_active_model() -> str:
    return _get_brain().model
