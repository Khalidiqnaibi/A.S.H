# LLM.py
import os
import re
import json
import requests
import threading
import contextlib
from dotenv import load_dotenv
from typing import Optional, List, Sequence, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)      # never closed: the budget ran out mid-thought
_THINK_CLOSE = re.compile(r"\A.*?</think>", re.DOTALL | re.IGNORECASE)   # the opening tag was left out
# Reasoning that was written as the answer, with no tags: "Okay, so the user is
# asking...", "Let me break this down", "I need to figure out what to do here".
_REASONING = re.compile(
    r"^\s*(?:(?:okay|ok|alright|hmm+|so|well)[,.]?\s+)+(?:the user|let me|i(?:'m| am) trying to|i need to (?:figure|think|understand|"
    r"determine|remember)|i remember (?:that )?(?:from|the)|first,? i)\b|"
    r"^\s*(?:let me (?:think|break|figure|see|start)|the user (?:is asking|asked|wants|mentioned|said)|"
    r"i need to (?:figure|think|understand|determine))\b", re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Drop inline <think>...</think> blocks emitted by reasoning models --
    including one that was never closed, or never opened."""
    if not text:
        return ""
    t = _THINK_RE.sub("", text)
    if "<think>" in t.lower():
        t = _THINK_OPEN.sub("", t)
    elif "</think>" in t.lower():
        t = _THINK_CLOSE.sub("", t)
    return t


def looks_like_reasoning(text: str) -> bool:
    """True if the text is a model thinking aloud rather than answering."""
    return bool(text) and bool(_REASONING.match(text))

# Load env
load_dotenv(override=False)

# Default endpoints
DEFAULT_HF_MODEL = os.environ.get("MISTRAL_HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
DEFAULT_HF_URL = "https://api-inference.huggingface.co/models"

DEFAULT_OPENROUTER_MODEL = os.environ.get("MISTRAL_OPENROUTER_MODEL", "mistralai/mistral-nemo")
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_LOCAL_URL = os.environ.get("MISTRAL_LOCAL_URL", "http://localhost:5005/chat/completions")

# Ollama defaults
DEFAULT_OLLAMA_MODEL = os.environ.get("MISTRAL_OLLAMA_MODEL", "mistral:instruct")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434/api/chat"


class LLM(BaseChatModel):
    """
    ChatModel-compatible Mistral wrapper.
    """

    def __init__(
        self,
        mode: str = "openrouter",  # "hf", "openrouter", "local", "ollama"
        temperature: float = 0.0,
        max_tokens: int = 512,
        hf_model: Optional[str] = None,
        hf_token:str =None,
        openrouter_model: Optional[str] = None,
        openrouter_key : str = None,
        timeout: int = 60,
        local_url: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._mode = mode
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._hf_model = hf_model or DEFAULT_HF_MODEL
        self._hf_token = hf_token

        self._openrouter_model = openrouter_model or DEFAULT_OPENROUTER_MODEL
        self._openrouter_key = openrouter_key

        self._local_url = local_url or DEFAULT_LOCAL_URL
        self._ollama_url = ollama_url or DEFAULT_OLLAMA_URL
        self._ollama_model = ollama_model or DEFAULT_OLLAMA_MODEL
        self._timeout = timeout

        self._bound_tools: Optional[Any] = None
        # A second, smaller Ollama model for small talk (ASH_OLLAMA_LIGHT_MODEL or daemon.json llm.light_model).
        self._light_model: str = os.environ.get("ASH_OLLAMA_LIGHT_MODEL", "").strip()
        self._route = threading.local()

        if self._mode == "hf" and not self._hf_token:
            raise RuntimeError("HUGGINGFACE_API_TOKEN not found for mode='hf'")
        if self._mode == "openrouter" and not self._openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY not found for mode='openrouter'")

    @property
    def _llm_type(self) -> str:
        return "mistral-chat"

    def bind_tools(self, tools: Any):
        """
        LangChain agents call this.
        We store tools but do not force structured tool calling
        (our prompts already control behavior).
        """
        self._bound_tools = tools
        return self

    def _messages_to_dicts(self, messages: Sequence[BaseMessage]) -> List[dict]:
        """Converts LangChain messages to native API dictionaries, sanitizing bad characters."""
        api_messages = []
        for msg in messages:
            # Clean non-breaking spaces (\xa0) that crash strict tokenizers
            clean_content = msg.content.replace("\xa0", " ").strip()
            
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": clean_content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": clean_content})
            else:
                api_messages.append({"role": "user", "content": clean_content})
                
        return api_messages

    def _messages_to_prompt(self, messages: Sequence[BaseMessage]) -> str:
        """Fallback string compiler for HuggingFace pipeline requirements."""
        prompt = ""
        for msg in messages:
            clean_content = msg.content.replace("\xa0", " ").strip()
            role = "SYSTEM" if isinstance(msg, SystemMessage) else "ASH" if isinstance(msg, AIMessage) else "USER"
            prompt += f"[{role}]\n{clean_content}\n\n"
        return prompt.strip()

    def _call_hf(self, prompt: str) -> str:
        url = f"{DEFAULT_HF_URL}/{self._hf_model}"
        headers = {"Authorization": f"Bearer {self._hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self._max_tokens,
                "temperature": float(self._temperature),
            },
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
            if r.status_code != 200:
                print(f"[LLM ERROR] HF API returned status {r.status_code}: {r.text}")
                print(f"[LLM] request shape: requests.post({url}, headers={headers}, json={payload}, timeout={self._timeout})")
                return f"API Error: {r.status_code} - {r.text}"
            
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            return str(data)
        except Exception as e:
            print(f"[LLM ERROR] HF request failed: {e}")
            return f"Error calling HF API: {e}"

    def _call_openrouter(self, messages: List[dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self._openrouter_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._openrouter_model,
            "messages": messages,
            "max_tokens": int(self._max_tokens),
            "temperature": float(self._temperature),
        }
        try:
            r = requests.post(DEFAULT_OPENROUTER_URL, headers=headers, json=payload, timeout=self._timeout)
            
            if r.status_code != 200:
                print(f"[LLM ERROR] OpenRouter returned status {r.status_code}: {r.text}")
                print(f"[LLM] request shape: requests.post({DEFAULT_OPENROUTER_URL}, headers={headers}, json={payload}, timeout={self._timeout})")
                return f"[LLM] API Error: {r.status_code} - {r.text}"
            
            data = r.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            elif "error" in data:
                return f"Error from LLM provider: {data['error'].get('message', str(data['error']))}"
            else:
                return f"Error: Unexpected response format from LLM."
        except Exception as e:
            print(f"[LLM ERROR] OpenRouter request failed: {e}")
            return f"Error calling OpenRouter API: {e}"

    def _call_local(self, messages: List[dict]) -> str:
        payload = {
            "model": self._openrouter_model,
            "messages": messages,
            "max_tokens": int(self._max_tokens),
            "temperature": float(self._temperature),
        }
        try:
            r = requests.post(self._local_url, json=payload, timeout=self._timeout)
            
            if r.status_code != 200:
                print(f"[LLM ERROR] Local API returned status {r.status_code}: {r.text}")
                print(f"[LLM] request shape: requests.post({self._local_url}, json={payload}, timeout={self._timeout})")
                return f"[LLM] Local API Error: {r.status_code} - {r.text}"
                
            data = r.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"].get("content", str(data))
            else:
                return str(data)
        except Exception as e:
            print(f"[LLM ERROR] Local request failed: {e}")
            return f"Error calling Local API: {e}"

    def set_light_model(self, name: str):
        self._light_model = (name or "").strip()

    @contextlib.contextmanager
    def route(self, kind: str = ""):
        """`with llm.route("light"):` answers with the small model for this thread, if one is set
        (ollama only). Anything else, or no light model, leaves the main model in use."""
        old = getattr(self._route, "model", "")
        self._route.model = self._light_model if (kind == "light" and self._mode == "ollama") else ""
        try:
            yield bool(self._route.model)
        finally:
            self._route.model = old

    def _model_now(self) -> str:
        return getattr(self._route, "model", "") or self._ollama_model

    def _call_ollama(self, messages: List[dict]) -> str:
        """Answer with the model this thread is routed to. If that is the small model and it
        fails or says nothing, the main model answers instead -- small talk must never come
        back as an error. A small model that is not installed is switched off for good."""
        model = self._model_now()
        out = self._call_ollama_model(messages, model)
        if model != self._ollama_model and (not out or out.startswith("[LLM]")):
            if out and ("404" in out or "not found" in out.lower() or "Error calling" in out):
                print(f"[LLM] small model {model!r} is not usable -- using the main model from now on.")
                self._light_model = ""
            return self._call_ollama_model(messages, self._ollama_model)
        return out

    def _call_ollama_model(self, messages: List[dict], model: str) -> str:
        payload = {
            "model": model,
            "messages": messages,
            # Non-streaming: one JSON object back instead of an NDJSON stream.
            # The line loop below still handles a stream if a proxy forces one.
            "stream": False,
            "options": {
                "temperature": float(self._temperature),
                "num_predict": int(self._max_tokens),
            },
        }
        # ASH_OLLAMA_NUM_GPU: how many layers to put on the GPU; "0" keeps the
        # model off the graphics card entirely (daemon.json: llm.gpu = false),
        # for a machine whose GPU driver can't be trusted under load.
        # keep_alive: Ollama unloads a model after 5 idle minutes, and the next
        # question then pays a whole load (up to two minutes on a small GPU).
        # ASH is always on, so the model stays resident for hours by default;
        # ASH_OLLAMA_KEEP_ALIVE overrides ("5m", "-1" forever, "0" unload at once).
        payload["keep_alive"] = os.environ.get("ASH_OLLAMA_KEEP_ALIVE", "").strip() or "6h"
        n_gpu = os.environ.get("ASH_OLLAMA_NUM_GPU", "").strip()
        if n_gpu.lstrip("-").isdigit():
            payload["options"]["num_gpu"] = int(n_gpu)
        try:
            # Reasoning models (deepseek-r1, qwen3) think first, and on a small
            # token budget the thinking is all there is -- and it was being
            # spoken as the answer. Asked not to think; a model that has no
            # such option says so and the request is sent again without it.
            if os.environ.get("ASH_OLLAMA_THINK", "").strip().lower() not in ("1", "true", "yes", "on"):
                payload["think"] = False
            r = requests.post(self._ollama_url, json=payload, timeout=self._timeout)
            if r.status_code == 400 and "think" in payload and "think" in (r.text or "").lower():
                payload.pop("think", None)
                r = requests.post(self._ollama_url, json=payload, timeout=self._timeout)

            if r.status_code != 200:
                print(f"[LLM ERROR] Ollama API returned status {r.status_code}: {r.text}")
                print(f"[LLM] request shape: requests.post({self._ollama_url}, json={payload}, timeout={self._timeout})")
                return f"[LLM] Ollama API Error: {r.status_code} - {r.text}"

            response_text = ""
            thinking_text = ""
            for line in r.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message") or {}
                response_text += msg.get("content") or ""
                # Reasoning models (qwen3, deepseek-r1, ...) put chain-of-thought
                # in a separate field, or inline in <think> tags.
                thinking_text += msg.get("thinking") or ""

            response_text = _strip_think(response_text).strip()

            # A reasoning model that burned its whole num_predict budget on
            # thinking returns empty content. The thinking is NOT an answer
            # ("Okay, so the user is asking who is messi...") and is never
            # returned as one: an empty reply is the honest "no answer".
            if not response_text and thinking_text:
                print(
                    "[LLM WARN] Ollama returned only reasoning tokens -- raise max_tokens "
                    f"(currently {self._max_tokens}) or use a non-reasoning model."
                )
            if looks_like_reasoning(response_text):
                print("[LLM WARN] the model answered with its own reasoning; dropped.")
                return ""

            return response_text
        except Exception as e:
            print(f"[LLM ERROR] Ollama request failed: {e}")
            return f"[LLM] Error calling Ollama API: {e}"

    def warm(self, light: bool = False) -> bool:
        """Load the Ollama model into memory now, so the first real question
        does not pay for it. One token, no thinking. True if the model answered.
        `light=True` loads the small-talk model instead (False if none is set)."""
        if self._mode != "ollama" or (light and not self._light_model):
            return False
        payload = {
            "model": self._light_model if light else self._ollama_model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0.0},
            "keep_alive": os.environ.get("ASH_OLLAMA_KEEP_ALIVE", "").strip() or "6h",
        }
        n_gpu = os.environ.get("ASH_OLLAMA_NUM_GPU", "").strip()
        if n_gpu.lstrip("-").isdigit():
            payload["options"]["num_gpu"] = int(n_gpu)
        try:
            r = requests.post(self._ollama_url, json=payload, timeout=max(self._timeout, 120))
            if r.status_code == 400 and "think" in (r.text or "").lower():
                return False
            return r.status_code == 200
        except Exception:
            return False

    def _call(self, messages: Sequence[BaseMessage]) -> str:
        if self._mode == "hf":
            prompt = self._messages_to_prompt(messages)
            return self._call_hf(prompt)
        
        # OpenRouter, Local, and Ollama all consume structural dictionary lists natively
        api_messages = self._messages_to_dicts(messages)
        
        if self._mode == "openrouter":
            return self._call_openrouter(api_messages)
        elif self._mode == "local":
            return self._call_local(api_messages)
        elif self._mode == "ollama":
            return self._call_ollama(api_messages)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        # Pass the full message sequence down directly
        text = self._call(messages)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text)
                )
            ]
        )


if __name__ == "__main__":
    print("Running MistralLLM ChatModel self-test...\n")
    mode = os.environ.get("MISTRAL_TEST_MODE", "openrouter")

    llm = LLM(mode=mode, temperature=0.7, max_tokens=64)

    messages = [
        HumanMessage(content="Write a short haiku about the moon.")
    ]

    result = llm.invoke(messages)
    print("\nLLM Output:\n")
    print(result.content)