# LLM.py
import os
import json
import requests
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
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:4444/api/chat")


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

    def _messages_to_prompt(self, messages: Sequence[BaseMessage]) -> str:
        """
        Convert chat messages into a single prompt string.
        """
        parts: List[str] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"[SYSTEM]\n{msg.content}")
            elif isinstance(msg, HumanMessage):
                parts.append(f"[USER]\n{msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"[ASSISTANT]\n{msg.content}")
            else:
                parts.append(msg.content)

        return "\n\n".join(parts)

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
                return f"API Error: {r.status_code} - {r.text}"
            
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            return str(data)
        except Exception as e:
            print(f"[LLM ERROR] HF request failed: {e}")
            return f"Error calling HF API: {e}"

    def _call_openrouter(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._openrouter_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(self._max_tokens),
            "temperature": float(self._temperature),
        }
        try:
            r = requests.post(DEFAULT_OPENROUTER_URL, headers=headers, json=payload, timeout=self._timeout)
            
            # Catch bad API keys, out of credits, or model errors safely
            if r.status_code != 200:
                print(f"[LLM ERROR] OpenRouter returned status {r.status_code}: {r.text}")
                return f"API Error: {r.status_code} - {r.text}"
            
            data = r.json()
            
            # Safely extract 'choices' without throwing a KeyError
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            elif "error" in data:
                print(f"[LLM ERROR] OpenRouter API Error Payload: {data['error']}")
                return f"Error from LLM provider: {data['error'].get('message', str(data['error']))}"
            else:
                print(f"[LLM ERROR] Unexpected response format: {data}")
                return f"Error: Unexpected response format from LLM."
        except Exception as e:
            print(f"[LLM ERROR] OpenRouter request failed: {e}")
            return f"Error calling OpenRouter API: {e}"

    def _call_local(self, prompt: str) -> str:
        payload = {
            "model": self._openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(self._max_tokens),
            "temperature": float(self._temperature),
        }
        try:
            r = requests.post(self._local_url, json=payload, timeout=self._timeout)
            
            if r.status_code != 200:
                print(f"[LLM ERROR] Local API returned status {r.status_code}: {r.text}")
                return f"Local API Error: {r.status_code} - {r.text}"
                
            data = r.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"].get("content", str(data))
            else:
                return str(data)
        except Exception as e:
            print(f"[LLM ERROR] Local request failed: {e}")
            return f"Error calling Local API: {e}"

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self._ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": float(self._temperature),
                "num_predict": int(self._max_tokens),
            },
        }

        try:
            r = requests.post(self._ollama_url, json=payload, timeout=self._timeout)
            
            if r.status_code != 200:
                print(f"[LLM ERROR] Ollama API returned status {r.status_code}: {r.text}")
                return f"Ollama API Error: {r.status_code} - {r.text}"

            response_text = ""
            for line in r.text.splitlines():
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        response_text += data["message"]["content"]
                except json.JSONDecodeError:
                    continue

            return response_text.strip()
        except Exception as e:
            print(f"[LLM ERROR] Ollama request failed: {e}")
            return f"Error calling Ollama API: {e}"

    def _call(self, prompt: str) -> str:
        if self._mode == "hf":
            return self._call_hf(prompt)
        elif self._mode == "openrouter":
            return self._call_openrouter(prompt)
        elif self._mode == "local":
            return self._call_local(prompt)
        elif self._mode == "ollama":
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        prompt = self._messages_to_prompt(messages)
        text = self._call(prompt)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text)
                )
            ]
        )


if __name__ == "__main__":
    print("Running MistralLLM ChatModel self-test...\n")
    mode = os.environ.get("MISTRAL_TEST_MODE", "ollama")

    llm = LLM(mode=mode, temperature=0.7, max_tokens=64)

    messages = [
        HumanMessage(content="Write a short haiku about the moon.")
    ]

    result = llm.invoke(messages)
    print("\nLLM Output:\n")
    print(result.content)