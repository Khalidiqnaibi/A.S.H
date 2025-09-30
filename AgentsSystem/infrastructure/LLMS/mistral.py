import os
import requests
from dotenv import load_dotenv
from typing import Optional, List
from langchain.llms.base import LLM
from langchain.schema import LLMResult

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

# Tokens
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")


class MistralLLM(LLM):
    class Config:
        extra = "allow"

    def __init__(
        self,
        mode: str = "openrouter",  # "hf", "openrouter", "local", "ollama"
        temperature: float = 0.0,
        max_tokens: int = 512,
        hf_model: Optional[str] = None,
        openrouter_model: Optional[str] = None,
        timeout: int = 60,
        local_url: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.hf_model = hf_model or DEFAULT_HF_MODEL
        self.openrouter_model = openrouter_model or DEFAULT_OPENROUTER_MODEL
        self.local_url = local_url or DEFAULT_LOCAL_URL
        self.ollama_url = ollama_url or DEFAULT_OLLAMA_URL
        self.ollama_model = ollama_model or DEFAULT_OLLAMA_MODEL
        self.timeout = timeout

        if self.mode == "hf" and not HF_TOKEN:
            raise RuntimeError("HUGGINGFACE_API_TOKEN not found for mode='hf'")
        if self.mode == "openrouter" and not OPENROUTER_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not found for mode='openrouter'")

    @property
    def _llm_type(self) -> str:
        return "mistral"

    def _call_hf(self, prompt: str) -> str:
        url = f"{DEFAULT_HF_URL}/{self.hf_model}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": self.max_tokens, "temperature": float(self.temperature)}}
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)

    def _call_openrouter(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": self.openrouter_model, "messages": messages, "max_tokens": int(self.max_tokens), "temperature": float(self.temperature)}
        r = requests.post(DEFAULT_OPENROUTER_URL, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _call_local(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": self.openrouter_model, "messages": messages, "max_tokens": int(self.max_tokens), "temperature": float(self.temperature)}
        r = requests.post(self.local_url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", str(data))

    def _call_ollama(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "options": {
                "temperature": float(self.temperature),
                "num_predict": int(self.max_tokens),
            }
        }

        r = requests.post(self.ollama_url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()

        # Ollama may return multiple JSON objects separated by newline
        response_text = ""
        for line in r.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # normal chat completion object
                if "message" in data and "content" in data["message"]:
                    response_text += data["message"]["content"]
            except json.JSONDecodeError:
                # ignore lines that are not valid JSON
                continue

        return response_text.strip()


    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.mode == "hf":
            return self._call_hf(prompt)
        elif self.mode == "openrouter":
            return self._call_openrouter(prompt)
        elif self.mode == "local":
            return self._call_local(prompt)
        elif self.mode == "ollama":
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = [[{"text": self._call(p, stop)}] for p in prompts]
        return LLMResult(generations=generations, llm_output={"token_usage": {}})


if __name__ == "__main__":
    import json
    print("Running MistralLLM self-test...\n")
    mode = os.environ.get("MISTRAL_TEST_MODE", "ollama")
    print(f"Testing mode: {mode}")
    try:
        llm = MistralLLM(mode=mode, temperature=0.7, max_tokens=64)
        prompt = "Write a short haiku about the moon."
        print(f"\nPrompt: {prompt}")
        output = llm._call(prompt)
        print("\nLLM Output:")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\nError: {e}")