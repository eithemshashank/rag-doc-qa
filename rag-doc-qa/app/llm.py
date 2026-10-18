"""Step 3a: the language model that writes the final answer."""
import json
import urllib.error
import urllib.request
from typing import Callable, Protocol


class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        """Return the model's text reply to the prompt."""


class GeminiLLM:
    """Google Gemini through the official `google-genai` package."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Put it in a .env file: GEMINI_API_KEY=your_key"
            )
        from google import genai  # imported lazily

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model, contents=prompt
        )
        return (response.text or "").strip()


class GroqLLM:
    """Groq's free tier, through its OpenAI-compatible HTTP API (no extra package).

    Useful as a backup if Gemini is unavailable for your account.
    """

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str, sender: Callable | None = None):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Put it in a .env file: GROQ_API_KEY=your_key"
            )
        self._api_key = api_key
        self._model = model
        self._send = sender or self._http_post

    def _http_post(self, url: str, headers: dict, payload: dict) -> tuple[int, str]:
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode("utf-8", errors="replace")

    def generate(self, prompt: str) -> str:
        status, body = self._send(
            self.URL,
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # Groq sits behind Cloudflare, which rejects the default urllib agent
                "User-Agent": "rag-doc-qa/1.0",
            },
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        if status != 200:
            raise RuntimeError(f"Groq API error {status}: {body[:300]}")
        return json.loads(body)["choices"][0]["message"]["content"].strip()


def make_llm(provider: str, config) -> "LLM":
    """Pick the language model named by LLM_PROVIDER in your .env file."""
    provider = provider.lower().strip()
    if provider == "gemini":
        return GeminiLLM(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    if provider == "groq":
        return GroqLLM(config.GROQ_API_KEY, config.GROQ_MODEL)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'groq'.")


class FakeLLM:
    """Test double: returns a fixed reply and remembers the prompts it got."""

    def __init__(self, reply: str = "FAKE ANSWER"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply
