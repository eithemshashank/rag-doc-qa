import json

import pytest

from app import config
from app.llm import GeminiLLM, GroqLLM, make_llm


def test_groq_sends_prompt_and_parses_reply():
    seen = {}

    def fake_sender(url, headers, payload):
        seen.update(url=url, headers=headers, payload=payload)
        body = {"choices": [{"message": {"content": "  Hello there  "}}]}
        return 200, json.dumps(body)

    llm = GroqLLM("secret-key", "some-model", sender=fake_sender)
    assert llm.generate("Say hi") == "Hello there"
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"]["Authorization"] == "Bearer secret-key"
    assert seen["payload"]["model"] == "some-model"
    assert seen["payload"]["messages"][0]["content"] == "Say hi"


def test_groq_error_is_raised_without_leaking_the_key():
    llm = GroqLLM("secret-key", "m", sender=lambda *a: (401, '{"error":"bad key"}'))
    with pytest.raises(RuntimeError) as err:
        llm.generate("x")
    assert "401" in str(err.value)
    assert "secret-key" not in str(err.value)


def test_missing_keys_raise_helpful_errors():
    with pytest.raises(ValueError):
        GroqLLM("", "m")
    with pytest.raises(ValueError):
        GeminiLLM("", "m")


def test_make_llm_picks_provider_and_rejects_unknown():
    assert isinstance(make_llm("groq", type("C", (), {"GROQ_API_KEY": "k", "GROQ_MODEL": "m"})), GroqLLM)
    with pytest.raises(ValueError):
        make_llm("openai", config)
