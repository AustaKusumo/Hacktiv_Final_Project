from types import SimpleNamespace

import pytest

from datalens.gemini_service import ChatSettings, GeminiError, ask_gemini, build_system_instruction


class FakeInteractions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="Insight teruji", id="interaction-123")


class FakeClient:
    def __init__(self):
        self.interactions = FakeInteractions()


def test_system_instruction_contains_persona_settings():
    instruction = build_system_instruction(
        ChatSettings(language="English", style="Akademik", expertise="Expert", length="Ringkas")
    )

    assert "English" in instruction
    assert "Akademik" in instruction
    assert "Expert" in instruction
    assert "Jangan mengarang angka" in instruction


def test_ask_gemini_sends_context_and_memory_id():
    fake_client = FakeClient()

    text, interaction_id = ask_gemini(
        api_key="secret",
        model="gemini-test",
        question="Apa insight utama?",
        dataset_context='{"rows": 10}',
        settings=ChatSettings(temperature=0.2),
        previous_interaction_id="previous-1",
        client_factory=lambda **_: fake_client,
    )

    assert text == "Insight teruji"
    assert interaction_id == "interaction-123"
    assert fake_client.interactions.request["previous_interaction_id"] == "previous-1"
    assert fake_client.interactions.request["generation_config"] == {"temperature": 0.2}
    assert '"rows": 10' in fake_client.interactions.request["input"]


def test_ask_gemini_rejects_empty_api_key():
    with pytest.raises(GeminiError, match="belum diisi"):
        ask_gemini(
            api_key=" ",
            model="gemini-test",
            question="test",
            dataset_context="{}",
            settings=ChatSettings(),
        )

