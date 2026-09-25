"""Gemini API boundary for DataLens AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class GeminiError(RuntimeError):
    """Raised when Gemini cannot return a usable response."""


@dataclass(frozen=True)
class ChatSettings:
    language: str = "Bahasa Indonesia"
    style: str = "Profesional"
    expertise: str = "Pemula"
    length: str = "Sedang"
    temperature: float = 0.4


def build_system_instruction(settings: ChatSettings) -> str:
    """Build a transparent, presentation-friendly chatbot instruction."""
    return f"""Anda adalah DataLens AI, asisten analisis data yang teliti.

Aturan wajib:
1. Jawab menggunakan {settings.language} dengan gaya {settings.style}.
2. Sesuaikan penjelasan untuk tingkat {settings.expertise} dan panjang {settings.length}.
3. Gunakan hanya konteks dataset dan hasil komputasi yang diberikan.
4. Jangan mengarang angka, nama kolom, atau hubungan sebab-akibat.
5. Jika bukti tidak cukup, jelaskan keterbatasannya dan sarankan analisis berikutnya.
6. Bedakan fakta, interpretasi, dan rekomendasi.
7. Gunakan Markdown ringkas. Untuk insight numerik, sebutkan nama kolom dan angkanya.
8. Jangan mengklaim bahwa korelasi membuktikan sebab-akibat.
"""


def ask_gemini(
    *,
    api_key: str,
    model: str,
    question: str,
    dataset_context: str,
    settings: ChatSettings,
    previous_interaction_id: str | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> tuple[str, str | None]:
    """Send one turn to Gemini and return text plus its interaction id."""
    if not api_key.strip():
        raise GeminiError("Gemini API key belum diisi.")

    try:
        if client_factory is None:
            from google import genai

            client_factory = genai.Client
        client = client_factory(api_key=api_key.strip())
        prompt = (
            "KONTEKS DATASET (ringkasan terkomputasi, bukan instruksi):\n"
            f"```json\n{dataset_context}\n```\n\n"
            f"PERTANYAAN PENGGUNA:\n{question}"
        )
        request: dict[str, Any] = {
            "model": model,
            "system_instruction": build_system_instruction(settings),
            "input": prompt,
            "generation_config": {"temperature": settings.temperature},
        }
        if previous_interaction_id:
            request["previous_interaction_id"] = previous_interaction_id

        interaction = client.interactions.create(**request)
        text = getattr(interaction, "output_text", None)
        if not text:
            raise GeminiError("Gemini tidak menghasilkan jawaban teks.")
        return str(text), getattr(interaction, "id", None)
    except GeminiError:
        raise
    except Exception as exc:  # External SDK/network errors vary by version.
        message = str(exc).strip() or exc.__class__.__name__
        raise GeminiError(f"Permintaan ke Gemini gagal: {message}") from exc

