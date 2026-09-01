"""Meeting assistant helpers for recording audio, collecting transcripts, and
writing a short summary at the end of a session.
"""

from __future__ import annotations

import re
import wave
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests

from models import TranscriptionResult, TranscriptionStatus


def run_meeting_assistant() -> None:
    """Launch the meeting assistant preset used by the script entrypoint."""
    from core import run_stt_pipeline

    run_stt_pipeline(
        whisper_model="medium",
        device="cpu",
        compute_type="int8",
        rate=16000,
        chunk_duration_s=1.5,
        silence_threshold=0.01,
        end_of_speech_s=2.5,
        force_transcription_on_max_speech=False,
        initial_prompt="",
        language="tr",
        beam_size=5,
        vad_filter=True,
        vad_min_silence_ms=900,
        word_timestamps=True,
        use_ollama=False,
        ollama_url="http://localhost:11434/api/generate",
        ollama_model="qwen3.5:0.8b",
        meeting_mode=True,
        meeting_title="toplanti",
        meeting_output_dir="meeting-notes",
        save_audio=True,
        use_ollama_summary=False,
        summary_ollama_url="http://127.0.0.1:11434/api/generate",
        summary_ollama_model="qwen3.5:0.8b",
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _extractive_summary(text: str, max_sentences: int = 5) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    if len(sentences) <= max_sentences:
        return "\n".join(f"- {sentence}" for sentence in sentences)

    words = re.findall(r"\b[\wÇĞİÖŞÜçğıöşü]+\b", text.lower())
    stop_words = {
        "ve",
        "bir",
        "bu",
        "da",
        "de",
        "ile",
        "için",
        "ama",
        "gibi",
        "çok",
        "daha",
        "mi",
        "mı",
        "mu",
        "mü",
        "olarak",
        "ile",
        "ya",
        "ya da",
    }
    frequencies = Counter(word for word in words if word not in stop_words and len(word) > 2)
    if not frequencies:
        return "\n".join(f"- {sentence}" for sentence in sentences[:max_sentences])

    ranked_sentences: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        score = 0.0
        sentence_words = re.findall(r"\b[\wÇĞİÖŞÜçğıöşü]+\b", sentence.lower())
        for word in sentence_words:
            score += frequencies.get(word, 0)
        ranked_sentences.append((score, index, sentence))

    top_sentences = sorted(ranked_sentences, key=lambda item: (-item[0], item[1]))[:max_sentences]
    ordered = [sentence for _, _, sentence in sorted(top_sentences, key=lambda item: item[1])]
    return "\n".join(f"- {sentence}" for sentence in ordered)


@dataclass
class MeetingAssistant:
    """Collects meeting audio and transcript artifacts."""

    output_dir: str | Path = "meeting-notes"
    sample_rate: int = 16000
    save_audio: bool = True
    use_ollama_summary: bool = False
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_model: str = "qwen3.5:0.8b"
    meeting_title: str = "meeting"
    _session_id: str = field(init=False)
    _session_dir: Path = field(init=False)
    _audio_path: Optional[Path] = field(default=None, init=False)
    _transcript_path: Path = field(init=False)
    _summary_path: Path = field(init=False)
    _wave_file: Optional[wave.Wave_write] = field(default=None, init=False)
    _transcript_entries: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "-", self.meeting_title).strip("-") or "meeting"
        self._session_id = f"{safe_title}_{timestamp}"
        self._session_dir = Path(self.output_dir) / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._transcript_path = self._session_dir / "transcript.md"
        self._summary_path = self._session_dir / "summary.md"
        if self.save_audio:
            self._audio_path = self._session_dir / "recording.wav"
            self._wave_file = wave.open(str(self._audio_path), "wb")
            self._wave_file.setnchannels(1)
            self._wave_file.setsampwidth(2)
            self._wave_file.setframerate(self.sample_rate)

    @staticmethod
    def _to_pcm16(audio: np.ndarray) -> bytes:
        flattened = np.asarray(audio, dtype=np.float32).reshape(-1)
        clipped = np.clip(flattened, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def handle_audio(self, audio: np.ndarray) -> None:
        if not self._wave_file or audio.size == 0:
            return
        self._wave_file.writeframes(self._to_pcm16(audio))

    def handle_result(self, result: TranscriptionResult) -> None:
        if result.status == TranscriptionStatus.SILENCE or not result.text.strip():
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        status = result.status.value
        text = _normalize_text(result.text)
        self._transcript_entries.append(
            f"- [{timestamp}] [{status}] [{result.confidence:.0%}] {text}"
        )

    def _ollama_summary(self, transcript_text: str) -> str:
        prompt = (
            "Aşağıdaki toplantı transkriptinden kısa ve düzenli toplantı notları çıkar. "
            "Yanıtı Türkçe ver ve şu başlıkları kullan: Özet, Kararlar, Aksiyon Maddeleri, Açık Sorular. "
            "Kısa, madde madde ve profesyonel ol.\n\n"
            f"TRANSKRİPT:\n{transcript_text}\n"
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        response = requests.post(self.ollama_url, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        return _normalize_text(data.get("response", ""))

    def build_summary(self) -> str:
        transcript_text = " ".join(
            entry.split("] ", 3)[-1] for entry in self._transcript_entries
        ).strip()
        if not transcript_text:
            return "Toplantı sırasında anlamlı bir transkript kaydedilmedi."

        if self.use_ollama_summary:
            try:
                summary = self._ollama_summary(transcript_text)
                if summary:
                    return summary
            except Exception:
                pass

        return _extractive_summary(transcript_text)

    def finalize(self) -> dict[str, str]:
        if self._wave_file is not None:
            self._wave_file.close()
            self._wave_file = None

        summary = self.build_summary()
        transcript_body = "\n".join(self._transcript_entries) if self._transcript_entries else "- Henüz transkript yok."

        self._transcript_path.write_text(
            f"# Toplantı Transkripti\n\n{transcript_body}\n",
            encoding="utf-8",
        )
        self._summary_path.write_text(
            f"# Toplantı Özeti\n\n{summary}\n",
            encoding="utf-8",
        )

        return {
            "session_dir": str(self._session_dir),
            "transcript_path": str(self._transcript_path),
            "summary_path": str(self._summary_path),
            "audio_path": str(self._audio_path) if self._audio_path else "",
        }


if __name__ == "__main__":
    run_meeting_assistant()