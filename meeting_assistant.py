"""Meeting assistant helpers for recording audio, collecting transcripts, and
writing a short summary at the end of a session.
"""

from __future__ import annotations

import re
import time
import wave
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests

from models import TranscriptionResult, TranscriptionStatus
from diarization import DiarizationPipeline


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


def _normalize_summary_lines(text: str) -> str:
    """LLM özetinin satır/başlık yapısını (## Konu, - madde) korur; sadece
    satır içi fazla boşlukları ve art arda gelen fazla boş satırları toparlar.
    """
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if line == "" and (not cleaned or cleaned[-1] == ""):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _format_elapsed(seconds: float) -> str:
    """Kayıt başından itibaren geçen süreyi HH:MM:SS olarak biçimlendirir."""
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
    use_diarization: bool = True  # Speaker diarization aç/kapat
    _session_id: str = field(init=False)
    _session_dir: Path = field(init=False)
    _audio_path: Optional[Path] = field(default=None, init=False)
    _transcript_path: Path = field(init=False)
    _summary_path: Path = field(init=False)
    _wave_file: Optional[wave.Wave_write] = field(default=None, init=False)
    _transcript_entries: list[str] = field(default_factory=list, init=False)
    # Her satırın ham metni (formatlanmamış) - konuşmacı bazlı birleştirme
    # yaparken _transcript_entries'i yeniden ayrıştırmak yerine bunu kullanır.
    _transcript_texts: list[str] = field(default_factory=list, init=False)
    # Her transkript satırının, toplantı/kayıt başlangıcından itibaren geçen
    # saniyesi - diarization segmentleriyle aynı zaman tabanında (dosya
    # başından itibaren) olduğu için doğru konuşmacı eşlemesi bunlarla yapılır.
    _transcript_elapsed: list[float] = field(default_factory=list, init=False)
    _start_monotonic: float = field(default_factory=time.monotonic, init=False)
    _diarization_pipeline: Optional[DiarizationPipeline] = field(default=None, init=False)
    # Batch modda diarizasyon Whisper transkripsiyonuyla paralel, ayrı bir
    # thread'de çalıştırılıp sonucu buraya önceden konabilir - finalize()
    # o zaman diarizasyonu kendisi çalıştırmak yerine bunu kullanır.
    _precomputed_diarization_segments: Optional[list] = field(default=None, init=False)

    def __post_init__(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "-", self.meeting_title).strip("-") or "meeting"
        self._session_id = f"{safe_title}_{timestamp}"
        self._session_dir = Path(self.output_dir) / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._transcript_path = self._session_dir / "transcript.md"
        self._summary_path = self._session_dir / "summary.md"
        
        # Diarization pipeline'ını initialize et
        self._diarization_pipeline = DiarizationPipeline(use_diarization=self.use_diarization)
        
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
        # SILENCE veya boş metni at, geri kalan her şeyi (ACCEPTED/LOW_CONFIDENCE/REJECTED) kabul et
        if result.status == TranscriptionStatus.SILENCE or not result.text.strip():
            return
        elapsed = time.monotonic() - self._start_monotonic
        self._add_entry(elapsed, result)

    def add_segment(self, elapsed_seconds: float, result: TranscriptionResult) -> None:
        """Dosya/batch modu için: segmentin kaydın başından itibaren gerçek
        zaman konumunu (saniye) bilerek ekler - diarization ile doğru eşleşir.
        """
        if result.status == TranscriptionStatus.SILENCE or not result.text.strip():
            return
        self._add_entry(elapsed_seconds, result)

    def set_diarization_segments(self, segments: list) -> None:
        """Diarizasyon Whisper transkripsiyonuyla paralel, ayrı bir thread'de
        önceden çalıştırılmışsa sonucu buraya verin - finalize() bu durumda
        diarizasyonu tekrar çalıştırmaz.
        """
        self._precomputed_diarization_segments = segments

    def _add_entry(self, elapsed_seconds: float, result: TranscriptionResult) -> None:
        timestamp = _format_elapsed(elapsed_seconds)
        text = _normalize_text(result.text)
        self._transcript_entries.append(f"- [{timestamp}] {text}")
        self._transcript_texts.append(text)
        self._transcript_elapsed.append(elapsed_seconds)

    def _ollama_summary(self, transcript_text: str) -> str:
        """Ollama ile özet yap - timeout varsa fallback yap"""
        # Önce Ollama'nın çalışıp çalışmadığını kontrol et
        try:
            response = requests.get(f"{self.ollama_url.rsplit('/api', 1)[0]}/api/tags", timeout=2)
            if response.status_code != 200:
                raise Exception("Ollama API bağlantısız")
        except Exception as e:
            raise Exception(f"Ollama erişilemez: {e}")

        prompt = (
            "Aşağıdaki toplantı transkriptinden ÇOK KISA bir özet çıkar.\n\n"
            "KURALLAR (kesinlikle uy):\n"
            "- EN FAZLA 8 MADDE yaz. Asla daha fazla yazma.\n"
            "- Her madde TEK cümle olsun.\n"
            "- Aynı bilgiyi iki kere yazma.\n"
            "- Selamlaşma, 'kolay gelsin', teşekkür, sırayla herkesin ne yaptığını anlattığı "
            "rutin durum güncellemelerini YAZMA - sadece gerçek karar, sorun veya yapılacak "
            "iş varsa onu yaz.\n"
            "- Transkriptte geçmeyen hiçbir bilgi uydurma.\n"
            "- Başlık kullanma, sadece '- ' ile başlayan madde listesi yaz.\n"
            "- Türkçe yaz.\n\n"
            f"TRANSKRİPT:\n{transcript_text}\n"
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            # "thinking" destekleyen modeller (ör. qwen3.5), tüm token bütçesini
            # muhakeme aşamasında tüketip asıl cevabı hiç üretmeden kesilebiliyor
            # (response boş dönüyor). think=False bunu kapatıp doğrudan cevap üretir.
            "think": False,
            "options": {
                # temperature=0.0 (tam "greedy" seçim) küçük modellerde aynı
                # cümleyi/bloğu sonsuza kadar tekrarlama döngüsüne girmeye çok
                # yatkın - hafif bir rastgelelik bunu büyük ölçüde engelliyor.
                "temperature": 0.2,
                # Az önce kullanılan kelime/kalıpları cezalandırır - tekrar
                # döngüsüne karşı asıl savunma bu.
                "repeat_penalty": 1.3,
                # Kaçarsa bile çıktının sonsuza kadar uzayıp durmasını engeller.
                "num_predict": 400,
            },
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)  # 60 saniye timeout
            response.raise_for_status()
            data = response.json()
            # DİKKAT: _normalize_text kullanma - o satır sonlarını da boşluğa
            # çevirir ve modelin ürettiği başlık/madde yapısını (## Konu,
            # - madde vb.) tek bir metin yığınına dönüştürür.
            return _normalize_summary_lines(data.get("response", ""))
        except requests.exceptions.Timeout:
            raise Exception("Ollama timeout (30s) - çok uzun metin olabilir")
        except requests.exceptions.ConnectionError:
            raise Exception("Ollama bağlantı hatası")

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
            except Exception as e:
                # Ollama başarısız, extractive'e düş
                pass

        # Fallback: Extractive summary
        return _extractive_summary(transcript_text)

    def finalize(self) -> dict[str, str]:
        try:
            if self._wave_file is not None:
                self._wave_file.close()
                self._wave_file = None

            summary = self.build_summary()
            transcript_body = "\n".join(self._transcript_entries) if self._transcript_entries else "- Henüz transkript yok."

            # Diarization'ı çalıştır (ses dosyası varsa) - eğer paralel olarak
            # önceden çalıştırılıp set_diarization_segments() ile verildiyse
            # burada tekrar çalıştırmadan onu kullan.
            transcript_header = "# Toplantı Transkripti\n\n"
            if self._audio_path and self._diarization_pipeline and self._diarization_pipeline.use_diarization:
                try:
                    segments = (
                        self._precomputed_diarization_segments
                        if self._precomputed_diarization_segments is not None
                        else self._diarization_pipeline.get_speaker_segments(str(self._audio_path), sr=self.sample_rate)
                    )
                    if segments and self._transcript_entries:
                        # Transkripte konuşmacı bilgisi ekle
                        transcript_body = self._add_speaker_labels(segments)
                        transcript_header = "# Toplantı Transkripti (Konuşmacı Bilgisi ile)\n\n"
                except Exception as e:
                    print(f"Diarization hatası: {e}", flush=True)

            self._transcript_path.write_text(
                f"{transcript_header}{transcript_body}\n",
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
        except Exception as e:
            print(f"HATA finalize'de: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Fallback: En azından boş dosyalar oluştur
            try:
                self._transcript_path.write_text("# Toplantı Transkripti\n\n- Hata oluştu.\n", encoding="utf-8")
                self._summary_path.write_text("# Toplantı Özeti\n\n- Özet oluşturulamadı.\n", encoding="utf-8")
            except:
                pass
            return {
                "session_dir": str(self._session_dir),
                "transcript_path": str(self._transcript_path),
                "summary_path": str(self._summary_path),
                "audio_path": str(self._audio_path) if self._audio_path else "",
            }

    def _add_speaker_labels(self, segments: list) -> str:
        """
        Transkript satırlarına konuşmacı bilgilerini ekler ve aynı
        konuşmacıya ait art arda gelen satırları tek bir paragrafta birleştirir
        (ör. bir daily'de bir kişinin peş peşe söylediği 6 kısa cümle, 6 ayrı
        madde yerine tek bir konuşma turu olarak görünür).

        Segments: [(start_sec, end_sec, speaker_no), ...] - kaydın başından
        itibaren saniye cinsinden, tıpkı _transcript_elapsed gibi. Satırdaki
        görüntülenen HH:MM:SS metnini yeniden ayrıştırmak yerine (bu, canlı
        modda duvar saatiyle karıştırılıp yanlış eşleşmelere yol açıyordu),
        doğrudan aynı zaman tabanındaki _transcript_elapsed listesini kullanır.
        """
        turns: list[tuple[int, float, list[str]]] = []  # (speaker_no, ilk_zaman, metinler)

        for text, elapsed in zip(self._transcript_texts, self._transcript_elapsed):
            speaker_no = self._diarization_pipeline.get_speaker_at_time(segments, elapsed)
            # speaker_no == 0 (bilinmeyen/eşleşmeyen) satırları asla birbirine
            # birleştirme - farklı gerçek konuşmacılara ait olabilirler.
            if turns and speaker_no > 0 and turns[-1][0] == speaker_no:
                turns[-1][2].append(text)
            else:
                turns.append((speaker_no, elapsed, [text]))

        lines = []
        for speaker_no, elapsed, texts in turns:
            timestamp = _format_elapsed(elapsed)
            prefix = f"[Konuşmacı {speaker_no}] " if speaker_no > 0 else ""
            lines.append(f"- {prefix}[{timestamp}] {' '.join(texts)}")

        return "\n".join(lines)


if __name__ == "__main__":
    run_meeting_assistant()