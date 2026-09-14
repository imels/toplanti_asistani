"""
Türkçe Konuşmadan Metne (STT) pipeline'ı.

Mimari:
  STTPipeline (sınıf)  → İç OOP motor
  run_stt_pipeline()   → Geriye uyumlu fonksiyon arayüzü (main.py bunu çağırır)

Felsefe: TAHMİN ETME, REDDET.
  Güven düşükse → "Anlayamadım, tekrar eder misiniz?" demek,
  yanlış transkripsiyon vermekten her zaman daha iyidir.
"""

import os

# ── Ortam Değişkenleri ─────────────────────────────────────────────────────
# huggingface_hub bu değerleri import anında modül seviyesinde sabitlere okur,
# bu yüzden huggingface_hub'ı (transitive olarak faster_whisper üzerinden)
# import etmeden ÖNCE ayarlanmaları şart — aksi halde XET indirme yolu
# devre dışı kalmaz ve indirme (yavaş/engelli ağlarda) donmuş gibi görünür.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"

import re
import time
import queue
import logging
import concurrent.futures
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

from config import STTConfig, DEFAULT_TECHNICAL_TERMS, BANKING_TERMS
from models import TranscriptionResult, TranscriptionStatus, SegmentDetail
from meeting_assistant import MeetingAssistant
from filters import HallucinationFilter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# STTPipeline — Ana Motor
# ═══════════════════════════════════════════════════════════════════════════


class STTPipeline:
    """
    Gerçek zamanlı STT pipeline'ı.

    Özellikler:
      - Model ısınması (JIT derleme için sessiz ses transkripsiyon)
      - Kayan pencere tampon (sonsuz büyümeyi engeller)
      - Çok katmanlı halüsinasyon filtresi
      - Enerji kapısı (sessizliği Whisper'a göndermez)
      - Güven bazlı sonuç sınıflandırma (ACCEPTED/LOW_CONFIDENCE/REJECTED/SILENCE)
      - Kuyruk taşma koruması
    """

    def __init__(
        self,
        config: STTConfig,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None,
        silent: bool = False,
    ):
        self.config = config
        self.silent = silent
        # Silent mode'da sadece dosyaya yaz, terminal çıktı yapma
        if silent:
            self.on_result = on_result or (lambda x: None)  # Boş callback
        else:
            self.on_result = on_result or self._default_callback
        self.on_audio_chunk = on_audio_chunk
        self.halucination_filter = HallucinationFilter(config)

        # Ses durumu
        self._buffer = np.array([], dtype=np.float32)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=200)
        self._max_buffer_samples = int(config.sample_rate * config.max_buffer_s)

        # Konuşma tespiti durumu
        self._speech_active: bool = False
        self._silence_samples: int = 0
        self._end_of_speech_samples = int(config.sample_rate * config.end_of_speech_s)
        self._max_speech_samples = int(config.sample_rate * config.max_speech_s)

        # Model indirme durumunu kontrol et
        local_exists = self._check_local_model_exists(config.whisper_model)
        if local_exists:
            logger.info(
                "Model yerel diskte bulundu. İnternet kontrolü bypass ediliyor (local_files_only=True)."
            )

        # Model yükle
        logger.info(
            "Model yükleniyor: %s (%s/%s)",
            config.whisper_model,
            config.device,
            config.compute_type,
        )
        self._model = WhisperModel(
            config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            local_files_only=local_exists,
        )

        # CUDA JIT derlemesini tetikle
        self._warmup()
        logger.info("Pipeline hazır.")

    # ── Model Kontrolü  ─────────────────────────────────────────────────────
    def _check_local_model_exists(self, model_name: str) -> bool:
        """
        Hugging Face cache dizinini kontrol ederek modelin yerelde olup olmadığını denetler.
        """
        # Kullanıcının model indirdiği varsayılan dizin
        cache_dir = os.path.expanduser(r"~/.cache/huggingface/hub")

        # Kısa model isimlerinin repo karşılıkları
        repo_map = {
            "tiny": "Systran/faster-whisper-tiny",
            "tiny.en": "Systran/faster-whisper-tiny.en",
            "base": "Systran/faster-whisper-base",
            "base.en": "Systran/faster-whisper-base.en",
            "small": "Systran/faster-whisper-small",
            "small.en": "Systran/faster-whisper-small.en",
            "medium": "Systran/faster-whisper-medium",
            "medium.en": "Systran/faster-whisper-medium.en",
            "large-v1": "Systran/faster-whisper-large-v1",
            "large-v2": "Systran/faster-whisper-large-v2",
            "large-v3": "Systran/faster-whisper-large-v3",
            "large": "Systran/faster-whisper-large-v3",
        }

        repo_id = repo_map.get(model_name, model_name)
        formatted_repo = f"models--{repo_id.replace('/', '--')}"
        model_path = os.path.join(cache_dir, formatted_repo)

        # Dizin ve snapshot kontrolü — sadece klasör değil, asıl model ağırlığının
        # (model.bin) var olduğuna bak. Aksi halde yarım kalmış bir indirme
        # "yerelde var" sanılıp local_files_only=True ile internetten
        # tamamlanması engellenir ve yükleme çöker.
        if os.path.exists(model_path):
            snapshots_dir = os.path.join(model_path, "snapshots")
            if os.path.exists(snapshots_dir):
                for snapshot in os.listdir(snapshots_dir):
                    if os.path.exists(os.path.join(snapshots_dir, snapshot, "model.bin")):
                        return True

        # Doğrudan yerel bir klasör yolu verilmişse
        if os.path.exists(model_name):
            return True

        return False

    # ── Model Isınması ─────────────────────────────────────────────────────

    def _warmup(self):
        """1 saniyelik sessiz ses ile CUDA JIT derlemesini tetikle.
        İlk gerçek transkripsiyon böylece hızlı olur."""
        logger.info("Model ısınma çalıştırılıyor...")
        warmup_audio = np.zeros(self.config.sample_rate, dtype=np.float32)
        try:
            segments, _ = self._model.transcribe(
                warmup_audio,
                language=self.config.language,
                beam_size=1,  # Hızlı ısınma
                vad_filter=False,  # VAD'a gerek yok
            )
            # Jeneratörü tüket → gerçekten çıkarım çalışsın
            for _ in segments:
                pass
            logger.info("Model ısınması tamamlandı.")
        except Exception as e:
            logger.warning("Isınma hatası (göz ardı ediliyor): %s", e)

    # ── Ses Giriş Yönetimi ────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — veriyi kuyruğa ekle, engelleme yapma."""
        if status:
            logger.warning("Mikrofon hatası: %s", status)
        # Çok kanallı giriş (ör. mikrofon + BlackHole aggregate cihazı) → mono'ya indirge
        mono = indata if indata.shape[1] == 1 else indata.mean(axis=1, keepdims=True)
        try:
            self._audio_queue.put_nowait(mono.copy())
        except queue.Full:
            # Kuyruk doluysa en eski chunk'ı at, yenisini ekle
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(mono.copy())
            except queue.Full:
                logger.warning("Audio kuyruk taşması — chunk atlandı")

    def _drain_queue(self) -> np.ndarray:
        """Kuyruktan tüm bekleyen ses verilerini oku. Gecikme birikimini önler."""
        chunks = []
        try:
            while True:
                chunks.append(np.squeeze(self._audio_queue.get_nowait()))
        except queue.Empty:
            pass

        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)

    # ── Otomatik Kazanç Kontrolü (AGC) ─────────────────────────────────────

    @staticmethod
    def _apply_gain(audio: np.ndarray, target_peak: float = 0.6, max_gain: float = 8.0) -> np.ndarray:
        """
        Sessiz mikrofon girişini yükselt. Gerçek sinyali (tepe > 1e-4) hedef
        tepe seviyesine taşır; tam sessizliği (arka plan gürültü tabanı)
        yükseltmeden bırakır ki enerji kapısı yanlışlıkla tetiklenmesin.
        """
        if audio.size == 0:
            return audio
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-4:
            return audio
        gain = min(target_peak / peak, max_gain)
        if gain <= 1.0:
            return audio
        return np.clip(audio * gain, -1.0, 1.0)

    # ── Tampon Yönetimi ────────────────────────────────────────────────────

    def _trim_buffer(self):
        """Kayan pencere: tamponu max_buffer_samples'a kırp."""
        if len(self._buffer) > self._max_buffer_samples:
            excess = len(self._buffer) - self._max_buffer_samples
            self._buffer = self._buffer[excess:]
            logger.debug("Tampon kırpıldı: %d örnek silindi", excess)

    def _clear_buffer(self):
        """Tampon ve konuşma durumunu sıfırla."""
        self._buffer = np.array([], dtype=np.float32)
        self._speech_active = False
        self._silence_samples = 0
        self.halucination_filter.reset()

    # ── Transkripsiyon ─────────────────────────────────────────────────────

    def _whisper_params(self) -> dict:
        return dict(
            language=self.config.language,
            beam_size=self.config.beam_size,
            temperature=self.config.temperature,
            vad_filter=self.config.vad_filter,
            vad_parameters=(
                dict(min_silence_duration_ms=self.config.vad_min_silence_ms)
                if self.config.vad_filter
                else None
            ),
            initial_prompt=(
                self.config.initial_prompt if self.config.initial_prompt else None
            ),
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=self.config.word_timestamps,
            suppress_blank=self.config.suppress_blank,
        )

    def _transcribe(self) -> TranscriptionResult:
        """Tam filtreleme zinciri ile transkripsiyon çalıştır (canlı tampon)."""
        start_time = time.perf_counter()

        segments, _info = self._model.transcribe(self._buffer, **self._whisper_params())

        # Segmentleri filtrele
        valid_segments, confidence = self.halucination_filter.filter_segments(segments)

        # Metni birleştir
        text = " ".join(seg.text for seg in valid_segments).strip()

        processing_ms = (time.perf_counter() - start_time) * 1000.0

        return self.halucination_filter.build_result(
            text,
            valid_segments,
            confidence,
            processing_ms,
        )

    def transcribe_file_segments(self, audio_path: str) -> List[Tuple[float, TranscriptionResult]]:
        """Bitmiş bir ses dosyasını (canlı mikrofon değil) segment segment transkript eder.

        Her segment kendi (dosya başından itibaren saniye) zaman damgasıyla
        döner - bu, konuşmacı ayrıştırma (diarization) sonuçlarının doğru
        segmente eşlenebilmesi için gerekli. Tek bir metin bloğu değil, çok
        konuşmacılı bir kayıtta her cümlenin kendi konuşmacısıyla
        etiketlenebilmesini sağlar.

        faster-whisper dosya yolunu doğrudan kabul edip kendi VAD'ıyla
        segmentlere ayırdığı için, canlı akıştaki sessizlik/max-speech
        tahminlerine gerek kalmaz.

        Whisper'ın ham segmentleri (~4-9 saniyelik VAD parçaları) genelde bir
        cümlenin ortasında bitiyor. Bu yüzden segmentleri cümle sonu
        noktalamasına (. ! ?) kadar biriktirip tek bir transkript satırı
        olarak veriyoruz - diarizasyon için gereken zaman bilgisi, biriken
        grubun İLK segmentinin başlangıcından alınır.
        """
        segments, _info = self._model.transcribe(audio_path, **self._whisper_params())

        SENTENCE_END = re.compile(r'[.!?]["\')]?\s*$')
        MAX_GROUP_SPAN_S = 40.0  # noktalama hiç gelmezse sonsuza kadar biriktirmeyi engelle

        def is_sentence_end(seg_text: str) -> bool:
            # "..." / "…" konuşmacının cümleyi yarım bırakıp duraksadığını
            # gösterir (Whisper'ın kendi konvansiyonu) - gerçek cümle sonu
            # değil, aksi halde "Bu altı ay... acaba... yapay zeka..." gibi
            # tek bir düşünce yanlışlıkla üçe/dörde bölünür.
            if seg_text.endswith("...") or seg_text.endswith("…"):
                return False
            return bool(SENTENCE_END.search(seg_text))

        results: List[Tuple[float, TranscriptionResult]] = []
        buffer: list = []
        buffer_start: Optional[float] = None

        def flush() -> None:
            if not buffer:
                return
            valid_segments, confidence = self.halucination_filter.filter_segments(buffer)
            text = " ".join(s.text for s in valid_segments).strip()
            result = self.halucination_filter.build_result(text, valid_segments, confidence, 0.0)
            if result.status != TranscriptionStatus.SILENCE and result.text:
                results.append((buffer_start, result))

        for seg in segments:
            if buffer_start is None:
                buffer_start = seg.start
            buffer.append(seg)
            if is_sentence_end(seg.text.strip()) or (seg.end - buffer_start) >= MAX_GROUP_SPAN_S:
                flush()
                buffer = []
                buffer_start = None
        flush()

        return results

    # ── Dolgu Kelimesi Temizleme (Ollama) ──────────────────────────────────

    def _clean_filler_words(self, text: str) -> str:
        """Ollama ile metinden 'eee, hmm' gibi dolgu kelimelerini temizle."""
        prompt = (
            "Your task is to correct a text generated by an STT (Speech-to-Text) model in Turkish. You must ONLY apply the following 4 corrections:\n"
            "1. Remove meaningless filler words and hesitation sounds such as 'eee', 'ııı', 'hmm', 'şey'.\n"
            "2. Correct the phonetic Turkish spelling of foreign proper nouns to their original correct spelling (e.g., 'göte' -> 'goethe', 'şilegel' -> 'schlegel', 'pihte' -> 'fichte').\n"
            "3. Fix spacing errors for combined or separated words and numbers (e.g., 'onyedi' -> 'on yedi', 'meşetteki' -> 'meşheddeki').\n"
            "4. You MUST STRICTLY follow Turkish spelling and grammar rules.\n"
            "NEVER change the original meaning of the sentence, and NEVER add or remove any other words. Return ONLY the cleaned and corrected text, without any explanations, conversational filler, or comments.\n"
            f'Original Text: "{text}"\n'
            "Corrected Text:"
        )
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # thinking modelleri tüm bütçeyi muhakemede tüketip boş cevap dönebiliyor
            "options": {"temperature": 0.0},
        }
        try:
            start_t = time.perf_counter()
            resp = requests.post(
                self.config.ollama_url,
                json=payload,
                timeout=30,
            ).json()
            result = resp.get("response", "").strip()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.debug("Ollama yanıt: '%s' (%.0fms)", result, elapsed_ms)

            return result
        except Exception as e:
            logger.error("Ollama hatası: %s", e)
            return text

    # ── Varsayılan Callback ────────────────────────────────────────────────

    @staticmethod
    def _default_callback(result: TranscriptionResult):
        """Renkli terminal çıktısı ile varsayılan sonuç işleyici."""
        if result.status == TranscriptionStatus.SILENCE:
            return

        status_map = {
            TranscriptionStatus.ACCEPTED: "\n✅ [{conf:.0%} | {ms:.0f}ms] {text}",
            TranscriptionStatus.LOW_CONFIDENCE: "\n⚠️  [{conf:.0%} | {ms:.0f}ms] {text}  (doğrulama gerekli)",
            TranscriptionStatus.REJECTED: "",  # Reddedilenleri kullanıcıya gösterme
        }

        template = status_map.get(result.status, "")
        if template:
            print(
                template.format(
                    conf=result.confidence,
                    ms=result.processing_time_ms,
                    text=result.text,
                )
            )
        elif result.status == TranscriptionStatus.REJECTED:
            logger.info(
                "Reddedildi [%.0f%%]: '%s'",
                result.confidence * 100,
                result.text,
            )

    def _do_transcribe_and_output(self):
        """
        Konuşma sonu tespit edildiğinde çağrılır.
        Tamponu TEK SEFERDE transkript eder, sonucu iletir,
        gerekirse Ollama ile dolgu kelimelerini temizler ve tamponu temizler.
        """
        if self._buffer.size == 0:
            logger.debug("Buffer boş, transkripsiyon yapılmıyor")
            return

        result = self._transcribe()

        # Debug: Her transkripsiyon sonucunu logla
        logger.info(f"Transkripsiyon: status={result.status}, text='{result.text}', confidence={result.confidence:.2f}")

        # Sonucu her zaman ilet (SILENCE bile)
        self.on_result(result)

        if result.status == TranscriptionStatus.SILENCE or not result.text:
            # Ses vardı ama anlamlı metin çıkmadı → temizle, devam et
            logger.warning("SILENCE veya boş metin - transkript kaydedildi ama işlemeye devam etmiyor")
            self._clear_buffer()
            return

        full_text = result.text.strip()

        # ── Ollama Dolgu Kelimesi Temizleme ───────────────────────────
        if self.config.use_ollama:
            cleaned_text = self._clean_filler_words(full_text)
            if not self.silent:
                print(f"\n[Ollama ile Temizlenmiş Metin]: {cleaned_text}")
            full_text = cleaned_text

        if not self.silent:
            print(f"\n>>> NİHAİ İSTEK YAKALANDI: {full_text}\n")
            print("--- Yeni Cümle Bekleniyor ---\n")

        # Ses tamponunu her zaman temizle → bir sonraki konuşma taze başlar
        self._clear_buffer()

    # ── Ana Döngü ──────────────────────────────────────────────────────────

    def start(self):
        """
        Pipeline'ı başlat. KeyboardInterrupt'a kadar engeller.

        Mimari: KONUŞMA-BAZLI İŞLEME
          1. Ses varsa → biriktir (transkripsiyon YAPMA)
          2. Sessizlik 0.8s aştı → konuşma bitti → TEK SEFERDE transkript et
          3. Tamponu temizle → sonraki konuşmayı bekle

        Bu sayede her cümle sadece 1 kez işlenir. Sıfır tekrar.
        """
        print("\n🎧 Sistem Hazır! Konuşmaya başlayın (Çıkmak için CTRL+C)\n")

        # Cihazın gerçek giriş kanal sayısını kullan (ör. mikrofon + BlackHole
        # aggregate cihazı birden çok kanal taşır) → _audio_callback mono'ya indirger
        if self.config.audio_device is not None:
            device_info = sd.query_devices(self.config.audio_device)
            input_channels = max(1, int(device_info["max_input_channels"]))
        else:
            input_channels = 1

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=input_channels,
            dtype="float32",
            callback=self._audio_callback,
            device=self.config.audio_device,
            # Whisper CPU'yu yoğun kullanırken callback'in zamanında yetişememesi
            # (xrun) ve aggregate cihazlardaki saat sapması kaynaklı kopmalara karşı
            # daha geniş donanım tamponu.
            latency="high",
        ):
            try:
                while True:
                    # İlk chunk'ı bekle (engelleyici)
                    first_chunk = np.squeeze(self._audio_queue.get())

                    # Kuyrukta biriken ek chunk'ları hemen oku
                    extra = self._drain_queue()
                    new_audio = (
                        np.concatenate([first_chunk, extra])
                        if extra.size > 0
                        else first_chunk
                    )
                    new_audio = self._apply_gain(new_audio)

                    if self.on_audio_chunk is not None:
                        self.on_audio_chunk(new_audio)

                    has_energy = self.halucination_filter.check_energy(new_audio)

                    # Her durumda tampona ekle
                    self._buffer = np.concatenate((self._buffer, new_audio))
                    self._trim_buffer()

                    if has_energy:
                        # ── Konuşma algılandı ────────────────────────
                        if not self._speech_active:
                            self._speech_active = True
                            print("🎙️", end="", flush=True)
                        self._silence_samples = 0

                        # Güvenlik: çok uzun kesintisiz konuşma → zorla transkript et
                        if (
                            self.config.force_transcription_on_max_speech
                            and len(self._buffer) >= self._max_speech_samples
                        ):
                            logger.info(
                                "Maks konuşma süresi aşıldı (%.1fs), zorla transkript ediliyor.",
                                self.config.max_speech_s,
                            )
                            self._do_transcribe_and_output()

                    elif self._speech_active:
                        # ── Konuşuyordu, şimdi sessiz ─────────────────
                        self._silence_samples += len(new_audio)

                        if self._silence_samples >= self._end_of_speech_samples:
                            # Konuşma sonu! Tek seferde transkript et.
                            self._do_transcribe_and_output()

                    # else: konuşma yok, enerji yok → beklemeye devam
            except KeyboardInterrupt:
                if not self.silent:
                    print("\nÇıkış yapılıyor...")
                if self._buffer.size > 0:
                    # Son cümle henüz sessizlik ile kapanmadıysa bile kaydet.
                    self._do_transcribe_and_output()

    def stop(self):
        """Pipeline'ı durdur ve kaynakları temizle."""
        self._clear_buffer()
        logger.info("Pipeline durduruldu.")


# ═══════════════════════════════════════════════════════════════════════════
# clean_filler_words — Geriye Uyumlu Fonksiyon
# ═══════════════════════════════════════════════════════════════════════════


def clean_filler_words(text, ollama_url, ollama_model):
    """Eski arayüz yerine dolgu kelimesi temizleme için güncellendi."""
    prompt = (
        "Görevin: Aşağıdaki metinden 'eee', 'ııı', 'hmm', 'şey' gibi düşünme belirten "
        "anlamsız sesleri ve dolgu (filler) kelimelerini temizlemektir.\n"
        "Cümlenin asıl anlamını bozma, başka kelime ekleme veya çıkarma.\n"
        "Sadece temizlenmiş metni döndür, başka hiçbir açıklama veya yorum yapma.\n"
        f'Orijinal Metin: "{text}"\n'
        "Temizlenmiş Metin:"
    )
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # thinking modelleri tüm bütçeyi muhakemede tüketip boş cevap dönebiliyor
        "options": {"temperature": 0.0},
    }
    try:
        start_t = time.time()
        response = requests.post(ollama_url, json=payload, timeout=5).json()
        result = response.get("response", "").strip()
        print(f" [Ollama Temizleme Hızı: {time.time() - start_t:.2f}s]")
        return result
    except Exception as e:
        print(f"[Ollama Hatası]: {e}")
        return text


# ═══════════════════════════════════════════════════════════════════════════
# run_stt_pipeline — Geriye Uyumlu Ana Arayüz
# ═══════════════════════════════════════════════════════════════════════════


def run_stt_pipeline(
    whisper_model="large-v3",
    device="cpu",
    compute_type="int8",
    use_ollama=True,
    ollama_url="http://localhost:11434/api/generate",
    ollama_model="qwen3.5:0.8b",
    rate=16000,
    audio_device=None,
    chunk_duration_s=1.5,
    silence_threshold=0.01,
    end_of_speech_s=2.5,
    max_speech_s=8.0,
    initial_prompt="",
    language="tr",
    vad_filter=True,
    word_timestamps=True,
    # ── Ek halüsinasyon filtreleme parametreleri (varsayılanlarla) ──────
    beam_size=5,
    max_buffer_s=15.0,
    energy_threshold=None,
    no_speech_threshold=0.3,
    avg_logprob_threshold=-0.7,
    compression_ratio_max=2.0,
    accept_confidence=0.75,
    low_confidence_min=0.50,
    vad_min_silence_ms=300,
    force_transcription_on_max_speech=True,
    on_result=None,
    meeting_mode=False,
    meeting_title="meeting",
    meeting_output_dir="meeting-notes",
    save_audio=True,
    use_ollama_summary=False,
    summary_ollama_url="http://127.0.0.1:11434/api/generate",
    summary_ollama_model="qwen3.5:0.8b",
    use_diarization=True,
    silent=False,
):
    """
    Türkçe STT pipeline'ı.

    Geriye uyumlu fonksiyon arayüzü — main.py'deki çağrı şekli korunur.
    Tüm orijinal parametreler aynen kabul edilir. Yeni parametreler
    halüsinasyon filtreleme sistemini kontrol eder.

    Args:
        whisper_model: Whisper model adı (large-v3, distil-large-v3, vb.)
        device: Hesaplama cihazı (cuda, cpu)
        compute_type: Hesaplama tipi (float16, int8, vb.)
        use_ollama: Ollama dolgu kelimesi temizlemeyi aç/kapat (True/False)
        ollama_url: Ollama API adresi
        ollama_model: Ollama model adı
        rate: Ses örnekleme hızı (Hz)
        audio_device: sounddevice giriş cihazı (index veya isim). None → sistem varsayılan mikrofon.
            Teams gibi bir uygulamanın sesini de yakalamak için BlackHole + Aggregate Device
            kurup buraya o cihazın adını/index'ini verin.
        chunk_duration_s: Anlık işleme süresi (saniye)
        max_speech_s: Kesintisiz konuşma sınırı (saniye) — aşılırsa zorla transkript edilir
        silence_threshold: Sessizlik eşiği (enerji kapısı olarak kullanılır)
        initial_prompt: Whisper ön-promptu (kelime dağarcığı yönlendirme)
        language: Dil kodu
        vad_filter: VAD filtresi aç/kapat
        word_timestamps: Kelime zaman damgaları

        beam_size: Beam genişliği (doğruluk için 5+)
        max_buffer_s: Maksimum tampon süresi (saniye)
        energy_threshold: RMS enerji kapısı eşiği (None ise silence_threshold kullanılır)
        no_speech_threshold: no_speech_prob üst sınırı
        avg_logprob_threshold: avg_logprob alt sınırı
        compression_ratio_max: compression_ratio üst sınırı
        accept_confidence: ACCEPTED durum güven eşiği
        low_confidence_min: LOW_CONFIDENCE alt sınırı
        vad_min_silence_ms: VAD minimum sessizlik süresi (ms)
        on_result: Sonuç callback'i (None ise varsayılan terminal çıktısı)
        meeting_mode: Toplantı kaydı ve özet üretimi aç/kapat
        meeting_title: Kaydedilen toplantı klasörü adı
        meeting_output_dir: Çıktıların yazılacağı klasör
        save_audio: Ham ses kaydını WAV olarak yaz
        use_ollama_summary: Toplantı özeti için Ollama kullan
        summary_ollama_url: Özet için Ollama API adresi
        summary_ollama_model: Özet için Ollama model adı
    """
    # energy_threshold belirtilmemişse, silence_threshold'u kullan
    effective_energy = (
        energy_threshold if energy_threshold is not None else silence_threshold
    )

    effective_prompt = initial_prompt

    config = STTConfig(
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        sample_rate=rate,
        audio_device=audio_device,
        chunk_duration_s=chunk_duration_s,
        max_buffer_s=max_buffer_s,
        energy_threshold=effective_energy,
        end_of_speech_s=end_of_speech_s,
        max_speech_s=max_speech_s,
        beam_size=beam_size,
        temperature=0.0,
        language=language,
        vad_filter=vad_filter,
        vad_min_silence_ms=vad_min_silence_ms,
        word_timestamps=word_timestamps,
        suppress_blank=True,
        condition_on_previous_text=False,
        initial_prompt=effective_prompt,
        no_speech_threshold=no_speech_threshold,
        avg_logprob_threshold=avg_logprob_threshold,
        compression_ratio_max=compression_ratio_max,
        accept_confidence=accept_confidence,
        low_confidence_min=low_confidence_min,
        use_ollama=use_ollama,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        silence_reset_s=15.0,
        force_transcription_on_max_speech=force_transcription_on_max_speech,
    )

    # Logging ayarla (silent mode'da WARNING, aksi takdirde DEBUG)
    log_level = logging.WARNING if silent else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Tüm loggers'ı sessiz mode'da sustur
    if silent:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        logging.getLogger("diarization").setLevel(logging.WARNING)
        logging.getLogger("pyannote").setLevel(logging.WARNING)
    else:
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    print(f"[{whisper_model}] Yükleniyor...")
    meeting_assistant = None
    if meeting_mode:
        meeting_assistant = MeetingAssistant(
            output_dir=meeting_output_dir,
            sample_rate=rate,
            save_audio=save_audio,
            use_ollama_summary=use_ollama_summary,
            ollama_url=summary_ollama_url,
            ollama_model=summary_ollama_model,
            meeting_title=meeting_title,
            use_diarization=use_diarization,
        )

    combined_on_result = on_result
    if meeting_assistant is not None:
        previous_on_result = combined_on_result

        def combined_on_result(result: TranscriptionResult) -> None:
            meeting_assistant.handle_result(result)
            if previous_on_result is not None:
                previous_on_result(result)

    pipeline = STTPipeline(
        config=config,
        on_result=combined_on_result,
        on_audio_chunk=meeting_assistant.handle_audio if meeting_assistant else None,
        silent=silent,
    )

    try:
        pipeline.start()
    finally:
        if meeting_assistant is not None:
            artifacts = meeting_assistant.finalize()
            if not silent:
                print("\nToplantı çıktıları kaydedildi:")
                print(f"- Klasör: {artifacts['session_dir']}")
                print(f"- Transkript: {artifacts['transcript_path']}")
                print(f"- Özet: {artifacts['summary_path']}")
                if artifacts["audio_path"]:
                    print(f"- Ses kaydı: {artifacts['audio_path']}")


# ═══════════════════════════════════════════════════════════════════════════
# transcribe_recording — Bitmiş Bir Ses Dosyasını İşle (Canlı Mikrofon Değil)
# ═══════════════════════════════════════════════════════════════════════════


def transcribe_recording(
    audio_path: str,
    # Batch modda gerçek zamanlı CPU kısıtı yok (kayıt bitmiş bir dosya,
    # canlı mikrofonla yarışma derdi yok), bu yüzden daha isabetli
    # transkripsiyon için "small" yerine "medium" kullanılabilir.
    whisper_model: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "tr",
    beam_size: int = 5,
    initial_prompt: str = "",
    meeting_output_dir: str = "meeting-notes",
    meeting_title: str = "kayit",
    use_ollama_summary: bool = True,
    summary_ollama_url: str = "http://127.0.0.1:11434/api/generate",
    summary_ollama_model: str = "qwen3.5:0.8b",
    use_diarization: bool = False,
) -> dict:
    """Daha önce kaydedilmiş bir ses dosyasını (Teams kaydı, telefon kaydı vb.)
    tek seferde transkript edip özetler. Canlı mikrofon akışı kullanmaz,
    bu yüzden sessizlik/max-speech tahminlerine ve gerçek zamanlı CPU
    kısıtlarına gerek yoktur.
    """
    # Yazılım/teknoloji ve bankacılık terimleri her transkripsiyonda otomatik
    # kullanılır; burada verilen initial_prompt (ör. web arayüzündeki ek
    # terimler) bunlara eklenir, üzerine yazmaz.
    effective_prompt = f"{DEFAULT_TECHNICAL_TERMS} {BANKING_TERMS}"
    if initial_prompt:
        effective_prompt = f"{effective_prompt} {initial_prompt}"

    config = STTConfig(
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        beam_size=beam_size,
        temperature=0.0,
        vad_filter=True,
        word_timestamps=True,
        suppress_blank=True,
        condition_on_previous_text=False,
        initial_prompt=effective_prompt,
        # STTConfig'in varsayılan eşikleri canlı akış için sıkı tutulmuştu.
        # Burada Whisper'ın kendi VAD'ı zaten sessizliği ayıklıyor, bu yüzden
        # segment eşiklerini main.py'deki canlı moddaki gibi gevşetiyoruz -
        # aksi halde kısa/gürültülü kayıtlarda tüm segmentler reddedilip
        # transkript tamamen boş çıkabiliyor.
        no_speech_threshold=0.95,
        avg_logprob_threshold=-1.0,
        compression_ratio_max=10.0,
        accept_confidence=0.5,
        low_confidence_min=0.3,
    )

    pipeline = STTPipeline(config=config, silent=True)

    meeting_assistant = MeetingAssistant(
        output_dir=meeting_output_dir,
        sample_rate=config.sample_rate,
        save_audio=False,
        use_ollama_summary=use_ollama_summary,
        ollama_url=summary_ollama_url,
        ollama_model=summary_ollama_model,
        meeting_title=meeting_title,
        use_diarization=use_diarization,
    )
    # Diarization, canlı kayıttaki recording.wav yerine doğrudan verilen dosyayı kullansın.
    meeting_assistant._audio_path = Path(audio_path)

    # Whisper transkripsiyonu ve diarizasyon birbirinden bağımsız işlemler
    # (ikisi de aynı ses dosyasını okuyor, biri diğerinin çıktısına ihtiyaç
    # duymuyor). Sırayla değil paralel çalıştırarak toplam süreyi
    # (whisper_süresi + diarizasyon_süresi) yerine max(ikisi) yapıyoruz.
    diarization_future = None
    if meeting_assistant._diarization_pipeline and meeting_assistant._diarization_pipeline.use_diarization:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        diarization_future = executor.submit(
            meeting_assistant._diarization_pipeline.get_speaker_segments,
            audio_path,
            sr=config.sample_rate,
        )

    segment_results = pipeline.transcribe_file_segments(audio_path)

    if diarization_future is not None:
        try:
            meeting_assistant.set_diarization_segments(diarization_future.result())
        except Exception as e:
            logger.warning("Paralel diarizasyon başarısız: %s", e)
        finally:
            executor.shutdown(wait=False)

    for start_sec, result in segment_results:
        meeting_assistant.add_segment(start_sec, result)
    artifacts = meeting_assistant.finalize()

    accepted = [r for _, r in segment_results if r.status == TranscriptionStatus.ACCEPTED]
    overall_status = (
        "ACCEPTED" if accepted
        else "LOW_CONFIDENCE" if any(r.status == TranscriptionStatus.LOW_CONFIDENCE for _, r in segment_results)
        else "REJECTED" if segment_results
        else "SILENCE"
    )
    overall_confidence = (
        sum(r.confidence for _, r in segment_results) / len(segment_results)
        if segment_results else 0.0
    )

    return {
        **artifacts,
        "text": " ".join(r.text for _, r in segment_results).strip(),
        "confidence": overall_confidence,
        "status": overall_status,
    }
