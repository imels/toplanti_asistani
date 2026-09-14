from core import run_stt_pipeline

if __name__ == "__main__":
    # Toplantı asistanı modu: konuşmayı kaydeder, transkript ve özet üretir.
    run_stt_pipeline(
        # ── 1. Model Yapılandırması ──────────────────────────────────────
        whisper_model="small",              # TEST: CPU yükünü azaltıp kopmanın CPU kaynaklı olup olmadığını görmek için
        device="cpu",                       # "cuda" veya "cpu"
        compute_type="int8",                # float16, int8, bfloat16, int4

        # ── 2. Ses Akışı Yönetimi ────────────────────────────────────────
        rate=16000,                         # Örnekleme hızı (Hz)
        audio_device=4,                     # "Teams Toplantı Girişi" aggregate cihazı (mikrofon + BlackHole)
                                             # (cihazları görmek için: python list_audio_devices.py)
        chunk_duration_s=1.5,               # Anlık işleme paket süresi (saniye)
        silence_threshold=0.001,            # RMS gürültü/sessizlik kapısı (ÇOK DÜŞÜK - her şeyi kabul et)
        end_of_speech_s=2.5,                # Cümle bitişi için sessizlik eşiği (saniye)
        max_speech_s=10.0,                  # Kesintisiz konuşma sınırı (saniye) - aşılırsa zorla transkript
        force_transcription_on_max_speech=True, # Maksimum süre aşımında zorla çeviri

        # ── 3. Whisper Çıkarım Ayarları ──────────────────────────────────
        initial_prompt="",                  # Kelime dağarcığı yönlendirme (opsiyonel)
        language="tr",                      # Hedef dil
        beam_size=5,                        # Arama genişliği (Toplantı akışı için daha dengeli)

        # ── 4. Ses Aktivite Filtresi (VAD) ───────────────────────────────
        vad_filter=False,                   # VAD filtresini KAPAT (çok katı olabilir)
        vad_min_silence_ms=500,             # VAD minimum sessizlik (AZALTILDI - daha az sessiz)
        word_timestamps=True,               # Kelime zaman damgaları (Zorunlu - halüsinasyon engeller)

        # ── 5. Ollama Dolgu Kelimesi Temizleme ───────────────────────────
        use_ollama=False,                   # Ollama ile 'eee, hmm' gibi kelimeleri temizleme
        ollama_url="http://localhost:11434/api/generate",
        ollama_model="qwen3.5:0.8b",

        # ── 6. Toplantı Asistanı Çıktıları ────────────────────────────────
        meeting_mode=True,
        meeting_title="toplanti",
        meeting_output_dir="meeting-notes",
        save_audio=True,
        use_ollama_summary=True,  # Ollama ile özet yap
        summary_ollama_url="http://127.0.0.1:11434/api/generate",
        summary_ollama_model="qwen3.5:0.8b",
        
        # ── 7. Güven Eşikleri ────────────
        accept_confidence=0.5,              # 50% üzeri kabul et
        low_confidence_min=0.3,             # 30-50% arası LOW_CONFIDENCE
        no_speech_threshold=0.95,           # Çok yüksek - neredeyse her şeyi konuşma kabul et
        avg_logprob_threshold=-1.0,         # Çok toleranslı
        compression_ratio_max=10.0,         # Çok toleranslı
        
        # ── 8. Konuşmacı Ayrıştırma (Speaker Diarization) ───────────────────
        use_diarization=False,              # PyAnnote devre dışı (HF gated model onayı gerekiyor, şimdilik ertelendi)
        
        # ── 9. Sessiz Mode ────────────────────────────────────────────────
        silent=False,                       # TEST: terminalde canlı transkript göster (test sonrası True'ya alın)
    )
