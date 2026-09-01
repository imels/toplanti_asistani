from core import run_stt_pipeline

if __name__ == "__main__":
    # Toplantı asistanı modu: konuşmayı kaydeder, transkript ve özet üretir.
    run_stt_pipeline(
        # ── 1. Model Yapılandırması ──────────────────────────────────────
        whisper_model="medium",
        device="cpu",                       # "cuda" veya "cpu"
        compute_type="int8",                # float16, int8, bfloat16, int4

        # ── 2. Ses Akışı Yönetimi ────────────────────────────────────────
        rate=16000,                         # Örnekleme hızı (Hz)
        chunk_duration_s=1.5,               # Anlık işleme paket süresi (saniye)
        silence_threshold=0.01,             # RMS gürültü/sessizlik kapısı
        end_of_speech_s=2.5,                # Cümle bitişi için sessizlik eşiği (saniye)
        force_transcription_on_max_speech=False, # Maksimum süre aşımında zorla çeviri

        # ── 3. Whisper Çıkarım Ayarları ──────────────────────────────────
        initial_prompt="",                  # Kelime dağarcığı yönlendirme (Boşsa varsayılan bankacılık promptu)
        language="tr",                      # Hedef dil
        beam_size=5,                        # Arama genişliği (Toplantı akışı için daha dengeli)

        # ── 4. Ses Aktivite Filtresi (VAD) ───────────────────────────────
        vad_filter=True,                    # VAD filtresini aktif et
        vad_min_silence_ms=900,             # Kısa duraksamaları bölmemek için daha toleranslı
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
        use_ollama_summary=False,
        summary_ollama_url="http://127.0.0.1:11434/api/generate",
        summary_ollama_model="qwen3.5:0.8b",
    )
