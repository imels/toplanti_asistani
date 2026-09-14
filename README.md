# Toplantı Asistanı

Ses kaydından (toplantı, daily, konuşma vb.) otomatik olarak **transkript**, **konuşmacı ayrımı (diarizasyon)** ve **özet** üretir. Türkçe konuşmaya göre ayarlı.

## Özellikler

- **Web arayüzü** ([webapp.py](webapp.py)): tarayıcıdan ses dosyası yükleme ya da doğrudan tarayıcıda kayıt alma (mikrofon + paylaşılan sekmenin sesi).
- **Transkripsiyon**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) ile Türkçe konuşmadan metne çeviri, halüsinasyon/dolgu kelimesi filtreleme.
- **Konuşmacı ayrımı**: [pyannote.audio](https://github.com/pyannote/pyannote-audio) ile "kim ne zaman konuştu" tespiti, aynı konuşmacının art arda gelen cümlelerini tek paragrafta birleştirir.
- **Özet**: [Ollama](https://ollama.com) üzerinden yerel bir LLM ile kısa, madde madde toplantı özeti çıkarır.
- **PDF indirme**: transkript ve özeti başlık + tarihli PDF olarak indirir.
- **Teknik terim desteği**: yazılım/DevOps/Agile ve bankacılık alanına özgü terimleri Whisper'a önceden veriyorum (bkz. [config.py](config.py) → `DEFAULT_TECHNICAL_TERMS`, `BANKING_TERMS`).

## Mimari

İki kullanım modu var:

1. **Batch/dosya modu** — web arayüzünün kullandığı mod. Bitmiş bir ses dosyasını (kayıt, upload) işler. Whisper transkripsiyonu ve diarizasyon paralel çalışır, gerçek zamanlı kısıt yoktur. Giriş noktası: `core.transcribe_recording()`.
2. **Canlı mikrofon modu** (`main.py`) — mikrofonu sürekli dinler, konuşma bitince anında transkript eder. Gerçek zamanlı CPU kısıtları vardır (bkz. Bilinen Sınırlamalar). Giriş noktası: `core.run_stt_pipeline()`.

### Dosya yapısı

| Dosya | Görevi |
|---|---|
| `webapp.py` | Flask web sunucusu — yükleme/kayıt arayüzü, `/process`, `/download` route'ları |
| `core.py` | `STTPipeline` (Whisper motoru), `transcribe_recording()` (batch akış), `run_stt_pipeline()` (canlı akış) |
| `meeting_assistant.py` | `MeetingAssistant` — transkript/özet dosyalarını üretir, konuşmacı etiketleme, Ollama özet çağrısı |
| `diarization.py` | `DiarizationPipeline` — pyannote ile konuşmacı ayrımı, PyTorch/speechbrain uyumluluk düzeltmeleri |
| `filters.py` | `HallucinationFilter` — Whisper çıktısını güven skoruna göre filtreleme |
| `config.py` | `STTConfig`, teknik terim listeleri, halüsinasyon kalıpları |
| `pdf_export.py` | Markdown transkript/özeti PDF'e çevirme |
| `models.py` | `TranscriptionResult`, `TranscriptionStatus` veri sınıfları |
| `main.py` | Canlı mikrofon modu için terminal giriş noktası |
| `list_audio_devices.py` | Sistemdeki ses cihazlarını listeleyen yardımcı script |
| `benchmark.py` | Whisper performans testi (web akışından bağımsız) |

## Kurulum

```bash
pip3 install -r requirements.txt
```

### Diarizasyon için HuggingFace token'ı

`pyannote/speaker-diarization-3.1` gated bir model:

1. [huggingface.co](https://huggingface.co) hesabı.
2. [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) ve [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) sayfalarında kullanım koşullarının kabulü.
3. [hf.co/settings/tokens](https://hf.co/settings/tokens) üzerinden bir Read token.
4. Kalıcı tanım:
   ```bash
   echo 'export HF_TOKEN=hf_xxxxxxxx' >> ~/.zshrc
   source ~/.zshrc
   ```

### Ollama (özet için)

```bash
ollama pull qwen3.5:0.8b
```

Ollama `http://127.0.0.1:11434` üzerinde çalışır, ayrıca başlatma gerekmez.

## Kullanım

### Web arayüzü

```bash
python3 webapp.py
```

`http://127.0.0.1:5000`:
- Dosya yükleme (wav/mp3/m4a/mp4/webm/caf ve diğer yaygın formatlar), veya
- "Kaydı Başlat" ile tarayıcıdan doğrudan kayıt (mikrofon + paylaşılan sekmenin sesi). Sekme sesi paylaşımı Chrome/Edge'de çalışır, Safari bunu desteklemiyor.

İşlem bitince transkript, özet ve PDF indirme linkleri sayfada görünür. Sonuçları ayrıca `meeting-notes/<oturum-adı>/` klasörüne kaydediyorum.

### Canlı mikrofon modu

```bash
python3 main.py
```

Mikrofonu sürekli dinler, konuşma bitince anında transkript eder. `Ctrl+C` ile durdurunca özet çıkarır.

## Bilinen Sınırlamalar

- Diarizasyon kalitesi ses kalitesine bağlı — gürültülü/sıkıştırılmış (webm) kayıtlarda pyannote gerçekte olduğundan fazla "konuşmacı" bulabiliyor. `get_speaker_segments(..., num_speakers=N)` katılımcı sayısını sabitler.
- Özet kalitesi kullanılan Ollama modeline bağlı — `qwen3.5:0.8b` küçük bir model, uzun/çok kişili toplantılarda tekrar/uydurma yapabiliyor.
- Canlı mikrofon modunda büyük Whisper modelleri (medium/large) CPU'da ses kopmasına yol açabiliyor — gerçek zamanlı ses yakalama ile model çıkarımı aynı CPU'yu paylaşıyor. Batch modda bu kısıt yok.
- Diarizasyon sadece "Konuşmacı 1/2/3" gibi anonim etiketler veriyor, gerçek isim bilgisi yok (ses kümeleme, kimlik doğrulama değil).
- Web arayüzü tek seferde tek isteği işliyor (Flask `threaded=True` değil) — webapp bir dosyayı işlerken başka bir istek kuyrukta bekliyor.
- `webapp.py` her istekte Whisper ve diarizasyon modelini sıfırdan yüklüyor, bu ek gecikmeye neden oluyor.
