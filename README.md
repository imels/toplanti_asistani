# Toplantı Asistanı

Ses kaydından (toplantı, daily, konuşma vb.) otomatik olarak **transkript**, **konuşmacı ayrımı (diarizasyon)** ve **özet** üreten bir araç. Türkçe konuşma için optimize edilmiştir.

## Özellikler

- **Web arayüzü** ([webapp.py](webapp.py)): tarayıcıdan ses dosyası yükleme ya da doğrudan tarayıcıda kayıt alma (mikrofon + paylaşılan sekmenin sesi).
- **Transkripsiyon**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) ile Türkçe konuşmadan metne çeviri, halüsinasyon/dolgu kelimesi filtreleme.
- **Konuşmacı ayrımı**: [pyannote.audio](https://github.com/pyannote/pyannote-audio) ile "kim ne zaman konuştu" tespiti, aynı konuşmacının art arda gelen cümleleri tek paragrafta birleştirilir.
- **Özet**: [Ollama](https://ollama.com) üzerinden yerel bir LLM ile kısa, madde madde toplantı özeti.
- **PDF indirme**: transkript ve özeti başlık + tarihli PDF olarak indirme.
- **Teknik terim desteği**: yazılım/DevOps/Agile ve bankacılık alanına özgü terimler Whisper'a önceden tanıtılır (bkz. [config.py](config.py) → `DEFAULT_TECHNICAL_TERMS`, `BANKING_TERMS`).

## Mimari

İki ayrı kullanım modu var:

1. **Batch/dosya modu (önerilen, web arayüzü bunu kullanır)** — bitmiş bir ses dosyasını (kayıt, upload) işler. Whisper transkripsiyonu ve diarizasyon **paralel** çalıştırılır, gerçek zamanlı kısıt yoktur. Giriş noktası: `core.transcribe_recording()`.
2. **Canlı mikrofon modu (eski, main.py)** — mikrofonu sürekli dinler, konuşma bitince anında transkript eder. Gerçek zamanlı CPU kısıtları vardır (bkz. Bilinen Sınırlamalar). Giriş noktası: `core.run_stt_pipeline()`.

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

`pyannote/speaker-diarization-3.1` gated (onay gerektiren) bir model:

1. [huggingface.co](https://huggingface.co) hesabı açın.
2. [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) ve [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) sayfalarında kullanım koşullarını kabul edin.
3. [hf.co/settings/tokens](https://hf.co/settings/tokens) üzerinden bir Read token oluşturun.
4. Kalıcı olarak tanımlayın:
   ```bash
   echo 'export HF_TOKEN=hf_xxxxxxxx' >> ~/.zshrc
   source ~/.zshrc
   ```

### Ollama (özet için)

```bash
ollama pull qwen3.5:0.8b   # ya da daha iyi kalite için: qwen3.5:4b
```

Ollama'nın `http://127.0.0.1:11434` üzerinde çalışıyor olması yeterli, ayrıca başlatmaya gerek yok.

## Kullanım

### Web arayüzü (önerilen)

```bash
python3 webapp.py
```

Tarayıcıda `http://127.0.0.1:5000` adresine gidin:
- **Dosya yükleyin** (wav/mp3/m4a/mp4/webm/caf ve diğer yaygın formatlar), **veya**
- **"Kaydı Başlat"** ile tarayıcıdan doğrudan kayıt alın (Chrome/Edge önerilir — paylaşım penceresinde **"Sekme"**yi seçip **"Sekme sesini paylaş"**ı işaretleyin).

İşlem bitince transkript, özet ve PDF indirme linkleri sayfada görünür. Sonuçlar ayrıca `meeting-notes/<oturum-adı>/` klasörüne de kaydedilir.

### Canlı mikrofon modu

```bash
python3 main.py
```

Sürekli mikrofonu dinler, konuşma bitince anında transkript eder. `Ctrl+C` ile durdurunca özet üretilir.

## Bilinen Sınırlamalar

- **Diarizasyon kalitesi ses kalitesine çok bağlı** — gürültülü/sıkıştırılmış (webm) kayıtlarda pyannote gerçekte olduğundan fazla "konuşmacı" bulabilir. Katılımcı sayısını biliyorsanız `get_speaker_segments(..., num_speakers=N)` ile belirtmek doğruluğu artırır.
- **Özet kalitesi kullanılan Ollama modeline bağlı** — `qwen3.5:0.8b` çok küçük bir model, uzun/çok kişili toplantılarda tekrar/uydurma yapabiliyor. Daha iyi sonuç için `qwen3.5:4b` gibi daha büyük bir model önerilir.
- **Canlı mikrofon modunda büyük Whisper modelleri (medium/large) CPU'da ses kopmasına yol açabilir** — gerçek zamanlı ses yakalama ile model çıkarımı aynı CPU'yu paylaştığı için. Batch modda bu sorun yoktur.
- **İsim bilgisi yok** — diarizasyon sadece "Konuşmacı 1/2/3" gibi anonim etiketler verir, gerçek isimleri bilemez (ses kümeleme, kimlik doğrulama değildir).
- **Web arayüzü tek seferde tek isteği işler** (Flask `threaded=True` değil) — bir dosya işlenirken başka bir sekmeden istek atarsanız kuyrukta bekler.
- **Model her istekte yeniden yükleniyor** — `webapp.py` her dosya için Whisper ve diarizasyon modelini sıfırdan yüklüyor; sunucuya taşınırken modellerin sürekli açık bir servis olarak çalıştırılması (her istekte yeniden yükleme yapılmaması) performansı ciddi iyileştirir.
