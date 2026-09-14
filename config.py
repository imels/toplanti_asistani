"""
STT yapılandırması.

Tüm pipeline parametreleri, güven eşikleri ve Türkçe halüsinasyon
kalıpları burada merkezi olarak tanımlanır.
"""

import re
from dataclasses import dataclass
from typing import Optional, Union


# ─── Varsayılan Teknik Terim Listesi ──────────────────────────────────────────
# Whisper'a önceden verilen bu terimler, yazılım/teknoloji toplantılarında sık
# geçen kelimelerin doğru yazılmasına yardımcı olur (initial_prompt olarak
# kullanılır). Kendi alanınıza göre bu listeyi düzenleyin/genişletin.
DEFAULT_TECHNICAL_TERMS = (
    # Genel yazılım/mimari terimleri
    "API, backend, frontend, veritabanı, sunucu, entegrasyon, versiyon, "
    "deployment, sprint, bug, güncelleme, algoritma, yapay zeka, bulut, "
    "mikroservis, DevOps, repository, commit, pull request, framework, "
    "kütüphane, endpoint, token, önbellek, konteyner, pipeline, test ortamı, "
    "staging, production, webhook, OAuth, JWT, SSO, REST API, GraphQL, gRPC, "
    "WebSocket, load balancer, CI/CD, unit test, entegrasyon testi, "
    "client, müşteri, change, change request, change management, ticket, "
    "approval, CAB, rollout, environment, config, "
    # Bulut/altyapı araçları
    "AWS, Azure, Google Cloud, GCP, Kubernetes, Docker, Terraform, Ansible, "
    "Nginx, Jenkins, GitHub Actions, CircleCI, "
    # Sürüm kontrol/işbirliği araçları
    "GitHub, GitLab, Bitbucket, Jira, Confluence, Slack, Notion, Figma, "
    "Trello, Asana, "
    # Veritabanları ve vektör veritabanları
    "PostgreSQL, MySQL, SQLite, MongoDB, Redis, Elasticsearch, Kafka, "
    "RabbitMQ, Qdrant, Pinecone, Weaviate, Milvus, Chroma, vektör veritabanı, "
    # Yapay zeka / makine öğrenmesi
    "makine öğrenmesi, derin öğrenme, LLM, büyük dil modeli, GPT, Claude, "
    "Gemini, embedding, gömme vektörü, RAG, transformer, fine-tuning, "
    "PyTorch, TensorFlow, Hugging Face, LangChain, OpenAI, Anthropic, "
    "Whisper, Ollama, prompt, token limiti, "
    # Diller ve framework'ler
    "Python, JavaScript, TypeScript, Java, Go, Rust, Swift, Kotlin, "
    "React, Vue, Angular, Next.js, Node.js, Django, Flask, FastAPI, "
    "Spring Boot, .NET, "
    # İzleme, test, geliştirme araçları
    "Grafana, Prometheus, Datadog, Sentry, Postman, Swagger, Selenium, "
    "Cypress, VSCode, IntelliJ, "
    # Git / sürüm kontrolü iş akışı
    "commit, push, pull, merge, merge conflict, rebase, branch, checkout, "
    "clone, fork, pull request, merge request, code review, cherry-pick, "
    "squash, revert, stash, tag, changelog, hotfix, rollback, release, "
    # Agile / Scrum toplantı terimleri
    "daily, daily stand-up, sprint planning, sprint review, retrospective, "
    "backlog, backlog grooming, refinement, story point, epic, kanban, "
    "scrum, product owner, demo, roadmap, milestone, "
    # Diğer yaygın geliştirme terimleri
    "pair programming, refactor, teknik borç, regression, "
    "QA, UAT, on-call, incident, root cause analysis."
)

# Bankacılık/finans toplantılarında sık geçen terimler. DEFAULT_TECHNICAL_TERMS
# ile birlikte kullanılır (bkz. core.py transcribe_recording).
BANKING_TERMS = (
    "vadeli hesap, vadesiz hesap, hesap numarası, IBAN, SWIFT, havale, EFT, "
    "FAST, kredi kartı, banka kartı, mevduat, faiz oranı, Türk Lirası, TL, "
    "döviz kuru, hesap bakiyesi, ekstre, para transferi, transfer et, yatır, "
    "çek, senet, ödeme, taksit, kredi, kredi notu, kredi limiti, POS, ATM, "
    "şube, otomatik ödeme talimatı, dijital bankacılık, mobil bankacılık, "
    "internet bankacılığı, açık bankacılık, kripto para, BDDK, TCMB, SPK, "
    "risk yönetimi, likidite, teminat, komisyon, swap, tahvil, bono, portföy."
)

# ─── Türkçe Whisper Halüsinasyon Kalıpları ────────────────────────────────────
# Whisper'ın eğitim verisindeki Türkçe altyazı dosyalarından ve YouTube
# içeriklerinden kaynaklanan bilinen sahte çıktılar.
# Sessizlikte veya gürültüde model bu kalıpları "uydurur".
TURKISH_HALLUCINATION_PATTERNS: list[re.Pattern] = [
    # Altyazı meta verileri
    re.compile(r"\baltyazı\b", re.IGNORECASE),
    re.compile(r"\bM\.K\.?\b"),
    re.compile(r"\bsubtitle\b", re.IGNORECASE),
    # Çizgi film / müzik referansları
    re.compile(r"\bçizgi\s*film\b", re.IGNORECASE),
    re.compile(r"\bmüzik\s*çalıyor\b", re.IGNORECASE),
    re.compile(r"\bmüziği\b", re.IGNORECASE),
    # YouTube / sosyal medya kalıpları
    re.compile(r"\babone\s*ol\b", re.IGNORECASE),
    re.compile(r"\bbeğen\b", re.IGNORECASE),
    re.compile(r"\biyi\s*seyirler\b", re.IGNORECASE),
    re.compile(r"\byoutube\b", re.IGNORECASE),
    re.compile(r"\bwww\b", re.IGNORECASE),
    re.compile(r"\.com\b", re.IGNORECASE),
    # Tek başına "teşekkürler" (cümle içinde olursa sorun yok)
    re.compile(r"^\s*teşekkürler\.?\s*$", re.IGNORECASE),
    # "İzlediğiniz için teşekkür" kalıbı (video sonu halüsinasyonu)
    re.compile(r"izlediğiniz\s*için\s*teşekkür", re.IGNORECASE),
    # Müzik sembolleri
    re.compile(r"[♪🎵🎶]"),
]

# Türkçe dolgu/duraksama sesleri - "şey" gibi gerçek anlamı olan kelimeler
# kasıtlı olarak dışarıda bırakıldı (yanlışlıkla gerçek içerik silinmesin diye).
TURKISH_FILLER_PATTERN = re.compile(
    r"\b(?:e{2,}|ı{2,}|ee|ıh+|hı+m*|h[ıi]mm+|aa+|öö+)\b",
    re.IGNORECASE,
)

@dataclass
class STTConfig:
    """STT pipeline yapılandırması."""

    # ── Model ──────────────────────────────────────────────────
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"

    # ── Ses ────────────────────────────────────────────────────
    audio_device: Optional[Union[int, str]] = None  # sounddevice giriş cihazı (index/isim). None → sistem varsayılanı
    sample_rate: int = 16000
    chunk_duration_s: float = 1  # Düşük gecikme için 1s (eskiden 1.5)
    max_buffer_s: float = 8.0  # Kayan pencere sınırı (8s → maks ~800ms işlem süresi)
    energy_threshold: float = 0.0001  # RMS enerji kapısı (ÇOK DÜŞÜK - hemen her şeyi kabul et)
    end_of_speech_s: float = 1.2  # Konuşma sonu sessizlik eşiği (saniye) - Nefes alma/düşünme payı için 1.2s idealdir.
    max_speech_s: float = 8.0  # Kesintisiz konuşma sınırı → zorla transkript et
    force_transcription_on_max_speech: bool = (
        False  # Zorla transkript etmeyi açıp kapatmak için
    )

    # ── Whisper Ayarları ───────────────────────────────────────
    beam_size: int = 16  # Maksimum hız için 1 (Greedy decoding). 5 çok yavaştır.
    temperature: float = 0.0  # Deterministik çıktı
    language: str = "tr"
    vad_filter: bool = True
    vad_min_silence_ms: int = 600  # VAD sessizlik eşiği
    word_timestamps: bool = True  # Hizalama zorunluluğu → halüsinasyon azaltır
    suppress_blank: bool = True  # Boş token bastırma
    condition_on_previous_text: bool = False  # Geri besleme döngüsünü kır

    initial_prompt: str = ""

    # ── Güven Eşikleri ─────────────────────────────────────────
    no_speech_threshold: float = 0.3  # Sıkı (eskiden 0.5)
    avg_logprob_threshold: float = -0.7  # Sıkı eşik
    compression_ratio_max: float = 2.0  # Tekrarlı döngüleri yakala
    accept_confidence: float = 0.80  # ≥ 0.75 → ACCEPTED
    low_confidence_min: float = 0.50  # 0.50-0.74 → LOW_CONFIDENCE, < 0.50 → REJECTED

    # ── Ollama Dolgu Kelimesi (Filler) Temizleme ───────────────
    use_ollama: bool = False
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_model: str = "qwen3.5:0.8b"
    silence_reset_s: float = 15.0  # Otomatik sıfırlama süresi (kullanım dışı)
