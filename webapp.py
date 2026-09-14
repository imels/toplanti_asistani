"""Basit web arayüzü: bir ses kaydı yükle, transkript + özet al.

Canlı mikrofon dinlemez — sadece bitmiş bir ses dosyasını (Teams kaydı,
telefon kaydı vb.) işler. Çalıştırmak için:

    python3 webapp.py

sonra tarayıcıda http://127.0.0.1:5000 adresine gidin.
"""

import io
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, render_template_string, request, send_file
from werkzeug.utils import secure_filename

from core import transcribe_recording
from pdf_export import markdown_to_pdf_bytes

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MEETING_NOTES_DIR = Path("meeting-notes").resolve()
MEETING_NOTES_DIR.mkdir(exist_ok=True)
DOWNLOADABLE_FILES = {"transcript.md", "summary.md"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

PAGE_STYLE = """
<style>
  :root {
    --paper: #F5F6F4;
    --surface: #FFFFFF;
    --ink: #1A211E;
    --muted: #6B756F;
    --line: #E2E6E1;
    --accent: #2E6F68;
    --accent-ink: #1D4A45;
    --accent-soft: rgba(46,111,104,0.08);
    --accent-soft-strong: rgba(46,111,104,0.14);
    --warn: #B5652E;
    --warn-soft: rgba(181,101,46,0.10);
    --bad: #B4443C;
    --bad-soft: rgba(180,68,60,0.10);
    --radius: 14px;
    --shadow: 0 1px 2px rgba(20,25,22,0.04), 0 8px 24px rgba(20,25,22,0.06);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #141815;
      --surface: #1B211E;
      --ink: #E8ECE6;
      --muted: #93A19A;
      --line: #2A322D;
      --accent: #57B0A5;
      --accent-ink: #8FD4CB;
      --accent-soft: rgba(87,176,165,0.12);
      --accent-soft-strong: rgba(87,176,165,0.20);
      --warn: #E0965A;
      --warn-soft: rgba(224,150,90,0.12);
      --bad: #E07A72;
      --bad-soft: rgba(224,122,114,0.12);
      --shadow: 0 1px 2px rgba(0,0,0,0.2), 0 8px 24px rgba(0,0,0,0.35);
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--paper);
    color: var(--ink);
    margin: 0;
    padding: 48px 20px 80px;
    line-height: 1.5;
  }
  .wrap { max-width: 760px; margin: 0 auto; }
  header.page-head { margin-bottom: 28px; }
  h1 { font-size: 24px; font-weight: 650; margin: 0 0 6px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 14.5px; margin: 0; }

  .card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 26px;
    margin-top: 18px;
    box-shadow: var(--shadow);
  }
  .card h2 { font-size: 15px; font-weight: 650; margin: 0; display: flex; align-items: center; gap: 8px; }
  .card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .download-btn {
    font-size: 12.5px; font-weight: 600; color: var(--accent-ink);
    background: var(--accent-soft); padding: 5px 12px; border-radius: 7px;
    text-decoration: none; white-space: nowrap;
  }
  .download-btn:hover { background: var(--accent-soft-strong); }

  /* ── Dropzone ──
     Not: kesikli (dashed) border + border-radius kombinasyonu Safari/WebKit'te
     düzensiz, kalın parçalar halinde render oluyor (bilinen bir motor hatası).
     Bu yüzden düz (solid) ince bir çizgi kullanıyoruz. */
  .dropzone {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    border: 1.5px solid var(--line);
    border-radius: 12px;
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color .15s ease, background .15s ease;
    background: var(--paper);
  }
  .dropzone.drag { border-color: var(--accent); background: var(--accent-soft); }
  .dropzone .icon { font-size: 28px; margin-bottom: 8px; }
  .dropzone .primary-text { font-weight: 600; font-size: 15px; }
  .dropzone .secondary-text { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .dropzone .filename {
    margin-top: 14px; font-size: 13.5px; font-weight: 600; color: var(--accent-ink);
    background: var(--accent-soft); border-radius: 8px; padding: 6px 12px; display: none;
  }
  input[type=file] { display: none; }

  .record-row { display: flex; align-items: center; gap: 14px; }
  .record-btn {
    background: var(--bad-soft, var(--accent-soft)); color: var(--bad, var(--accent-ink));
    border: 1px solid var(--line); padding: 11px 20px; border-radius: 9px;
    font-size: 14.5px; font-weight: 600; cursor: pointer;
  }
  .record-btn.recording {
    background: var(--bad); color: #fff; animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
  .record-timer {
    font-family: "SF Mono", ui-monospace, monospace; font-size: 14px;
    font-weight: 600; color: var(--muted);
  }

  .divider {
    display: flex; align-items: center; gap: 12px; margin: 22px 0;
    color: var(--muted); font-size: 12.5px;
  }
  .divider::before, .divider::after {
    content: ""; flex: 1; height: 1px; background: var(--line);
  }

  button.primary {
    background: var(--accent); color: #fff; border: none;
    padding: 12px 22px; border-radius: 9px; font-size: 15px; font-weight: 600;
    cursor: pointer; width: 100%; margin-top: 18px;
    display: flex; align-items: center; justify-content: center; gap: 10px;
    transition: background .15s ease;
  }
  button.primary:hover { background: var(--accent-ink); }
  button.primary:disabled { opacity: 0.75; cursor: default; }

  .spinner {
    width: 16px; height: 16px; border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff;
    animation: spin 0.7s linear infinite; display: none;
  }
  button.primary.loading .spinner { display: inline-block; }
  button.primary.loading .btn-label::after { content: "İşleniyor…"; }
  button.primary.loading .btn-label { visibility: hidden; position: relative; }
  button.primary.loading .btn-label::after { visibility: visible; position: absolute; left: 0; right: 0; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .hint { color: var(--muted); font-size: 12.5px; margin-top: 10px; text-align: center; }

  .alert {
    margin-top: 16px; padding: 14px 16px; border-radius: 10px;
    background: var(--bad-soft); color: var(--bad); font-size: 14px;
    border-left: 3px solid var(--bad);
  }

  /* ── Result page ── */
  .meta-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
  .pill {
    font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px;
    background: var(--accent-soft); color: var(--accent-ink);
  }
  .pill.status-LOW_CONFIDENCE { background: var(--warn-soft); color: var(--warn); }
  .pill.status-REJECTED, .pill.status-SILENCE { background: var(--bad-soft); color: var(--bad); }

  pre.content {
    white-space: pre-wrap; word-wrap: break-word;
    background: var(--paper); border: 1px solid var(--line);
    padding: 16px; border-radius: 10px; font-size: 14px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    max-height: 480px; overflow-y: auto; margin: 0;
  }

  .back-link {
    display: inline-block; margin-top: 22px; color: var(--accent-ink);
    font-size: 14px; font-weight: 600; text-decoration: none;
  }
  .back-link:hover { text-decoration: underline; }
</style>
"""

UPLOAD_FORM = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Toplantı Asistanı</title>{{ style|safe }}</head>
<body>
  <div class="wrap">
    <header class="page-head">
      <h1>Toplantı Asistanı</h1>
      <p class="sub">Bir ses kaydı yükleyin — transkript, özet ve konuşmacı ayrımı otomatik üretilsin.</p>
    </header>

    <div class="card">
      <div class="record-section">
        <div class="record-row">
          <button type="button" class="record-btn" id="record-btn">🔴 Kaydı Başlat</button>
          <span class="record-timer" id="record-timer" hidden>00:00</span>
        </div>
        <p class="hint" style="text-align:left; margin-top:10px;">
          Teams'i tarayıcıda (teams.microsoft.com) açın, "Kaydı Başlat"a basın, açılan pencerede
          <strong>"Sekme"</strong>yi seçip Teams sekmesini işaretleyin ve <strong>"Sekme sesini
          paylaş"</strong> kutucuğunu açın. Hem sizin sesiniz hem karşı taraf kaydedilir — hiçbir
          kurulum gerekmez.
        </p>
      </div>

      <div class="divider"><span>veya bir dosya yükleyin</span></div>

      <form id="upload-form" method="post" action="/process" enctype="multipart/form-data">
        <label class="dropzone" id="dropzone" for="file-input">
          <div class="icon">🎙️</div>
          <div class="primary-text">Dosyayı sürükleyin ya da seçmek için tıklayın</div>
          <div class="secondary-text">wav, mp3, m4a, mp4, caf ve diğer yaygın ses/video formatları</div>
          <div class="filename" id="filename-badge"></div>
        </label>
        <input type="file" name="audio" id="file-input" accept="audio/*,video/*" required>

        <button type="submit" class="primary" id="submit-btn">
          <span class="spinner"></span>
          <span class="btn-label">Yükle ve İşle</span>
        </button>
        <p class="hint">İşlem birkaç dakika sürebilir, sayfadan ayrılmayın.</p>
      </form>
      {% if error %}<div class="alert">{{ error }}</div>{% endif %}
    </div>
  </div>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const filenameBadge = document.getElementById('filename-badge');
    const form = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');

    function showFilename(file) {
      if (!file) return;
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      filenameBadge.textContent = `${file.name} · ${sizeMb} MB`;
      filenameBadge.style.display = 'inline-block';
    }

    fileInput.addEventListener('change', () => showFilename(fileInput.files[0]));

    ['dragenter', 'dragover'].forEach(evt =>
      dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('drag'); })
    );
    ['dragleave', 'drop'].forEach(evt =>
      dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('drag'); })
    );
    dropzone.addEventListener('drop', e => {
      const dropped = e.dataTransfer.files;
      if (dropped.length > 0) {
        fileInput.files = dropped;
        showFilename(dropped[0]);
      }
    });

    form.addEventListener('submit', () => {
      if (!fileInput.files.length) return;
      submitBtn.classList.add('loading');
      submitBtn.disabled = true;
    });

    // ── Tarayıcıdan doğrudan kayıt: mikrofon + paylaşılan sekmenin sesi ──
    const recordBtn = document.getElementById('record-btn');
    const recordTimer = document.getElementById('record-timer');
    let mediaRecorder = null;
    let recordedChunks = [];
    let activeStreams = [];
    let audioCtx = null;
    let timerInterval = null;
    let recordStart = null;

    function updateTimer() {
      const elapsed = Math.floor((Date.now() - recordStart) / 1000);
      const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const s = String(elapsed % 60).padStart(2, '0');
      recordTimer.textContent = `${m}:${s}`;
    }

    function cleanupStreams() {
      activeStreams.forEach(stream => stream.getTracks().forEach(track => track.stop()));
      activeStreams = [];
      if (audioCtx) { audioCtx.close(); audioCtx = null; }
    }

    function setRecordingUI(isRecording) {
      recordBtn.textContent = isRecording ? '⏹ Kaydı Durdur' : '🔴 Kaydı Başlat';
      recordBtn.classList.toggle('recording', isRecording);
      recordTimer.hidden = !isRecording;
    }

    async function startRecording() {
      // Sıra önemli: getDisplayMedia "kullanıcı hareketinden doğrudan
      // çağrılmalı" kuralına tabi. Önce mikrofon için await edilirse, o
      // bekleme sırasında tarayıcı kullanıcı hareketi bağlamının süresini
      // dolmuş sayıp getDisplayMedia'yı reddediyor. Bu yüzden önce
      // getDisplayMedia, sonra mikrofon istenir.
      let displayStream, micStream;
      try {
        displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      } catch (err) {
        alert('Sekme paylaşımı iptal edildi ya da izin verilmedi: ' + err.message);
        return;
      }
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err) {
        displayStream.getTracks().forEach(t => t.stop());
        alert('Mikrofon izni verilmedi: ' + err.message);
        return;
      }

      if (displayStream.getAudioTracks().length === 0) {
        alert('Seçtiğiniz sekme/pencere ses paylaşmıyor. Paylaşım penceresinde "Sekme"yi seçip ' +
              '"Sekme sesini paylaş" kutucuğunu işaretleyip tekrar deneyin. Şimdilik sadece ' +
              'mikrofonunuz kaydedilecek.');
      }

      activeStreams = [micStream, displayStream];
      audioCtx = new AudioContext();
      const dest = audioCtx.createMediaStreamDestination();
      audioCtx.createMediaStreamSource(micStream).connect(dest);
      if (displayStream.getAudioTracks().length > 0) {
        audioCtx.createMediaStreamSource(new MediaStream(displayStream.getAudioTracks())).connect(dest);
      }

      recordedChunks = [];
      mediaRecorder = new MediaRecorder(dest.stream, { mimeType: 'audio/webm' });
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'audio/webm' });
        if (blob.size === 0) {
          alert('Kayıt boş çıktı (0 byte) - hiç ses yakalanamadı. Lütfen tekrar deneyin ve ' +
                'kaydı en az birkaç saniye açık tutun.');
          cleanupStreams();
          clearInterval(timerInterval);
          setRecordingUI(false);
          return;
        }
        // Her kayda benzersiz, zaman damgalı bir isim veriyoruz - aksi halde
        // hep aynı "kayit.webm" adı kullanılıp bir önceki kayıt sessizce
        // üzerine yazılıyordu.
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        const file = new File([blob], `kayit-${stamp}.webm`, { type: 'audio/webm' });
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        showFilename(file);
        cleanupStreams();
        clearInterval(timerInterval);
        setRecordingUI(false);
      };
      mediaRecorder.start();

      // Kullanıcı paylaşımı tarayıcının kendi "paylaşımı durdur" çubuğundan
      // keserse kaydı da otomatik bitir.
      displayStream.getVideoTracks()[0].addEventListener('ended', stopRecording);

      recordStart = Date.now();
      updateTimer();
      timerInterval = setInterval(updateTimer, 1000);
      setRecordingUI(true);
    }

    function stopRecording() {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
    }

    recordBtn.addEventListener('click', () => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
      } else {
        startRecording();
      }
    });
  </script>
</body></html>
"""

RESULT_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonuç — Toplantı Asistanı</title>{{ style|safe }}</head>
<body>
  <div class="wrap">
    <header class="page-head">
      <h1>{{ filename }}</h1>
      <div class="meta-row">
        <span class="pill">Güven: %{{ "%.0f"|format(confidence * 100) }}</span>
        <span class="pill status-{{ status }}">{{ status }}</span>
      </div>
    </header>

    <div class="card">
      <div class="card-head">
        <h2>📝 Özet</h2>
        <a class="download-btn" href="/download/{{ session_id }}/summary.md" download>⬇ İndir</a>
      </div>
      <pre class="content">{{ summary }}</pre>
    </div>

    <div class="card">
      <div class="card-head">
        <h2>🗒️ Transkript</h2>
        <a class="download-btn" href="/download/{{ session_id }}/transcript.md" download>⬇ İndir</a>
      </div>
      <pre class="content">{{ transcript }}</pre>
    </div>

    <a class="back-link" href="/">← Yeni bir dosya işle</a>
  </div>
</body></html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_FORM, style=PAGE_STYLE, error=None)


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("audio")
    if file is None or file.filename == "":
        return render_template_string(UPLOAD_FORM, style=PAGE_STYLE, error="Lütfen bir dosya seçin.")

    filename = secure_filename(file.filename)
    saved_path = UPLOAD_DIR / filename
    file.save(saved_path)

    try:
        artifacts = transcribe_recording(
            str(saved_path),
            meeting_title=Path(filename).stem,
            use_diarization=True,
        )
    except Exception as e:
        return render_template_string(
            UPLOAD_FORM, style=PAGE_STYLE,
            error=f"Dosya işlenemedi: {e}",
        )

    return render_template_string(
        RESULT_PAGE,
        style=PAGE_STYLE,
        filename=filename,
        confidence=artifacts["confidence"],
        status=artifacts["status"],
        summary=Path(artifacts["summary_path"]).read_text(encoding="utf-8"),
        transcript=Path(artifacts["transcript_path"]).read_text(encoding="utf-8"),
        session_id=Path(artifacts["session_dir"]).name,
    )


def _parse_session_id(session_id: str) -> tuple[str, str]:
    """'{toplantı-adı}_{YYYYMMDD}_{HHMMSS}' formatındaki oturum klasörü
    adından okunabilir bir başlık ve tarih çıkarır (PDF başlığı için)."""
    match = re.match(r"^(.*)_(\d{8})_(\d{6})$", session_id)
    if not match:
        return session_id, ""
    title_raw, date_part, time_part = match.groups()
    title = title_raw.replace("_", " ").replace("-", " ").strip() or "Toplantı"
    try:
        dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        date_str = dt.strftime("%d.%m.%Y %H:%M")
    except ValueError:
        date_str = ""
    return title, date_str


@app.route("/download/<session_id>/<filename>")
def download(session_id, filename):
    if filename not in DOWNLOADABLE_FILES:
        abort(404)
    file_path = (MEETING_NOTES_DIR / secure_filename(session_id) / filename).resolve()
    if MEETING_NOTES_DIR not in file_path.parents or not file_path.is_file():
        abort(404)

    meeting_title, meeting_date = _parse_session_id(session_id)
    pdf_bytes = markdown_to_pdf_bytes(
        file_path.read_text(encoding="utf-8"),
        meeting_title=meeting_title,
        meeting_date=meeting_date,
    )
    pdf_name = Path(filename).stem + ".pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=pdf_name,
    )


if __name__ == "__main__":
    # use_reloader=False: speechbrain'in opsiyonel k2 bağımlılığı için attığı
    # ImportError, Werkzeug'un dosya-izleme reloader'ını çökertiyor. Debug hata
    # sayfaları için debug=True kalıyor, sadece otomatik yeniden başlatma kapalı.
    app.run(debug=True, port=5000, use_reloader=False)
