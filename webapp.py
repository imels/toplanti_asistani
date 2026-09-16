"""Web arayüzü: bir ses kaydı yükle ya da tarayıcıdan kaydet, transkript +
özet al. Sol menüde geçmiş kayıtlar listelenir, tıklanınca transkript/özet/
orijinal ses birlikte görüntülenir.

Canlı mikrofon dinlemez — sadece bitmiş bir ses dosyasını (Teams kaydı,
telefon kaydı vb.) işler. Çalıştırmak için:

    python3 webapp.py

sonra tarayıcıda http://127.0.0.1:5000 adresine gidin.
"""

from __future__ import annotations

import io
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for
from werkzeug.utils import secure_filename

from core import transcribe_recording
from pdf_export import markdown_to_pdf_bytes

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MEETING_NOTES_DIR = Path("meeting-notes").resolve()
MEETING_NOTES_DIR.mkdir(exist_ok=True)
DOWNLOADABLE_FILES = {"transcript.md", "summary.md"}

AUDIO_MIME_TYPES = {
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".webm": "audio/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".caf": "audio/x-caf",
}

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
    line-height: 1.5;
  }

  /* ── Sayfa iskeleti: sol menü + ana içerik ── */
  .layout { display: flex; align-items: stretch; min-height: 100vh; }
  .sidebar {
    width: 260px; flex-shrink: 0; background: var(--surface);
    border-right: 1px solid var(--line); padding: 60px 14px 22px;
    transition: width .16s ease, padding .16s ease, opacity .16s ease;
    overflow: hidden;
  }
  .sidebar.collapsed { width: 0; padding-left: 0; padding-right: 0; border-right: none; opacity: 0; }
  .sidebar-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
  .sidebar-title {
    font-size: 12px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
    color: var(--muted); white-space: nowrap;
  }
  /* Panel daraltılınca içindeki her şeyle birlikte kaybolmasın diye,
     açma/kapama düğmesi panelin DIŞINDA, sabit konumda duruyor - aksi
     halde panel bir kere kapatılınca geri açacak bir yer kalmıyordu. */
  .sidebar-toggle-fixed {
    position: fixed; top: 14px; left: 14px; z-index: 40;
    background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow);
    color: var(--muted); cursor: pointer; font-size: 15px; line-height: 1;
    width: 34px; height: 34px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
  }
  .sidebar-toggle-fixed:hover { background: var(--accent-soft); color: var(--accent-ink); }
  .new-btn {
    display: flex; align-items: center; gap: 8px; background: var(--accent); color: #fff;
    padding: 10px 14px; border-radius: 9px; font-size: 14px; font-weight: 600;
    text-decoration: none; margin-bottom: 18px; white-space: nowrap;
  }
  .new-btn:hover { background: var(--accent-ink); }
  .session-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
  .session-row { display: flex; align-items: stretch; gap: 2px; }
  .session-item {
    display: block; flex: 1; min-width: 0; padding: 9px 10px; border-radius: 8px;
    text-decoration: none; color: var(--ink);
  }
  .session-item:hover { background: var(--paper); }
  .session-item.active { background: var(--accent-soft); color: var(--accent-ink); }
  .session-item .s-title {
    display: block; font-size: 13.5px; font-weight: 600; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .session-item .s-date { display: block; font-size: 11px; color: var(--muted); margin-top: 2px; }
  .session-item.active .s-date { color: var(--accent-ink); opacity: .8; }
  .delete-form { display: flex; margin: 0; }
  .delete-btn, .rename-btn {
    background: none; border: none; cursor: pointer; color: var(--muted);
    font-size: 14px; line-height: 1; padding: 0 8px; border-radius: 7px; opacity: 0;
    transition: opacity .12s ease, background .12s ease, color .12s ease;
  }
  .delete-btn { font-size: 18px; }
  .session-row:hover .delete-btn, .session-row:hover .rename-btn { opacity: 1; }
  .delete-btn:hover { background: var(--bad-soft); color: var(--bad); }
  .rename-btn:hover { background: var(--accent-soft); color: var(--accent-ink); }
  .session-row.editing .session-item { display: none; }
  .rename-form { display: flex; flex: 1; min-width: 0; margin: 0; }
  .rename-form[hidden] { display: none; }
  .rename-input {
    flex: 1; min-width: 0; font: inherit; font-size: 13.5px; font-weight: 600;
    padding: 8px 9px; border-radius: 8px; border: 1px solid var(--accent);
    background: var(--paper); color: var(--ink);
  }
  .rename-input:focus { outline: none; }
  .empty-hint { color: var(--muted); font-size: 12.5px; padding: 8px 10px; }
  .main-area { flex: 1; min-width: 0; padding: 48px 20px 80px; }

  @media (max-width: 720px) {
    .sidebar { position: fixed; top: 0; left: 0; bottom: 0; z-index: 30; box-shadow: var(--shadow); }
    .sidebar.collapsed { display: none; }
  }

  .wrap { max-width: 760px; margin: 0 auto; }
  header.page-head { margin-bottom: 28px; }
  h1 { font-size: 24px; font-weight: 650; margin: 0 0 6px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 14.5px; margin: 0; }
  .title-row { display: flex; align-items: center; gap: 8px; }
  .title-row h1 { margin: 0 0 6px; }
  .header-rename-btn { font-size: 16px; opacity: .55; padding: 4px 8px; margin-bottom: 6px; }
  .header-rename-btn:hover { opacity: 1; }
  .header-rename-form { margin: 0 0 6px; max-width: 480px; }
  .header-rename-input { font-size: 22px; font-weight: 650; padding: 6px 10px; }

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

  audio.player { width: 100%; margin-top: 2px; }

  /* ── Sabit alt ses oynatıcı çubuğu (geçmiş kayıt sayfası) ── */
  .bottom-player {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 25;
    background: var(--surface); border-top: 1px solid var(--line); box-shadow: var(--shadow);
    padding: 12px 24px 16px; display: flex; flex-direction: column; align-items: center; gap: 8px;
  }
  .bottom-player .ap-seek-row { width: 100%; max-width: 720px; }
  .ap-controls { display: flex; align-items: center; justify-content: center; gap: 14px; }
  .ap-btn {
    background: var(--accent-soft); color: var(--accent-ink); border: none;
    border-radius: 999px; cursor: pointer; display: flex; align-items: center;
    justify-content: center; font-size: 13px; font-weight: 700;
  }
  .ap-btn:hover { background: var(--accent-soft-strong); }
  .ap-btn.ap-skip { width: 40px; height: 40px; font-size: 11px; flex-direction: column; gap: 1px; line-height: 1; }
  .ap-btn.ap-play { width: 52px; height: 52px; font-size: 20px; background: var(--accent); color: #fff; }
  .ap-btn.ap-play:hover { background: var(--accent-ink); }
  .ap-seek-row { display: flex; align-items: center; gap: 10px; }
  .ap-time {
    font-family: "SF Mono", ui-monospace, monospace; font-size: 12px;
    color: var(--muted); min-width: 38px; text-align: center;
  }
  .ap-seek {
    flex: 1; -webkit-appearance: none; appearance: none; height: 5px;
    border-radius: 999px; background: var(--line); outline: none; cursor: pointer;
  }
  .ap-seek::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent); cursor: pointer;
  }
  .ap-seek::-moz-range-thumb {
    width: 14px; height: 14px; border-radius: 50%; background: var(--accent);
    border: none; cursor: pointer;
  }

  /* ── Transkript satırları (konuşmacı avatarlı, sese senkron) ── */
  .transcript-rows { display: flex; flex-direction: column; }
  .t-row {
    display: grid; grid-template-columns: 62px 30px 116px 1fr; gap: 4px 12px;
    align-items: start; padding: 11px 10px; border-radius: 8px; cursor: pointer;
  }
  .t-row:hover { background: var(--paper); }
  .t-row.active { background: var(--accent-soft); }
  .t-time {
    font-family: "SF Mono", ui-monospace, monospace; font-size: 12.5px;
    font-weight: 600; color: var(--accent-ink); padding-top: 2px;
  }
  .t-avatar {
    width: 26px; height: 26px; border-radius: 50%; color: #fff;
    font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center;
  }
  .t-speaker { font-size: 13.5px; font-weight: 700; color: var(--ink); padding-top: 3px; }
  .t-text { grid-column: 4; font-size: 14px; color: var(--ink); line-height: 1.55; }

  .autoscroll-toggle {
    display: flex; align-items: center; gap: 10px; margin-top: 14px;
    font-size: 13px; font-weight: 600; color: var(--ink); cursor: pointer; user-select: none;
  }
  .autoscroll-toggle input { position: absolute; opacity: 0; width: 0; height: 0; }
  .toggle-track {
    width: 36px; height: 20px; border-radius: 999px; background: var(--line);
    position: relative; transition: background .15s ease; flex-shrink: 0;
  }
  .toggle-thumb {
    position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%;
    background: #fff; box-shadow: var(--shadow); transition: transform .15s ease;
  }
  .autoscroll-toggle input:checked + .toggle-track { background: var(--accent); }
  .autoscroll-toggle input:checked + .toggle-track .toggle-thumb { transform: translateX(16px); }

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
  .filename {
    margin-top: 14px; font-size: 13.5px; font-weight: 600; color: var(--accent-ink);
    background: var(--accent-soft); border-radius: 8px; padding: 6px 12px; display: none;
    text-align: center;
  }
  input[type=file] { display: none; }

  /* ── Kaydet / Dosya Yükle seçim kartları ── */
  .mode-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .mode-card {
    background: var(--paper); border: 1.5px solid var(--line); border-radius: 14px;
    padding: 30px 14px 18px; cursor: pointer; display: flex; flex-direction: column;
    align-items: center; gap: 16px; transition: border-color .15s ease, background .15s ease;
    font-family: inherit;
  }
  .mode-card:hover { border-color: var(--accent); }
  .mode-card.active { border-color: var(--accent); background: var(--accent-soft); }
  .mode-card-visual {
    width: 60px; height: 60px; border-radius: 50%; background: var(--accent);
    display: flex; align-items: center; justify-content: center; font-size: 26px; color: #fff;
    flex-shrink: 0;
  }
  .mode-label { font-size: 14px; font-weight: 700; color: var(--ink); text-align: center; line-height: 1.3; }

  .tab-panel {
    display: flex; flex-direction: column; align-items: center; gap: 18px;
    width: 100%; padding: 4px 0 6px;
  }
  /* Yazar CSS'i (display:flex) tarayıcının [hidden] varsayılanını ezip
     paneli her zaman görünür bırakıyordu - burada açıkça geri kazandırıyoruz. */
  .tab-panel[hidden] { display: none; }
  .record-row { display: flex; align-items: center; justify-content: center; gap: 14px; }
  .record-btn {
    background: var(--accent); color: #fff;
    border: none; padding: 13px 28px; border-radius: 999px;
    font-size: 15px; font-weight: 700; cursor: pointer; box-shadow: var(--shadow);
    transition: background .15s ease;
  }
  .record-btn:hover { background: var(--accent-ink); }
  .record-btn.recording {
    background: var(--bad); animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
  .record-timer {
    font-family: "SF Mono", ui-monospace, monospace; font-size: 15px;
    font-weight: 700; color: var(--ink); background: var(--paper);
    border: 1px solid var(--line); padding: 7px 14px; border-radius: 8px;
  }
  .note {
    display: flex; gap: 10px; align-items: flex-start; text-align: left; width: 100%;
    background: var(--accent-soft); border-radius: 10px; padding: 14px 16px;
    font-size: 13px; color: var(--ink); line-height: 1.55;
  }
  .note .note-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
  .note strong { color: var(--accent-ink); }

  .source-toggle {
    display: flex; gap: 4px; background: var(--paper); border: 1px solid var(--line);
    border-radius: 999px; padding: 4px;
  }
  .source-btn {
    background: none; border: none; cursor: pointer; padding: 8px 16px; border-radius: 999px;
    font-size: 13px; font-weight: 600; color: var(--muted); white-space: nowrap;
    transition: background .15s ease, color .15s ease;
  }
  .source-btn.active { background: var(--accent); color: #fff; }
  .source-btn:hover:not(.active) { color: var(--ink); }
  @media (max-width: 480px) { .source-toggle { flex-direction: column; width: 100%; } }

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

  .meta-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
  .pill {
    font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px;
    background: var(--accent-soft); color: var(--accent-ink);
  }

  pre.content {
    white-space: pre-wrap; word-wrap: break-word;
    background: var(--paper); border: 1px solid var(--line);
    padding: 16px; border-radius: 10px; font-size: 14px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    max-height: 480px; overflow-y: auto; margin: 0;
  }
</style>
"""

SHELL = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ page_title }} — Toplantı Asistanı</title>{{ style|safe }}</head>
<body>
  <button type="button" class="sidebar-toggle-fixed" id="sidebar-toggle" title="Menüyü aç/kapat">☰</button>
  <div class="layout">
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-head">
        <span class="sidebar-title">Kayıtlar</span>
      </div>
      <a class="new-btn" href="/">+ Yeni Kayıt</a>
      {% if sessions %}
      <ul class="session-list">
        {% for s in sessions %}
        <li class="session-row" data-session-id="{{ s.id }}">
          <a class="session-item{% if s.id == active_session %} active{% endif %}" href="/session/{{ s.id }}">
            <span class="s-title">{{ s.title }}</span>
            <span class="s-date">{{ s.date }}</span>
          </a>
          <form class="rename-form" method="post" action="/session/{{ s.id }}/rename" hidden>
            <input type="text" name="title" class="rename-input" value="{{ s.title }}" maxlength="120" autocomplete="off">
          </form>
          <button type="button" class="rename-btn" title="Adını değiştir" aria-label="Adını değiştir">✎</button>
          <form class="delete-form" method="post" action="/session/{{ s.id }}/delete">
            <button type="submit" class="delete-btn" title="Kaydı sil" aria-label="Kaydı sil">×</button>
          </form>
        </li>
        {% endfor %}
      </ul>
      {% else %}
      <p class="empty-hint">Henüz kayıt yok.</p>
      {% endif %}
    </aside>
    <main class="main-area">
      {{ content|safe }}
    </main>
  </div>
  <script>
    // Panel her sayfa yüklemesinde açık başlar - önceki oturumdan kalan bir
    // "daraltılmış" durumu hatırlamıyoruz, bu da panelin habersizce kapalı
    // görünmesine yol açıyordu. Daraltma butonu sadece o an için geçerlidir.
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
    });

    document.querySelectorAll('.delete-form').forEach(form => {
      form.addEventListener('submit', e => {
        if (!confirm('Bu kaydı silmek istediğinize emin misiniz? Bu işlem geri alınamaz.')) {
          e.preventDefault();
        }
      });
    });

    document.querySelectorAll('.sidebar .rename-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const row = btn.closest('.session-row');
        const form = row.querySelector('.rename-form');
        const input = form.querySelector('.rename-input');
        row.classList.add('editing');
        form.hidden = false;
        input.focus();
        input.select();
      });
    });

    document.querySelectorAll('.sidebar .rename-form').forEach(form => {
      const input = form.querySelector('.rename-input');
      const row = form.closest('.session-row');
      const cancel = () => {
        row.classList.remove('editing');
        form.hidden = true;
      };
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          form.requestSubmit();
        } else if (e.key === 'Escape') {
          cancel();
        }
      });
      input.addEventListener('blur', () => {
        // Kısa bir gecikme: submit tıklaması da blur tetikler, submit'in
        // önce işlenmesine izin ver.
        setTimeout(cancel, 150);
      });
      form.addEventListener('submit', e => {
        if (!input.value.trim()) {
          e.preventDefault();
          cancel();
        }
      });
    });
  </script>
</body></html>
"""

HOME_CONTENT = """
<div class="wrap">
  <header class="page-head">
    <h1>Toplantı Asistanı</h1>
    <p class="sub">Bir ses kaydı yükleyin — transkript, özet ve konuşmacı ayrımı otomatik üretilsin.</p>
  </header>

  <div class="card">
    <div class="mode-cards">
      <button type="button" class="mode-card" id="tab-btn-record" data-panel="record">
        <span class="mode-card-visual">🎙️</span>
        <span class="mode-label">Kaydet ve Metne Dönüştür</span>
      </button>
      <button type="button" class="mode-card" id="tab-btn-upload" data-panel="upload">
        <span class="mode-card-visual">⬆️</span>
        <span class="mode-label">Yükle ve Metne Dönüştür</span>
      </button>
    </div>

    <form id="upload-form" method="post" action="/process" enctype="multipart/form-data">
      <div class="tab-panel" id="panel-record" hidden>
        <div class="source-toggle" id="source-toggle">
          <button type="button" class="source-btn active" id="src-btn-both" data-source="both">🖥️ Ekran Sesi + Mikrofon</button>
          <button type="button" class="source-btn" id="src-btn-mic" data-source="mic">🎙️ Sadece Mikrofon</button>
        </div>
        <div class="record-row">
          <button type="button" class="record-btn" id="record-btn">🔴 Kaydı Başlat</button>
          <span class="record-timer" id="record-timer" hidden>00:00</span>
        </div>
        <audio class="player" id="preview-player" controls hidden></audio>
        <div class="note" id="source-note">
          <span class="note-icon">💡</span>
          <span>
            Teams'i tarayıcıda (teams.microsoft.com) açın, "Kaydı Başlat"a basın, açılan pencerede
            <strong>"Sekme"</strong>yi seçip Teams sekmesini işaretleyin ve <strong>"Sekme sesini
            paylaş"</strong> kutucuğunu açın. Hem sizin sesiniz hem karşı taraf kaydedilir — hiçbir
            kurulum gerekmez.
          </span>
        </div>
      </div>

      <div class="tab-panel" id="panel-upload" hidden>
        <label class="dropzone" id="dropzone" for="file-input">
          <div class="icon">📁</div>
          <div class="primary-text">Dosyayı sürükleyin ya da seçmek için tıklayın</div>
          <div class="secondary-text">wav, mp3, m4a, mp4, caf ve diğer yaygın ses/video formatları</div>
        </label>
      </div>

      <input type="file" name="audio" id="file-input" accept="audio/*,video/*" required>
      <div class="filename" id="filename-badge"></div>

      <div id="submit-area" hidden>
        <button type="submit" class="primary" id="submit-btn">
          <span class="spinner"></span>
          <span class="btn-label">Yükle ve İşle</span>
        </button>
        <p class="hint">GPU yok, işlem CPU'da çalışıyor - uzun kayıtlarda (10+ dk) bu 30-90 dakika sürebilir. Sayfayı yenilemeyin veya tekrar göndermeyin, aksi halde aynı kayıt iki kez işlenir.</p>
      </div>
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

  // ── Kaydet / Dosya Yükle kart seçimi: hiçbiri seçilmeden hiçbir panel
  // ── (ne kayıt kontrolleri ne yükleme alanı) görünmez.
  const modeCards = document.querySelectorAll('.mode-card');
  const panelRecord = document.getElementById('panel-record');
  const panelUpload = document.getElementById('panel-upload');
  const submitArea = document.getElementById('submit-area');
  modeCards.forEach(card => {
    card.addEventListener('click', () => {
      modeCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      const isRecord = card.dataset.panel === 'record';
      panelRecord.hidden = !isRecord;
      panelUpload.hidden = isRecord;
      submitArea.hidden = false;
    });
  });

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

  // ── Tarayıcıdan doğrudan kayıt: mikrofon + (isteğe bağlı) paylaşılan sekmenin sesi ──
  const recordBtn = document.getElementById('record-btn');
  const recordTimer = document.getElementById('record-timer');
  const previewPlayer = document.getElementById('preview-player');
  const sourceNote = document.getElementById('source-note');
  let mediaRecorder = null;
  let recordedChunks = [];
  let activeStreams = [];
  let audioCtx = null;
  let timerInterval = null;
  let recordStart = null;
  let previewUrl = null;
  let recordSource = 'both';

  const SOURCE_NOTES = {
    both: 'Teams\\'i tarayıcıda (teams.microsoft.com) açın, "Kaydı Başlat"a basın, açılan pencerede ' +
          '<strong>"Sekme"</strong>yi seçip Teams sekmesini işaretleyin ve <strong>"Sekme sesini ' +
          'paylaş"</strong> kutucuğunu açın. Hem sizin sesiniz hem karşı taraf kaydedilir — hiçbir ' +
          'kurulum gerekmez.',
    mic: 'Sadece bilgisayarınızın mikrofonu kaydedilir — karşı tarafın/sekmenin sesi dahil olmaz. ' +
         'Fiziksel bir toplantıda ya da kendi sesinizi kaydetmek için uygundur.',
  };

  document.querySelectorAll('.source-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      recordSource = btn.dataset.source;
      sourceNote.querySelector('span:last-child').innerHTML = SOURCE_NOTES[recordSource];
    });
  });

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
    const wantsScreen = recordSource === 'both';

    // Sıra önemli: getDisplayMedia "kullanıcı hareketinden doğrudan
    // çağrılmalı" kuralına tabi. Önce mikrofon için await edilirse, o
    // bekleme sırasında tarayıcı kullanıcı hareketi bağlamının süresini
    // dolmuş sayıp getDisplayMedia'yı reddediyor. Bu yüzden "Ekran Sesi +
    // Mikrofon" seçiliyse önce getDisplayMedia, sonra mikrofon istenir.
    let displayStream = null, micStream;
    if (wantsScreen) {
      try {
        displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      } catch (err) {
        alert('Sekme paylaşımı iptal edildi ya da izin verilmedi: ' + err.message);
        return;
      }
    }
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      if (displayStream) displayStream.getTracks().forEach(t => t.stop());
      alert('Mikrofon izni verilmedi: ' + err.message);
      return;
    }

    if (displayStream && displayStream.getAudioTracks().length === 0) {
      alert('Seçtiğiniz sekme/pencere ses paylaşmıyor. Paylaşım penceresinde "Sekme"yi seçip ' +
            '"Sekme sesini paylaş" kutucuğunu işaretleyip tekrar deneyin. Şimdilik sadece ' +
            'mikrofonunuz kaydedilecek.');
    }

    activeStreams = displayStream ? [micStream, displayStream] : [micStream];
    audioCtx = new AudioContext();
    const dest = audioCtx.createMediaStreamDestination();
    audioCtx.createMediaStreamSource(micStream).connect(dest);
    if (displayStream && displayStream.getAudioTracks().length > 0) {
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

      // Yüklemeden önce dinleyip kontrol edebilmek için müzik çalar gibi
      // oynat/duraklat/sar kontrolleriyle bir önizleme oynatıcısı.
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(blob);
      previewPlayer.src = previewUrl;
      previewPlayer.hidden = false;

      cleanupStreams();
      clearInterval(timerInterval);
      setRecordingUI(false);
    };
    mediaRecorder.start();

    // Yeni kayıt başlarken önceki önizlemeyi temizle.
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    previewPlayer.hidden = true;
    previewPlayer.removeAttribute('src');

    // Kullanıcı paylaşımı tarayıcının kendi "paylaşımı durdur" çubuğundan
    // keserse kaydı da otomatik bitir.
    if (displayStream) {
      displayStream.getVideoTracks()[0].addEventListener('ended', stopRecording);
    }

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
"""

SESSION_CONTENT = """
<div class="wrap">
  <header class="page-head">
    <div class="title-row">
      <h1 id="header-title-display">{{ title }}</h1>
      <button type="button" class="rename-btn header-rename-btn" id="header-rename-btn" title="Adını değiştir" aria-label="Adını değiştir">✎</button>
    </div>
    <form class="rename-form header-rename-form" id="header-rename-form" method="post" action="/session/{{ session_id }}/rename" hidden>
      <input type="text" name="title" class="rename-input header-rename-input" id="header-rename-input" value="{{ title }}" maxlength="120" autocomplete="off">
    </form>
    <div class="meta-row">
      {% if date %}<span class="pill">📅 {{ date }}</span>{% endif %}
      {% if audio_url %}<span class="pill" id="header-duration">⏱ --:--</span>{% endif %}
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

    {% if transcript_rows %}
    <div class="transcript-rows" id="transcript-rows">
      {% for row in transcript_rows %}
      <div class="t-row" data-time="{{ row.elapsed }}">
        <span class="t-time">{{ row.timestamp }}</span>
        {% if row.speaker %}
        <span class="t-avatar" style="background:{{ row.color }}">{{ row.speaker }}</span>
        <span class="t-speaker">Konuşmacı {{ row.speaker }}</span>
        {% endif %}
        <span class="t-text">{{ row.text }}</span>
      </div>
      {% endfor %}
    </div>
    {% if audio_url %}
    <label class="autoscroll-toggle">
      <input type="checkbox" id="autoscroll-check" checked>
      <span class="toggle-track"><span class="toggle-thumb"></span></span>
      Otomatik Kaydırma
    </label>
    {% endif %}
    {% else %}
    <pre class="content">{{ transcript }}</pre>
    {% endif %}
  </div>

  {% if audio_url %}<div style="height: 96px;"></div>{% endif %}
</div>

{% if audio_url %}
<audio id="ap-audio" src="{{ audio_url }}" preload="metadata"></audio>
<div class="bottom-player">
  <div class="ap-seek-row">
    <span class="ap-time" id="ap-current">0:00</span>
    <input type="range" class="ap-seek" id="ap-seek" min="0" max="0" step="0.1" value="0">
    <span class="ap-time" id="ap-duration">0:00</span>
  </div>
  <div class="ap-controls">
    <button type="button" class="ap-btn ap-skip" id="ap-back" title="10 saniye geri">⏪<span>10</span></button>
    <button type="button" class="ap-btn ap-play" id="ap-play" title="Oynat/Duraklat">▶</button>
    <button type="button" class="ap-btn ap-skip" id="ap-fwd" title="10 saniye ileri">⏩<span>10</span></button>
  </div>
</div>

<script>
  (() => {
    const audio = document.getElementById('ap-audio');
    const playBtn = document.getElementById('ap-play');
    const backBtn = document.getElementById('ap-back');
    const fwdBtn = document.getElementById('ap-fwd');
    const seek = document.getElementById('ap-seek');
    const currentLabel = document.getElementById('ap-current');
    const durationLabel = document.getElementById('ap-duration');
    const headerDuration = document.getElementById('header-duration');
    const rows = Array.from(document.querySelectorAll('.t-row'));
    const autoscrollCheck = document.getElementById('autoscroll-check');

    function formatTime(sec) {
      if (!isFinite(sec)) return '0:00';
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60).toString().padStart(2, '0');
      return `${m}:${s}`;
    }

    audio.addEventListener('loadedmetadata', () => {
      seek.max = audio.duration;
      durationLabel.textContent = formatTime(audio.duration);
      if (headerDuration) headerDuration.textContent = '⏱ ' + formatTime(audio.duration);
    });
    audio.addEventListener('timeupdate', () => {
      seek.value = audio.currentTime;
      currentLabel.textContent = formatTime(audio.currentTime);

      let activeRow = null;
      for (const row of rows) {
        if (parseFloat(row.dataset.time) <= audio.currentTime) activeRow = row;
        else break;
      }
      rows.forEach(r => r.classList.toggle('active', r === activeRow));
      if (activeRow && autoscrollCheck && autoscrollCheck.checked) {
        activeRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
    audio.addEventListener('play', () => { playBtn.textContent = '⏸'; });
    audio.addEventListener('pause', () => { playBtn.textContent = '▶'; });
    audio.addEventListener('ended', () => { playBtn.textContent = '▶'; });

    playBtn.addEventListener('click', () => {
      if (audio.paused) audio.play(); else audio.pause();
    });
    backBtn.addEventListener('click', () => {
      audio.currentTime = Math.max(0, audio.currentTime - 10);
    });
    fwdBtn.addEventListener('click', () => {
      audio.currentTime = Math.min(audio.duration || Infinity, audio.currentTime + 10);
    });
    seek.addEventListener('input', () => {
      audio.currentTime = parseFloat(seek.value);
    });

    rows.forEach(row => {
      row.addEventListener('click', () => {
        audio.currentTime = parseFloat(row.dataset.time);
        audio.play();
      });
    });
  })();
</script>
{% endif %}
<script>
  (function() {
    const titleDisplay = document.getElementById('header-title-display');
    const renameBtn = document.getElementById('header-rename-btn');
    const renameForm = document.getElementById('header-rename-form');
    const renameInput = document.getElementById('header-rename-input');
    if (!renameBtn) return;

    renameBtn.addEventListener('click', () => {
      titleDisplay.hidden = true;
      renameBtn.hidden = true;
      renameForm.hidden = false;
      renameInput.focus();
      renameInput.select();
    });
    const cancel = () => {
      titleDisplay.hidden = false;
      renameBtn.hidden = false;
      renameForm.hidden = true;
    };
    renameInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        e.preventDefault();
        renameForm.requestSubmit();
      } else if (e.key === 'Escape') {
        cancel();
      }
    });
    renameInput.addEventListener('blur', () => setTimeout(cancel, 150));
    renameForm.addEventListener('submit', e => {
      if (!renameInput.value.trim()) {
        e.preventDefault();
        cancel();
      }
    });
  })();
</script>
"""


def _session_title(session_dir: Path, session_id: str) -> tuple[str, str]:
    """Oturumun başlığını döndürür: kullanıcı adını değiştirdiyse meta.json'daki
    özel başlık, aksi halde klasör adından çıkarılan varsayılan başlık."""
    default_title, date_str = _parse_session_id(session_id)
    meta_path = session_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            custom_title = meta.get("title", "").strip()
            if custom_title:
                return custom_title, date_str
        except (json.JSONDecodeError, OSError):
            pass
    return default_title, date_str


def _parse_session_id(session_id: str) -> tuple[str, str]:
    """'{toplantı-adı}_{YYYYMMDD}_{HHMMSS}' formatındaki oturum klasörü
    adından okunabilir bir başlık ve tarih çıkarır."""
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


TRANSCRIPT_ROW_RE = re.compile(r"^- (?:\[Konuşmacı (\d+)\] )?\[(\d{2}):(\d{2}):(\d{2})\] (.*)$")
SPEAKER_COLORS = [
    "#6B4A3B", "#6D3FA0", "#2E6F68", "#B5652E",
    "#3A6EA5", "#8A3B5C", "#4A7A3B", "#5B5B8A",
]


def _parse_transcript_rows(text: str) -> list[dict]:
    """transcript.md satırlarını (ör. '- [Konuşmacı 1] [00:00:05] Merhaba')
    sese senkron, tıklanabilir satırlar halinde göstermek için ayrıştırır."""
    rows = []
    for line in text.splitlines():
        match = TRANSCRIPT_ROW_RE.match(line.strip())
        if not match:
            continue
        speaker, h, m, s, content = match.groups()
        speaker_no = int(speaker) if speaker else None
        rows.append({
            "elapsed": int(h) * 3600 + int(m) * 60 + int(s),
            "timestamp": f"{h}:{m}:{s}",
            "speaker": speaker_no,
            "color": SPEAKER_COLORS[(speaker_no - 1) % len(SPEAKER_COLORS)] if speaker_no else None,
            "text": content,
        })
    return rows


def list_sessions() -> list[dict]:
    """meeting-notes/ altındaki tamamlanmış oturumları, en yeni önde olacak
    şekilde sol menüde göstermek için listeler."""
    sessions = []
    for d in MEETING_NOTES_DIR.iterdir():
        if not d.is_dir() or not (d / "transcript.md").exists():
            continue
        title, date_str = _session_title(d, d.name)
        match = re.match(r"^(.*)_(\d{8})_(\d{6})$", d.name)
        if match:
            try:
                sort_key = datetime.strptime(match.group(2) + match.group(3), "%Y%m%d%H%M%S")
            except ValueError:
                sort_key = datetime.fromtimestamp(d.stat().st_mtime)
        else:
            sort_key = datetime.fromtimestamp(d.stat().st_mtime)
        sessions.append({"id": d.name, "title": title, "date": date_str, "_sort": sort_key})
    sessions.sort(key=lambda s: s["_sort"], reverse=True)
    for s in sessions:
        del s["_sort"]
    return sessions


def render_page(content_html: str, page_title: str = "Toplantı Asistanı", active_session: str | None = None) -> str:
    return render_template_string(
        SHELL,
        style=PAGE_STYLE,
        content=content_html,
        sessions=list_sessions(),
        active_session=active_session,
        page_title=page_title,
    )


@app.route("/", methods=["GET"])
def index():
    content = render_template_string(HOME_CONTENT, error=None)
    return render_page(content, page_title="Toplantı Asistanı")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("audio")
    if file is None or file.filename == "":
        content = render_template_string(HOME_CONTENT, error="Lütfen bir dosya seçin.")
        return render_page(content, page_title="Toplantı Asistanı")

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
        content = render_template_string(HOME_CONTENT, error=f"Dosya işlenemedi: {e}")
        return render_page(content, page_title="Toplantı Asistanı")

    session_dir = Path(artifacts["session_dir"])
    session_id = session_dir.name

    # Orijinal ses dosyasını oturum klasörüne kopyala - sonradan bu kayda
    # geri dönüp dinleyebilmek için (transcribe_recording save_audio=False
    # kullanıyor, ses ayrıca kalıcı olarak saklanmıyordu).
    audio_ext = Path(filename).suffix or ".bin"
    try:
        shutil.copyfile(saved_path, session_dir / f"original{audio_ext}")
    except OSError:
        pass

    return redirect(url_for("session_view", session_id=session_id))


@app.route("/session/<session_id>")
def session_view(session_id):
    session_dir = (MEETING_NOTES_DIR / secure_filename(session_id)).resolve()
    if MEETING_NOTES_DIR not in session_dir.parents or not session_dir.is_dir():
        abort(404)
    transcript_path = session_dir / "transcript.md"
    summary_path = session_dir / "summary.md"
    if not transcript_path.is_file() or not summary_path.is_file():
        abort(404)

    title, date_str = _session_title(session_dir, session_id)

    audio_url = None
    original_audio = next(session_dir.glob("original.*"), None)
    if original_audio is not None:
        audio_url = url_for("session_audio", session_id=session_id, filename=original_audio.name)

    transcript_text = transcript_path.read_text(encoding="utf-8")
    content = render_template_string(
        SESSION_CONTENT,
        title=title,
        date=date_str,
        session_id=session_id,
        audio_url=audio_url,
        summary=summary_path.read_text(encoding="utf-8"),
        transcript=transcript_text,
        transcript_rows=_parse_transcript_rows(transcript_text),
    )
    return render_page(content, page_title=title, active_session=session_id)


@app.route("/session/<session_id>/delete", methods=["POST"])
def delete_session(session_id):
    session_dir = (MEETING_NOTES_DIR / secure_filename(session_id)).resolve()
    if MEETING_NOTES_DIR not in session_dir.parents or not session_dir.is_dir():
        abort(404)
    shutil.rmtree(session_dir)
    return redirect(url_for("index"))


@app.route("/session/<session_id>/rename", methods=["POST"])
def rename_session(session_id):
    session_dir = (MEETING_NOTES_DIR / secure_filename(session_id)).resolve()
    if MEETING_NOTES_DIR not in session_dir.parents or not session_dir.is_dir():
        abort(404)
    new_title = request.form.get("title", "").strip()
    if new_title:
        meta_path = session_dir / "meta.json"
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}
        meta["title"] = new_title
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    # Sol menüden veya oturum sayfasının kendisinden çağrılabilir - hangi
    # sayfadaysa oraya geri dön.
    referrer = urlparse(request.referrer or "")
    if referrer.netloc == request.host and referrer.path.startswith("/"):
        return redirect(referrer.path)
    return redirect(url_for("index"))


@app.route("/session-audio/<session_id>/<filename>")
def session_audio(session_id, filename):
    safe_name = secure_filename(filename)
    if not safe_name.startswith("original."):
        abort(404)
    file_path = (MEETING_NOTES_DIR / secure_filename(session_id) / safe_name).resolve()
    if MEETING_NOTES_DIR not in file_path.parents or not file_path.is_file():
        abort(404)
    mimetype = AUDIO_MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return send_file(file_path, mimetype=mimetype)


@app.route("/download/<session_id>/<filename>")
def download(session_id, filename):
    if filename not in DOWNLOADABLE_FILES:
        abort(404)
    file_path = (MEETING_NOTES_DIR / secure_filename(session_id) / filename).resolve()
    if MEETING_NOTES_DIR not in file_path.parents or not file_path.is_file():
        abort(404)

    meeting_title, meeting_date = _session_title(file_path.parent, session_id)
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
