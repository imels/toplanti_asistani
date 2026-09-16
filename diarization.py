"""
Speaker Diarization modülü - PyAnnote Audio kullanarak kişi ayrıştırma
"""

import os
import sys
import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import multiprocessing

    import torch
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True

    # PyTorch varsayılan olarak tüm çekirdekleri kullanmayabiliyor (bu makinede
    # 8 çekirdek varken 6 thread kullanıyordu) - diarizasyonun en ağır kısmı
    # (matris çarpımları) için tüm çekirdekleri açıkça talep ediyoruz.
    torch.set_num_threads(multiprocessing.cpu_count())
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("PyAnnote Audio kurulu değil - diarization devre dışı")


def _defuse_speechbrain_lazy_modules() -> None:
    """speechbrain, k2 gibi opsiyonel bağımlılıkları sys.modules'a "tembel"
    (henüz gerçekten import edilmemiş) modüller olarak kaydediyor. PyTorch
    Lightning checkpoint yüklerken inspect.stack() çağırıyor; bu da CPython'ın
    iç mekanizması yüzünden yüklü TÜM modüllerin __file__ özelliğini okumaya
    çalışıyor ve bu tembel modülleri tetikleyip gerçek import'u deniyor - k2
    gibi opsiyonel paket kurulu değilse bu ImportError'a düşüp checkpoint
    yüklemeyi tamamen çökertiyor (diarization ile hiç ilgisi olmamasına
    rağmen). Henüz yüklenmemiş bu modüllere sahte bir __file__ koyup
    getattr'ın hiç tembel yükleme koduna düşmesini engelliyoruz.
    """
    try:
        from speechbrain.utils.importutils import LazyModule
    except ImportError:
        return
    for mod in list(sys.modules.values()):
        if isinstance(mod, LazyModule) and mod.lazy_module is None:
            mod.__dict__.setdefault("__file__", f"<lazy:{mod.target}>")


def _load_pretrained_pipeline(model_name: str, hf_token: str):
    """PyTorch 2.6+, torch.load()'ın varsayılanını weights_only=True yaptı.
    pyannote'un checkpoint'i bu yeni katı listede olmayan birden fazla nesne
    (TorchVersion, pyannote.audio.core.task.Specifications, ...) taşıyor ve
    bunları teker teker allowlist'lemek yerine, sadece bu yükleme sırasında
    torch.load'u weights_only=False ile çağıracak şekilde geçici olarak
    yamalıyoruz. Kaynak resmi/güvenilen pyannote deposu olduğu için güvenli.

    Not: pyannote, torch.load'ı doğrudan değil, PyTorch Lightning'in kendi
    _load() sarmalayıcısı üzerinden çağırıyor ve bu sarmalayıcı
    weights_only=None'ı AÇIKÇA geçiyor (parametre eksik değil, değeri None).
    Bu yüzden kwargs.setdefault(...) işe yaramaz - anahtar zaten var demektir.
    Değeri ne olursa olsun zorla False'a çekmemiz gerekiyor.
    """
    original_load = torch.load

    def _load_full(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _load_full
    _defuse_speechbrain_lazy_modules()
    try:
        return Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    finally:
        torch.load = original_load


class DiarizationPipeline:
    """PyAnnote kullanarak ses dosyasında konuşmacı tespiti."""

    def __init__(self, use_diarization: bool = True):
        self.use_diarization = use_diarization and PYANNOTE_AVAILABLE
        self.pipeline = None

        if self.use_diarization:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning(
                    "HF_TOKEN ortam değişkeni yok - pyannote/speaker-diarization-3.1 "
                    "gated bir model, token olmadan indirilemez. Diarization devre dışı kalacak."
                )
                self.use_diarization = False
                return
            try:
                self.pipeline = _load_pretrained_pipeline(
                    "pyannote/speaker-diarization-3.1", hf_token
                )
                logger.info("Diarization pipeline yüklendi")
            except Exception as e:
                logger.warning(f"Diarization pipeline yüklenemedi: {e}")
                logger.info("Diarization özelliği devre dışı kalacak - transkripsiyon normal devam edecek")
                self.use_diarization = False

    def get_speaker_segments(
        self,
        audio_path: str,
        sr: int = 16000,
        num_speakers: Optional[int] = None,
    ) -> List[Tuple[float, float, int]]:
        """
        Ses dosyasını analiz et ve konuşmacı segmentlerini döndür.

        num_speakers: Kesin konuşmacı sayısı biliniyorsa verin - pyannote'un
            kendi tahminini atlar. Verilmezse pyannote kendi doğal tahminini
            kullanır (üst sınır dayatılmaz - kalabalık toplantılarda gerçek
            konuşmacı sayısı fazla olabilir).

        Returns:
            List[(start_sec, end_sec, speaker_id), ...]
        """
        if not self.use_diarization or not self.pipeline:
            return []

        try:
            # torchaudio.load() m4a/mp4 gibi konteynerleri açamıyor
            # ("Format not recognised"). faster-whisper'ın zaten transkripsiyon
            # için kullandığı PyAV tabanlı çözücüyü kullanıyoruz - hemen her
            # formatı açabiliyor ve doğrudan istenen örnekleme hızına
            # (mono, float32) çeviriyor, ayrıca resample'a gerek kalmıyor.
            from faster_whisper.audio import decode_audio

            audio = decode_audio(audio_path, sampling_rate=sr)
            waveform = torch.from_numpy(audio).unsqueeze(0)

            diarization_kwargs = {"num_speakers": num_speakers} if num_speakers else {}
            try:
                diarization = self.pipeline({"waveform": waveform, "sample_rate": sr}, **diarization_kwargs)
            except Exception as pipe_err:
                logger.warning(f"Diarization pipe hatası: {pipe_err}")
                return []

            # Sonuçları parse et
            segments = []
            speaker_map = {}  # speaker_id → speaker_no
            next_speaker_no = 1

            for segment, track, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_map:
                    speaker_map[speaker] = next_speaker_no
                    next_speaker_no += 1

                speaker_no = speaker_map[speaker]
                segments.append((segment.start, segment.end, speaker_no))

            logger.info(f"Diarization tamamlandı: {len(segments)} segment, {len(speaker_map)} konuşmacı")
            return segments

        except Exception as e:
            logger.error(f"Diarization hatası: {e}")
            return []

    def get_speaker_at_time(self, segments: List[Tuple[float, float, int]], time_sec: float) -> int:
        """Belirli bir zaman anında hangi konuşmacı konuşuyor?

        pyannote segmentleri arasında küçük boşluklar bırakabiliyor (ör. iki
        konuşmacı arasındaki mikro-sessizlik); sorgulanan an tam o boşluğa
        denk gelirse, hemen "bilinmeyen konuşmacı"ya düşmek yerine, kısa bir
        tolerans içinde en yakın segmente bakıyoruz - aksi halde transkriptte
        gereksiz yere çok sayıda etiketsiz satır birikip, aslında bilinen bir
        konuşmacıya ait sözler etiketsiz/kopuk görünüyordu.
        """
        GAP_TOLERANCE_S = 0.75
        for start, end, speaker_no in segments:
            if start <= time_sec <= end:
                return speaker_no
        best_speaker = 0
        best_distance = GAP_TOLERANCE_S
        for start, end, speaker_no in segments:
            distance = start - time_sec if time_sec < start else time_sec - end
            if distance < best_distance:
                best_distance = distance
                best_speaker = speaker_no
        return best_speaker

    def get_dominant_speaker(
        self, segments: List[Tuple[float, float, int]], start_sec: float, end_sec: float
    ) -> int:
        """[start_sec, end_sec] aralığıyla en çok çakışan konuşmacıyı döndürür.

        Tek bir ana (ör. segmentin başlangıcı) bakmak yerine örtüşme süresine
        göre karar vermek daha sağlam - konuşmacı geçişi tam segmentin
        başında olduğunda ya da diarizasyon sınırları Whisper segmentiyle
        birebir örtüşmediğinde tek nokta bakışı yanlış konuşmacıyı seçebiliyor.
        """
        overlap_by_speaker: dict[int, float] = {}
        for seg_start, seg_end, speaker_no in segments:
            overlap = min(end_sec, seg_end) - max(start_sec, seg_start)
            if overlap > 0:
                overlap_by_speaker[speaker_no] = overlap_by_speaker.get(speaker_no, 0.0) + overlap
        if not overlap_by_speaker:
            return 0
        return max(overlap_by_speaker, key=overlap_by_speaker.get)
