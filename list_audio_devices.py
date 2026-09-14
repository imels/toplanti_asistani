"""Mevcut ses giriş cihazlarını listeler.

BlackHole + Aggregate Device kurulumundan sonra, oluşturduğunuz
combined cihazın index'ini/adını bulmak için çalıştırın:

    python list_audio_devices.py
"""

import sounddevice as sd

if __name__ == "__main__":
    print(sd.query_devices())
    print(f"\nVarsayılan giriş cihazı: {sd.default.device[0]}")
