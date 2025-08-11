import os
import sounddevice as sd
import numpy as np
import librosa
import pickle

MODEL_DIR = "audiomodels"

def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def save_model(name, features):
    ensure_model_dir()
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(features, f)

def load_models():
    ensure_model_dir()
    models = {}
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            path = os.path.join(MODEL_DIR, file)
            with open(path, "rb") as f:
                features = pickle.load(f)
                name = os.path.splitext(file)[0]
                models[name] = features
    return models

def list_devices():
    devices = sd.query_devices()
    mic_devices = []
    loopback_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and 'loopback' not in dev['name'].lower():
            mic_devices.append((i, dev['name']))
        if dev['max_input_channels'] > 0 and 'loopback' in dev['name'].lower():
            loopback_devices.append((i, dev['name']))
    print("Mikrofonlar:")
    for i, name in mic_devices:
        print(f"{i}: {name}")
    print("Sistem sesi (loopback) cihazları:")
    for i, name in loopback_devices:
        print(f"{i}: {name}")
    return mic_devices, loopback_devices

def find_wasapi_loopback_device():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        # hostapi 7 = WASAPI Windows audio API, loopback içeren input kanalı
        if d['hostapi'] == 7 and d['max_input_channels'] > 0 and 'loopback' in d['name'].lower():
            return i
    return None

def record_audio(duration=3, sr=16000, device=None, use_loopback=False):
    print(f"Kayıt başladı... ({duration} saniye)")
    try:
        if use_loopback:
            if device is None:
                device = find_wasapi_loopback_device()
                if device is None:
                    print("WASAPI loopback cihazı bulunamadı! Windows'ta 'Stereo Mix' etkin veya VB-Cable kurulu olmalı.")
                    return None
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=2, dtype='float32',
                           device=device, blocking=True,
                           extra_settings=sd.WasapiSettings(loopback=True))
        else:
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32', device=device)
            sd.wait()
        print("Kayıt bitti.")
        return audio.flatten()
    except Exception as e:
        print(f"Hata: {e}")
        return None

def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def recognize(features, models, threshold=50):
    best_match = None
    best_distance = float("inf")
    for name, feat in models.items():
        dist = np.linalg.norm(features - feat)
        if dist < best_distance:
            best_distance = dist
            best_match = name
    if best_distance < threshold:
        return best_match
    else:
        return None

def main():
    models = load_models()
    mic_devices, loopback_devices = list_devices()
    print("\nSes kaynağı seç:\n1) Mikrofon\n2) Sistem Sesi (Loopback)")
    source_choice = input("Seçimin (1 veya 2): ")

    if source_choice == '1':
        device_id = int(input("Mikrofon cihaz ID'si: "))
        if device_id not in [d[0] for d in mic_devices]:
            print("Geçersiz mikrofon seçimi.")
            return
        use_loopback = False
    elif source_choice == '2':
        # Loopback cihaz seçimi yerine otomatik bulma:
        device_id = find_wasapi_loopback_device()
        if device_id is None:
            print("Loopback cihazı bulunamadı! 'Stereo Mix' etkinleştirilmeli veya VB-Cable kurulmalı.")
            return
        print(f"Otomatik bulunan loopback cihazı: {device_id}")
        use_loopback = True
    else:
        print("Geçersiz seçim.")
        return

    while True:
        try:
            duration = float(input("Kayıt süresi (saniye, örn: 10 veya 30): "))
            if duration <= 0:
                print("Süre pozitif olmalı.")
                continue
            break
        except ValueError:
            print("Lütfen geçerli bir sayı gir.")

    print("""
Modlar:
1) Tanıma
2) Kaydetme
3) Modele ekleme (Yeni kişi)
Çıkmak için 'q' yaz.
""")

    while True:
        mode = input("Mod seç (1/2/3/q): ")
        if mode == 'q':
            print("Çıkılıyor...")
            break

        if mode == '1':
            audio = record_audio(duration=duration, device=device_id, use_loopback=use_loopback)
            if audio is None:
                continue
            features = extract_features(audio)
            name = recognize(features, models)
            if name:
                print(f"Tanındı: {name}")
            else:
                print("Tanımlanamayan kişi!")
        elif mode == '2':
            audio = record_audio(duration=duration, device=device_id, use_loopback=use_loopback)
            if audio is None:
                continue
            filename = input("Kaydedilecek dosya adı (örnek: ses.wav): ")
            import soundfile as sf
            sf.write(filename, audio, 16000)
            print(f"{filename} kaydedildi.")
        elif mode == '3':
            audio = record_audio(duration=duration, device=device_id, use_loopback=use_loopback)
            if audio is None:
                continue
            features = extract_features(audio)
            new_name = input("Kişi adı: ")
            save_model(new_name, features)
            models[new_name] = features
            print(f"{new_name} modele eklendi.")
        else:
            print("Geçersiz mod seçimi.")

if __name__ == "__main__":
    main()
