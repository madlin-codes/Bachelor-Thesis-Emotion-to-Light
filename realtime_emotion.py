import os #prüft ob die Modelldatei existiert
import time #Zeitmessung für Hop-Abstände oder LED-Sendeintervalle
import queue #zwischenspeichert Audioblöcke aus dem Mikrofon-Callback
import numpy as np #numerische Verarbeitung von Audiodaten
import joblib #lädt das gespeicherte Modell-Bundle
import sounddevice as sd #Mikrofonaufnahme in Echtzeit
import librosa #Audioverarbeitung und Merkmalsextraktion
from collections import Counter, deque #für Voting und Zählen der Emotionen

# pandas prüfung für feature Vektor als data frame
try:
    import pandas as pd
except Exception:
    pd = None

#prüfung für led ausgabe
try:
    import serial
except Exception:
    serial = None



# Modellpfad SVM mit extra Features
BUNDLE_PATH = r"C:\Users\madli\Desktop\Emotion_Echtzeit\Finales_Echtzeitbundle_SVM_Extrafeature.joblib"

# Samplingrate und Mono Kanal
SR = 16000 
CHANNELS = 1

#Fenster & Hopgröße
WINDOW_SEC = 2.7 # 
HOP_SEC = 0.7 #alle 0.7 sek neue Vorhersage

# Trainings-Notebook Parameter Offline SVM Anpassung an Echtzeitmodell
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
N_MFCC = 13

# Preprocessing wie im Offline Modell
TRIM_TOP_DB = 20
TRIM_FRAME_LENGTH = 2048
TRIM_HOP_LENGTH = 512
USE_TRIM = True #true
USE_PEAK_NORM = True #true
MIN_LEN = 2048  

# Gates falls Signal zu leise
SILENCE_RMS = 0.003 # mittlere energie
MIN_PEAK = 0.008 # ausschlag

USE_SPEECH_GATE = False 
SPEECH_RMS_TH = 0.012
SPEECH_MIN_RATIO = 0.25 #0.3

# Stabilität der Vorhersage
USE_VOTE = True #true
VOTE_N = 2 #7 wie viele vorhersagen mitzählen, häufigste gewinnt

#Stabilität für neue Emotion sonst bleibt alte bestehen
USE_HOLD = False #true
HOLD_SEC = 0.0 #1.0

#Switch zu neutral bei Vergleich 2 top Scores, Abstand zu klein dann Vorhersage unsicher -> neutral
USE_UNCERTAIN_TO_NEUTRAL = False
UNCERTAIN_GAP = 0.6 # 0.6

# Debug Ausgaben
PRINT_AUDIO_DEBUG = True #zeigt RMS, Peak,Clipping
PRINT_SCORE_DEBUG = True #zeigt 3 beste Klassenwerte
PRINT_MISSING_FEATURES = False #zeigt falls erwartete Features fehlen

# Mikro festlegen, Razor Mini 3
MIC_NAME_CONTAINS = "Razer"
MIC_DEVICE_INDEX = 11 #Auf 11 lassen damit es erkannt wird


# LED / SERIAL CONFIG
USE_LED = True
SERIAL_PORT = "COM3" #Port -> Gerätemanager checken ob dort auch COM§ steht, am laptop oben links einstecken!
SERIAL_BAUD = 115200 #Baudrate
LED_SEND_INTERVAL = 0.25 #Alle 0,25 sek wird neu gesendet

#RGB-Farbwerte Zuordnung
EMOTION_TO_RGB = {
    "anger": (255, 0, 0),
    "fear": (180, 0, 255),
    "sadness": (0, 0, 255),
    "happiness": (255, 180, 0),
    "neutral": (255, 255, 255),
    "boredom": (0, 255, 180),
    "disgust": (0, 255, 0),
}

#Audio Queue puffert Audioblöcke zwischen Mikrofon Callbck & Schleife
audio_q = queue.Queue(maxsize=8)


# AUDIO CALLBACK
# Bei Aufnahmeproblemen wir alter Block verworfen und neuer gespeichert, System bleibt echtzeitfähig
def callback(indata, frames, time_info, status):
    if status:
        print(status)

    try:
        audio_q.put_nowait(indata.copy())
    except queue.Full:
        try:
            audio_q.get_nowait()
        except queue.Empty:
            pass
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            pass



# DEVICE HELPERS
# zeigt Index, Gerätename, Kanalanzhal, Host API -> für MIC_DEVICE_INDEX
def list_input_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
 
    print("Available INPUT devices")
    
    for i, d in enumerate(devices):
        if int(d.get("max_input_channels", 0)) > 0:
            host = hostapis[d["hostapi"]]["name"] if "hostapi" in d else ""
            print(f"[{i:02d}] {d['name']}  | channels={d['max_input_channels']} | host={host}")
    print("==============================\n")

# Sucht Gerät mit Name Razor
def pick_input_device(name_contains: str, fallback_index=None):
    devices = sd.query_devices()
    needle = (name_contains or "").strip().lower()
    candidates = []

    for i, d in enumerate(devices):
        if int(d.get("max_input_channels", 0)) <= 0:
            continue
        n = d["name"].lower()
        if needle and needle in n:
            candidates.append((i, d["name"]))

    if fallback_index is not None:
        try:
            d = sd.query_devices(fallback_index)
            if int(d.get("max_input_channels", 0)) > 0:
                return fallback_index
        except Exception:
            pass

    if candidates:
        return candidates[0][0]
    return None



# PREPROCESSING
# Peak Ausschlag, Signal wird durch größten Wert geteilt um Lautstärken vergleichbarer zu machen
def peak_normalize(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)

# Bereitet Signal für Merkmalsvektor vor
# Umwandlung in float32
# Ungültige Werte wie NaN oder Inf werden zu 0
def preprocess_live_audio(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
#optionales Trmming
    if USE_TRIM:
        try:
            y, _ = librosa.effects.trim(
                y,
                top_db=TRIM_TOP_DB,
                frame_length=TRIM_FRAME_LENGTH,
                hop_length=TRIM_HOP_LENGTH
            )
        except Exception:
            pass
#optional peak normalisierung
    if USE_PEAK_NORM:
        y = peak_normalize(y)
#Mindestlänge prüfen
    if len(y) < MIN_LEN:
        y = np.pad(y, (0, MIN_LEN - len(y)))

    return y.astype(np.float32)



# SMALL UTILS
# Berechnet für 1D Vektor Mittelwert, Standardabweichung, Minimum, Maximum wie im Offline Modell
#Falls None, leer, ungültig -> 0
def safe_stats_1d(x: np.ndarray):
    if x is None:
        return 0.0, 0.0, 0.0, 0.0

    x = np.asarray(x).reshape(-1)

    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    x = x[np.isfinite(x)]

    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(np.nanmean(x)),
        float(np.nanstd(x)),
        float(np.nanmin(x)),
        float(np.nanmax(x)),
    )

#Berechnet RMS Wert. Maß für mitterle Energie
def rms_level(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0

#Größter Ausschlag
def peak_level(y: np.ndarray) -> float:
    return float(np.max(np.abs(y))) if y.size else 0.0

#Prozentanteil der Samples an Obergrenze
def clip_percent(y: np.ndarray, thr=0.98) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) > thr) * 100.0)

#Frame RMS, wie viel vom Fenster enthält sprachliche Aktivität
def compute_speech_ratio(y: np.ndarray) -> float:
    try:
        r = librosa.feature.rms(
            y=y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True
        )[0]
        if r.size == 0:
            return 0.0
        return float(np.mean(r > SPEECH_RMS_TH))
    except Exception:
        return 0.0



# LED HELPERS
def open_serial():
    if not USE_LED:
        return None

    if serial is None:
        print("pyserial nicht installiert. LED-Ausgabe deaktiviert.")
        return None

    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2.0)
        print(f"LED Serial verbunden: {SERIAL_PORT} @ {SERIAL_BAUD}")
        return ser
    except Exception as e:
        print(f"LED Serial konnte nicht geöffnet werden: {e}")
        return None

#Farbe senden
def send_led_color(ser, emotion: str):
    if ser is None:
        return

    rgb = EMOTION_TO_RGB.get(str(emotion), (255, 255, 255))
    msg = f"{rgb[0]},{rgb[1]},{rgb[2]}\n"

    try:
        ser.write(msg.encode("utf-8"))
    except Exception as e:
        print(f"LED senden fehlgeschlagen: {e}")



# FEATURE EXTRACTION
# Wie bei finales SVM Extra Feature
def extract_llds_live(y: np.ndarray, sr: int):
    feats = {}

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )
    feats["mfcc"] = mfcc

    # RMS
    rms = librosa.feature.rms(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        center=True
    )
    feats["rms"] = rms

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        center=True
    )
    feats["zcr"] = zcr

    # Pitch
    try:
        pitch_hz = librosa.yin(
            y=y,
            fmin=50,
            fmax=400,   # angepasst an Notebook
            sr=sr,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH
        )
        pitch_hz_masked = np.where(np.isfinite(pitch_hz), pitch_hz, np.nan)
        feats["pitch_hz"] = pitch_hz_masked
    except Exception:
        pass

    # RMS dB
    try:
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        feats["rms_db"] = rms_db
    except Exception:
        pass

    # Spectral Centroid
    feats["spec_centroid"] = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    # Spectral
    feats["spec_bandwidth"] = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    # Chroma
    feats["chroma"] = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Spectral Contrast
    feats["spec_contrast"] = librosa.feature.spectral_contrast(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Tonnetz
    try:
        feats["tonnetz"] = librosa.feature.tonnetz(y=y, sr=sr)
    except Exception:
        pass

    return feats

# Aggregation zu statistischen Features 
def aggregate_llds_live(feats, use_minmax=True):
    feature_vector = {}

    for name, arr in feats.items():
        arr = np.asarray(arr)

        # 1D
        if arr.ndim == 1:
            m, s, mn, mx = safe_stats_1d(arr)
            feature_vector[f"{name}_mean"] = m
            feature_vector[f"{name}_std"] = s
            if use_minmax:
                feature_vector[f"{name}_min"] = mn
                feature_vector[f"{name}_max"] = mx

        # 2D
        elif arr.ndim == 2:
            for i in range(arr.shape[0]):
                x = arr[i, :]
                m, s, mn, mx = safe_stats_1d(x)
                feature_vector[f"{name}_{i:02d}_mean"] = m
                feature_vector[f"{name}_{i:02d}_std"] = s
                if use_minmax:
                    feature_vector[f"{name}_{i:02d}_min"] = mn
                    feature_vector[f"{name}_{i:02d}_max"] = mx

    return feature_vector

#Feature Vektor für das Modell
#Vorverarbeitung, LLD Extraktion, Aggregation, Sortierung für Training
def extract_feature_vector(y: np.ndarray, sr: int, expected_cols):
    y = preprocess_live_audio(y, sr)

    feats = extract_llds_live(y, sr)
    vec = aggregate_llds_live(feats, use_minmax=True)

    for k, v in list(vec.items()):
        if not np.isfinite(v):
            vec[k] = 0.0

    missing = [c for c in expected_cols if c not in vec]
    if PRINT_MISSING_FEATURES and missing:
        print(f"MISSING FEATURES: {len(missing)} / {len(expected_cols)}")
        print("   examples:", missing[:12])
#Sortierung genau wie im Training
    row = [float(vec.get(c, 0.0)) for c in expected_cols]
# Ein Feature Vektor pro Vorhersage
    if pd is not None:
        X = pd.DataFrame([row], columns=expected_cols)
    else:
        X = np.array([row], dtype=float)

    return X



# SCORE HELPERS
# Modell Scores mit Abstand zur anderen Klasse
def get_top_scores(predictor, X):
    func = None

    if hasattr(predictor, "decision_function"):
        func = predictor.decision_function
    elif hasattr(predictor, "predict_proba"):
        func = predictor.predict_proba

    if func is None:
        return None, None

    try:
        scores = func(X)
    except Exception:
        scores = func(np.asarray(X))

    scores = np.array(scores).reshape(-1)

    classes = None
    if hasattr(predictor, "classes_"):
        classes = predictor.classes_
    else:
        try:
            classes = predictor.named_steps["svm"].classes_
        except Exception:
            classes = None

    if classes is None or len(scores) != len(classes):
        return None, None

    return np.array(classes), scores

# 3 Top Scores
def print_top3_scores(predictor, X):
    classes, scores = get_top_scores(predictor, X)
    if classes is None or scores is None:
        return

    top = np.argsort(scores)[::-1][:3]
    print("   top3:", [(str(classes[i]), float(scores[i])) for i in top])

    if USE_UNCERTAIN_TO_NEUTRAL and len(top) >= 2:
        gap = float(scores[top[0]] - scores[top[1]])
        print(f"   gap(top1-top2)={gap:.3f}")

#Prüft Abstand zwischen Top1 und Top2, unsicher -> neutral
def apply_uncertain_to_neutral(predictor, X, raw_label):
    if not USE_UNCERTAIN_TO_NEUTRAL:
        return raw_label

    classes, scores = get_top_scores(predictor, X)
    if classes is None or scores is None:
        return raw_label

    order = np.argsort(scores)[::-1]
    if len(order) < 2:
        return raw_label

    gap = float(scores[order[0]] - scores[order[1]])
    if gap < UNCERTAIN_GAP:
        if "neutral" in set(map(str, classes)):
            return "neutral"

    return raw_label



# MAIN
def main():
    list_input_devices()

    if MIC_DEVICE_INDEX is not None:
        device = MIC_DEVICE_INDEX
    else:
        device = pick_input_device(MIC_NAME_CONTAINS, fallback_index=None)

    if device is None:
        print("Konnte Mikro nicht automatisch finden.")
        print("Setze MIC_DEVICE_INDEX oben auf den richtigen Index aus der Liste.")
        return

    d = sd.query_devices(device)
    default_sr = int(d.get("default_samplerate", SR))

    print(f"Using input device index: {device}")
    print(f"Device name: {d['name']}")
    print(f"Device default SR: {default_sr} Hz\n")

    if not os.path.exists(BUNDLE_PATH):
        print(f"Bundle nicht gefunden: {BUNDLE_PATH}")
        return

    bundle = joblib.load(BUNDLE_PATH)
    print(f"Bundle loaded from: {BUNDLE_PATH}")
    print("Bundle keys:", list(bundle.keys()))

    expected_cols = bundle.get("feature_names")
    if expected_cols is None:
        raise ValueError("Bundle enthält kein 'feature_names'.")

    predictor = bundle.get("pipeline", None)
    if predictor is None:
        predictor = bundle.get("model", None)
    if predictor is None:
        raise ValueError("Bundle enthält weder 'pipeline' noch 'model'.")

    print("Predictor type:", type(predictor))
    print("Expected features:", len(expected_cols))
    print("First 15 features:", expected_cols[:15])

    ser = open_serial()

    print("\n ....Realtime läuft.... (Strg+C zum Stop)")
    print(f"   Window/Hop: {WINDOW_SEC:.1f}s / {HOP_SEC:.1f}s")
    print(f"   Gates: SILENCE_RMS<{SILENCE_RMS}  MIN_PEAK<{MIN_PEAK}")
    if USE_SPEECH_GATE:
        print(f"   SpeechGate: TH={SPEECH_RMS_TH}  MIN_RATIO={SPEECH_MIN_RATIO}")
    print(f"   Preprocess: TRIM={USE_TRIM} | PEAK_NORM={USE_PEAK_NORM} | MIN_LEN={MIN_LEN}")
    print(f"   Vote: {USE_VOTE} (N={VOTE_N}) | Hold: {USE_HOLD} ({HOLD_SEC:.1f}s)")
    print(f"   Uncertain->neutral: {USE_UNCERTAIN_TO_NEUTRAL} (gap<{UNCERTAIN_GAP})")
    if USE_LED:
        print(f"   LED: {SERIAL_PORT} @ {SERIAL_BAUD}")
    print()

    window_samples = int(WINDOW_SEC * SR)
    hop_samples = int(HOP_SEC * SR)

    ring = np.zeros(window_samples, dtype=np.float32)
    filled = 0
    last_pred_t = 0.0

    raw_hist = deque(maxlen=max(1, VOTE_N))
    last_output_label = None
    last_output_t = 0.0
    last_led_send_t = 0.0
    last_led_label = None

    emotion_counter = Counter()
    idx = 0

    try:
        with sd.InputStream(
            samplerate=SR,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
            blocksize=hop_samples,
            device=device
        ):
            while True:
                chunk = audio_q.get().reshape(-1)
                n = len(chunk)

                if n >= window_samples:
                    ring[:] = chunk[-window_samples:]
                    filled = window_samples
                else:
                    ring = np.roll(ring, -n)
                    ring[-n:] = chunk
                    filled = min(window_samples, filled + n)

                now = time.time()

                if filled < window_samples:
                    continue

                if (now - last_pred_t) < HOP_SEC:
                    continue
                last_pred_t = now

                r = rms_level(ring)
                p = peak_level(ring)
                c = clip_percent(ring, 0.98)

                if PRINT_AUDIO_DEBUG:
                    print(f"[audio] rms={r:.4f} peak={p:.3f} clip%={c:.2f}")

                if r < SILENCE_RMS or p < MIN_PEAK:
                    continue

                if USE_SPEECH_GATE:
                    sr_ratio = compute_speech_ratio(ring)
                    if PRINT_AUDIO_DEBUG:
                        print(f"   speech_ratio={sr_ratio:.2f}")
                    if sr_ratio < SPEECH_MIN_RATIO:
                        continue

                try:
                    X = extract_feature_vector(ring.copy(), SR, expected_cols)
                    raw = predictor.predict(X)[0]
                    raw = str(raw)
                except Exception as e:
                    print(f"Prediction/Feature Fehler: {e}")
                    continue

                raw2 = apply_uncertain_to_neutral(predictor, X, raw)

                if USE_VOTE:
                    raw_hist.append(raw2)
                    voted = Counter(raw_hist).most_common(1)[0][0]
                else:
                    voted = raw2

                if USE_HOLD and last_output_label is not None:
                    if voted != last_output_label and (now - last_output_t) < HOLD_SEC:
                        voted = last_output_label

                idx += 1
                last_output_label = voted
                last_output_t = now

                emotion_counter[voted] += 1
                print(f"[{idx:04d}] ➡️ {voted} (raw={raw2})")

                if USE_LED and ser is not None:
                    if (voted != last_led_label) or ((now - last_led_send_t) >= LED_SEND_INTERVAL):
                        send_led_color(ser, voted)
                        last_led_send_t = now
                        last_led_label = voted

                if PRINT_SCORE_DEBUG:
                    print_top3_scores(predictor, X)

    except KeyboardInterrupt:
        print("\nAbgebrochen mit Strg+C.")

    finally:
        if ser is not None:
            try:
                send_led_color(ser, "neutral")
                ser.close()
            except Exception:
                pass

    print("\n==============================")
    print("SUMMARY – Emotion Häufigkeiten")
    print("==============================")
    total = sum(emotion_counter.values())

    classes = None
    try:
        classes = predictor.classes_
    except Exception:
        try:
            classes = predictor.named_steps["svm"].classes_
        except Exception:
            classes = None

    labels = list(map(str, classes)) if classes is not None else sorted(emotion_counter.keys())
    for lab in labels:
        c = emotion_counter.get(lab, 0)
        pct = (c / total * 100) if total else 0.0
        print(f"{lab:12s}  {c:3d}  ({pct:5.1f}%)")
    print("==============================\n")


if __name__ == "__main__":
    main()