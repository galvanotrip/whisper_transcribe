"""Transcribe every audio/video file in a folder, picking the ASR engine by language.

Flow:
  1. Find files that still need work (no .txt yet, or a .txt with a .checkpoint).
  2. Detect each file's language with a small Whisper model on 3 samples
     taken from different parts of the file. Results are cached in lang_cache.json.
  3. Group files by engine and load ONE engine at a time (8 GB RAM budget):
       he                      -> ivrit-ai/whisper-large-v3-turbo-ct2 (faster-whisper)
       25 European languages   -> NVIDIA Parakeet TDT 0.6B v3 int8 (onnx-asr)
       anything else           -> Whisper medium with the detected language
  4. Transcribe chunk by chunk. After each chunk the text is appended to <name>.txt
     and the processed position is saved to <name>.txt.checkpoint, so a stopped
     run resumes where it left off.

Checkpoint format: "s=<seconds processed>". Old checkpoints written by the
previous version contain a bare chunk index (60-second chunks) and are converted.
"""

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

SUPPORTED_EXT = ('.mp4', '.mp3', '.wav', '.mkv')
LOG_FILE = "log.txt"
LANG_CACHE_FILE = "lang_cache.json"
SAMPLE_RATE = 16000
LEGACY_CHUNK_SECONDS = 60  # chunk size used by the previous version of this script

DETECT_MODEL = os.environ.get("DETECT_MODEL", "base")
DETECT_SAMPLE_SECONDS = 30
DETECT_POSITIONS = (0.2, 0.5, 0.8)  # skip intros/music at the very start

HEBREW_MODEL = os.environ.get("HEBREW_MODEL", "ivrit-ai/whisper-large-v3-turbo-ct2")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "medium")
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"

# Chunk sizes per engine. Whisper keeps context inside a chunk, so longer is better
# for quality. Parakeet memory grows with chunk length, so it gets short chunks.
WHISPER_CHUNK_SECONDS = int(os.environ.get("WHISPER_CHUNK_SECONDS", "300"))
PARAKEET_CHUNK_SECONDS = int(os.environ.get("PARAKEET_CHUNK_SECONDS", "60"))

PARAKEET_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
}


def handle_sigint(sig, frame):
    print("\n🚫 Остановка по Ctrl+C. До связи! 👋", flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


def log(message):
    print(message, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# ---------- audio ----------

def get_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def load_segment(path, start, length):
    """Decode [start, start+length) seconds to 16 kHz mono float32 without temp files."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-t", str(length),
         "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, np.int16).astype(np.float32) / 32768.0


# ---------- checkpoint ----------

def read_checkpoint(path):
    """Return seconds already processed (0 if none). Understands the legacy format."""
    if not os.path.exists(path):
        return 0.0
    raw = open(path).read().strip()
    if raw.startswith("s="):
        return float(raw[2:])
    if raw.isdigit():  # legacy: number of finished 60-second chunks
        return float(int(raw) * LEGACY_CHUNK_SECONDS)
    raise ValueError(f"unknown checkpoint format in {path}: {raw!r}")


def write_checkpoint(path, seconds):
    with open(path, "w") as f:
        f.write(f"s={seconds:.0f}")


# ---------- language detection ----------

def load_lang_cache():
    if os.path.exists(LANG_CACHE_FILE):
        with open(LANG_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_lang_cache(cache):
    with open(LANG_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def detect_language(model, path, duration):
    """Sum language probabilities over several samples; return (lang, prob)."""
    if duration <= DETECT_SAMPLE_SECONDS * len(DETECT_POSITIONS):
        starts = [0.0]
    else:
        starts = [max(0.0, duration * p - DETECT_SAMPLE_SECONDS / 2) for p in DETECT_POSITIONS]
    totals = {}
    for start in starts:
        audio = load_segment(path, start, DETECT_SAMPLE_SECONDS)
        if audio.size == 0:
            continue
        _, _, all_probs = model.detect_language(audio=audio)
        for lang, prob in all_probs:
            totals[lang] = totals.get(lang, 0.0) + prob
    if not totals:
        raise RuntimeError("no audio decoded for language detection")
    lang = max(totals, key=totals.get)
    return lang, totals[lang] / len(starts)


def route(lang):
    if lang == "he":
        return "hebrew"
    if lang in PARAKEET_LANGS:
        return "parakeet"
    return "fallback"


# ---------- engines ----------

class WhisperEngine:
    def __init__(self, model_name):
        from faster_whisper import WhisperModel
        self.name = model_name
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.chunk_seconds = WHISPER_CHUNK_SECONDS

    def transcribe(self, audio, lang):
        segments, _ = self.model.transcribe(audio, language=lang, vad_filter=True)
        return "\n".join(s.text.strip() for s in segments)


class ParakeetEngine:
    def __init__(self):
        import onnx_asr
        self.name = PARAKEET_MODEL
        self.model = onnx_asr.load_model(PARAKEET_MODEL, quantization="int8")
        self.chunk_seconds = PARAKEET_CHUNK_SECONDS

    def transcribe(self, audio, lang):
        return self.model.recognize(audio, sample_rate=SAMPLE_RATE).strip()


def make_engine(kind):
    if kind == "hebrew":
        return WhisperEngine(HEBREW_MODEL)
    if kind == "parakeet":
        return ParakeetEngine()
    return WhisperEngine(FALLBACK_MODEL)


# ---------- main work ----------

def transcribe_file(engine, path, lang, output_file):
    checkpoint_path = output_file + ".checkpoint"
    duration = get_duration(path)
    pos = read_checkpoint(checkpoint_path)
    if pos > 0:
        log(f"  ⏩ Продолжаю с {pos/60:.1f} мин")
    started = time.time()

    while duration - pos >= 1.0:  # skip sub-second tails left by container rounding
        length = min(engine.chunk_seconds, duration - pos)
        audio = load_segment(path, pos, length)
        text = engine.transcribe(audio, lang) if audio.size else ""
        with open(output_file, "a", encoding="utf-8") as f:
            if text:
                f.write(text + "\n")
        pos += length
        write_checkpoint(checkpoint_path, pos)
        elapsed = (time.time() - started) / 60
        log(f"  ⏳ {pos/duration*100:5.1f}%  {pos/60:.1f}/{duration/60:.1f} мин аудио, прошло {elapsed:.1f} мин")
        gc.collect()

    os.remove(checkpoint_path)
    log(f"✅ Сохранено: {output_file}")


def pending_files(folder):
    result = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not (name.lower().endswith(SUPPORTED_EXT) and os.path.isfile(path)):
            continue
        out = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(out) and not os.path.exists(out + ".checkpoint"):
            continue
        result.append((path, out))
    return result


def process_folder(folder):
    todo = pending_files(folder)
    log(f"📋 Файлов к обработке: {len(todo)}")
    if not todo:
        return 0

    # 1. Language detection with a small model, cached between runs.
    cache = load_lang_cache()
    need = [(p, o) for p, o in todo if p not in cache]
    if need:
        from faster_whisper import WhisperModel
        detector = WhisperModel(DETECT_MODEL, device="cpu", compute_type="int8")
        for path, _ in need:
            try:
                lang, prob = detect_language(detector, path, get_duration(path))
                cache[path] = {"lang": lang, "prob": round(prob, 3)}
                save_lang_cache(cache)
                log(f"🌐 {os.path.basename(path)}: {lang} ({prob:.2f})")
            except Exception as e:
                log(f"❌ Не удалось определить язык {path}: {e}")
        del detector
        gc.collect()

    # 2. Group by engine so only one model sits in memory at a time.
    groups = {"hebrew": [], "parakeet": [], "fallback": []}
    for path, out in todo:
        if path in cache:
            groups[route(cache[path]["lang"])].append((path, out, cache[path]["lang"]))
    log("📦 " + ", ".join(f"{k}: {len(v)}" for k, v in groups.items()))

    failed = len(todo) - sum(len(v) for v in groups.values())
    for kind, files in groups.items():
        if not files:
            continue
        engine = make_engine(kind)
        log(f"🧠 Модель: {engine.name}")
        for path, out, lang in files:
            log(f"🟡 Обработка [{lang}]: {os.path.basename(path)}")
            try:
                transcribe_file(engine, path, lang, out)
            except Exception as e:
                failed += 1
                log(f"❌ Ошибка на {path}: {e}")
        del engine
        gc.collect()
    return failed


if __name__ == "__main__":
    print("=== START ===", flush=True)
    parser = argparse.ArgumentParser(description="Transcribe audio/video files in a folder.")
    parser.add_argument("folder", nargs="?", default="/Volumes/Transcend/video")
    args = parser.parse_args()

    failed = process_folder(args.folder)
    if failed:
        log(f"=== DONE WITH ERRORS: {failed} файл(ов) не обработано ===")
        sys.exit(1)
    print("=== DONE ===", flush=True)
