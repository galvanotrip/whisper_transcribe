"""Transcribe every audio/video file in a folder, picking the ASR engine by language.

Flow:
  1. Find files that still need work (no .txt yet, or a .txt with a .checkpoint).
  2. Detect each file's language with a small Whisper model on 3 samples
     taken from different parts of the file. Results are cached in
     <folder>/.lang_cache.json (keys: path relative to <folder>) after every file, so a crash does not repeat detection.
  3. Group files by engine and load ONE engine at a time (8 GB RAM budget):
       he                      -> ivrit-ai/whisper-large-v3-turbo-ct2 (faster-whisper)
       25 European languages   -> NVIDIA Parakeet TDT 0.6B v3 int8 (onnx-asr)
       anything else           -> Whisper medium with the detected language
  4. Transcribe chunk by chunk. After each chunk the text is appended to <name>.txt
     and the processed position is saved to <name>.txt.checkpoint, so a stopped
     run resumes where it left off.

Checkpoint format: "s=<seconds processed> b=<size of .txt in bytes>". On resume the
.txt is cut back to b, so a crash between writing text and saving the checkpoint
never duplicates a chunk. Older formats ("s=<seconds>" or a bare 60-second chunk
index) are still understood.
"""

import argparse
import gc
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np

YTDLP_STREAM = re.compile(r"\.f\d+\.\w+$")  # e.g. "title.f401.mp4"
SUPPORTED_EXT = ('.mp4', '.mp3', '.wav', '.mkv')
LOG_FILE = "log.txt"
LANG_CACHE_NAME = ".lang_cache.json"  # stored inside the media folder, next to the files
SAMPLE_RATE = 16000
CPU_THREADS = int(os.environ.get("CPU_THREADS", "4"))  # i5-4278U: 2 cores / 4 threads
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
    """Return (seconds processed, txt size in bytes or None). Understands old formats."""
    if not os.path.exists(path):
        return 0.0, None
    raw = open(path).read().strip()
    if raw.isdigit():  # legacy: number of finished 60-second chunks
        return float(int(raw) * LEGACY_CHUNK_SECONDS), None
    fields = dict(part.split("=", 1) for part in raw.split())
    if "s" not in fields:
        raise ValueError(f"unknown checkpoint format in {path}: {raw!r}")
    size = int(fields["b"]) if "b" in fields else None
    return float(fields["s"]), size


def write_checkpoint(path, seconds, txt_size):
    """Write atomically: a crash mid-write leaves the previous checkpoint intact."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"s={seconds:.0f} b={txt_size}")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------- language detection ----------

LEGACY_LANG_CACHE = "lang_cache.json"  # previous location: script working dir, full-path keys


def load_lang_cache(path):
    """Cache of detected languages, keyed by file name. Survives crashes and restarts."""
    if not os.path.exists(path):
        if os.path.exists(LEGACY_LANG_CACHE):  # one-time import from the previous version
            with open(LEGACY_LANG_CACHE, encoding="utf-8") as f:
                old = json.load(f)
            log(f"🌐 Импорт {len(old)} языков из {LEGACY_LANG_CACHE}")
            return {os.path.basename(k): v for k, v in old.items()}  # old run covered only the root folder
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        log(f"⚠️ Кеш языков повреждён ({e}), определяю заново")
        return {}


def save_lang_cache(path, cache):
    """Atomic write after every file: a crash loses at most the file in progress."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8",
                                  cpu_threads=CPU_THREADS)
        self.chunk_seconds = WHISPER_CHUNK_SECONDS

    def transcribe(self, audio, lang):
        segments, _ = self.model.transcribe(audio, language=lang, vad_filter=True)
        return "\n".join(s.text.strip() for s in segments)


class ParakeetEngine:
    def __init__(self):
        import onnx_asr
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = CPU_THREADS
        self.name = PARAKEET_MODEL
        self.model = onnx_asr.load_model(PARAKEET_MODEL, quantization="int8", sess_options=opts)
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
    pos, txt_size = read_checkpoint(checkpoint_path)
    if txt_size is not None and os.path.exists(output_file) and os.path.getsize(output_file) > txt_size:
        # text of an unfinished chunk was written but its checkpoint was not: drop it
        with open(output_file, "r+b") as f:
            f.truncate(txt_size)
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
            f.flush()
            os.fsync(f.fileno())
        pos += length
        write_checkpoint(checkpoint_path, pos, os.path.getsize(output_file))
        elapsed = (time.time() - started) / 60
        log(f"  ⏳ {pos/duration*100:5.1f}%  {pos/60:.1f}/{duration/60:.1f} мин аудио, прошло {elapsed:.1f} мин")
        gc.collect()

    os.remove(checkpoint_path)
    log(f"✅ Сохранено: {output_file}")


def pending_files(folder):
    """All media files in folder and its subfolders (<channel>/<playlist>/...) without a finished .txt."""
    result = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            path = os.path.join(root, name)
            if name.startswith(".") or not name.lower().endswith(SUPPORTED_EXT):
                continue
            if YTDLP_STREAM.search(name):  # intermediate stream of an unfinished download
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
    cache_path = os.path.join(folder, LANG_CACHE_NAME)
    cache = load_lang_cache(cache_path)
    def key(p):  # path relative to the media folder; for files in its root this is the bare name
        return os.path.relpath(p, folder)
    need = [(p, o) for p, o in todo if key(p) not in cache]
    log(f"🌐 Язык уже известен: {len(todo) - len(need)}, определить: {len(need)}")
    if need:
        from faster_whisper import WhisperModel
        detector = WhisperModel(DETECT_MODEL, device="cpu", compute_type="int8",
                                cpu_threads=CPU_THREADS)
        for path, _ in need:
            try:
                lang, prob = detect_language(detector, path, get_duration(path))
                cache[key(path)] = {"lang": lang, "prob": round(prob, 3)}
                save_lang_cache(cache_path, cache)
                log(f"🌐 {key(path)}: {lang} ({prob:.2f})")
            except Exception as e:
                log(f"❌ Не удалось определить язык {path}: {e}")
        del detector
        gc.collect()

    # 2. Group by engine so only one model sits in memory at a time.
    groups = {"hebrew": [], "parakeet": [], "fallback": []}
    for path, out in todo:
        if key(path) in cache:
            lang = cache[key(path)]["lang"]
            groups[route(lang)].append((path, out, lang))
    log("📦 " + ", ".join(f"{k}: {len(v)}" for k, v in groups.items()))

    failed = len(todo) - sum(len(v) for v in groups.values())
    for kind, files in groups.items():
        if not files:
            continue
        engine = make_engine(kind)
        log(f"🧠 Модель: {engine.name}")
        for path, out, lang in files:
            log(f"🟡 Обработка [{lang}]: {os.path.relpath(path, folder)}")
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
