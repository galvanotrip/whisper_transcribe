import os
import math
import ffmpeg
import gc
import time
import signal
import sys
import argparse
from faster_whisper import WhisperModel

CHUNK_SECONDS = 60
SUPPORTED_EXT = ('.mp4', '.mp3', '.wav', '.mkv')
TEMP_DIR = "temp_chunks"
LOG_FILE = "log.txt"

# === Ctrl+C обработка ===
def handle_sigint(sig, frame):
    print("\n🚫 Остановка по Ctrl+C. До связи! 👋")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

os.makedirs(TEMP_DIR, exist_ok=True)

def log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def get_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe["format"]["duration"])
    except ffmpeg.Error as e:
        raise Exception(f"FFprobe error: {e.stderr.decode()}")

def split_audio(input_file, chunk_length):
    duration = get_duration(input_file)
    chunks = []

    for i in range(math.ceil(duration / chunk_length)):
        out_path = os.path.join(TEMP_DIR, f"{os.path.basename(input_file)}_chunk_{i}.wav")
        ffmpeg.input(input_file, ss=i * chunk_length, t=chunk_length).output(
            out_path, ac=1, ar='16000'
        ).run(overwrite_output=True, quiet=True)
        chunks.append(out_path)

    return chunks

def print_progress(current, total):
    percent = int((current / total) * 100)
    bar = "#" * (percent // 5) + "-" * (20 - percent // 5)
    print(f"  ⏳ [{bar}] {percent}% ", end="\r")

def transcribe_file(input_file, output_file):
    log(f"🟡 Обработка: {os.path.basename(input_file)}")

    try:
        model = WhisperModel("medium", compute_type="int8")
        chunks = split_audio(input_file, CHUNK_SECONDS)

        checkpoint_path = output_file + ".checkpoint"
        last_chunk = 0
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                last_chunk = int(f.read().strip())

        for i, chunk in enumerate(chunks):
            if i < last_chunk:
                log(f"  ⏩ Пропуск chunk {i+1} (уже обработан)")
                os.remove(chunk)
                continue

            print_progress(i + 1, len(chunks))
            segments, _ = model.transcribe(chunk)
            text_part = "\n".join([seg.text for seg in segments])

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(text_part + "\n")

            with open(checkpoint_path, "w") as f:
                f.write(str(i + 1))

            os.remove(chunk)
            gc.collect()
            time.sleep(1)

        print()
        log(f"✅ Сохранено: {output_file}")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        del model
        gc.collect()
        time.sleep(2)

    except Exception as e:
        log(f"❌ Ошибка на {input_file}: {e}")

def process_folder(folder_path):
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXT)
        and os.path.isfile(os.path.join(folder_path, f))
    ]

    for f in files:
        input_path = os.path.join(folder_path, f)
        output_path = os.path.splitext(input_path)[0] + ".txt"

        if os.path.exists(output_path) and not os.path.exists(output_path + ".checkpoint"):
            log(f"⏩ Уже есть: {os.path.basename(output_path)} — пропускаем")
            continue

        transcribe_file(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio/video files in a folder.")
    parser.add_argument("folder", nargs="?", default="/Volumes/Transcend/video", help="Path to the folder with files")
    args = parser.parse_args()

    folder = args.folder
    process_folder(folder)
