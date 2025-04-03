import os
import math
import tempfile
import ffmpeg
from faster_whisper import WhisperModel

CHUNK_SECONDS = 300  # 5 минут

def get_duration(file_path):
    probe = ffmpeg.probe(file_path)
    return float(probe["format"]["duration"])

def split_audio(input_file, chunk_length):
    temp_dir = tempfile.mkdtemp()
    duration = get_duration(input_file)
    chunks = []

    for i in range(0, math.ceil(duration / chunk_length)):
        out_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        ffmpeg.input(input_file, ss=i * chunk_length, t=chunk_length).output(
            out_path, ac=1, ar='16000'
        ).run(overwrite_output=True, quiet=True)
        chunks.append(out_path)

    return chunks

def transcribe_chunk(model, audio_path):
    segments, _ = model.transcribe(audio_path)
    return "\n".join([seg.text for seg in segments])

def transcribe_long_audio(file_path, output_txt="transcription.txt"):
    print("📦 Загружаем модель...")
    model = WhisperModel("base", compute_type="int8")  # можешь попробовать "small" или "medium"

    print("✂️ Режем на части...")
    chunks = split_audio(file_path, CHUNK_SECONDS)

    final_text = ""
    for idx, chunk in enumerate(chunks):
        print(f"🎧 Обработка части {idx + 1}/{len(chunks)}...")
        text = transcribe_chunk(model, chunk)
        final_text += text + "\n"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"\n✅ Готово! Сохранил в: {output_txt}")

if __name__ == "__main__":
    input_file = "./test.mp4"  # замени на свой путь
    transcribe_long_audio(input_file)