# ====== PATCH: ALLOW ALL REPOS FOR NEUCODEC ======
import os, re

def patch_neucodec():
    import neucodec
    import inspect
    
    path = inspect.getfile(neucodec)  # đường dẫn neucodec/main.py
    print("Patching file:", path)

    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Nếu đã patch rồi thì bỏ qua
    if "# PATCHED_ALLOW_ALL" in code:
        print("Already patched.")
        return

    # Tìm dòng assert và comment nó
    patched = re.sub(
        r"assert model_id[^\n]+",
        "# PATCHED_ALLOW_ALL: assert removed to allow custom repos",
        code
    )

    # Ghi lại
    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)

    print("Patch applied successfully!")

# Thực thi patch
patch_neucodec()

# ====== TIẾP TỤC CODE TTS ======
from vieneu_tts import VieNeuTTS
import soundfile as sf
import os

input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi. Giáo viên sử dụng video, bài tập tương tác và thảo luận trực tuyến để nâng cao hiệu quả học tập.",
    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não và chế độ dinh dưỡng lành mạnh, giúp giảm tốc độ suy giảm trí nhớ ở người cao tuổi.",
    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn, kết hợp yếu tố tâm lý sâu sắc khiến người đọc luôn hồi hộp theo dõi diễn biến câu chuyện.",
    "Các nhà khoa học nghiên cứu gen người phát hiện những đột biến mới liên quan đến bệnh di truyền. Điều này giúp nâng cao khả năng chẩn đoán và điều trị.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

def main(
    backbone="truongboykkhd/VieNeu-TTS-1000h",
    codec="truongboykkhd/neucodec"
):
    ref_audio_path = "./sample/id_0001.wav"
    ref_text_path = "./sample/id_0001.txt"

    ref_text_raw = open(ref_text_path, "r", encoding="utf-8").read()

    tts = VieNeuTTS(
        backbone_repo=backbone,
        backbone_device="cuda",
        codec_repo=codec,
        codec_device="cuda"
    )

    print("Encoding reference audio...")
    ref_codes = tts.encode_reference(ref_audio_path)

    for i, text in enumerate(input_texts, 1):
        print(f"Generating audio {i}/{len(input_texts)}...")
        wav = tts.infer(text, ref_codes, ref_text_raw)
        sf.write(f"./output_audio/output_{i}.wav", wav, 24000)
        print(f"✓ Saved output_{i}.wav")

if __name__ == "__main__":
    main()
