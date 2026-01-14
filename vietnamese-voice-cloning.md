# Voice Cloning Tiếng Việt - Hướng Dẫn Chi Tiết

## Mục Lục
1. [Tổng Quan](#1-tổng-quan)
2. [Các Model Tiếng Việt Native](#2-các-model-tiếng-việt-native)
3. [So Sánh Chi Tiết](#3-so-sánh-chi-tiết)
4. [Hướng Dẫn Tự Train Model](#4-hướng-dẫn-tự-train-model)
5. [Dataset Tiếng Việt](#5-dataset-tiếng-việt)
6. [Recommendations](#6-recommendations)

---

## 1. Tổng Quan

### 1.1 Tại Sao Cần Model Tiếng Việt Native?

Tiếng Việt có những đặc điểm ngôn ngữ đặc biệt mà các model đa ngôn ngữ thường xử lý không tốt:

| Đặc điểm | Mô tả | Thách thức |
|----------|-------|------------|
| **6 thanh điệu** | Ngang, huyền, sắc, hỏi, ngã, nặng | Model phải phân biệt và tạo đúng thanh điệu |
| **Nguyên âm phức** | ư, ơ, â, ê, ô, ă | Phát âm khác biệt với tiếng Anh |
| **Phụ âm đặc biệt** | đ, ng, nh, kh, gh, gi | Cần training data tiếng Việt |
| **Dấu thanh** | àáảãạ, èéẻẽẹ... | Phải giữ nguyên trong text processing |

### 1.2 Bảng So Sánh Tổng Hợp

| Model | Tiếng Việt | Voice Clone | VRAM | Inference | License | Recommend |
|-------|------------|-------------|------|-----------|---------|-----------|
| **VieNeu-TTS** | Native (1000h) | 3-5 giây | 4-8GB | CPU Real-time | Apache 2.0 | **#1** |
| **F5-TTS-Vietnamese** | Fine-tuned (1000h) | Zero-shot | 8GB+ | Fast | Apache 2.0 | **#2** |
| **VietTTS** | Native | Yes | 6-8GB | Fast | Apache 2.0 | **#3** |
| **viXTTS** | Fine-tuned | 6 giây | 8-16GB | <150ms | Non-commercial | #4 |
| **VietVoice-TTS** | Native | Yes | 4-6GB | Fast | MIT | #5 |
| **Vi-SparkTTS** | Fine-tuned | Yes | 6-8GB | Fast | Custom | #6 |
| **GPT-SoVITS** | Community | 1 phút train | 8GB+ | RTF 0.028 | MIT | Alternative |

---

## 2. Các Model Tiếng Việt Native

### 2.1 VieNeu-TTS (Khuyến Nghị #1)

**GitHub**: https://github.com/pnnbao97/VieNeu-TTS

#### Thông Tin
- **Developer**: pnnbao97 (Việt Nam)
- **Training data**: 1000 giờ tiếng Việt (443,641 samples)
- **Architecture**: NeuTTS (Neural TTS)
- **Release**: 2025

#### Model Variants

| Model | Size | Chất lượng | Tốc độ | License |
|-------|------|------------|--------|---------|
| VieNeu-TTS 0.5B | 2GB | Cao nhất | Very Fast | Apache 2.0 |
| VieNeu-TTS 0.3B | 1.2GB | Cao | 2x faster | CC BY-NC |
| VieNeu-TTS-0.3B-q4-gguf | 400MB | Khá | Extreme | CC BY-NC |
| VieNeu-TTS-0.3B-q8-gguf | 800MB | Tốt | Ultra Fast | CC BY-NC |

#### Đặc Điểm Nổi Bật
1. **CPU Real-time**: Chạy được trên CPU thông thường
2. **Voice Cloning**: Chỉ cần 3-5 giây audio reference
3. **Code-switching**: Hỗ trợ Vietnamese-English xen kẽ
4. **LoRA Support**: Có thể fine-tune cho custom voice
5. **Docker Ready**: Có sẵn Docker image

#### Cài Đặt

```bash
# Clone repo
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS

# Cài đặt dependencies
pip install -r requirements.txt

# Download model từ Hugging Face
# Model sẽ tự động download khi chạy lần đầu
```

#### Sử Dụng

```python
# vieneu_example.py
from vieneu_tts import VieNeuTTS

# Initialize model
model = VieNeuTTS.from_pretrained("pnnbao97/VieNeu-TTS-0.5B")

# Zero-shot voice cloning
audio = model.generate(
    text="Xin chào các bạn, đây là giọng nói được nhân bản.",
    reference_audio="reference.wav",  # 3-5 giây
    language="vi"
)

# Save output
audio.save("output.wav")
```

#### Docker Deployment

```bash
# CPU mode (không cần GPU)
docker compose --profile cpu up

# GPU mode (cần NVIDIA Container Toolkit)
docker compose --profile gpu up

# Access: http://localhost:7860
```

---

### 2.2 F5-TTS-Vietnamese (Khuyến Nghị #2)

**Hugging Face**: https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice

#### Thông Tin
- **Base model**: F5-TTS
- **Fine-tuned on**: viVoice + VLSP 2021/2022/2023
- **Training time**: 1.5 tháng trên RTX 3090
- **Versions**: 100h và 1000h

#### Đặc Điểm
- Zero-shot voice cloning
- Hỗ trợ tiếng Việt chuẩn
- Output 24kHz quality
- Apache 2.0 License

#### Cài Đặt

```bash
# Install F5-TTS
pip install f5-tts

# Download Vietnamese model
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="hynt/F5-TTS-Vietnamese-ViVoice",
    filename="model_1000h.safetensors"
)
```

#### Sử Dụng

```python
# f5_vietnamese.py
from f5_tts import F5TTS
import torch

class F5VietnameseTTS:
    def __init__(self, model_path):
        self.model = F5TTS.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, text, reference_audio, output_path):
        """
        Generate Vietnamese speech với voice cloning.

        Args:
            text: Text tiếng Việt cần đọc
            reference_audio: File audio reference (3-10 giây)
            output_path: Path lưu output
        """
        audio = self.model.synthesize(
            text=text,
            reference_audio=reference_audio,
            language="vi"
        )
        audio.save(output_path)
        return output_path

# Usage
tts = F5VietnameseTTS("path/to/model_1000h.safetensors")
tts.generate(
    text="Hôm nay trời đẹp quá!",
    reference_audio="my_voice.wav",
    output_path="output.wav"
)
```

---

### 2.3 VietTTS (dangvansam)

**GitHub**: https://github.com/dangvansam/viet-tts
**Hugging Face**: https://huggingface.co/dangvansam/viet-tts

#### Đặc Điểm
- OpenAI API compatible format
- Docker deployment với GPU support
- Text normalization via Vinorm
- Active development

#### API Usage

```bash
# Start server
docker run -d -p 8298:8298 --gpus all dangvansam/viet-tts

# Generate speech (OpenAI compatible)
curl -X POST http://localhost:8298/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Xin chào Việt Nam!",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output output.wav
```

#### Python Usage

```python
# viettts_example.py
import requests

def generate_vietnamese_speech(text, output_path):
    """Generate speech using VietTTS API."""
    response = requests.post(
        "http://localhost:8298/v1/audio/speech",
        json={
            "input": text,
            "voice": "default",
            "response_format": "wav"
        }
    )

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path

# Usage
generate_vietnamese_speech(
    "Đây là test giọng nói tiếng Việt.",
    "output.wav"
)
```

---

### 2.4 viXTTS

**GitHub**: https://github.com/thinhlpg/vixtts-demo

#### Thông Tin
- Fine-tuned từ XTTS-v2.0.3
- Sử dụng viVoice dataset
- Expanded tokenizer cho tiếng Việt
- Hỗ trợ 18 ngôn ngữ

#### Lưu Ý
- Câu dưới 10 từ có thể output không ổn định
- Non-commercial license (Coqui)
- Project không còn actively maintained

#### Sử Dụng

```python
# vixtts_example.py
from vixtts import ViXTTS

# Load model
model = ViXTTS.from_pretrained()

# Generate với voice cloning
audio = model.generate(
    text="Xin chào, tôi là một trợ lý ảo.",
    speaker_wav="reference.wav",
    language="vi"
)

audio.save("output.wav")
```

---

### 2.5 VietVoice-TTS

**GitHub**: https://github.com/nguyenvulebinh/VietVoice-TTS

#### Đặc Điểm
- Multiple voice options: gender, accent, emotion
- Northern/Southern Vietnamese accents
- MIT License (commercial friendly)
- Voice cloning support

#### Sử Dụng

```python
# vietvoice_example.py
from vietvoicetts import VietVoiceTTS

tts = VietVoiceTTS()

# Generate với các options
tts.synthesize(
    text="Chào mừng bạn đến với Việt Nam!",
    voice="female",           # male/female
    accent="north",           # north/south
    emotion="happy",          # neutral/happy/sad
    style="news",             # news/conversation
    output_path="output.wav"
)

# Voice cloning
tts.clone_and_speak(
    text="Đây là giọng nói của tôi.",
    reference_audio="my_voice.wav",
    output_path="cloned_output.wav"
)
```

---

### 2.6 Vi-SparkTTS-0.5B

**Hugging Face**: https://huggingface.co/DragonLineageAI/Vi-SparkTTS-0.5B

#### Architecture
- Spark-TTS với BiCodec
- Semantic tokens cho linguistic content
- Global tokens cho speaker attributes
- Chain-of-thought (CoT) generation

#### Đặc Điểm
- Zero-shot voice cloning
- Controllable: gender, pitch, speaking rate
- Train trên viVoice dataset
- HuggingFace Transformers compatible

```python
# spark_vietnamese.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "DragonLineageAI/Vi-SparkTTS-0.5B"
)
tokenizer = AutoTokenizer.from_pretrained(
    "DragonLineageAI/Vi-SparkTTS-0.5B"
)

# Generate
inputs = tokenizer("Xin chào các bạn!", return_tensors="pt")
outputs = model.generate(**inputs)
```

---

### 2.7 Dia-Vietnamese (Multi-speaker)

**GitHub**: https://github.com/TuananhCR/Dia-Finetuning-Vietnamese

#### Đặc Điểm
- Fine-tuned từ Nari Labs Dia 1.6B
- Multi-speaker: North-male, South-male, North-female, South-female
- Output: 44.1kHz mono WAV/FLAC
- Controllable parameters

#### Sử Dụng

```python
# dia_vietnamese.py
from dia_vietnamese import DiaVietnamese

model = DiaVietnamese.from_pretrained(
    "cosrigel/dia-finetuning-vnese"
)

# Generate với speaker selection
audio = model.generate(
    text="Hà Nội là thủ đô của Việt Nam.",
    speaker="north-female",  # north-male, south-male, north-female, south-female
    temperature=0.8,
    top_p=0.9,
    cfg_scale=3.0
)

audio.save("output.wav")
```

---

## 3. So Sánh Chi Tiết

### 3.1 Chất Lượng Phát Âm Tiếng Việt

| Model | Dấu thanh | Âm đặc biệt | Ngữ điệu | Code-switch | Overall |
|-------|-----------|-------------|----------|-------------|---------|
| VieNeu-TTS | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **9.5/10** |
| F5-TTS-VN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | **8.5/10** |
| VietTTS | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | **8/10** |
| viXTTS | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | **7/10** |
| VietVoice | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | **7/10** |
| GPT-SoVITS | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **6.5/10** |

### 3.2 Voice Cloning Quality

| Model | Data cần | Zero-shot | Similarity | Training time |
|-------|----------|-----------|------------|---------------|
| VieNeu-TTS | 3-5 giây | ✅ Yes | 85%+ | 0 |
| F5-TTS-VN | 3-10 giây | ✅ Yes | 80%+ | 0 |
| viXTTS | 6 giây | ✅ Yes | 75%+ | 0 |
| VietTTS | 5-10 giây | ✅ Yes | 75%+ | 0 |
| GPT-SoVITS | 1 phút | Few-shot | 90%+ | 30-60 phút |
| VietVoice | 5-10 giây | ✅ Yes | 70%+ | 0 |

### 3.3 Hardware Requirements

| Model | Min VRAM | CPU Support | Inference Speed | Cold Start |
|-------|----------|-------------|-----------------|------------|
| VieNeu-TTS GGUF | 2GB | ✅ Real-time | <0.5 RTF | <10s |
| VieNeu-TTS 0.5B | 4GB | ✅ Yes | <0.3 RTF | <15s |
| VietVoice | 4GB | ✅ Yes | Fast | <10s |
| VietTTS | 6GB | Limited | Fast | 15-30s |
| F5-TTS-VN | 8GB | ❌ No | ~0.8 RTF | 20-40s |
| viXTTS | 8GB | Slow | ~1.0 RTF | 30-60s |

### 3.4 License & Commercial Use

| Model | License | Commercial | Personal | Attribution |
|-------|---------|------------|----------|-------------|
| VieNeu-TTS 0.5B | Apache 2.0 | ✅ Yes | ✅ Yes | Optional |
| VieNeu-TTS 0.3B | CC BY-NC | ❌ No | ✅ Yes | Required |
| F5-TTS-VN | Apache 2.0 | ✅ Yes | ✅ Yes | Optional |
| VietTTS | Apache 2.0 | ✅ Yes | ✅ Yes | Optional |
| VietVoice | MIT | ✅ Yes | ✅ Yes | Not required |
| viXTTS | Coqui | ❌ No | ✅ Yes | Required |
| GPT-SoVITS | MIT | ✅ Yes | ✅ Yes | Not required |

---

## 4. Hướng Dẫn Tự Train Model

### 4.1 Tổng Quan Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA COLLECTION                                              │
│  ├── Thu thập audio tiếng Việt (clean, mono, 24kHz+)            │
│  ├── Nguồn: YouTube, podcast, audiobook, recordings              │
│  └── Số lượng: 10-1000+ giờ tùy mục đích                        │
│                           │                                      │
│                           ▼                                      │
│  2. DATA PREPROCESSING                                           │
│  ├── Noise removal (Facebook demucs)                            │
│  ├── Volume normalization                                        │
│  ├── Split into clips (3-30 giây)                               │
│  └── Filter quality (remove bad samples)                         │
│                           │                                      │
│                           ▼                                      │
│  3. TRANSCRIPTION                                                │
│  ├── ASR: Whisper, PhoWhisper                                   │
│  ├── Manual verification                                         │
│  └── Text normalization (Vinorm)                                │
│                           │                                      │
│                           ▼                                      │
│  4. TRAINING                                                     │
│  ├── Select base model (F5-TTS, VITS, etc.)                     │
│  ├── Configure training params                                   │
│  ├── Train on GPU (RTX 3090+)                                   │
│  └── Monitor loss, save checkpoints                             │
│                           │                                      │
│                           ▼                                      │
│  5. EVALUATION                                                   │
│  ├── MOS score (human evaluation)                               │
│  ├── WER (word error rate)                                       │
│  └── Speaker similarity test                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Requirements

| Mục đích | Data cần | Chất lượng | Thời gian train |
|----------|----------|------------|-----------------|
| Zero-shot cloning | 3-30 giây | 16kHz+, clean | 0 |
| Few-shot (GPT-SoVITS) | 1-5 phút | 24kHz+, clean | 30-60 phút |
| LoRA fine-tune | 10-30 phút | 24kHz+, clean | 2-4 giờ |
| Full fine-tune | 10-100 giờ | 24kHz+, transcribed | 1-7 ngày |
| Train from scratch | 500-1000+ giờ | 24kHz+, transcribed | 2-4 tuần |

### 4.3 Chi Tiết Từng Bước

#### Bước 1: Thu Thập Data

```python
# data_collection.py
import yt_dlp
import os

def download_youtube_audio(url, output_dir):
    """Download audio từ YouTube."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Download từ YouTube playlist
urls = [
    "https://www.youtube.com/watch?v=...",  # Podcast tiếng Việt
    "https://www.youtube.com/watch?v=...",  # Audiobook
]

for url in urls:
    download_youtube_audio(url, "raw_audio/")
```

#### Bước 2: Preprocessing

```python
# preprocessing.py
import subprocess
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np

class AudioPreprocessor:
    def __init__(self, target_sr=24000):
        self.target_sr = target_sr

    def remove_noise(self, input_path, output_path):
        """Remove background noise using Facebook Demucs."""
        cmd = [
            "demucs",
            "--two-stems=vocals",
            "-o", "denoised/",
            input_path
        ]
        subprocess.run(cmd, capture_output=True)

        # Move vocals to output
        vocals_path = Path("denoised/htdemucs") / Path(input_path).stem / "vocals.wav"
        if vocals_path.exists():
            subprocess.run(["mv", str(vocals_path), output_path])

    def normalize_audio(self, input_path, output_path):
        """Normalize volume and convert to mono."""
        audio, sr = librosa.load(input_path, sr=self.target_sr, mono=True)

        # Peak normalization
        audio = audio / np.max(np.abs(audio)) * 0.95

        sf.write(output_path, audio, self.target_sr)

    def split_audio(self, input_path, output_dir, min_len=3, max_len=30):
        """Split audio vào clips dựa trên silence."""
        from pydub import AudioSegment
        from pydub.silence import split_on_silence

        audio = AudioSegment.from_file(input_path)

        chunks = split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=-40,
            keep_silence=200
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for i, chunk in enumerate(chunks):
            duration = len(chunk) / 1000  # seconds

            if min_len <= duration <= max_len:
                output_path = output_dir / f"clip_{i:05d}.wav"
                chunk.export(output_path, format="wav")
                saved.append(output_path)

        return saved

# Usage
preprocessor = AudioPreprocessor()

# Full pipeline
for audio_file in Path("raw_audio/").glob("*.wav"):
    # 1. Remove noise
    denoised = f"denoised/{audio_file.name}"
    preprocessor.remove_noise(str(audio_file), denoised)

    # 2. Normalize
    normalized = f"normalized/{audio_file.name}"
    preprocessor.normalize_audio(denoised, normalized)

    # 3. Split into clips
    clips = preprocessor.split_audio(normalized, f"clips/{audio_file.stem}/")
    print(f"Created {len(clips)} clips from {audio_file.name}")
```

#### Bước 3: Transcription

```python
# transcription.py
import whisper
from vinorm import TTSnorm
import json
from pathlib import Path

class VietnameseTranscriber:
    def __init__(self, model_size="large-v3"):
        self.model = whisper.load_model(model_size)
        self.normalizer = TTSnorm()

    def transcribe(self, audio_path):
        """Transcribe Vietnamese audio."""
        result = self.model.transcribe(
            audio_path,
            language="vi",
            task="transcribe"
        )
        return result["text"]

    def normalize_text(self, text):
        """Normalize Vietnamese text for TTS."""
        # Expand abbreviations, numbers, etc.
        normalized = self.normalizer(text)
        return normalized

    def process_directory(self, input_dir, output_file):
        """Process all clips and create metadata."""
        input_dir = Path(input_dir)
        metadata = []

        for audio_file in sorted(input_dir.glob("*.wav")):
            # Transcribe
            text = self.transcribe(str(audio_file))

            # Normalize
            normalized = self.normalize_text(text)

            metadata.append({
                "audio_file": str(audio_file),
                "text": text,
                "normalized_text": normalized
            })

            print(f"Processed: {audio_file.name}")

        # Save metadata
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

# Usage
transcriber = VietnameseTranscriber()
metadata = transcriber.process_directory("clips/", "metadata.json")
```

#### Bước 4: Training

##### Option A: Fine-tune VieNeu-TTS với LoRA

```python
# train_vieneu_lora.py
from vieneu_tts import VieNeuTTS, LoRAConfig, Trainer

# Load base model
model = VieNeuTTS.from_pretrained("pnnbao97/VieNeu-TTS-0.5B")

# LoRA config
lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

# Apply LoRA
model.add_lora(lora_config)

# Training config
trainer = Trainer(
    model=model,
    train_dataset="clips/",
    metadata_file="metadata.json",
    output_dir="lora_output/",
    learning_rate=5e-5,
    num_epochs=10,
    batch_size=4,
    gradient_accumulation_steps=4,
)

# Train
trainer.train()

# Save
model.save_pretrained("my_vietnamese_voice/")
```

##### Option B: Fine-tune F5-TTS

```python
# train_f5_vietnamese.py
from f5_tts import F5TTS, TrainingConfig

# Config
config = TrainingConfig(
    base_model="SWivid/F5-TTS",
    train_data_dir="clips/",
    metadata_file="metadata.json",
    output_dir="f5_vietnamese/",

    # Training params
    learning_rate=5e-6,
    batch_size=3200,  # frames
    num_steps=100000,
    save_every=10000,

    # Audio params
    sample_rate=24000,
    hop_length=256,

    # Vietnamese specific
    vocab_file="vietnamese_vocab.txt",
)

# Train
trainer = F5TTS.create_trainer(config)
trainer.train()
```

##### Option C: GPT-SoVITS (Easiest - WebUI)

```bash
# Clone GPT-SoVITS
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# Install
pip install -r requirements.txt

# Run WebUI
python webui.py
# Mở http://localhost:9874
```

**WebUI Steps:**
1. Upload audio files (1-5 phút clean audio)
2. Click "One-click format" để process
3. Start SoVITS training (15-30 phút)
4. Start GPT training (15-30 phút)
5. Test inference

#### Bước 5: Evaluation

```python
# evaluate.py
import torch
import numpy as np
from scipy.io import wavfile
from resemblyzer import VoiceEncoder, preprocess_wav

class ModelEvaluator:
    def __init__(self):
        self.voice_encoder = VoiceEncoder()

    def speaker_similarity(self, reference_audio, generated_audio):
        """Calculate speaker similarity score."""
        # Load and preprocess
        ref_wav = preprocess_wav(reference_audio)
        gen_wav = preprocess_wav(generated_audio)

        # Get embeddings
        ref_embed = self.voice_encoder.embed_utterance(ref_wav)
        gen_embed = self.voice_encoder.embed_utterance(gen_wav)

        # Cosine similarity
        similarity = np.dot(ref_embed, gen_embed) / (
            np.linalg.norm(ref_embed) * np.linalg.norm(gen_embed)
        )

        return similarity * 100  # Percentage

    def word_error_rate(self, reference_text, generated_text):
        """Calculate WER using Levenshtein distance."""
        import jiwer
        return jiwer.wer(reference_text, generated_text) * 100

# Usage
evaluator = ModelEvaluator()

# Test speaker similarity
similarity = evaluator.speaker_similarity(
    "reference.wav",
    "generated.wav"
)
print(f"Speaker Similarity: {similarity:.1f}%")
```

### 4.4 Training Tips cho Tiếng Việt

1. **Data Quality quan trọng nhất**
   - Clean audio, không có background noise
   - Single speaker per dataset
   - Consistent recording conditions

2. **Vietnamese-specific processing**
   ```python
   # Giữ nguyên dấu thanh
   text = "Xin chào các bạn"  # Correct
   text = "Xin chao cac ban"  # Wrong - mất dấu

   # Normalize abbreviations
   "TP.HCM" -> "Thành phố Hồ Chí Minh"
   "VN" -> "Việt Nam"
   ```

3. **Tokenizer cho tiếng Việt**
   - Extend vocab với Vietnamese characters
   - Include all diacritics: àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ

4. **Hyperparameters**
   - Learning rate: 5e-6 đến 1e-5
   - Batch size: Tùy VRAM, 4-16 samples
   - Gradient accumulation: 4-8 steps

---

## 5. Dataset Tiếng Việt

### 5.1 Public Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **viVoice** | 1000h | Multi-speaker Vietnamese | Hugging Face |
| **VLSP 2021/22/23** | 200h+ | Competition data | VLSP.org |
| **PhoAudiobook** | 941h | High-quality audiobook | VinAI |
| **VIVOS** | 15h | Read speech | OpenSLR |
| **viet-tts-dataset** | 35.9h | Google TTS generated | Hugging Face |
| **VinBigdata VLSP** | 100h+ | News, conversation | VinBigData |

### 5.2 Download Scripts

```python
# download_datasets.py
from datasets import load_dataset

# viVoice (1000h)
dataset = load_dataset("viVoice/viVoice")

# VIVOS (15h)
dataset = load_dataset("vivos", split="train")

# viet-tts-dataset (35.9h)
dataset = load_dataset("ntt123/viet-tts-dataset")

# Save locally
dataset.save_to_disk("vietnamese_tts_data/")
```

### 5.3 Tự Thu Thập Data

```python
# record_data.py
import sounddevice as sd
import soundfile as sf
import numpy as np

def record_sample(duration, sample_rate=24000, output_path="recording.wav"):
    """Record audio sample."""
    print(f"Recording for {duration} seconds...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    sf.write(output_path, audio, sample_rate)
    print(f"Saved to {output_path}")
    return output_path

# Recording session
texts = [
    "Xin chào, tôi là một trợ lý ảo.",
    "Hôm nay trời đẹp quá!",
    "Việt Nam là một đất nước xinh đẹp.",
    # ... thêm các câu khác
]

for i, text in enumerate(texts):
    print(f"\nĐọc câu {i+1}: {text}")
    input("Nhấn Enter khi sẵn sàng...")
    record_sample(5, output_path=f"recordings/sample_{i:03d}.wav")
```

---

## 6. Recommendations

### 6.1 Cho Web App Tiếng Việt

| Scenario | Recommended Model | Lý do |
|----------|-------------------|-------|
| **Production (Commercial)** | VieNeu-TTS 0.5B | Apache 2.0, CPU real-time, chất lượng cao |
| **Production (Budget)** | VieNeu-TTS 0.3B GGUF | Siêu nhẹ, chạy được server rẻ |
| **Highest Quality** | F5-TTS-Vietnamese | 1000h training, voice cloning tốt nhất |
| **Easy Integration** | VietTTS | OpenAI API compatible |
| **Multi-accent** | Dia-Vietnamese | North/South, male/female |

### 6.2 Decision Matrix

```
Bạn cần commercial license?
│
├── CÓ ──▶ VieNeu-TTS 0.5B (Apache 2.0)
│         hoặc VietTTS (Apache 2.0)
│         hoặc VietVoice (MIT)
│
└── KHÔNG ──▶ Bạn có GPU?
              │
              ├── CÓ ──▶ F5-TTS-Vietnamese (chất lượng cao nhất)
              │
              └── KHÔNG ──▶ VieNeu-TTS 0.3B GGUF (CPU real-time)
```

### 6.3 Recommended Architecture cho Web App

```
┌─────────────────────────────────────────────────────────────────┐
│                 WEB APP TIẾNG VIỆT                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FRONTEND (React/Vue)                                           │
│  ├── Upload MP3 (voice reference)                               │
│  ├── Upload DOC/PDF (content)                                   │
│  └── Audio player + download                                    │
│                                                                  │
│  BACKEND (FastAPI)                                              │
│  ├── File processing                                            │
│  ├── Text extraction (Vietnamese OCR)                           │
│  └── Queue management (Celery + Redis)                          │
│                                                                  │
│  TTS ENGINE (VieNeu-TTS)                                        │
│  ├── Zero-shot voice cloning                                    │
│  ├── Vietnamese text processing                                 │
│  └── Audio generation                                           │
│                                                                  │
│  DEPLOYMENT                                                      │
│  ├── CPU: VieNeu-TTS GGUF (cheap hosting)                       │
│  └── GPU: RunPod/Modal (better quality)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

1. [VieNeu-TTS - GitHub](https://github.com/pnnbao97/VieNeu-TTS)
2. [F5-TTS Vietnamese - Hugging Face](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice)
3. [VietTTS - GitHub](https://github.com/dangvansam/viet-tts)
4. [viXTTS Demo - GitHub](https://github.com/thinhlpg/vixtts-demo)
5. [VietVoice-TTS - GitHub](https://github.com/nguyenvulebinh/VietVoice-TTS)
6. [Vi-SparkTTS - Hugging Face](https://huggingface.co/DragonLineageAI/Vi-SparkTTS-0.5B)
7. [Dia Vietnamese - GitHub](https://github.com/TuananhCR/Dia-Finetuning-Vietnamese)
8. [GPT-SoVITS - GitHub](https://github.com/RVC-Boss/GPT-SoVITS)
9. [PhoAudiobook - VinAI](https://research.vinai.io/)
10. [viVoice Dataset - Hugging Face](https://huggingface.co/datasets)

---

*Document Version: 1.0*
*Last Updated: January 2026*
