# Hướng Dẫn Sử Dụng Google Colab cho Voice Cloning

## 🚀 Quick Start Guide

### Bước 1: Mở Google Colab
1. Truy cập: https://colab.research.google.com
2. Đăng nhập bằng Google account

### Bước 2: Upload Notebook
1. Click **File** → **Upload notebook**
2. Upload file `colab_voice_cloning_demo.ipynb` (trong thư mục gốc project)
3. Hoặc kéo thả file vào Colab

### Bước 3: Bật GPU
1. Click **Runtime** → **Change runtime type**
2. Chọn **Hardware accelerator** → **GPU**
3. Chọn **GPU type** → **T4** (free tier)
4. Click **Save**

### Bước 4: Chạy Demo
1. Chạy cells theo thứ tự (Shift + Enter)
2. Cell đầu tiên sẽ check GPU
3. Cell thứ 2 cài đặt XTTS-v2 (~2-3 phút)
4. Upload file MP3 giọng nói (tối thiểu 6 giây)
5. Nhập text tiếng Anh để tạo giọng
6. Download kết quả

---

## 📋 Workflow Hoàn Chỉnh

### 1. Chuẩn Bị Reference Audio
```
Requirements:
✅ Duration: 6-30 giây (10-15s là optimal)
✅ Quality: Clear, minimal background noise
✅ Format: MP3, WAV, M4A, hoặc bất kỳ audio format
✅ Single speaker
✅ Good pronunciation
```

### 2. Clone Voice
```python
# Trong Colab notebook:
# 1. Upload reference audio
# 2. Run voice cloning cell
# 3. Input your text
# 4. Download output
```

### 3. Batch Processing (Nhiều text)
```python
# List texts cần convert
texts = [
    "First sentence here.",
    "Second sentence here.",
    # ... more texts
]

# Tự động generate tất cả
# Download all outputs
```

### 4. PDF to Audiobook
```python
# Upload PDF
# Auto extract text
# Split into chunks
# Generate audio for each chunk
# Merge into final MP3
```

---

## ⚡ Performance & Limitations

### Google Colab Free Tier

**GPU Available:**
- Tesla T4 (16GB VRAM)
- Không phải lúc nào cũng có
- Runtime limit: 12 giờ liên tục

**Speed:**
- 5-10 giây per sentence
- Batch: ~100 sentences/giờ

**Storage:**
- Files bị xóa khi disconnect
- Cần download results trước khi đóng

**Usage Limits:**
- ~12 giờ GPU/ngày (không chính thức)
- Có thể bị disconnect khi idle lâu

### Google Colab Pro ($10/tháng)

**Improvements:**
✅ GPU tốt hơn (T4/V100/A100)
✅ Priority access to GPUs
✅ 24 giờ runtime
✅ More storage
✅ Background execution

**Worth it nếu:**
- Dùng thường xuyên (>20 giờ/tháng)
- Cần reliability cao
- Processing volume lớn

---

## 🎯 Use Cases & Examples

### Use Case 1: English Learning App
```
Goal: Tạo audio cho vocabulary flashcards
Input: 500 English sentences
Process: Batch generation
Output: 500 WAV files
Time: ~1 giờ on T4 GPU
```

### Use Case 2: Audiobook Creation
```
Goal: Convert PDF textbook to audiobook
Input: 200-page PDF
Process: Extract → Split → Generate → Merge
Output: Single MP3 audiobook
Time: 2-4 giờ depending on length
```

### Use Case 3: Podcast Voice Cloning
```
Goal: Clone podcast host voice
Input: 30-second clip from podcast
Process: Voice clone + generate new content
Output: New episodes with same voice
```

---

## 🔧 Troubleshooting

### "GPU not available"
```
Solution:
1. Runtime → Change runtime type → GPU
2. Restart runtime
3. Nếu vẫn không có: đợi vài giờ (free tier limit)
```

### "CUDA out of memory"
```
Solution:
1. Restart runtime
2. Reduce batch size
3. Process smaller chunks
4. Upgrade to Colab Pro
```

### "Disconnected from runtime"
```
Causes:
- Idle quá lâu (90 phút)
- Đạt 12 giờ limit
- Overuse GPU quota

Prevention:
- Save outputs frequently
- Use auto-save scripts
- Avoid leaving idle
```

### "Poor voice quality"
```
Solutions:
1. Use better reference audio (longer, clearer)
2. Ensure reference is 10-15 seconds
3. Remove background noise from reference
4. Try different text (simpler sentences first)
```

---

## 💡 Tips & Best Practices

### Reference Audio:
- **Length**: 10-15 giây là sweet spot
- **Quality**: Studio quality > Phone recording
- **Content**: Expressive speech > Monotone
- **Language**: Cùng ngôn ngữ với output (English)

### Text Input:
- **Sentence length**: 10-20 từ là tốt nhất
- **Punctuation**: Dùng dấu câu đúng (. , ! ?)
- **Numbers**: Viết text thay vì số (twenty-one vs 21)
- **Abbreviations**: Viết đầy đủ (United States vs US)

### Batch Processing:
- Split large texts into 500-character chunks
- Add 500ms pause between chunks
- Save intermediate results
- Use descriptive filenames

### Storage Management:
```python
# Auto-download all outputs
from google.colab import files
import glob

for file in glob.glob("outputs/*.wav"):
    files.download(file)
```

---

## 🚀 Next Steps

### 1. Test với Colab Free
- Upload notebook
- Test với vài samples
- Đánh giá chất lượng

### 2. Nếu Hài Lòng:
**Option A: Continue với Colab Pro**
- $10/tháng
- Better GPU access
- For regular use

**Option B: Deploy to Production**
- RunPod: $20-50/tháng
- Dedicated GPU
- Always available
- For heavy use (>100 giờ/tháng)

### 3. Integrate vào App:
```python
# Example API wrapper
def voice_clone(text, speaker_wav):
    # Run on Colab
    # Return audio URL
    pass
```

---

## 📊 Cost Comparison

| Solution | Setup | Monthly Cost | Best For |
|----------|-------|--------------|----------|
| **Colab Free** | 5 min | $0 | Testing, light use (<10 giờ/tháng) |
| **Colab Pro** | 5 min | $10 | Regular use (10-50 giờ/tháng) |
| **RunPod** | 30 min | $20-50 | Heavy use, production (>50 giờ/tháng) |
| **ElevenLabs API** | 2 min | $5-99 | No GPU, need convenience |

---

## ❓ FAQs

**Q: Có thể dùng tiếng Việt không?**
A: XTTS-v2 hỗ trợ tiếng Việt nhưng chất lượng kém hơn tiếng Anh. Cân nhắc dùng Fish Speech cho tiếng Việt.

**Q: Reference audio có cần phải tiếng Anh?**
A: Không nhất thiết, nhưng nếu output là tiếng Anh thì reference tiếng Anh sẽ tốt hơn.

**Q: Có thể lưu model đã clone?**
A: XTTS-v2 là zero-shot, không lưu model. Chỉ cần lưu reference audio.

**Q: Bao nhiêu text có thể generate trong 12 giờ?**
A: Khoảng 2000-5000 sentences tùy độ dài, đủ cho 1 audiobook nhỏ.

**Q: Có thể chạy nhiều sessions cùng lúc?**
A: Với free tier: Không. Với Pro: Có (limited).

---

**🎉 Bắt đầu ngay với file `colab_voice_cloning_demo.ipynb`!**
