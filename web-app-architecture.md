# Kiến Trúc Web App Voice Cloning Tiếng Việt

## Mục Lục
1. [Tổng Quan](#1-tổng-quan)
2. [Tech Stack](#2-tech-stack)
3. [System Architecture](#3-system-architecture)
4. [Backend Implementation](#4-backend-implementation)
5. [Frontend Implementation](#5-frontend-implementation)
6. [File Processing](#6-file-processing)
7. [Queue System](#7-queue-system)
8. [Database Schema](#8-database-schema)
9. [API Endpoints](#9-api-endpoints)
10. [Deployment](#10-deployment)
11. [Cost Estimation](#11-cost-estimation)

---

## 1. Tổng Quan

### 1.1 User Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       USER FLOW                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. ĐĂNG KÝ/ĐĂNG NHẬP                                           │
│     └── User tạo account                                        │
│                                                                  │
│  2. TẠO VOICE PROFILE                                           │
│     ├── Upload file MP3 (giọng nói của user)                   │
│     ├── System xử lý: denoise, normalize                        │
│     └── Voice profile được lưu                                  │
│                                                                  │
│  3. UPLOAD TÀI LIỆU                                             │
│     ├── Upload file DOC/PDF                                     │
│     ├── System extract text                                     │
│     └── Vietnamese text processing                              │
│                                                                  │
│  4. GENERATE AUDIO                                              │
│     ├── Chọn voice profile đã tạo                              │
│     ├── TTS engine generate audio                               │
│     └── User nghe/download kết quả                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Features

| Feature | Mô tả |
|---------|-------|
| **Voice Profile Creation** | User upload MP3, system tạo voice profile |
| **Document Upload** | Hỗ trợ DOC, DOCX, PDF |
| **Text-to-Speech** | Generate audio với giọng đã clone |
| **Real-time Progress** | WebSocket cập nhật tiến độ |
| **Audio Management** | Lưu trữ, download, delete audio |
| **Multi-language** | Hỗ trợ Vietnamese, English |

---

## 2. Tech Stack

### 2.1 Recommended Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        TECH STACK                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FRONTEND                                                        │
│  ├── React 18 + TypeScript                                      │
│  ├── Vite (build tool)                                          │
│  ├── TailwindCSS (styling)                                      │
│  ├── React Query (state management)                             │
│  ├── Socket.io-client (WebSocket)                               │
│  └── Uppy (file upload)                                         │
│                                                                  │
│  BACKEND                                                         │
│  ├── FastAPI (web framework)                                    │
│  ├── PostgreSQL (database)                                      │
│  ├── Redis (cache & message broker)                             │
│  ├── Celery (task queue)                                        │
│  ├── SQLAlchemy (ORM)                                           │
│  └── Pydantic (validation)                                      │
│                                                                  │
│  TTS ENGINE                                                      │
│  ├── VieNeu-TTS (recommended for Vietnamese)                    │
│  ├── OR: GPT-SoVITS                                             │
│  └── OR: F5-TTS-Vietnamese                                      │
│                                                                  │
│  INFRASTRUCTURE                                                  │
│  ├── Docker + Docker Compose                                    │
│  ├── RunPod / Modal (GPU compute)                               │
│  ├── AWS S3 / MinIO (file storage)                              │
│  └── Nginx (reverse proxy)                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Dependencies

#### Backend (requirements.txt)
```txt
# Web Framework
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
python-socketio==5.11.0

# Database
sqlalchemy==2.0.30
asyncpg==0.29.0
alembic==1.13.0

# Task Queue
celery==5.4.0
redis==5.0.0
flower==2.0.0

# File Processing
python-docx==1.1.0
pymupdf==1.24.0
python-pptx==0.6.23
pytesseract==0.3.10

# Audio Processing
torch==2.2.0
torchaudio==2.2.0
librosa==0.10.2
soundfile==0.12.1
pydub==0.25.1

# Vietnamese Text
vinorm==2.0.7
underthesea==6.8.0

# TTS
vieneu-tts>=0.1.0
# OR: f5-tts>=0.1.0
# OR: coqui-tts>=0.22.0

# Utilities
pydantic==2.7.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
boto3==1.34.0
httpx==0.27.0
```

#### Frontend (package.json)
```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.23.0",
    "@tanstack/react-query": "^5.40.0",
    "socket.io-client": "^4.7.0",
    "@uppy/core": "^3.11.0",
    "@uppy/react": "^3.3.0",
    "@uppy/xhr-upload": "^3.6.0",
    "axios": "^1.7.0",
    "zustand": "^4.5.0"
  },
  "devDependencies": {
    "typescript": "^5.4.0",
    "vite": "^5.2.0",
    "tailwindcss": "^3.4.0",
    "@types/react": "^18.3.0"
  }
}
```

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐  │
│    │   Browser    │────────▶│    Nginx     │────────▶│   Frontend   │  │
│    │   (User)     │         │   (Proxy)    │         │   (React)    │  │
│    └──────────────┘         └──────┬───────┘         └──────────────┘  │
│                                    │                                    │
│                                    │ /api/*                             │
│                                    ▼                                    │
│                            ┌──────────────┐                            │
│                            │   FastAPI    │                            │
│                            │   Backend    │                            │
│                            └──────┬───────┘                            │
│                                   │                                     │
│           ┌───────────────────────┼───────────────────────┐            │
│           │                       │                       │            │
│           ▼                       ▼                       ▼            │
│    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐     │
│    │  PostgreSQL  │       │    Redis     │       │   S3/MinIO   │     │
│    │  (Database)  │       │(Cache/Queue) │       │  (Storage)   │     │
│    └──────────────┘       └──────┬───────┘       └──────────────┘     │
│                                  │                                     │
│                                  │                                     │
│                                  ▼                                     │
│                          ┌──────────────┐                              │
│                          │   Celery     │                              │
│                          │   Workers    │                              │
│                          └──────┬───────┘                              │
│                                 │                                      │
│                                 ▼                                      │
│                          ┌──────────────┐                              │
│                          │  TTS Engine  │                              │
│                          │ (VieNeu-TTS) │                              │
│                          └──────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **Nginx** | Reverse proxy, SSL termination, static files | Nginx 1.25 |
| **Frontend** | User interface, file upload, audio playback | React 18 |
| **FastAPI** | REST API, WebSocket, authentication | FastAPI 0.115 |
| **PostgreSQL** | User data, voice profiles, job tracking | PostgreSQL 16 |
| **Redis** | Cache, session, Celery message broker | Redis 7 |
| **Celery** | Background task processing | Celery 5.4 |
| **S3/MinIO** | File storage (audio, documents) | MinIO/AWS S3 |
| **TTS Engine** | Voice cloning, speech synthesis | VieNeu-TTS |

---

## 4. Backend Implementation

### 4.1 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app entry point
│   ├── config.py                    # Configuration settings
│   │
│   ├── api/                         # API routes
│   │   ├── __init__.py
│   │   ├── auth.py                  # Authentication endpoints
│   │   ├── voice_profiles.py        # Voice profile management
│   │   ├── documents.py             # Document upload/processing
│   │   ├── generation.py            # TTS generation
│   │   └── jobs.py                  # Job status endpoints
│   │
│   ├── models/                      # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── voice_profile.py
│   │   ├── document.py
│   │   ├── generation.py
│   │   └── job.py
│   │
│   ├── schemas/                     # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── voice_profile.py
│   │   ├── document.py
│   │   └── generation.py
│   │
│   ├── services/                    # Business logic
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── voice_clone_service.py
│   │   ├── document_service.py
│   │   ├── tts_service.py
│   │   └── storage_service.py
│   │
│   ├── workers/                     # Celery tasks
│   │   ├── __init__.py
│   │   ├── celery_app.py
│   │   └── tasks.py
│   │
│   ├── websocket/                   # WebSocket handlers
│   │   ├── __init__.py
│   │   └── manager.py
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── vietnamese_text.py
│       ├── audio_processor.py
│       └── file_utils.py
│
├── alembic/                         # Database migrations
├── tests/                           # Unit tests
├── Dockerfile
├── requirements.txt
└── docker-compose.yml
```

### 4.2 Main Application

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import auth, voice_profiles, documents, generation, jobs
from app.config import settings
from app.database import engine, Base
from app.websocket.manager import WebSocketManager

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown
    await engine.dispose()

# Create app
app = FastAPI(
    title="Vietnamese Voice Cloning API",
    description="API for voice cloning and TTS in Vietnamese",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
ws_manager = WebSocketManager()

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(voice_profiles.router, prefix="/api/voice-profiles", tags=["Voice Profiles"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(generation.router, prefix="/api/generation", tags=["Generation"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### 4.3 Configuration

```python
# app/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Vietnamese Voice Cloning"
    DEBUG: bool = False
    SECRET_KEY: str

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/voice_clone"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Storage
    STORAGE_TYPE: str = "local"  # "local" or "s3"
    S3_BUCKET: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_ENDPOINT: str = ""

    # File paths
    UPLOAD_DIR: str = "/data/uploads"
    VOICE_PROFILES_DIR: str = "/data/voice_profiles"
    GENERATED_AUDIO_DIR: str = "/data/generated"

    # TTS
    TTS_MODEL: str = "vieneu-tts"  # "vieneu-tts", "f5-tts", "gpt-sovits"
    TTS_DEVICE: str = "cuda"  # "cuda" or "cpu"

    # Limits
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    MAX_AUDIO_DURATION: int = 300  # 5 minutes
    MAX_DOCUMENT_PAGES: int = 100

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"

settings = Settings()
```

### 4.4 Voice Profile Service

```python
# app/services/voice_clone_service.py
import torch
import torchaudio
from pathlib import Path
import uuid
from typing import Optional

from app.config import settings
from app.utils.audio_processor import AudioProcessor

class VoiceCloneService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load TTS model based on configuration."""
        if settings.TTS_MODEL == "vieneu-tts":
            from vieneu_tts import VieNeuTTS
            self._model = VieNeuTTS.from_pretrained(
                "pnnbao97/VieNeu-TTS-0.5B",
                device=settings.TTS_DEVICE
            )
        elif settings.TTS_MODEL == "f5-tts":
            from f5_tts import F5TTS
            self._model = F5TTS.from_pretrained(
                "hynt/F5-TTS-Vietnamese-ViVoice",
                device=settings.TTS_DEVICE
            )
        elif settings.TTS_MODEL == "gpt-sovits":
            # GPT-SoVITS implementation
            pass

        print(f"Loaded {settings.TTS_MODEL} model on {settings.TTS_DEVICE}")

    async def create_voice_profile(
        self,
        audio_path: str,
        user_id: str,
        profile_name: str
    ) -> dict:
        """
        Create voice profile from uploaded audio.

        Args:
            audio_path: Path to uploaded audio file
            user_id: User ID
            profile_name: Name for the voice profile

        Returns:
            Voice profile metadata
        """
        profile_id = str(uuid.uuid4())
        processor = AudioProcessor()

        # Preprocess audio
        processed_path = await processor.preprocess_for_cloning(
            audio_path,
            output_dir=settings.VOICE_PROFILES_DIR,
            profile_id=profile_id
        )

        # Validate audio quality
        quality_score = await processor.assess_quality(processed_path)

        return {
            "profile_id": profile_id,
            "user_id": user_id,
            "name": profile_name,
            "audio_path": processed_path,
            "duration": await processor.get_duration(processed_path),
            "quality_score": quality_score,
            "status": "ready"
        }

    async def generate_speech(
        self,
        text: str,
        voice_profile_path: str,
        output_path: Optional[str] = None,
        language: str = "vi"
    ) -> str:
        """
        Generate speech using cloned voice.

        Args:
            text: Text to synthesize
            voice_profile_path: Path to voice profile audio
            output_path: Optional output path
            language: Language code ("vi" or "en")

        Returns:
            Path to generated audio
        """
        if output_path is None:
            output_path = Path(settings.GENERATED_AUDIO_DIR) / f"{uuid.uuid4()}.wav"

        # Load reference audio
        reference, sr = torchaudio.load(voice_profile_path)

        # Generate based on model type
        if settings.TTS_MODEL == "vieneu-tts":
            output = self._model.generate(
                text=text,
                reference_audio=reference,
                language=language
            )
        elif settings.TTS_MODEL == "f5-tts":
            output = self._model.synthesize(
                text=text,
                reference_audio=voice_profile_path,
                language=language
            )

        # Save output
        torchaudio.save(str(output_path), output, sr)

        return str(output_path)

    async def generate_long_text(
        self,
        text: str,
        voice_profile_path: str,
        output_dir: str,
        max_chunk_length: int = 500
    ) -> list:
        """
        Generate speech for long text by splitting into chunks.

        Args:
            text: Long text to synthesize
            voice_profile_path: Path to voice profile
            output_dir: Directory for output files
            max_chunk_length: Maximum characters per chunk

        Returns:
            List of generated audio paths
        """
        from app.utils.vietnamese_text import VietnameseTextProcessor

        # Split text into chunks
        processor = VietnameseTextProcessor()
        chunks = processor.split_into_chunks(text, max_chunk_length)

        outputs = []
        for i, chunk in enumerate(chunks):
            output_path = Path(output_dir) / f"chunk_{i:04d}.wav"

            await self.generate_speech(
                text=chunk,
                voice_profile_path=voice_profile_path,
                output_path=str(output_path)
            )

            outputs.append(str(output_path))

        return outputs
```

### 4.5 Document Service

```python
# app/services/document_service.py
import PyPDF2
from docx import Document
from pathlib import Path
import re
from typing import Optional

from app.utils.vietnamese_text import VietnameseTextProcessor

class DocumentService:
    def __init__(self):
        self.text_processor = VietnameseTextProcessor()

    async def extract_text(self, file_path: str) -> str:
        """
        Extract text from document (PDF, DOC, DOCX).

        Args:
            file_path: Path to document file

        Returns:
            Extracted and cleaned text
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            text = await self._extract_from_pdf(file_path)
        elif extension in [".doc", ".docx"]:
            text = await self._extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        # Clean and process Vietnamese text
        text = self.text_processor.clean(text)
        text = self.text_processor.normalize(text)

        return text

    async def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""

        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)

                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"
                    else:
                        # Fallback to OCR for scanned pages
                        text += await self._ocr_page(file_path, page_num)

        except Exception as e:
            raise ValueError(f"Failed to extract PDF: {str(e)}")

        return text

    async def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)

    async def _ocr_page(self, pdf_path: Path, page_num: int) -> str:
        """OCR a scanned PDF page using Tesseract."""
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import io

        # Convert PDF page to image
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR

        img = Image.open(io.BytesIO(pix.tobytes()))

        # OCR with Vietnamese language
        text = pytesseract.image_to_string(img, lang="vie")

        doc.close()
        return text

    def get_word_count(self, text: str) -> int:
        """Get word count for Vietnamese text."""
        words = text.split()
        return len(words)

    def estimate_audio_duration(self, text: str, words_per_minute: int = 150) -> float:
        """
        Estimate audio duration in seconds.

        Args:
            text: Text content
            words_per_minute: Speaking rate

        Returns:
            Estimated duration in seconds
        """
        word_count = self.get_word_count(text)
        return (word_count / words_per_minute) * 60
```

### 4.6 Vietnamese Text Processor

```python
# app/utils/vietnamese_text.py
import re
from vinorm import TTSnorm
from typing import List

class VietnameseTextProcessor:
    def __init__(self):
        self.normalizer = TTSnorm()

    def clean(self, text: str) -> str:
        """Clean Vietnamese text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep Vietnamese diacritics
        vietnamese_chars = (
            r'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ'
            r'ùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆ'
            r'ÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ'
        )
        pattern = f'[^\\w\\s{vietnamese_chars}.,!?;:\'"()-]'
        text = re.sub(pattern, '', text)

        return text.strip()

    def normalize(self, text: str) -> str:
        """
        Normalize Vietnamese text for TTS.
        - Expand abbreviations
        - Convert numbers to words
        - Handle special cases
        """
        return self.normalizer(text)

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Vietnamese sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_into_chunks(self, text: str, max_length: int = 500) -> List[str]:
        """
        Split text into chunks for TTS processing.

        Args:
            text: Full text
            max_length: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def detect_language(self, text: str) -> str:
        """
        Detect if text is Vietnamese or English.

        Returns:
            "vi" for Vietnamese, "en" for English
        """
        vietnamese_chars = set(
            'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ'
            'ùúủũụưừứửữựỳýỷỹỵđ'
        )

        text_lower = text.lower()
        vietnamese_count = sum(1 for c in text_lower if c in vietnamese_chars)

        # If more than 1% of chars are Vietnamese-specific
        if vietnamese_count / max(len(text), 1) > 0.01:
            return "vi"
        return "en"
```

---

## 5. Frontend Implementation

### 5.1 Project Structure

```
frontend/
├── src/
│   ├── main.tsx                     # Entry point
│   ├── App.tsx                      # Main app component
│   ├── index.css                    # Global styles
│   │
│   ├── components/                  # Reusable components
│   │   ├── Layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Footer.tsx
│   │   ├── VoiceProfile/
│   │   │   ├── VoiceUploader.tsx
│   │   │   ├── VoiceProfileCard.tsx
│   │   │   └── VoiceProfileList.tsx
│   │   ├── Document/
│   │   │   ├── DocumentUploader.tsx
│   │   │   ├── DocumentPreview.tsx
│   │   │   └── DocumentList.tsx
│   │   ├── Generation/
│   │   │   ├── GenerationForm.tsx
│   │   │   ├── ProgressIndicator.tsx
│   │   │   └── AudioPlayer.tsx
│   │   └── Common/
│   │       ├── Button.tsx
│   │       ├── Card.tsx
│   │       ├── Modal.tsx
│   │       └── Toast.tsx
│   │
│   ├── pages/                       # Page components
│   │   ├── Home.tsx
│   │   ├── Dashboard.tsx
│   │   ├── VoiceProfiles.tsx
│   │   ├── Documents.tsx
│   │   ├── Generate.tsx
│   │   └── History.tsx
│   │
│   ├── hooks/                       # Custom hooks
│   │   ├── useAuth.ts
│   │   ├── useVoiceProfiles.ts
│   │   ├── useDocuments.ts
│   │   ├── useGeneration.ts
│   │   └── useWebSocket.ts
│   │
│   ├── services/                    # API services
│   │   ├── api.ts
│   │   ├── authService.ts
│   │   ├── voiceService.ts
│   │   └── generationService.ts
│   │
│   ├── store/                       # State management
│   │   ├── authStore.ts
│   │   └── appStore.ts
│   │
│   └── types/                       # TypeScript types
│       └── index.ts
│
├── public/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

### 5.2 Voice Uploader Component

```tsx
// src/components/VoiceProfile/VoiceUploader.tsx
import React, { useState, useCallback } from 'react';
import Uppy from '@uppy/core';
import { Dashboard } from '@uppy/react';
import XHRUpload from '@uppy/xhr-upload';
import { useWebSocket } from '../../hooks/useWebSocket';

interface VoiceUploaderProps {
  onUploadComplete: (profileId: string) => void;
}

export function VoiceUploader({ onUploadComplete }: VoiceUploaderProps) {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');

  // WebSocket for progress updates
  const { lastMessage } = useWebSocket(
    taskId ? `ws://localhost:8000/ws/progress/${taskId}` : null
  );

  // Handle WebSocket messages
  React.useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      setProgress(data.progress);
      setStatus(data.message);

      if (data.status === 'SUCCESS') {
        onUploadComplete(data.profile_id);
      }
    }
  }, [lastMessage, onUploadComplete]);

  // Configure Uppy
  const uppy = React.useMemo(() => {
    return new Uppy({
      restrictions: {
        maxFileSize: 100 * 1024 * 1024, // 100MB
        maxNumberOfFiles: 1,
        allowedFileTypes: ['audio/*'],
      },
      autoProceed: false,
    })
      .use(XHRUpload, {
        endpoint: '/api/voice-profiles/upload',
        fieldName: 'file',
      })
      .on('upload-success', (file, response) => {
        const { task_id } = response.body;
        setTaskId(task_id);
      });
  }, []);

  return (
    <div className="voice-uploader">
      <h2 className="text-xl font-bold mb-4">Tạo Voice Profile</h2>

      <div className="instructions mb-4 p-4 bg-blue-50 rounded">
        <h3 className="font-semibold">Hướng dẫn:</h3>
        <ul className="list-disc ml-4 mt-2">
          <li>Upload file MP3 hoặc WAV chứa giọng nói của bạn</li>
          <li>Độ dài tối thiểu: 5 giây, tối đa: 5 phút</li>
          <li>Chất lượng tốt nhất: giọng rõ ràng, không có tạp âm</li>
        </ul>
      </div>

      <Dashboard
        uppy={uppy}
        proudlyDisplayPoweredByUppy={false}
        showProgressDetails
        locale={{
          strings: {
            dropPasteFiles: 'Kéo thả file hoặc %{browse}',
            browse: 'chọn file',
          },
        }}
      />

      {taskId && (
        <div className="progress-section mt-4">
          <div className="progress-bar bg-gray-200 rounded-full h-4">
            <div
              className="bg-blue-500 rounded-full h-4 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-2 text-sm text-gray-600">{status}</p>
        </div>
      )}
    </div>
  );
}
```

### 5.3 Generation Form Component

```tsx
// src/components/Generation/GenerationForm.tsx
import React, { useState } from 'react';
import { useVoiceProfiles } from '../../hooks/useVoiceProfiles';
import { useGeneration } from '../../hooks/useGeneration';
import { AudioPlayer } from './AudioPlayer';
import { ProgressIndicator } from './ProgressIndicator';

export function GenerationForm() {
  const { profiles, isLoading: loadingProfiles } = useVoiceProfiles();
  const { generate, progress, status, audioUrl, isGenerating } = useGeneration();

  const [selectedProfile, setSelectedProfile] = useState('');
  const [text, setText] = useState('');
  const [documentId, setDocumentId] = useState('');

  const handleGenerate = async () => {
    await generate({
      voiceProfileId: selectedProfile,
      text: text || undefined,
      documentId: documentId || undefined,
    });
  };

  return (
    <div className="generation-form max-w-2xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Tạo Audio</h2>

      {/* Voice Profile Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">
          Chọn Voice Profile
        </label>
        <select
          value={selectedProfile}
          onChange={(e) => setSelectedProfile(e.target.value)}
          className="w-full p-3 border rounded-lg"
          disabled={loadingProfiles}
        >
          <option value="">-- Chọn giọng nói --</option>
          {profiles?.map((profile) => (
            <option key={profile.id} value={profile.id}>
              {profile.name}
            </option>
          ))}
        </select>
      </div>

      {/* Text Input */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">
          Nhập văn bản (hoặc chọn tài liệu đã upload)
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Nhập văn bản tiếng Việt cần đọc..."
          className="w-full p-3 border rounded-lg h-40"
          disabled={!!documentId}
        />
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={!selectedProfile || (!text && !documentId) || isGenerating}
        className="w-full py-3 bg-blue-500 text-white rounded-lg font-semibold
                   hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
      >
        {isGenerating ? 'Đang xử lý...' : 'Tạo Audio'}
      </button>

      {/* Progress */}
      {isGenerating && (
        <ProgressIndicator progress={progress} status={status} />
      )}

      {/* Audio Player */}
      {audioUrl && (
        <div className="mt-6">
          <h3 className="font-semibold mb-2">Kết quả:</h3>
          <AudioPlayer src={audioUrl} />
        </div>
      )}
    </div>
  );
}
```

### 5.4 WebSocket Hook

```tsx
// src/hooks/useWebSocket.ts
import { useEffect, useState, useCallback, useRef } from 'react';

interface WebSocketMessage {
  data: string;
}

export function useWebSocket(url: string | null) {
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [readyState, setReadyState] = useState<number>(WebSocket.CLOSED);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!url) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setReadyState(WebSocket.OPEN);
    };

    ws.onmessage = (event) => {
      setLastMessage(event);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      setReadyState(WebSocket.CLOSED);
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const sendMessage = useCallback((message: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(message);
    }
  }, []);

  return {
    lastMessage,
    readyState,
    sendMessage,
    isConnected: readyState === WebSocket.OPEN,
  };
}
```

---

## 6. File Processing

### 6.1 Audio Processor

```python
# app/utils/audio_processor.py
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import subprocess
from typing import Optional

class AudioProcessor:
    def __init__(self, target_sr: int = 24000):
        self.target_sr = target_sr

    async def preprocess_for_cloning(
        self,
        input_path: str,
        output_dir: str,
        profile_id: str
    ) -> str:
        """
        Preprocess audio for voice cloning.

        Steps:
        1. Load and convert to mono
        2. Resample to target sample rate
        3. Remove background noise
        4. Normalize volume
        5. Trim silence
        """
        output_path = Path(output_dir) / f"{profile_id}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio
        audio, sr = librosa.load(input_path, sr=self.target_sr, mono=True)

        # Remove noise using noisereduce
        import noisereduce as nr
        audio = nr.reduce_noise(y=audio, sr=sr)

        # Normalize
        audio = librosa.util.normalize(audio)

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Save
        sf.write(str(output_path), audio, sr)

        return str(output_path)

    async def get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        audio, sr = librosa.load(audio_path, sr=None)
        return len(audio) / sr

    async def assess_quality(self, audio_path: str) -> float:
        """
        Assess audio quality for voice cloning.

        Returns:
            Quality score from 0 to 100
        """
        audio, sr = librosa.load(audio_path, sr=self.target_sr)

        # Check SNR (Signal-to-Noise Ratio)
        rms = np.sqrt(np.mean(audio ** 2))

        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)

        # Check duration (optimal: 5-30 seconds)
        duration = len(audio) / sr
        duration_score = min(100, max(0, 100 - abs(duration - 15) * 2))

        # Calculate overall score
        snr_score = min(100, rms * 1000)
        clipping_score = 100 - (clipping_ratio * 1000)

        overall = (snr_score * 0.3 + clipping_score * 0.3 + duration_score * 0.4)

        return round(overall, 2)

    async def concatenate_audio(
        self,
        audio_paths: list,
        output_path: str,
        silence_duration: float = 0.5
    ) -> str:
        """Concatenate multiple audio files with silence between."""
        all_audio = []

        for path in audio_paths:
            audio, sr = librosa.load(path, sr=self.target_sr)
            all_audio.append(audio)

            # Add silence
            silence = np.zeros(int(silence_duration * self.target_sr))
            all_audio.append(silence)

        # Remove last silence
        all_audio = all_audio[:-1]

        # Concatenate
        combined = np.concatenate(all_audio)

        # Save
        sf.write(output_path, combined, self.target_sr)

        return output_path
```

---

## 7. Queue System

### 7.1 Celery Configuration

```python
# app/workers/celery_app.py
from celery import Celery
from app.config import settings

celery_app = Celery(
    "voice_clone",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"]
)

celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone
    timezone='Asia/Ho_Chi_Minh',
    enable_utc=True,

    # Task tracking
    task_track_started=True,
    result_extended=True,

    # Timeouts
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,

    # Retry
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Queues
    task_routes={
        'app.workers.tasks.process_voice_profile': {'queue': 'voice_processing'},
        'app.workers.tasks.generate_speech': {'queue': 'tts_generation'},
        'app.workers.tasks.process_document': {'queue': 'document_processing'},
    },

    # Concurrency
    worker_concurrency=2,  # Limit for GPU tasks
    worker_prefetch_multiplier=1,
)
```

### 7.2 Celery Tasks

```python
# app/workers/tasks.py
from celery import shared_task
from app.workers.celery_app import celery_app
from app.services.voice_clone_service import VoiceCloneService
from app.services.document_service import DocumentService
from app.utils.audio_processor import AudioProcessor
import asyncio

@celery_app.task(bind=True, max_retries=3)
def process_voice_profile(self, temp_path: str, user_id: str, profile_name: str):
    """Process uploaded voice profile."""
    try:
        self.update_state(
            state='PROCESSING',
            meta={'progress': 10, 'message': 'Đang kiểm tra file audio...'}
        )

        # Initialize services
        voice_service = VoiceCloneService()
        loop = asyncio.new_event_loop()

        # Create voice profile
        self.update_state(
            state='PROCESSING',
            meta={'progress': 30, 'message': 'Đang xử lý audio...'}
        )

        result = loop.run_until_complete(
            voice_service.create_voice_profile(temp_path, user_id, profile_name)
        )

        self.update_state(
            state='PROCESSING',
            meta={'progress': 90, 'message': 'Đang hoàn tất...'}
        )

        return {
            'status': 'SUCCESS',
            'progress': 100,
            'message': 'Voice profile đã sẵn sàng!',
            'profile_id': result['profile_id'],
            'quality_score': result['quality_score']
        }

    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'message': f'Lỗi: {str(e)}'}
        )
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def generate_speech(
    self,
    text: str,
    voice_profile_id: str,
    language: str = 'vi'
):
    """Generate speech from text."""
    try:
        self.update_state(
            state='PROCESSING',
            meta={'progress': 10, 'message': 'Đang chuẩn bị...'}
        )

        voice_service = VoiceCloneService()
        loop = asyncio.new_event_loop()

        # Get voice profile path
        voice_profile_path = f"/data/voice_profiles/{voice_profile_id}.wav"

        self.update_state(
            state='PROCESSING',
            meta={'progress': 30, 'message': 'Đang tạo audio...'}
        )

        # Generate speech
        if len(text) > 500:
            # Long text - split into chunks
            outputs = loop.run_until_complete(
                voice_service.generate_long_text(
                    text=text,
                    voice_profile_path=voice_profile_path,
                    output_dir=f"/data/generated/{voice_profile_id}/"
                )
            )

            # Concatenate
            processor = AudioProcessor()
            output_path = loop.run_until_complete(
                processor.concatenate_audio(
                    outputs,
                    f"/data/generated/{voice_profile_id}_final.wav"
                )
            )
        else:
            output_path = loop.run_until_complete(
                voice_service.generate_speech(
                    text=text,
                    voice_profile_path=voice_profile_path,
                    language=language
                )
            )

        self.update_state(
            state='PROCESSING',
            meta={'progress': 90, 'message': 'Đang hoàn tất...'}
        )

        return {
            'status': 'SUCCESS',
            'progress': 100,
            'message': 'Audio đã được tạo thành công!',
            'audio_path': output_path
        }

    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True)
def process_document(self, file_path: str, user_id: str):
    """Process uploaded document."""
    try:
        self.update_state(
            state='PROCESSING',
            meta={'progress': 10, 'message': 'Đang đọc tài liệu...'}
        )

        doc_service = DocumentService()
        loop = asyncio.new_event_loop()

        # Extract text
        text = loop.run_until_complete(
            doc_service.extract_text(file_path)
        )

        self.update_state(
            state='PROCESSING',
            meta={'progress': 70, 'message': 'Đang phân tích văn bản...'}
        )

        word_count = doc_service.get_word_count(text)
        estimated_duration = doc_service.estimate_audio_duration(text)

        return {
            'status': 'SUCCESS',
            'progress': 100,
            'message': 'Tài liệu đã được xử lý!',
            'text': text,
            'word_count': word_count,
            'estimated_duration': estimated_duration
        }

    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise
```

---

## 8. Database Schema

### 8.1 SQLAlchemy Models

```python
# app/models/user.py
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    voice_profiles = relationship("VoiceProfile", back_populates="user")
    documents = relationship("Document", back_populates="user")
    generations = relationship("Generation", back_populates="user")


# app/models/voice_profile.py
class VoiceProfile(Base):
    __tablename__ = "voice_profiles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    audio_path = Column(String, nullable=False)
    duration = Column(Float)  # seconds
    quality_score = Column(Float)
    status = Column(String, default="processing")  # processing, ready, failed
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="voice_profiles")
    generations = relationship("Generation", back_populates="voice_profile")


# app/models/document.py
class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)  # pdf, docx
    text_content = Column(Text)
    word_count = Column(Integer)
    status = Column(String, default="processing")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="documents")


# app/models/generation.py
class Generation(Base):
    __tablename__ = "generations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    voice_profile_id = Column(String, ForeignKey("voice_profiles.id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id"))
    text_input = Column(Text)
    audio_path = Column(String)
    duration = Column(Float)  # seconds
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="generations")
    voice_profile = relationship("VoiceProfile", back_populates="generations")
```

---

## 9. API Endpoints

### 9.1 API Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Auth** | | |
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login user |
| GET | `/api/auth/me` | Get current user |
| **Voice Profiles** | | |
| POST | `/api/voice-profiles/upload` | Upload voice file |
| GET | `/api/voice-profiles` | List user's profiles |
| GET | `/api/voice-profiles/{id}` | Get profile details |
| DELETE | `/api/voice-profiles/{id}` | Delete profile |
| **Documents** | | |
| POST | `/api/documents/upload` | Upload document |
| GET | `/api/documents` | List user's documents |
| GET | `/api/documents/{id}` | Get document details |
| DELETE | `/api/documents/{id}` | Delete document |
| **Generation** | | |
| POST | `/api/generation/create` | Create new generation |
| GET | `/api/generation/{id}` | Get generation status |
| GET | `/api/generation/{id}/audio` | Download audio file |
| GET | `/api/generation/history` | Get generation history |
| **Jobs** | | |
| GET | `/api/jobs/{task_id}` | Get job status |
| WS | `/ws/progress/{task_id}` | WebSocket progress |

### 9.2 API Implementation

```python
# app/api/voice_profiles.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.voice_profile import VoiceProfile
from app.services.auth_service import get_current_user
from app.workers.tasks import process_voice_profile
from app.schemas.voice_profile import VoiceProfileResponse, VoiceProfileCreate

router = APIRouter()

@router.post("/upload", response_model=dict)
async def upload_voice_profile(
    file: UploadFile = File(...),
    name: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload voice file for cloning."""

    # Validate file
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Invalid file format")

    if file.size > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(status_code=413, detail="File too large")

    # Save temporary file
    import tempfile
    temp_path = tempfile.mktemp(suffix=".mp3")
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Dispatch task
    task = process_voice_profile.delay(
        temp_path=temp_path,
        user_id=current_user.id,
        profile_name=name
    )

    return {
        "task_id": task.id,
        "status": "processing",
        "message": "Voice profile is being processed"
    }


@router.get("/", response_model=list[VoiceProfileResponse])
async def list_voice_profiles(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all voice profiles for current user."""
    result = await db.execute(
        select(VoiceProfile)
        .where(VoiceProfile.user_id == current_user.id)
        .order_by(VoiceProfile.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{profile_id}", response_model=VoiceProfileResponse)
async def get_voice_profile(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get voice profile details."""
    result = await db.execute(
        select(VoiceProfile)
        .where(
            VoiceProfile.id == profile_id,
            VoiceProfile.user_id == current_user.id
        )
    )
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return profile


@router.delete("/{profile_id}")
async def delete_voice_profile(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete voice profile."""
    result = await db.execute(
        select(VoiceProfile)
        .where(
            VoiceProfile.id == profile_id,
            VoiceProfile.user_id == current_user.id
        )
    )
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Delete file
    import os
    if os.path.exists(profile.audio_path):
        os.remove(profile.audio_path)

    # Delete from DB
    await db.delete(profile)
    await db.commit()

    return {"message": "Profile deleted successfully"}
```

---

## 10. Deployment

### 10.1 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - VITE_API_URL=http://localhost:8000

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/voice_clone
      - REDIS_URL=redis://redis:6379
      - TTS_MODEL=vieneu-tts
      - TTS_DEVICE=cpu
    volumes:
      - ./data:/data

  # Celery Worker
  celery-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    command: celery -A app.workers.celery_app worker -l info -Q voice_processing,tts_generation,document_processing
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/voice_clone
      - REDIS_URL=redis://redis:6379
      - TTS_MODEL=vieneu-tts
      - TTS_DEVICE=cpu
    volumes:
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Celery Beat (Scheduler)
  celery-beat:
    build:
      context: ./backend
      dockerfile: Dockerfile
    command: celery -A app.workers.celery_app beat -l info
    depends_on:
      - redis

  # Flower (Celery monitoring)
  flower:
    image: mher/flower
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379
      - FLOWER_PORT=5555
    depends_on:
      - redis

  # PostgreSQL
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=voice_clone
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # MinIO (S3 compatible storage)
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data

  # Nginx (Reverse proxy)
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
  minio_data:
```

### 10.2 Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:3000;
    }

    upstream backend {
        server backend:8000;
    }

    server {
        listen 80;
        server_name localhost;

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
        }

        # API
        location /api {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;

            # Increase timeout for long operations
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        # WebSocket
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }

        # File upload size
        client_max_body_size 100M;
    }
}
```

### 10.3 Production Deployment (RunPod)

```bash
# Deploy to RunPod
# 1. Build and push Docker image
docker build -t your-registry/voice-clone:latest .
docker push your-registry/voice-clone:latest

# 2. Create RunPod pod with GPU
# Use RunPod UI or API to create pod with:
# - GPU: RTX 4090 (or A10G for cost optimization)
# - Docker image: your-registry/voice-clone:latest
# - Ports: 8000, 3000

# 3. Configure environment variables in RunPod
```

---

## 11. Cost Estimation

### 11.1 Infrastructure Costs

| Component | Option | Monthly Cost |
|-----------|--------|--------------|
| **GPU Compute** | | |
| RunPod Serverless | RTX 4090, ~100h/month | $44 |
| RunPod On-demand | RTX 4090, 24/7 | $316 |
| Modal | A10G, pay-per-use | ~$50-100 |
| **Database** | | |
| PostgreSQL (self-hosted) | Docker | $0 |
| AWS RDS | db.t3.micro | $15 |
| **Storage** | | |
| MinIO (self-hosted) | 500GB | $0 |
| AWS S3 | 500GB | ~$12 |
| **Redis** | | |
| Self-hosted | Docker | $0 |
| AWS ElastiCache | cache.t3.micro | $13 |

### 11.2 Estimated Monthly Costs

| Scale | Users | GPU Hours | Total Cost |
|-------|-------|-----------|------------|
| Development | 1-10 | 10h | ~$10-20 |
| Small | 100 | 50h | ~$50-80 |
| Medium | 1000 | 200h | ~$150-250 |
| Large | 10000 | 1000h | ~$500-1000 |

### 11.3 Cost Optimization Tips

1. **Use CPU for VieNeu-TTS GGUF**: Giảm 80% chi phí GPU
2. **Cache generated audio**: Tránh generate lại cùng content
3. **Serverless GPU**: Chỉ trả tiền khi có request
4. **Compress audio**: MP3 thay vì WAV
5. **Auto-delete old files**: Xóa audio sau 30 ngày

---

*Document Version: 1.0*
*Last Updated: January 2026*
