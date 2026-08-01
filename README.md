# MedScript AI 🏥

> **Privacy-first AI system for transcribing doctor handwriting into structured digital text.**

MedScript AI combines Computer Vision (Donut + BiLSTM-CTC) and Medical NLP (BiomedBERT) to extract medicine names, dosages, frequencies, and durations from handwritten prescription images.

⚠️ **Clinical Disclaimer:** This system is NOT for diagnostic use. It is a research tool and should not be used for actual medical decision-making.

## Architecture

```
[Prescription Image]
    ↓
[OpenCV Preprocessing] → Deskew, Normalize, CLAHE
    ↓
[Donut Vision Encoder] → Swin Transformer feature extraction
    ↓
[BiLSTM + CTC Decoder] → Temporal decoding for cursive text
    ↓
[Medical BERT NER] → Entity extraction (medicine, dosage, frequency)
    ↓
[Structured JSON Output]
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for web UI)
- Docker & Docker Compose (optional, for full stack)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install Python packages
pip install -e ".[dev]"
```

### 2. Generate Synthetic Training Data

Since no real dataset is available yet, generate synthetic prescriptions:

```bash
python scripts/generate_synthetic.py --count 5000 --output data/synthetic
```

### 3. Start the API (Development)

```bash
# Copy environment file
cp .env.example .env

# Start the API server
make serve
# or: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Web UI

```bash
cd web
npm install
npm run dev
```

Visit `http://localhost:3000` for the web UI, `http://localhost:8000/docs` for the API docs.

### 5. Full Stack with Docker

```bash
docker-compose up -d
```

## Project Structure

```
├── src/medscript/          # Core AI/ML code
│   ├── data/               # Data pipeline (download, synthetic, augmentation)
│   ├── models/             # Model architectures (Donut, BiLSTM-CTC, BERT NER)
│   ├── training/           # PyTorch Lightning training
│   ├── inference/          # Production inference
│   └── utils/              # Config, logging, medical vocabulary
├── api/                    # FastAPI backend
│   ├── core/               # Security (OAuth 2.0, JWT, RBAC)
│   └── v1/                 # API routes and schemas
├── web/                    # Next.js frontend
├── configs/                # YAML configuration files
├── notebooks/              # Colab training notebooks
└── infra/                  # Docker, K8s, monitoring
```

## API Endpoints

| Method | Endpoint | Description | Role Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/auth/register` | Register new user | Public |
| POST | `/api/v1/auth/login` | Login | Public |
| POST | `/api/v1/transcribe` | Transcribe prescription image | Transcriber |
| GET | `/api/v1/transcriptions` | List transcription history | Transcriber |
| POST | `/api/v1/feedback` | Submit human correction | Reviewer |
| POST | `/api/v1/collect/upload` | Upload dataset image | Collector |
| GET | `/api/v1/health` | Health check | Public |

## Sample Output

```json
{
  "transcription": "Amoxicillin 500mg TID for 7 days",
  "entities": [
    {"type": "medicine", "value": "Amoxicillin", "confidence": 0.94},
    {"type": "dosage", "value": "500mg", "confidence": 0.91},
    {"type": "frequency", "value": "TID", "confidence": 0.88},
    {"type": "duration", "value": "7 days", "confidence": 0.85}
  ],
  "model_version": "medscript-ai-v0.1"
}
```

## Training (Google Colab)

Since no local GPU is available, training is done via Google Colab:

1. Upload `notebooks/03_train_donut.ipynb` to Colab
2. Connect to a T4 GPU runtime
3. Mount Google Drive for checkpoint persistence
4. Run all cells — training takes ~4-6 hours on T4

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vision Encoder | Donut (Swin Transformer) |
| Sequence Decoder | BiLSTM + CTC |
| NLP / NER | BiomedBERT |
| API | FastAPI + Uvicorn |
| Auth | OAuth 2.0 + JWT + RBAC |
| Frontend | Next.js (React, TypeScript) |
| Database | PostgreSQL + MongoDB |
| Cache | Redis |
| Queue | Celery |
| Storage | MinIO (S3-compatible) |
| Training | PyTorch Lightning + Google Colab |


## Author

**Navketan Singh** 
