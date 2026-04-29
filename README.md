# 🎬 Video Summarizer AI: Multi-modal Pipeline for Automated Podcast Highlights

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, production-grade AI pipeline designed to transform long-form video podcasts into short, engaging highlight reels. This project integrates state-of-the-art multi-modal models for transcription, semantic analysis, neural voiceover, and intelligent B-roll retrieval, culminating in a fully automated video assembly process.

---

## 🌟 Key Features

*   **🎙️ Precision Transcription**: Powered by **WhisperX** (large-v3) for word-level timestamps and robust audio alignment.
*   **🧠 Intelligent Scripting**: Utilizes **Llama 3.3 (via Groq)** or local **Qwen2.5-14B** to identify semantic highlights and generate concise summary scripts.
*   **🗣️ Neural Voiceover**: High-fidelity speech synthesis using **Kokoro v1.0** (ONNX) for natural-sounding narration.
*   **🔍 Semantic Visual Retrieval (3-Arm Strategy)**: 
    - **Arm A (Random)**: Baseline for evaluation.
    - **Arm B (Caption-Cosine)**: Cross-modal matching using **Qwen2.5-VL** captions and Sentence-Transformers.
    - **Arm C (SigLIP 2 Direct)**: Zero-shot vision-language retrieval using Google's **SigLIP 2**.
*   **🎬 Automated Video Assembly**: Precise frame-level muxing with **FFmpeg**, featuring silence padding (200ms), re-encoding (H.264 CRF 20), and burned-in subtitles.
*   **📊 Evaluation Framework**: Automated metrics including **ROUGE-L**, **BERTScore**, and **CLIPScore**, complemented by an **LLM-as-Judge** scoring system.
*   **💻 Modern Web Interface**: A sleek React-based dashboard with real-time WebSocket progress tracking and comparative evaluation metrics.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Input Video] --> B[Phase 1: Ingestion & WhisperX Transcription]
    B --> C[Phase 2: LLM Summarization & Scripting]
    C --> D[Phase 3: Neural Voiceover Generation]
    D --> E[Phase 4: Semantic B-roll Retrieval]
    E --> F[Phase 5: Automated Video Assembly]
    F --> G[Final Summary Video]
    G --> H[Phase 6: Automated Evaluation]
    
    subgraph "Retrieval Arms"
        E1[Arm A: Random]
        E2[Arm B: Qwen2.5-VL + Cosine]
        E3[Arm C: SigLIP 2 Direct]
    end
    E --- E1
    E --- E2
    E --- E3
```

---

## 🧪 The 3-Arm Retrieval Strategy

A core component of this project is the comparative analysis of visual retrieval methods:

| Arm | Method | Tech Stack | Characteristics |
| :--- | :--- | :--- | :--- |
| **Arm A** | Random | `numpy.random` | Baseline control. |
| **Arm B** | Caption-Cosine | `Qwen2.5-VL-7B` + `Sentence-Transformers` | Semantic bridge via natural language descriptions. |
| **Arm C** | SigLIP 2 Direct | `google/siglip2-so400m` | Direct vision-language embedding alignment. |

---

## 🛠️ Tech Stack

### Backend
- **Core**: Python 3.11, FastAPI, Pydantic
- **ML/AI**: Torch 2.5.1, Transformers, WhisperX, Open-CLIP, SigLIP 2
- **Video/Audio**: FFmpeg (via `ffmpeg-python`), `pyscenedetect`, `librosa`
- **Evaluation**: `rouge-score`, `bert-score`, `clip-score`

### Frontend
- **Framework**: React 18, Vite
- **Styling**: Tailwind CSS
- **Communication**: WebSockets (Real-time progress), REST API

---

## 📊 Evaluation & Metrics

The pipeline includes a robust evaluation suite to measure the quality of generated summaries across multiple dimensions:

1.  **Information Quality**: ROUGE-L and BERTScore against original transcript.
2.  **Visual Alignment**: CLIPScore between summary script and retrieved B-roll.
3.  **Human-like Judgment**: LLM-as-Judge (using Llama 3.3 70B) scoring Information, Factualness, and Visuals on a 1-5 scale.

---

## 🚀 Getting Started

### Prerequisites
- **OS**: Ubuntu 22.04 or WSL2
- **GPU**: NVIDIA RTX 3090/4090 (24GB VRAM recommended for local LLM & VL models)
- **Tools**: Python 3.11, FFmpeg (with `libx264` and `libass`)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/video-summarizer-ai.git
   cd video-summarizer-ai
   ```

2. **Environment Setup**
   ```bash
   # Create conda environment or venv
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install .
   # Note: transformers must be installed from source for SigLIP 2 support
   pip install git+https://github.com/huggingface/transformers
   ```

3. **Configuration**
   Copy the example environment file and fill in your API keys (Groq, etc.):
   ```bash
   cp .env.example .env
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   python scripts/run_server.py
   ```

2. **Start the Frontend Development Server**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

## 📁 Folder Structure

```text
video-summarizer/
├── src/                # Core logic (Phases 1-6)
│   ├── api/            # FastAPI routes and WebSocket handlers
│   ├── eval/           # Metrics and LLM-as-Judge implementation
│   ├── models/         # Model wrappers (VRAM-safe loaders)
│   ├── utils/          # FFmpeg ops, VRAM management, IO helpers
│   └── pipeline.py     # Main orchestrator
├── frontend/           # React + Vite application
├── configs/            # YAML configuration for pipeline parameters
├── results/            # Ablation study results, plots, and stats
├── data/               # Persistent storage for raw/processed assets
└── tests/              # Comprehensive test suite
```

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*This project was developed as part of a thesis exploring multi-modal AI applications in automated content creation.*
