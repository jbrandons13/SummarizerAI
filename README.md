# Video Summarizer AI

AI pipeline to summarize long video podcasts into short, engaging videos with AI voiceover and semantic B-roll matching.

## Project Description
This project is part of a thesis work focused on automating the creation of video highlights. It uses a multi-modal approach:
1. **Transcription**: WhisperX for precise, word-aligned transcripts.
2. **Analysis**: LLM-based summarization to identify key segments.
3. **B-roll Matching**: Semantic search (CLIP/SigLIP) to find relevant B-roll footages.
4. **Assembly**: Automated video editing using FFmpeg.

## Setup Instructions

### Prerequisites
- Python 3.11
- CUDA 12.1 (for RTX 3090)
- FFmpeg installed on system

### Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install .
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Folder Structure
```bash
video-summarizer/
├── README.md           # Project description and setup
├── pyproject.toml      # Dependency management
├── .env.example        # Environment variables template
├── .gitignore          # Git exclusion rules
├── configs/            # Config files (YAML)
├── src/                # Source code
│   ├── pipeline.py     # Main pipeline orchestration
│   ├── utils/          # Utility modules (VRAM, FFmpeg, IO)
│   ├── models/         # Model loaders and wrappers
│   └── eval/           # Evaluation metrics
├── scripts/            # Utility scripts (download models, etc.)
├── data/               # Data storage (raw, intermediate, output)
└── tests/              # Unit and integration tests
```

## Hardware Requirements
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) recommended for large-v3 and VL models.
- **OS**: Ubuntu 22.04 (tested on WSL2).
