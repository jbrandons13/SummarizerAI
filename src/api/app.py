from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from src.api.routes import router
import os
from dotenv import load_dotenv

# Load .env file with override so hot-reloading picks up changes
load_dotenv(override=True)

app = FastAPI(title="Video Summarizer API")

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"], # Allow all for now, but explicitly mention dev ones
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directories if they don't exist
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/output").mkdir(parents=True, exist_ok=True)

# Static files mounting
# /static links to data/output/ for video serving
app.mount("/static", StaticFiles(directory="data/output"), name="static")

# Include routes
app.include_router(router, prefix="/api")

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Video Summarizer API is running"}
