from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from src.api.routes import router
import os

app = FastAPI(title="Video Summarizer API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
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
