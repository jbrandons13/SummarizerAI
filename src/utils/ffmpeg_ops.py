import ffmpeg
import os
import tempfile
from pathlib import Path
from typing import List
from src.exceptions import FFmpegError
import logging

logger = logging.getLogger(__name__)

def extract_audio(video_path: Path, out_wav_path: Path, sample_rate: int = 16000):
    """
    Extract audio from a video file and save it as a WAV file.
    
    Args:
        video_path: Path to the input video.
        out_wav_path: Path to save the extracted audio.
        sample_rate: Target sample rate for the output audio.
    """
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(out_wav_path), ac=1, ar=sample_rate)
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to extract audio: {e.stderr.decode()}")

def extract_frame_at(video_path: Path, timestamp: float, out_jpg_path: Path):
    """
    Extract a single frame from a video at a specific timestamp.
    
    Args:
        video_path: Path to the input video.
        timestamp: Time in seconds at which to extract the frame.
        out_jpg_path: Path to save the extracted frame.
    """
    try:
        (
            ffmpeg
            .input(str(video_path), ss=timestamp)
            .output(str(out_jpg_path), vframes=1)
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to extract frame at {timestamp}s: {e.stderr.decode()}")

def cut_video_segment(video_path: Path, start: float, end: float, out_path: Path):
    """
    Cut a segment from a video file. This always re-encodes the output.
    
    Args:
        video_path: Path to the input video.
        start: Start time in seconds.
        end: End time in seconds.
        out_path: Path to save the cut segment.
    """
    try:
        duration = end - start
        (
            ffmpeg
            .input(str(video_path), ss=start, t=duration)
            .output(str(out_path))
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to cut video segment [{start}, {end}]: {e.stderr.decode()}")

def concat_videos(video_paths: List[Path], out_path: Path):
    """
    Concatenate multiple video files using the concat demuxer.
    
    Args:
        video_paths: List of paths to the video files to concatenate.
        out_path: Path to save the concatenated video.
    """
    if not video_paths:
        return

    # Create a temporary file list for the concat demuxer
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file_path = f.name
        for p in video_paths:
            # Escape single quotes and use absolute paths
            abs_path = str(p.absolute()).replace("'", "'\\''")
            f.write(f"file '{abs_path}'\n")

    try:
        (
            ffmpeg
            .input(temp_file_path, format='concat', safe=0)
            .output(str(out_path), c='copy')
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to concatenate videos: {e.stderr.decode()}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def mux_video_audio(video_path: Path, audio_path: Path, out_path: Path):
    """
    Combine a video stream and an audio stream into a single file.
    
    Args:
        video_path: Path to the video file (source for video stream).
        audio_path: Path to the audio file (source for audio stream).
        out_path: Path to save the muxed output.
    """
    try:
        v = ffmpeg.input(str(video_path)).video
        a = ffmpeg.input(str(audio_path)).audio
        (
            ffmpeg
            .output(v, a, str(out_path), shortest=None)
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to mux video and audio: {e.stderr.decode()}")
