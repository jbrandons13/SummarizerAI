import ffmpeg
import os
import tempfile
from pathlib import Path
from typing import List, Optional
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

def cut_video_segment(video_path: Path, start: float, end: float, out_path: Path, reencode: bool = True):
    """
    Cut a segment from a video file.
    
    Args:
        video_path: Path to the input video.
        start: Start time in seconds.
        end: End time in seconds.
        out_path: Path to save the cut segment.
        reencode: If True, re-encodes to H.264 CRF 20 preset fast (silent).
    """
    try:
        duration = end - start
        input_args = {}
        output_args = {}
        
        if reencode:
            # Re-encode H.264 CRF 20 preset fast, no audio
            output_args = {
                'c:v': 'libx264',
                'crf': 20,
                'preset': 'fast',
                'an': None
            }
        else:
            output_args = {'c': 'copy'}

        (
            ffmpeg
            .input(str(video_path), ss=start, t=duration)
            .output(str(out_path), **output_args)
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

def generate_silence(duration: float, out_path: Path, sample_rate: int = 48000):
    """Generate a silence WAV file of specific duration."""
    try:
        (
            ffmpeg
            .input(f'anullsrc=r={sample_rate}:cl=stereo', f='lavfi', t=duration)
            .output(str(out_path))
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to generate silence: {e.stderr.decode()}")

def concat_audio_with_padding(audio_paths: List[Path], padding_duration: float, out_path: Path, sample_rate: int = 48000):
    """
    Concatenate audio clips with silence padding in between.
    """
    if not audio_paths:
        return

    # Create temporary silences and list
    temp_files = []
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_list:
        list_path = f_list.name
        
        # Create a silence clip
        silence_path = Path(tempfile.gettempdir()) / f"silence_{padding_duration}.wav"
        if not silence_path.exists():
            generate_silence(padding_duration, silence_path, sample_rate)
            temp_files.append(silence_path)
            
        for i, p in enumerate(audio_paths):
            abs_p = str(p.absolute()).replace("'", "'\\''")
            f_list.write(f"file '{abs_p}'\n")
            
            # Add padding after every clip except the last one (or maybe according to user preference)
            # User says "200ms silence padding", usually means between segments.
            if i < len(audio_paths) - 1 and padding_duration > 0:
                abs_s = str(silence_path.absolute()).replace("'", "'\\''")
                f_list.write(f"file '{abs_s}'\n")

    try:
        (
            ffmpeg
            .input(list_path, format='concat', safe=0)
            .output(str(out_path), acodec='pcm_s16le', ar=sample_rate, ac=2)
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to concatenate audio: {e.stderr.decode()}")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)

def mux_video_audio(video_path: Path, audio_path: Path, out_path: Path, subtitle_path: Optional[Path] = None, subtitle_style: Optional[str] = None):
    """
    Combine video and audio, optionally burning in subtitles with custom styling.
    """
    try:
        vi = ffmpeg.input(str(video_path))
        ai = ffmpeg.input(str(audio_path))
        
        if subtitle_path:
            # Re-encode video to burn in subtitles
            filter_args = {'filename': str(subtitle_path)}
            if subtitle_style:
                filter_args['force_style'] = subtitle_style
                
            v = vi.video.filter('subtitles', **filter_args)
            cmd = ffmpeg.output(v, ai.audio, str(out_path), vcodec='libx264', crf=20, preset='fast', acodec='aac', movflags='faststart')
        else:
            # Stream copy video, re-encode audio to aac for safety in MP4
            cmd = ffmpeg.output(vi.video, ai.audio, str(out_path), vcodec='copy', acodec='aac', movflags='faststart')
            
        logger.info(f"Running mux: {' '.join(cmd.get_args())}")
        cmd.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to mux: {e.stderr.decode()}")

def get_video_info(video_path: Path) -> dict:
    """Get video information using ffprobe."""
    try:
        probe = ffmpeg.probe(str(video_path))
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to probe video: {e.stderr.decode()}")
