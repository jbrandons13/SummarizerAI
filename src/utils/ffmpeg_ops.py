import ffmpeg
import os
import tempfile
import subprocess
import json
from pathlib import Path
from typing import List, Optional, Union
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

# Constants for hybrid duration strategy
MIN_SLOWDOWN_RATIO = 0.6  # don't slow below 0.6x speed (i.e. clip plays at 60% speed = 1.67x duration)

def probe_video_duration(video_path: Union[Path, str]) -> float:
    """Probe actual video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path)
            ],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        stderr = e.stderr.decode() if hasattr(e, "stderr") and e.stderr else str(e)
        raise FFmpegError(f"Failed to probe video duration: {stderr}")


def probe_video_resolution(video_path: Union[Path, str]) -> tuple[int, int]:
    """Probe video resolution (width, height) using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                str(video_path)
            ],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception as e:
        stderr = e.stderr.decode() if hasattr(e, "stderr") and e.stderr else str(e)
        raise FFmpegError(f"Failed to probe video resolution: {stderr}")


def scale_video_to_target(
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    target_width: int,
    target_height: int,
    fps: int = 30,
) -> None:
    """
    Scale video to target resolution with letterbox padding to preserve aspect ratio.
    
    LTX outputs are 768x512 (1.5:1). Source videos may be 640x320 (2:1), 1920x1080 (16:9), etc.
    Letterbox pad with black bars to preserve LTX content aspect ratio.
    Also normalize fps for clean concat.
    """
    # Build filter: scale to fit within target, pad to exact target size
    vf = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps},"
        f"setsar=1"
    )
    
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(input_path),
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-an",  # remove audio (segments are silent at this stage)
                str(output_path)
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        raise FFmpegError(f"Failed to scale video to target: {stderr}")


def extend_clip_to_duration(
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    target_duration_s: float,
    width: int,
    height: int,
    fps: int = 30,
) -> None:
    """
    Extend a short video clip to target duration using hybrid strategy:
    1. If clip duration >= target: trim to exact target.
    2. If MIN_SLOWDOWN_RATIO * target <= clip < target: pure slow playback to match.
    3. If clip < MIN_SLOWDOWN_RATIO * target: slow to MIN_SLOWDOWN_RATIO, then freeze last frame for remainder.
    """
    clip_dur = probe_video_duration(input_path)
    
    try:
        if clip_dur >= target_duration_s:
            # Case 1: trim
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(input_path),
                    "-t", f"{target_duration_s}",
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-an",
                    str(output_path)
                ],
                check=True, capture_output=True
            )
            return
        
        ratio = clip_dur / target_duration_s
        
        if ratio >= MIN_SLOWDOWN_RATIO:
            # Case 2: pure slow playback
            # setpts=PTS/speed where speed = clip_dur/target_duration (slow down)
            speed_factor = clip_dur / target_duration_s  # < 1.0 means slowdown
            vf = f"setpts={1.0/speed_factor:.6f}*PTS,fps={fps}"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(input_path),
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-an",
                    str(output_path)
                ],
                check=True, capture_output=True
            )
            return
        
        # Case 3: hybrid (slow to MIN_SLOWDOWN_RATIO, then freeze remainder)
        # Slowed clip duration = clip_dur / MIN_SLOWDOWN_RATIO
        # Freeze duration = target - slowed_duration
        slowed_dur = clip_dur / MIN_SLOWDOWN_RATIO
        freeze_dur = target_duration_s - slowed_dur
        
        # Step 3a: produce slowed version
        slowed_path = Path(output_path).with_suffix(".slowed.mp4")
        speed_factor = MIN_SLOWDOWN_RATIO
        vf_slow = f"setpts={1.0/speed_factor:.6f}*PTS,fps={fps}"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(input_path),
                "-vf", vf_slow,
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-an",
                str(slowed_path)
            ],
            check=True, capture_output=True
        )
        
        # Step 3b: extract last frame as image
        last_frame_path = Path(output_path).with_suffix(".lastframe.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-sseof", "-0.1",  # seek to ~0.1s from end
                "-i", str(slowed_path),
                "-vframes", "1",
                str(last_frame_path)
            ],
            check=True, capture_output=True
        )
        
        # Step 3c: create freeze clip from last frame
        freeze_path = Path(output_path).with_suffix(".freeze.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-loop", "1",
                "-i", str(last_frame_path),
                "-t", f"{freeze_dur}",
                "-vf", f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1",
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-an",
                str(freeze_path)
            ],
            check=True, capture_output=True
        )
        
        # Step 3d: concat slowed + freeze
        concat_list_path = Path(output_path).with_suffix(".concat.txt")
        concat_list_path.write_text(f"file '{slowed_path.resolve()}'\nfile '{freeze_path.resolve()}'\n")
        
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list_path),
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-an",
                str(output_path)
            ],
            check=True, capture_output=True
        )
        
        # Cleanup temp files
        for p in [slowed_path, last_frame_path, freeze_path, concat_list_path]:
            p.unlink(missing_ok=True)
            
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        raise FFmpegError(f"Failed to extend clip: {stderr}")


def cut_video_segment(
    video_path: Union[Path, str],
    start: float,
    end: float,
    out_path: Union[Path, str],
    reencode: bool = True,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    fps: int = 30,
) -> None:
    """
    Cut a segment from source video. If target_width/height provided, scale + letterbox.
    Output is always re-encoded (unless reencode is False and target_width/height are not provided)
    to ensure uniform codec & timestamps.
    """
    duration = end - start
    
    if target_width and target_height:
        vf = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={fps},"
            f"setsar=1"
        )
        reencode = True
    else:
        vf = f"fps={fps},setsar=1"
        
    try:
        if reencode:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", f"{start}",
                    "-i", str(video_path),
                    "-t", f"{duration}",
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    str(out_path)
                ],
                check=True, capture_output=True
            )
        else:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", f"{start}",
                    "-i", str(video_path),
                    "-t", f"{duration}",
                    "-c:v", "copy",
                    "-an",
                    str(out_path)
                ],
                check=True, capture_output=True
            )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        raise FFmpegError(f"Failed to cut video segment [{start}, {end}]: {stderr}")

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

def generate_silence(duration: float, out_path: Path, sample_rate: int = 48000, channels: int = 2):
    """Generate a silence WAV file of specific duration and audio properties."""
    layout = "stereo" if channels == 2 else "mono"
    try:
        (
            ffmpeg
            .input(f'anullsrc=r={sample_rate}:cl={layout}', f='lavfi', t=duration)
            .output(str(out_path))
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFmpegError(f"Failed to generate silence: {e.stderr.decode()}")

def concat_audio_with_padding(audio_paths: List[Path], padding_duration: float, out_path: Path, sample_rate: Optional[int] = None):
    """
    Concatenate audio clips with silence padding in between.
    """
    if not audio_paths:
        return

    # Probe first audio to match properties
    in_sample_rate = 48000
    in_channels = 2
    try:
        probe = ffmpeg.probe(str(audio_paths[0]))
        audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
        if audio_stream:
            in_sample_rate = int(audio_stream['sample_rate'])
            in_channels = int(audio_stream['channels'])
    except Exception:
        pass

    target_sample_rate = sample_rate if sample_rate is not None else in_sample_rate

    # Create temporary silences and list
    temp_files = []
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_list:
        list_path = f_list.name
        
        # Create a silence clip matching target properties to prevent corruption/desync
        layout_str = "stereo" if in_channels == 2 else "mono"
        silence_path = Path(tempfile.gettempdir()) / f"silence_{padding_duration}_{target_sample_rate}_{layout_str}.wav"
        if not silence_path.exists():
            generate_silence(padding_duration, silence_path, target_sample_rate, in_channels)
            temp_files.append(silence_path)
            
        for i, p in enumerate(audio_paths):
            abs_p = str(p.absolute()).replace("'", "'\\''")
            f_list.write(f"file '{abs_p}'\n")
            
            # Add padding after every clip except the last one
            if i < len(audio_paths) - 1 and padding_duration > 0:
                abs_s = str(silence_path.absolute()).replace("'", "'\\''")
                f_list.write(f"file '{abs_s}'\n")

    try:
        (
            ffmpeg
            .input(list_path, format='concat', safe=0)
            .output(str(out_path), acodec='pcm_s16le', ar=target_sample_rate, ac=in_channels)
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
