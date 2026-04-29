import os
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.schemas import (
    AudioManifest, KeyframesManifest, RetrievalOutput, 
    Phase5Output, Phase5SegmentMetadata
)
from src.utils.ffmpeg_ops import (
    cut_video_segment, concat_videos, concat_audio_with_padding, 
    mux_video_audio, get_video_info
)
from src.utils.io import load_json_as_model, save_model_as_json
from src.utils.vram import VRAMManager
from src.exceptions import FFmpegError, VideoSummarizerError

logger = logging.getLogger(__name__)

class Phase5Assembler:
    """Final phase: Precise video assembly and stitching."""

    def __init__(self, config: Dict[str, Any], vram_manager: Optional[VRAMManager] = None):
        self.config = config
        self.vram_manager = vram_manager
        self.temp_root = Path(config.get("temp_root", "/tmp/video_summarizer"))
        self.output_dir = Path(config.get("output_dir", "data/output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, original_video_path: Path, audio_manifest_path: Path, 
            keyframes_manifest_path: Path, retrieval_output_path: Path, 
            progress_callback: Any = None, original_filename: str = None) -> Phase5Output:
        """
        Execute Phase 5: Assembly of video and audio segments.
        """
        video_id = original_video_path.stem
        start_time = time.time()
        
        # 1. Load inputs
        audio_manifest = load_json_as_model(audio_manifest_path, AudioManifest)
        keyframes_manifest = load_json_as_model(keyframes_manifest_path, KeyframesManifest)
        retrieval_output = load_json_as_model(retrieval_output_path, RetrievalOutput)
        
        method = retrieval_output.retrieval_method
        temp_dir = self.temp_root / f"{video_id}_{method}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        video_segments_dir = temp_dir / "video_segments"
        video_segments_dir.mkdir(exist_ok=True)
        
        # Map scenes by ID for quick lookup
        scenes_map = {s.id: s for s in keyframes_manifest.scenes}
        
        # 2. Process segments
        video_segment_paths = []
        audio_segment_paths = []
        assembled_segments = []
        
        total_matches = len(retrieval_output.matches)
        logger.info(f"Assembling {total_matches} segments for {video_id} using {method}")
        
        for i, match in enumerate(retrieval_output.matches):
            if progress_callback:
                pct = int((i / total_matches) * 60)
                progress_callback.update(5, "Assembly", pct, f"Cutting segment {i+1}/{total_matches}")
                
            try:
                # Find matching audio
                audio_sentence = next((s for s in audio_manifest.sentences if s.id == match.sentence_id), None)
                if not audio_sentence:
                    logger.warning(f"No audio found for sentence {match.sentence_id}, skipping segment.")
                    continue
                
                scene = scenes_map.get(match.matched_scene_id)
                if not scene:
                    logger.warning(f"No scene found for ID {match.matched_scene_id}, skipping segment.")
                    continue
                
                audio_duration = audio_sentence.duration_seconds
                cut_start = scene.start_seconds
                cut_end = scene.start_seconds + audio_duration
                
                video_seg_path = video_segments_dir / f"seg_{i:03d}.mp4"
                
                # Cut video segment (re-encoded, silent)
                cut_video_segment(original_video_path, cut_start, cut_end, video_seg_path)
                
                video_segment_paths.append(video_seg_path)
                
                # Audio path is relative to intermediate dir in manifest
                audio_abs_path = audio_manifest_path.parent / audio_sentence.audio_path
                audio_segment_paths.append(audio_abs_path)
                
                # Metadata provenance
                assembled_segments.append(Phase5SegmentMetadata(
                    sentence_id=match.sentence_id,
                    text=audio_sentence.text,
                    source_scene_id=match.matched_scene_id,
                    source_time_range=[cut_start, cut_end],
                    audio_path=str(audio_sentence.audio_path),
                    similarity_score=match.score
                ))
                
            except Exception as e:
                logger.error(f"Failed to process segment {i}: {e}")
                continue
                
        if not video_segment_paths:
            raise VideoSummarizerError("No segments were successfully processed.")
            
        # 3. Concatenate video segments
        if progress_callback:
            progress_callback.update(5, "Assembly", 70, "Concatenating video segments")
            
        logger.info("Concatenating video segments...")
        concat_video_path = temp_dir / "concat_video_silent.mp4"
        concat_videos(video_segment_paths, concat_video_path)
        
        # 4. Concatenate audio segments with padding
        if progress_callback:
            progress_callback.update(5, "Assembly", 80, "Concatenating audio segments")
            
        logger.info("Concatenating audio segments with padding...")
        concat_audio_path = temp_dir / "concat_audio.wav"
        padding_ms = self.config.get("tts", {}).get("padding_ms", 200)
        concat_audio_with_padding(audio_segment_paths, padding_ms / 1000.0, concat_audio_path)
        
        # 5. Final Mux
        if progress_callback:
            progress_callback.update(5, "Assembly", 90, "Muxing final video and audio")
            
        logger.info("Muxing final video and audio...")
        if original_filename:
            final_output_path = self.output_dir / f"{video_id}_{original_filename}_summary_{method}.mp4"
        else:
            final_output_path = self.output_dir / f"{video_id}_summary_{method}.mp4"
        
        subtitle_path = None
        if self.config.get("subtitle", {}).get("enabled", False):
            subtitle_path = self._generate_srt(assembled_segments, temp_dir / "subtitles.srt")

        mux_video_audio(concat_video_path, concat_audio_path, final_output_path, subtitle_path=subtitle_path)
        
        # 6. Metadata Output
        total_duration = sum(s.source_time_range[1] - s.source_time_range[0] for s in assembled_segments)
        total_duration += (len(assembled_segments) - 1) * (padding_ms / 1000.0)
        
        peak_vram = self.vram_manager.get_peak_usage() if self.vram_manager else 0.0
        
        output_metadata = Phase5Output(
            video_id=video_id,
            output_path=str(final_output_path),
            method=method,
            total_duration_seconds=total_duration,
            segments=assembled_segments,
            total_processing_time_seconds=time.time() - start_time,
            peak_vram_gb=peak_vram
        )
        
        metadata_path = self.output_dir / f"{video_id}_summary_{method}_metadata.json"
        save_model_as_json(output_metadata, metadata_path)
        
        # 7. Cleanup
        if self.config.get("cleanup_temp", True):
            shutil.rmtree(temp_dir)
            
        logger.info(f"Phase 5 complete. Final video: {final_output_path}")
        
        if progress_callback:
            progress_callback.update(5, "Assembly", 100, f"Final duration: {total_duration:.2f}s")
            
        return output_metadata

    def _generate_srt(self, segments: List[Phase5SegmentMetadata], out_path: Path) -> Path:
        """Simple SRT generator from segments."""
        padding_s = self.config.get("tts", {}).get("padding_ms", 200) / 1000.0
        current_time = 0.0
        
        def format_ts(seconds):
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            msecs = int((seconds * 1000) % 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

        with open(out_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                duration = seg.source_time_range[1] - seg.source_time_range[0]
                start_srt = format_ts(current_time)
                end_srt = format_ts(current_time + duration)
                
                f.write(f"{i+1}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{seg.text}\n\n")
                
                current_time += duration + padding_s
                
        return out_path
