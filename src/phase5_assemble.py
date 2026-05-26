import os
import json
import logging
import shutil
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.schemas import (
    AudioManifest, KeyframesManifest, RetrievalOutput, 
    Phase5Output, Phase5SegmentMetadata
)
from src.utils.ffmpeg_ops import (
    cut_video_segment, concat_videos, concat_audio_with_padding, 
    mux_video_audio, get_video_info, probe_video_resolution,
    scale_video_to_target, extend_clip_to_duration
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
        
        # Load assignments JSON path or RetrievalOutput
        with open(retrieval_output_path, "r") as f:
            data = json.load(f)
            
        # Map scenes by ID for quick lookup
        scenes_map = {s.id: s for s in keyframes_manifest.scenes}
        
        from src.phase4_retrieve import Assignment
        assignments = []
        if isinstance(data, dict):
            method = data.get("retrieval_method", "sentence_level")
            if "groups" in data:
                for item in data["groups"]:
                    assignments.append(Assignment(
                        sentence_ids=item["sentence_ids"],
                        scene_id=item["scene_id"],
                        best_similarity=item["best_similarity"],
                        raw_cosine=item["raw_cosine"],
                        temporal_weight=item["temporal_weight"],
                        action=item["action"],
                        timestamp_hint_merged=tuple(item["timestamp_hint_merged"]),
                        similarity_trail=item.get("similarity_trail", [])
                    ))
            elif "matches" in data:
                # Convert sentence-level matches to Assignment objects
                for m in data["matches"]:
                    scene = scenes_map.get(m["matched_scene_id"])
                    if scene:
                        center_ts = m.get("best_frame_timestamp", 0.0) or (scene.start_seconds + scene.end_seconds) / 2.0
                    else:
                        center_ts = 0.0
                    assignments.append(Assignment(
                        sentence_ids=[m["sentence_id"]],
                        scene_id=m["matched_scene_id"],
                        best_similarity=m["score"],
                        raw_cosine=m["score"],
                        temporal_weight=1.0,
                        action="retrieve",
                        timestamp_hint_merged=(center_ts, center_ts),
                        similarity_trail=[]
                    ))
        else:
            method = "grouping_gate"
            for item in data:
                assignments.append(Assignment(
                    sentence_ids=item["sentence_ids"],
                    scene_id=item["scene_id"],
                    best_similarity=item["best_similarity"],
                    raw_cosine=item["raw_cosine"],
                    temporal_weight=item["temporal_weight"],
                    action=item["action"],
                    timestamp_hint_merged=tuple(item["timestamp_hint_merged"]),
                    similarity_trail=item.get("similarity_trail", [])
                ))

        temp_dir = self.temp_root / f"{video_id}_{method}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        video_segments_dir = temp_dir / "video_segments"
        video_segments_dir.mkdir(exist_ok=True)

        # Determine target resolution and target fps from original video
        target_width, target_height = probe_video_resolution(original_video_path)
        target_fps = 30
        
        # 2. Process segments
        video_segment_paths = []
        audio_segment_paths = []
        assembled_segments = []
        
        total_matches = len(assignments)
        logger.info(f"Assembling {total_matches} groups for {video_id} using {method}")
        
        for i, assignment in enumerate(assignments):
            if progress_callback:
                pct = int((i / total_matches) * 60)
                progress_callback.update(5, "Assembly", pct, f"Cutting segment {i+1}/{total_matches}")
                
            try:
                # Find matching audio(s) for the group
                group_sentences = [
                    next((s for s in audio_manifest.sentences if s.id == sid), None)
                    for sid in assignment.sentence_ids
                ]
                group_sentences = [s for s in group_sentences if s is not None]
                
                if not group_sentences:
                    logger.warning(f"No audio found for group {assignment.sentence_ids}, skipping segment.")
                    continue
                
                scene = scenes_map.get(assignment.scene_id)
                if not scene:
                    logger.warning(f"No scene found for ID {assignment.scene_id}, skipping segment.")
                    continue
                
                audio_duration = sum(s.duration_seconds for s in group_sentences)
                
                # Get video info for global boundaries
                video_info = get_video_info(original_video_path)
                total_video_duration = float(video_info.get('duration', 10000))
                
                padding_ms = self.config.get("tts", {}).get("silence_padding_ms", 200)
                padding_s = padding_ms / 1000.0
                
                # Target duration is audio + internal paddings
                # If N sentences in group, there are N-1 internal paddings in the final audio stream
                target_duration = audio_duration + (len(group_sentences) - 1) * padding_s
                
                # Use merged timestamp hint for centering the cut
                hint_lo, hint_hi = assignment.timestamp_hint_merged
                center_ts = (hint_lo + hint_hi) / 2.0
                
                # Calculate cut boundaries centered on center_ts
                cut_start = center_ts - (target_duration / 2)
                cut_end = cut_start + target_duration
                
                # Clamp to global video boundaries
                if cut_start < 0:
                    cut_start = 0
                    cut_end = min(total_video_duration, target_duration)
                elif cut_end > total_video_duration:
                    cut_end = total_video_duration
                    cut_start = max(0, cut_end - target_duration)
                    
                actual_v_duration = cut_end - cut_start
                
                # Check for generated clip if action is "generate"
                use_generated = False
                generated_dir = Path(self.config.get("paths", {}).get("intermediate_dir", "data/intermediate")) / video_id / "generated"
                generated_path = generated_dir / f"group_{i:03d}.mp4"
                
                if assignment.action == "generate":
                    if generated_path.exists():
                        logger.info(f"Using generated clip for group {i} from: {generated_path}")
                        use_generated = True
                    else:
                        logger.warning(
                            f"Generated clip missing at {generated_path}, falling back to retrieval for group {assignment.sentence_ids} "
                            f"(scene {assignment.scene_id}, sim={assignment.best_similarity:.3f})."
                        )
                
                video_seg_path = video_segments_dir / f"seg_{i:03d}.mp4"
                
                if use_generated:
                    # Step a: scale LTX clip to target resolution
                    scaled_clip_path = video_seg_path.with_suffix(".scaled.mp4")
                    scale_video_to_target(generated_path, scaled_clip_path, target_width, target_height, fps=target_fps)
                    
                    # Step b: extend to audio duration with hybrid strategy
                    extend_clip_to_duration(
                        scaled_clip_path, 
                        video_seg_path, 
                        target_duration_s=target_duration,
                        width=target_width, 
                        height=target_height,
                        fps=target_fps,
                    )
                    
                    # Cleanup
                    scaled_clip_path.unlink(missing_ok=True)
                    
                    # For generated clip, source_time_range is [0.0, target_duration]
                    seg_cut_start = 0.0
                    seg_cut_end = target_duration
                else:
                    # Cut video segment from source video (re-encoded, silent)
                    cut_video_segment(
                        original_video_path, 
                        cut_start, 
                        cut_end, 
                        video_seg_path,
                        target_width=target_width,
                        target_height=target_height,
                        fps=target_fps
                    )
                    seg_cut_start = cut_start
                    seg_cut_end = cut_end
                
                video_segment_paths.append(video_seg_path)
                
                # Audio paths for all sentences in group
                for s in group_sentences:
                    audio_abs_path = audio_manifest_path.parent / s.audio_path
                    audio_segment_paths.append(audio_abs_path)
                
                # Metadata provenance
                # Save the group audio duration (with internal paddings) in a temporary attribute
                group_audio_duration_with_internal_padding = audio_duration + (len(group_sentences) - 1) * padding_s
                
                assembled_segments.append(Phase5SegmentMetadata(
                    sentence_id=assignment.sentence_ids[0],
                    text=" ".join(s.text for s in group_sentences),
                    source_scene_id=assignment.scene_id,
                    best_frame_timestamp=center_ts,
                    source_time_range=[seg_cut_start, seg_cut_end],
                    audio_path=str(group_sentences[0].audio_path),
                    group_audio_duration=group_audio_duration_with_internal_padding,
                    similarity_score=assignment.best_similarity
                ))
                
            except Exception as e:
                logger.error(f"Failed to process segment {i}: {e}")
                continue
                
        if not video_segment_paths:
            raise VideoSummarizerError("No segments were successfully processed.")
                
        # 3. Concatenate video segments
        if progress_callback:
            progress_callback.update(5, "Assembly", 70, "Concatenating video segments")
            
        logger.info("Concatenating video segments with matching padding...")
        
        # We need to add video padding to match audio padding
        padding_ms = self.config.get("tts", {}).get("silence_padding_ms", 200)
        padding_s = padding_ms / 1000.0
        
        padded_video_segments = []
        for i, v_path in enumerate(video_segment_paths):
            padded_video_segments.append(v_path)
            
            seg_metadata = assembled_segments[i]
            v_dur = seg_metadata.source_time_range[1] - seg_metadata.source_time_range[0]
            a_dur = seg_metadata.group_audio_duration
            
            if i < len(video_segment_paths) - 1:
                # Spacer duration = gap between this group's audio and next group's audio
                # In concat_audio_with_padding, we add padding_s after every segment.
                # So the gap between groups is exactly padding_s.
                # Plus any mismatch within this group's video duration.
                spacer_duration = (a_dur + padding_s) - v_dur
                
                if spacer_duration > 0.05: # Only add if still significantly short (fallback)
                    spacer_path = temp_dir / f"spacer_{i:03d}.mp4"
                    self._generate_video_spacer(spacer_duration, spacer_path, target_width, target_height, fps=target_fps)
                    padded_video_segments.append(spacer_path)
            else:
                # Final group: check if we need a trailing spacer to match audio
                # FFmpeg mux might freeze last frame if audio is longer.
                # Adding a black spacer ensures a clean end.
                if a_dur > v_dur + 0.01:
                    spacer_duration = a_dur - v_dur
                    spacer_path = temp_dir / f"spacer_final.mp4"
                    self._generate_video_spacer(spacer_duration, spacer_path, target_width, target_height, fps=target_fps)
                    padded_video_segments.append(spacer_path)

        concat_video_path = temp_dir / "concat_video_silent.mp4"
        concat_videos(padded_video_segments, concat_video_path)
        
        # 4. Concatenate audio segments with padding
        if progress_callback:
            progress_callback.update(5, "Assembly", 80, "Concatenating audio segments")
            
        logger.info("Concatenating audio segments with padding...")
        concat_audio_path = temp_dir / "concat_audio.wav"
        concat_audio_with_padding(audio_segment_paths, padding_s, concat_audio_path)
        
        # 5. Final Mux
        if progress_callback:
            progress_callback.update(5, "Assembly", 90, "Muxing final video and audio")
            
        logger.info("Muxing final video and audio...")
        
        # Create per-job output folder
        job_output_dir = self.output_dir / video_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        if original_filename:
            final_output_path = job_output_dir / f"{original_filename}_summary_{method}.mp4"
            debug_output_path = job_output_dir / f"{original_filename}_summary_{method}_debug.mp4"
        else:
            final_output_path = job_output_dir / f"summary_{method}.mp4"
            debug_output_path = job_output_dir / f"summary_{method}_debug.mp4"
        
        subtitle_path = None
        subtitle_style = None
        if self.config.get("subtitle", {}).get("enabled", True): # Enabled by default for "modern" request
            subtitle_path = self._generate_srt(assembled_segments, temp_dir / "subtitles.srt")
            # Modern styling: Clean sans-serif, white text, semi-transparent black background box
            subtitle_style = self.config.get("subtitle", {}).get("style", 
                "FontName=Liberation Sans,FontSize=18,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=25"
            )

        mux_video_audio(concat_video_path, concat_audio_path, final_output_path, subtitle_path=subtitle_path, subtitle_style=subtitle_style)

        # Build segment timeline for debug overlay
        segments_timeline = []
        current_time = 0.0
        for i, seg_metadata in enumerate(assembled_segments):
            v_dur = seg_metadata.source_time_range[1] - seg_metadata.source_time_range[0]
            a_dur = seg_metadata.group_audio_duration
            
            duration_in_concat = v_dur
            if i < len(assembled_segments) - 1:
                spacer_duration = (a_dur + padding_s) - v_dur
                if spacer_duration > 0.05:
                    duration_in_concat += spacer_duration
            else:
                if a_dur > v_dur + 0.01:
                    spacer_duration = a_dur - v_dur
                    duration_in_concat += spacer_duration
                    
            segments_timeline.append({
                "start": current_time,
                "end": current_time + duration_in_concat,
                "action": assignments[i].action.upper(),
                "score": assignments[i].best_similarity,
                "group_id": i,
                "num_sentences": len(assignments[i].sentence_ids),
            })
            current_time += duration_in_concat

        # Generate debug-annotated output
        self._add_debug_annotation(final_output_path, debug_output_path, segments_timeline)
        
        # 6. Metadata Output
        total_duration = sum(s.source_time_range[1] - s.source_time_range[0] for s in assembled_segments)
        total_duration += (len(assembled_segments) - 1) * padding_s
        
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
        
        if original_filename:
            metadata_path = job_output_dir / f"{original_filename}_summary_{method}_metadata.json"
        else:
            metadata_path = job_output_dir / f"summary_{method}_metadata.json"
        save_model_as_json(output_metadata, metadata_path)
        
        # 7. Cleanup
        if self.config.get("cleanup_temp", True):
            shutil.rmtree(temp_dir)
            
        logger.info(f"Phase 5 complete. Final video: {final_output_path}")
        
        if progress_callback:
            progress_callback.update(5, "Assembly", 100, f"Final duration: {total_duration:.2f}s")
            
        return output_metadata

    def _generate_srt(self, segments: List[Phase5SegmentMetadata], out_path: Path) -> Path:
        """Simple SRT generator from segments, adjusted for new assembly logic."""
        padding_ms = self.config.get("tts", {}).get("silence_padding_ms", 200)
        padding_s = padding_ms / 1000.0
        current_time = 0.0
        
        def format_ts(seconds):
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            msecs = int((seconds * 1000) % 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

        def wrap_text(text, width=50):
            words = text.split()
            lines = []
            current_line = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > width:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_len = len(word)
                else:
                    current_line.append(word)
                    current_len += len(word) + 1
            if current_line:
                lines.append(" ".join(current_line))
            return "\n".join(lines)

        with open(out_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                v_dur = seg.source_time_range[1] - seg.source_time_range[0]
                
                # We align subtitle to the actual video segment
                start_srt = format_ts(current_time)
                end_srt = format_ts(current_time + v_dur)
                
                wrapped_text = wrap_text(seg.text)
                
                f.write(f"{i+1}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{wrapped_text}\n\n")
                
                # Advance current_time by the total length of this segment (clip + spacer)
                current_time += v_dur + padding_s
                
        return out_path

    def _generate_video_spacer(self, duration: float, out_path: Path, width: int, height: int, fps: int = 30):
        """Generate a black video segment with target resolution/fps."""
        import ffmpeg
        try:
            (
                ffmpeg
                .input(f'color=c=black:s={width}x{height}:r={fps}', f='lavfi', t=duration)
                .output(str(out_path), vcodec='libx264', pix_fmt='yuv420p')
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        except Exception as e:
            logger.error(f"Failed to generate video spacer: {e}")
            raise FFmpegError(f"Video spacer generation failed: {e}")

    def _add_debug_annotation(
        self,
        clean_video_path: Path | str,
        annotated_video_path: Path | str,
        segments: list[dict],
        bar_height: int = 40,
    ) -> None:
        """Add top-bar debug annotation to video. Subtitle in original remains untouched."""
        # Build drawtext filters: one per segment, gated by enable='between(t,start,end)'
        drawtext_parts = []
        
        # Background bar (semi-transparent black)
        bg_filter = (
            f"drawbox=x=0:y=0:w=iw:h={bar_height}:"
            f"color=black@0.6:t=fill"
        )
        drawtext_parts.append(bg_filter)
        
        for seg in segments:
            action = seg["action"]
            score = seg["score"]
            group_id = seg["group_id"]
            
            # Color: retrieve = cyan-ish, generate = orange
            color = "0x4FC3F7" if action == "RETRIEVE" else "0xFFB74D"
            
            num_sents = seg["num_sentences"]
            text = f"[{action}] | Score\\: {score:.3f} | Group {group_id} ({num_sents} sents)"
            # Escape special chars for ffmpeg drawtext
            text_escaped = text.replace(":", "\\:").replace("'", "\\'")
            
            drawtext = (
                f"drawtext=text='{text_escaped}':"
                f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf:"
                f"fontsize=18:"
                f"fontcolor={color}:"
                f"x=20:y=10:"
                f"enable='between(t,{seg['start']:.3f},{seg['end']:.3f})'"
            )
            drawtext_parts.append(drawtext)
        
        vf = ",".join(drawtext_parts)
        
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(clean_video_path),
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "copy",  # passthrough audio
                    str(annotated_video_path),
                ],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg drawtext failed. stderr: {e.stderr}")
            raise FFmpegError(f"Failed to add debug annotation: {e.stderr}")
