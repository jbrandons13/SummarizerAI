import json
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any
import soundfile as sf
import numpy as np
import nltk
import math

def merge_small_chunks(chunks, min_dur):
    """Merge any chunk < min_dur with its smaller neighbor."""
    if len(chunks) <= 1:
        return chunks
    
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(chunks):
            if c["dur"] < min_dur:
                # merge with smaller neighbor
                if i == 0:
                    neighbor_idx = 1
                elif i == len(chunks) - 1:
                    neighbor_idx = i - 1
                else:
                    left_dur = chunks[i-1]["dur"]
                    right_dur = chunks[i+1]["dur"]
                    neighbor_idx = i-1 if left_dur <= right_dur else i+1
                
                # merge c into neighbor
                merge_target = chunks[neighbor_idx]
                merge_target["text"] = (
                    c["text"] + " " + merge_target["text"]
                    if neighbor_idx > i
                    else merge_target["text"] + " " + c["text"]
                )
                merge_target["dur"] += c["dur"]
                merge_target["char_ratio"] += c["char_ratio"]
                chunks.pop(i)
                changed = True
                break
    return chunks

def balanced_word_split(text, total_duration, max_dur, min_dur):
    """
    Split text into N balanced chunks where N is minimum needed
    to keep each chunk <= max_dur.
    """
    # Step 1: minimum number of chunks needed
    n_chunks = math.ceil(total_duration / max_dur)
    
    # Step 2: target duration per chunk
    target_dur = total_duration / n_chunks
    
    # Step 3: split words proportionally
    words = text.split()
    total_chars = len(text)
    
    chunks = []
    words_per_chunk = len(words) / n_chunks  # float
    
    for i in range(n_chunks):
        start_word = int(round(i * words_per_chunk))
        end_word = int(round((i + 1) * words_per_chunk))
        chunk_words = words[start_word:end_word]
        chunk_text = " ".join(chunk_words)
        
        # Use char ratio for precise duration estimate
        chunk_chars = len(chunk_text)
        char_ratio = chunk_chars / total_chars if total_chars > 0 else 1.0 / n_chunks
        chunk_dur = total_duration * char_ratio
        
        chunks.append({
            "text": chunk_text,
            "dur": chunk_dur,
            "char_ratio": char_ratio,
        })
    
    # Step 4: edge-case merge (kalau ada chunk < min_dur)
    chunks = merge_small_chunks(chunks, min_dur)
    
    return chunks

def split_long_segment(text: str, dur: float, max_dur: float, min_dur: float) -> List[Dict[str, Any]]:
    sentences = nltk.sent_tokenize(text)
    total_chars = max(len(text), 1)
    chunks = []
    
    curr_sentences = []
    curr_chars = 0
    curr_start_char = 0
    
    for i, s in enumerate(sentences):
        s_len = len(s)
        space_len = 1 if i < len(sentences) - 1 else 0
        s_dur = dur * (s_len / total_chars)
        
        if s_dur > max_dur:
            if curr_sentences:
                chunks.append({
                    "text": " ".join(curr_sentences),
                    "start_ratio": curr_start_char / total_chars,
                    "end_ratio": (curr_start_char + curr_chars) / total_chars,
                    "dur": dur * (curr_chars / total_chars)
                })
                curr_start_char += curr_chars
                curr_sentences = []
                curr_chars = 0
                
            logging.warning(f"Sentence too long ({s_dur:.2f}s > {max_dur}s). Fallback to balanced word split.")
            
            b_chunks = balanced_word_split(s, s_dur, max_dur, min_dur)
            
            # Map chunk char ratios to the whole text
            w_start_ratio = curr_start_char / total_chars
            for bc in b_chunks:
                # rel_ratio is the ratio of this chunk's characters to the WHOLE text
                rel_ratio = bc["char_ratio"] * (s_len / total_chars)
                chunks.append({
                    "text": bc["text"],
                    "start_ratio": w_start_ratio,
                    "end_ratio": w_start_ratio + rel_ratio,
                    "dur": bc["dur"]
                })
                w_start_ratio += rel_ratio
                
            curr_start_char += s_len + space_len
            
        else:
            if dur * ((curr_chars + s_len) / total_chars) > max_dur and curr_sentences:
                chunks.append({
                    "text": " ".join(curr_sentences),
                    "start_ratio": curr_start_char / total_chars,
                    "end_ratio": (curr_start_char + curr_chars) / total_chars,
                    "dur": dur * (curr_chars / total_chars)
                })
                curr_start_char += curr_chars
                curr_sentences = [s]
                curr_chars = s_len + space_len
            else:
                curr_sentences.append(s)
                curr_chars += s_len + space_len
                
    if curr_sentences:
        chunks.append({
            "text": " ".join(curr_sentences),
            "start_ratio": curr_start_char / total_chars,
            "end_ratio": 1.0,
            "dur": dur * (1.0 - (curr_start_char / total_chars))
        })
        
    return chunks

def run_segmenter(
    video_id: str, 
    min_dur: float = 2.5, 
    max_dur: float = 6.0, 
    base_dir: str = "data/intermediate"
) -> str:
    """
    Run the shot segmenter for Phase 4.
    
    Reads Phase 2 summary_script.json and Phase 3 audio_manifest.json.
    Outputs shots.json containing 'shot units'.
    Handles audio concatenation and splitting using soundfile.
    """
    video_dir = Path(base_dir) / video_id
    script_path = video_dir / "summary_script.json"
    audio_manifest_path = video_dir / "audio_manifest.json"
    
    if not script_path.exists() or not audio_manifest_path.exists():
        raise FileNotFoundError(f"Missing input files in {video_dir}")
        
    with open(script_path, "r", encoding="utf-8") as f:
        script_data = json.load(f)
        
    with open(audio_manifest_path, "r", encoding="utf-8") as f:
        audio_data = json.load(f)
        
    audio_lookup = {}
    for seg in audio_data.get("sentences", []):
        seg_id = str(seg["id"])
        audio_lookup[seg_id] = {
            "duration_sec": seg["duration_seconds"],
            "audio_path": str(video_dir / seg["audio_path"] if not Path(seg["audio_path"]).is_absolute() else seg["audio_path"])
        }
        
    segments = script_data.get("sentences", [])
    
    out_dir = video_dir / "phase4"
    audio_out_dir = out_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    
    shots = []
    buffer = None
    
    def process_and_save_audio(audio_paths: List[str], shot_id: str, start_ratio: float = 0.0, end_ratio: float = 1.0) -> str:
        out_path = audio_out_dir / f"{shot_id}.wav"
        
        # Merge case
        if len(audio_paths) > 1:
            data = []
            sr = None
            for p in audio_paths:
                d, r = sf.read(p)
                if len(d.shape) == 1:
                    d = np.expand_dims(d, axis=1) # force 2D
                data.append(d)
                sr = r
            merged_data = np.concatenate(data, axis=0)
            sf.write(out_path, merged_data, sr)
            return str(out_path)
            
        # Split case
        if (start_ratio > 0.0 or end_ratio < 1.0) and len(audio_paths) == 1:
            d, sr = sf.read(audio_paths[0])
            if len(d.shape) == 1:
                d = np.expand_dims(d, axis=1)
            start_idx = int(len(d) * start_ratio)
            end_idx = int(len(d) * end_ratio)
            slice_data = d[start_idx:end_idx]
            sf.write(out_path, slice_data, sr)
            return str(out_path)
            
        # Normal case
        if len(audio_paths) == 1:
            shutil.copy2(audio_paths[0], out_path)
            return str(out_path)
            
        return ""

    for seg in segments:
        seg_id = str(seg["id"])
        if seg_id not in audio_lookup:
            continue
            
        dur = audio_lookup[seg_id]["duration_sec"]
        text = seg["text"]
        audio_path = audio_lookup[seg_id]["audio_path"]
        
        if buffer is None:
            buffer = {"text": text, "dur": dur, "src_ids": [seg_id], "audio_paths": [audio_path]}
        else:
            if buffer["dur"] < min_dur and (buffer["dur"] + dur) <= max_dur:
                buffer["text"] += " " + text
                buffer["dur"] += dur
                buffer["src_ids"].append(seg_id)
                buffer["audio_paths"].append(audio_path)
            else:
                shots.append(buffer)
                buffer = {"text": text, "dur": dur, "src_ids": [seg_id], "audio_paths": [audio_path]}
                
        # Split logic using nltk + recursive word-split fallback
        if buffer["dur"] > max_dur:
            split_chunks = split_long_segment(buffer["text"], buffer["dur"], max_dur, min_dur)
            for chunk in split_chunks:
                shots.append({
                    "text": chunk["text"],
                    "dur": chunk["dur"],
                    "src_ids": buffer["src_ids"].copy(),
                    "audio_paths": buffer["audio_paths"].copy(),
                    "start_ratio": chunk["start_ratio"],
                    "end_ratio": chunk["end_ratio"]
                })
            buffer = None

    if buffer:
        shots.append(buffer)
        
    formatted_shots = []
    for i, s in enumerate(shots):
        shot_id = f"shot_{i+1:03d}"
        
        audio_out_path = process_and_save_audio(
            s["audio_paths"], 
            shot_id, 
            start_ratio=s.get("start_ratio", 0.0), 
            end_ratio=s.get("end_ratio", 1.0)
        )
        
        formatted_shots.append({
            "shot_id": shot_id,
            "text": s["text"],
            "duration_sec": s["dur"],
            "source_segment_ids": s["src_ids"],
            "audio_path": str(Path(audio_out_path).relative_to(video_dir)) if audio_out_path else ""
        })
        
    out_file = out_dir / "shots.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "shots": formatted_shots
        }, f, indent=2, ensure_ascii=False)
        
    return str(out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--min-dur", type=float, default=2.5)
    parser.add_argument("--max-dur", type=float, default=6.0)
    args = parser.parse_args()
    
    out = run_segmenter(args.video_id, args.min_dur, args.max_dur)
    print(f"Segmenter finished. Output saved to {out}")
