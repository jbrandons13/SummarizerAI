import json
import logging
import re
import math
from pathlib import Path
from typing import Dict, Any, List

from src.models.llm_wrapper import LocalBackend
from src.utils.vram import VRAMManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a video director and storyboard artist. For the provided list of video shots, you must generate a visual storyboard JSON array.

RULES:
1. OUTPUT JSON ARRAY ONLY: No markdown formatting outside of the json block, no conversational filler. Return a valid JSON array of objects.
2. For each input shot, output EXACTLY ONE object in the array.
3. visual_description: 1-2 sentences describing the action or visuals on screen.
4. image_prompt: A text-to-image prompt. It MUST end EXACTLY with the string: "flat 2D educational vector scene, vivid colors"
5. key_entities: Array of 3-7 concrete nouns/concepts visible.
6. topic_tag: A snake_case identifier for the core topic. REUSE the EXACT same topic_tag across consecutive shots if they discuss the same overarching concept.
7. TAG GRANULARITY: topic_tag should be at a CONCEPTUAL level, not hyper-specific. Prefer broader tags that can group multiple consecutive shots. Examples:
   - GOOD: "rock_formation", "weathering", "rock_cycle_overview"
   - BAD (too specific): "sedimentary_rock_formation", "metamorphic_rock_at_los_cuernos", "weathering_in_winter"
   Aim for 6-12 unique topic_tags across a typical video, not 20+.
8. SCENES NOT DIAGRAMS: The image_prompt MUST describe a living, continuous scene (e.g., "A circular landscape scene of the rock cycle", "Cross-section view of the earth"). Do NOT describe infographics, diagrams, or UI panels.
9. NO TEXT OR GIBBERISH: Never request text, letters, captions, labels, numbers, signatures, or watermarks. The scene must convey the information entirely through natural visual composition.
10. SPLIT SHOTS: Some shots are marked "SPLIT: part X of N from the same sentence". These shots come from ONE sentence and share ONE concept, but they MUST be visually distinct -- use different framing (a wide establishing view vs a close-up detail) or sequential stages of the same process. They MUST keep the SAME topic_tag. Never output near-identical image_prompts for shots from the same sentence.

JSON SCHEMA PER SHOT:
{
  "shot_id": "string",
  "visual_description": "string",
  "image_prompt": "string",
  "key_entities": ["string", "string"],
  "topic_tag": "string"
}
"""

def extract_json(response: str) -> List[Dict[str, Any]]:
    response = response.strip()
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        
    if json_match:
        content = json_match.group(1)
    else:
        start = response.find('[')
        end = response.rfind(']')
        if start != -1 and end != -1:
            content = response[start:end+1]
        else:
            content = response
            
    content = re.sub(r',\s*([}\]])', r'\1', content)
    return json.loads(content)

def compute_split_hints(shots: List[Dict[str, Any]]) -> Dict[str, str]:
    """Consecutive shots sharing the SAME source_segment_ids are halves of one
    split sentence. Return shot_id -> a directive telling the storyboard LLM to
    make them visually distinct (wide vs close-up), so the split adds variety
    instead of near-duplicate frames. Shots with no sibling get no hint
    (behaviour unchanged)."""
    hints: Dict[str, str] = {}
    i, n = 0, len(shots)
    while i < n:
        key = tuple(shots[i].get("source_segment_ids", []))
        j = i
        while key and j + 1 < n and tuple(shots[j + 1].get("source_segment_ids", [])) == key:
            j += 1
        group = shots[i:j + 1]
        if len(group) > 1:
            total = len(group)
            for k, sh in enumerate(group):
                part = k + 1
                if part == 1:
                    role = "the WIDE, establishing view of this concept"
                elif part == total:
                    role = "a CLOSE-UP / detail of the SAME concept, clearly different framing from the earlier part, same setting and style"
                else:
                    role = "a different stage or angle of the SAME concept, distinct from the other parts, same setting and style"
                hints[sh["shot_id"]] = (
                    f"SPLIT: part {part} of {total} from the same sentence -- render {role}. "
                    "Keep the same topic_tag as the sibling part(s)."
                )
        i = j + 1
    return hints

def run_storyboard(video_id: str, base_dir: str = "data/intermediate") -> tuple[str, int, Dict[str, int]]:
    video_dir = Path(base_dir) / video_id
    shots_path = video_dir / "phase4" / "shots.json"
    summary_path = video_dir / "summary_script.json"
    out_path = video_dir / "phase4" / "storyboard.json"
    
    with open(shots_path, "r", encoding="utf-8") as f:
        shots_data = json.load(f)["shots"]

    split_hints = compute_split_hints(shots_data)
        
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)["sentences"]
        
    summary_lookup = {str(s["id"]): s["keywords"] for s in summary_data}
    
    chunk_size = 10
    n_chunks = math.ceil(len(shots_data) / chunk_size)
    
    vram_manager = VRAMManager()
    backend = LocalBackend(vram_manager, "Qwen/Qwen2.5-14B-Instruct-AWQ")
    
    all_storyboards = []
    fallback_count = 0
    previous_topic_tag = None
    
    for i in range(n_chunks):
        chunk_shots = shots_data[i*chunk_size:(i+1)*chunk_size]
        
        user_prompt = "Generate the storyboard for the following shots:\n\n"
        if previous_topic_tag:
            user_prompt += f"(Hint: The previous shot ended with topic_tag '{previous_topic_tag}'. Reuse it if the topic continues.)\n\n"
            
        for shot in chunk_shots:
            kws = []
            for sid in shot["source_segment_ids"]:
                kws.extend(summary_lookup.get(str(sid), []))
            kws = list(set(kws))
            
            user_prompt += f"SHOT ID: {shot['shot_id']}\n"
            user_prompt += f"TEXT: {shot['text']}\n"
            user_prompt += f"KEYWORDS: {', '.join(kws)}\n"
            if split_hints.get(shot['shot_id']):
                user_prompt += split_hints[shot['shot_id']] + "\n"
            user_prompt += "\n"
            
        try:
            logger.info(f"Generating storyboard for chunk {i+1}/{n_chunks}...")
            # retry logic
            success = False
            for attempt in range(3):
                try:
                    response = backend.generate(SYSTEM_PROMPT, user_prompt)
                    chunk_data = extract_json(response)
                    
                    if not isinstance(chunk_data, list):
                        raise ValueError("LLM did not return a JSON array.")
                        
                    # Basic validation
                    for item in chunk_data:
                        expected_end = "flat 2D educational vector scene, vivid colors"
                        if not item.get("image_prompt", "").endswith(expected_end):
                            item["image_prompt"] = item.get("image_prompt", "").rstrip(", ") + ", " + expected_end
                            
                        all_storyboards.append(item)
                        previous_topic_tag = item.get("topic_tag")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed: {e}")
            
            if not success:
                raise ValueError("All attempts failed.")
                
        except Exception as e:
            logger.error(f"Failed to process chunk {i}. Error: {e}")
            fallback_count += len(chunk_shots)
            for shot in chunk_shots:
                all_storyboards.append({
                    "shot_id": shot["shot_id"],
                    "visual_description": "Fallback visual description.",
                    "image_prompt": f"A generic illustration for {shot['text'][:30]}..., flat 2D educational vector scene, vivid colors",
                    "key_entities": ["fallback"],
                    "topic_tag": previous_topic_tag or "fallback_topic"
                })
                
    # Unload handled by backend.generate
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "shots": all_storyboards
        }, f, indent=2, ensure_ascii=False)
        
    # calculate tags
    tag_counts = {}
    for item in all_storyboards:
        tag = item.get("topic_tag", "unknown")
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
    return str(out_path), fallback_count, tag_counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    out_path, fallbacks, tags = run_storyboard(args.video_id)
    print(f"Storyboard saved to {out_path}")
    print(f"Fallbacks: {fallbacks}")
    print(f"Tags: {tags}")