import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.schemas import TranscriptSchema, SummaryScript, SummarySentence
from src.models.llm_wrapper import LLMBackend, GroqBackend, LocalBackend
from src.utils.io import load_json_as_model, save_model_as_json
from src.utils.text import clean_for_tts

logger = logging.getLogger(__name__)

STYLE_GUIDELINES = {
    "informative": "Be objective and clear. Focus on balanced summaries of the key points. Use a standard reporting tone.",
    "hook-driven": "Be dramatic and punchy. Start sentences with surprising claims. Use short, high-energy words suitable for social media.",
    "educational": "Be explanatory and logical. Use a teaching tone. Focus on the 'how' and 'why'. Include clear transitions between ideas."
}

SYSTEM_PROMPT = """You are a master scriptwriter. Your task is to output a single, valid JSON object containing a summarized video script.
STYLE: {style_description}

RULES:
1. JSON ONLY: No markdown formatting, no conversational filler.
2. HOOK: Start with a power statement. Never say "Starting video" or "In this clip".
3. TONE: {style_description}
4. SPELL NUMBERS: Say "seven minutes" not "7m".
5. LENGTH: Aim for UP TO {target_duration} seconds, but PRIORITIZE distinct, non-repetitive content over reaching this target. If the source does not contain enough distinct ideas, produce FEWER sentences. It is better to be shorter than to repeat or pad.
6. CONTENT ONLY: Summarize the MAIN TOPIC and FACTUAL CLAIMS of the video. Do NOT summarize vocabulary definitions, word explanations, pronunciation guides, or language teaching segments. Focus on WHAT the video is about, not HOW it teaches.
7. NO REPETITION: Each sentence must cover a DIFFERENT aspect or subtopic. Never rephrase the same point in multiple sentences.
7a. GRANULARITY: Aim for 7-10 seconds per sentence, each covering ONE specific concept, claim, or detail (do not pack multiple ideas into one sentence). The number of sentences MUST match the amount of distinct content in the source, NOT a fixed target. NEVER repeat or rephrase a point you have already made in order to reach the target length.
8. VISUAL KEYWORDS: The "keywords" field must contain VISUAL descriptions of what might appear on screen during that part of the video. Think: what would a viewer SEE? Use concrete nouns (e.g., "bar chart", "person running", "close-up of chip") not abstract concepts (e.g., "performance", "health", "innovation").
9. TIMESTAMP ACCURACY: The "source_timestamp_hint" must match the approximate time range in the transcript where the information for that sentence originally appears. Use the timestamps provided in the transcript. This is critical for visual matching.

SCHEMA:
{schema_json}"""

COMBINE_PROMPT = """You are a lead editor. Merge these partial summaries into one fluid final script.
STYLE: {style_description}

RULES:
1. JSON ONLY: No introductory text.
2. FLOW: Ensure sentences transition logically without repetition.
3. LENGTH: Aim for up to {target_duration} seconds total, but do NOT repeat or pad to reach it — drop redundant sentences and keep only distinct points.

SCHEMA:
{schema_json}"""

FEW_SHOT_EXAMPLE = """[Example input/output for format reference only]
Input: [00:15] Sleep quality is vital for brain function. [00:22] Most people need eight hours but quality wins. [00:38] Deep sleep is when memory consolidation happens. [00:55] REM sleep helps emotional regulation. [01:05] A new study from Harvard shows napping can reduce cortisol levels. [01:18] Cortisol is a stress hormone produced by the adrenal glands. [01:30] The word 'rejuvenate' means to restore energy. [01:45] Naps as short as 20 minutes already show measurable benefits. [02:00] Longer naps over 60 minutes may cause grogginess. [02:10] Brain scans revealed that nappers had larger hippocampal volume.
Output:
{
  "video_id": "demo",
  "target_duration": 90,
  "style": "informative",
  "backend_used": "local",
  "sentences": [
    {
      "id": 0,
      "text": "Sleep quality matters more than total hours, with new research highlighting why.",
      "estimated_duration_seconds": 8.0,
      "source_timestamp_hint": [15.0, 22.0],
      "keywords": ["person sleeping in bed", "alarm clock"]
    },
    {
      "id": 1,
      "text": "Deep sleep is the phase where the brain consolidates daily memories.",
      "estimated_duration_seconds": 7.0,
      "source_timestamp_hint": [38.0, 55.0],
      "keywords": ["brain waves graph", "memory neurons firing"]
    },
    {
      "id": 2,
      "text": "REM sleep plays a separate role in regulating emotions and mood.",
      "estimated_duration_seconds": 7.0,
      "source_timestamp_hint": [55.0, 65.0],
      "keywords": ["REM eye movement closeup", "mood chart"]
    },
    {
      "id": 3,
      "text": "Harvard researchers discovered that short naps significantly lower cortisol levels.",
      "estimated_duration_seconds": 8.0,
      "source_timestamp_hint": [65.0, 78.0],
      "keywords": ["Harvard logo", "scientist in lab"]
    },
    {
      "id": 4,
      "text": "Cortisol is the body's primary stress hormone, released by the adrenal glands.",
      "estimated_duration_seconds": 7.0,
      "source_timestamp_hint": [78.0, 90.0],
      "keywords": ["adrenal gland diagram", "cortisol molecule"]
    },
    {
      "id": 5,
      "text": "Even a twenty-minute nap can produce measurable cognitive benefits.",
      "estimated_duration_seconds": 7.0,
      "source_timestamp_hint": [105.0, 115.0],
      "keywords": ["20 minute timer", "person napping briefly"]
    },
    {
      "id": 6,
      "text": "However, naps longer than an hour often cause sleep inertia and grogginess.",
      "estimated_duration_seconds": 7.5,
      "source_timestamp_hint": [120.0, 130.0],
      "keywords": ["tired person yawning", "warning sign"]
    },
    {
      "id": 7,
      "text": "Brain scans of habitual nappers reveal a larger hippocampus linked to memory.",
      "estimated_duration_seconds": 7.5,
      "source_timestamp_hint": [130.0, 150.0],
      "keywords": ["MRI brain scan", "hippocampus diagram"]
    }
  ]
}
Note: The input segment about the word 'rejuvenate' was intentionally excluded because it is a vocabulary explanation, not factual content about the topic."""

class Phase2Summarizer:
    def __init__(self, backend: LLMBackend, config: Dict[str, Any]):
        self.backend = backend
        self.config = config

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Robustly extract JSON from LLM response, handling markdown fences and surrounding text."""
        # Remove whitespace
        response = response.strip()
        
        # Try finding markdown code block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            
        if json_match:
            content = json_match.group(1)
        else:
            # Fallback: find the first '{' and last '}'
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                content = response[start:end+1]
            else:
                content = response
        
        # Cleanup: sometimes LLMs add comments or trailing commas
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',\s*([}\]])', r'\1', content)
        
        # FIX: Replace hallucinated unquoted strings in 'id' field (like 'bk') with 0
        content = re.sub(r'"id":\s*[a-zA-Z]+\s*,', r'"id": 0,', content)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            with open('/tmp/raw_json_dump.txt', 'w') as f:
                f.write(content)
            logger.error(f"JSON Parsing Error at {e.lineno}:{e.colno}. Attempting secondary cleanup...")
            try:
                # Remove common non-json prefixes/suffixes that might have survived
                content = re.sub(r'^[^{]*', '', content)
                content = re.sub(r'[^}]*$', '', content)
                return json.loads(content)
            except:
                logger.error(f"Failed to parse JSON. Raw content preview: {content[:300]}...")
                raise e

    def _chunk_transcript(self, transcript: TranscriptSchema) -> List[str]:
        """Split transcript into compact text chunks if too large."""
        # Window size 5 min (300s) for more aggressive chunking, overlap 30s
        window_size = 300
        overlap = 30
        
        full_text = " ".join([s.text for s in transcript.segments])
        
        def format_compact(segments):
            return " | ".join([f"[{s['start']}] {s['text']}" for s in segments])

        # If less than 30k chars (~7.5k tokens), skip chunking but still use compact format
        if len(full_text) < 30000:
            return [format_compact([{"start": s.start, "text": s.text} for s in transcript.segments])]

        chunks = []
        current_start = 0
        while current_start < transcript.duration_seconds:
            current_end = current_start + window_size
            chunk_segments = [
                {"start": s.start, "text": s.text} 
                for s in transcript.segments 
                if s.start >= current_start and s.start < current_end
            ]
            if chunk_segments:
                chunks.append(format_compact(chunk_segments))
            
            if current_end >= transcript.duration_seconds:
                break
            current_start += (window_size - overlap)
            
        return chunks

    def run(self, transcript_path: Path, target_duration: int = 90, style: str = "informative", progress_callback: Any = None) -> Path:
        transcript = load_json_as_model(transcript_path, TranscriptSchema)
        video_id = transcript.video_id
        
        chunks = self._chunk_transcript(transcript)
        schema_json = SummaryScript.model_json_schema()
        style_desc = STYLE_GUIDELINES.get(style, STYLE_GUIDELINES["informative"])
        sys_prompt = SYSTEM_PROMPT.format(
            target_duration=target_duration, 
            schema_json=json.dumps(schema_json),
            style_description=style_desc
        )

        if len(chunks) == 1:
            # Single pass
            logger.info("Running single-pass summarization.")
            if progress_callback:
                progress_callback.update(2, "Summarization", 20, "Analyzing transcript")
            
            user_prompt = f"{FEW_SHOT_EXAMPLE}\n\nVIDEO_ID: {video_id}\nSTYLE: {style}\n\nTRANSCRIPT:\n"
            user_prompt += chunks[0]
            
            if progress_callback:
                progress_callback.update(2, "Summarization", 50, "Generating script via LLM")
                
            summary_data = self._generate_with_retry(sys_prompt, user_prompt)
        else:
            # Multi-pass
            logger.info(f"Running multi-pass summarization for {len(chunks)} chunks.")
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                pct = int(10 + (i / len(chunks)) * 60)
                if progress_callback:
                    progress_callback.update(2, "Summarization", pct, f"Summarizing chunk {i+1}/{len(chunks)}")
                
                logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
                chunk_user_prompt = f"CHUNK {i+1}/{len(chunks)} TRANSCRIPT:\n" + chunk
                chunk_summary = self._generate_with_retry(sys_prompt, chunk_user_prompt)
                chunk_summaries.append(chunk_summary)

            # Final merge
            logger.info("Combining chunk summaries into final script...")
            if progress_callback:
                progress_callback.update(2, "Summarization", 80, "Merging chunk summaries")
                
            final_user_prompt = f"{FEW_SHOT_EXAMPLE}\n\nCombine these partial summaries:\n" + json.dumps(chunk_summaries, indent=1)
            final_sys_prompt = COMBINE_PROMPT.format(
                target_duration=target_duration, 
                schema_json=json.dumps(schema_json),
                style_description=style_desc
            )
            summary_data = self._generate_with_retry(final_sys_prompt, final_user_prompt)

        # Ensure metadata is correct
        summary_data["video_id"] = video_id
        summary_data["target_duration"] = target_duration
        summary_data["style"] = style
        summary_data["backend_used"] = "groq" if isinstance(self.backend, GroqBackend) else "local"
        
        # Fix and reindex sequential IDs in case of hallucinated IDs
        for idx, sentence_data in enumerate(summary_data.get("sentences", [])):
            sentence_data["id"] = idx

        # Post-process for TTS cleaning
        for sentence_data in summary_data["sentences"]:
            sentence_data["text"] = clean_for_tts(sentence_data["text"])

        # Safe output path
        output_path = transcript_path.parent / "summary_script.json"
        save_model_as_json(SummaryScript(**summary_data), output_path)
        logger.info(f"Summary script generated at: {output_path}")
        
        if progress_callback:
            progress_callback.update(2, "Summarization", 100, "Phase 2 complete")
            
        return output_path


    def _generate_with_retry(self, system_prompt: str, user_prompt: str, retries: int = 3) -> Dict[str, Any]:
        last_error = None
        for attempt in range(retries):
            try:
                response = self.backend.generate(
                    system_prompt, user_prompt,
                    repetition_penalty=self.config.get("repetition_penalty", 1.1)
                )
                data = self._extract_json(response)
                # Minimal validation here, full validation at the end of run()
                if "sentences" not in data:
                    raise ValueError("JSON missing 'sentences' key")
                return data
            except Exception as e:
                logger.warning(f"Generation attempt {attempt+1} failed: {e}")
                last_error = e
        
        raise last_error
