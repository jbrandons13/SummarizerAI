import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.schemas import TranscriptSchema, SummaryScript, SummarySentence
from src.models.llm_wrapper import LLMBackend, GroqBackend, LocalBackend
from src.utils.io import load_json_as_model, save_model_as_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a video summarization script writer. Given a podcast/interview transcript with timestamps, write a concise voiceover script that captures the most important points.
RULES:

Output valid JSON only, no markdown fences.
Write SHORT sentences (8-18 words each) suitable for TTS narration.
Each sentence must be semantically self-contained (no "this", "that" without referent).
Target total spoken duration: {target_duration} seconds (assume 150 words/min).
For each sentence, provide "source_timestamp_hint" [start, end] pointing to the transcript range summarized.
Preserve speaker's claims faithfully. DO NOT invent facts, numbers, or quotes.
Write in the same language as the transcript.
Include 3-5 "keywords" per sentence for downstream retrieval.

OUTPUT SCHEMA: {schema_json}"""

COMBINE_PROMPT = """You are a final script editor. You have several partial summary scripts from different parts of a long video. 
Combine them into ONE COHESIVE script that fits the {target_duration}s time limit.
Ensure a smooth narrative flow.
Target total spoken duration: {target_duration} seconds.

RULES:
Output valid JSON only, no markdown fences.
Keep sentences short (8-18 words).
Maintain source_timestamp_hint accurately for each sentence.
Include 3-5 keywords per sentence.

OUTPUT SCHEMA: {schema_json}"""

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
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error. Raw content attempted: {content[:200]}...")
            raise e

    def _chunk_transcript(self, transcript: TranscriptSchema) -> List[List[Dict[str, Any]]]:
        # Window size 10 min (600s), overlap 1 min (60s)
        window_size = 600
        overlap = 60
        
        chunks = []
        current_start = 0
        
        # Basic character-based estimation for "tokens"
        full_text = " ".join([s.text for s in transcript.segments])
        # If less than 100k chars (~25k tokens), skip chunking
        if len(full_text) < 100000:
            return [[s.model_dump() for s in transcript.segments]]

        while current_start < transcript.duration_seconds:
            current_end = current_start + window_size
            chunk_segments = [
                {"start": s.start, "end": s.end, "text": s.text} 
                for s in transcript.segments 
                if s.start >= current_start and s.start < current_end
            ]
            if chunk_segments:
                chunks.append(chunk_segments)
            
            if current_end >= transcript.duration_seconds:
                break
            current_start += (window_size - overlap)
            
        return chunks

    def run(self, transcript_path: Path, target_duration: int = 90, style: str = "informative") -> Path:
        transcript = load_json_as_model(transcript_path, TranscriptSchema)
        video_id = transcript.video_id
        
        chunks = self._chunk_transcript(transcript)
        schema_json = SummaryScript.model_json_schema()
        sys_prompt = SYSTEM_PROMPT.format(target_duration=target_duration, schema_json=json.dumps(schema_json))

        if len(chunks) == 1:
            # Single pass
            logger.info("Running single-pass summarization.")
            user_prompt = f"VIDEO_ID: {video_id}\nSTYLE: {style}\n\nTRANSCRIPT:\n"
            user_prompt += json.dumps(chunks[0], indent=1)
            summary_data = self._generate_with_retry(sys_prompt, user_prompt)
        else:
            # Multi-pass
            logger.info(f"Running multi-pass summarization for {len(chunks)} chunks.")
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
                chunk_user_prompt = f"CHUNK {i+1}/{len(chunks)} TRANSCRIPT:\n" + json.dumps(chunk, indent=1)
                # For intermediate chunks, we use a slightly relaxed prompt or just same schema
                chunk_summary = self._generate_with_retry(sys_prompt, chunk_user_prompt)
                chunk_summaries.append(chunk_summary)

            # Final merge
            logger.info("Combining chunk summaries into final script...")
            final_user_prompt = "PARTIAL SUMMARIES:\n" + json.dumps(chunk_summaries, indent=1)
            final_sys_prompt = COMBINE_PROMPT.format(target_duration=target_duration, schema_json=json.dumps(schema_json))
            summary_data = self._generate_with_retry(final_sys_prompt, final_user_prompt)

        # Ensure metadata is correct
        summary_data["video_id"] = video_id
        summary_data["target_duration"] = target_duration
        summary_data["style"] = style
        summary_data["backend_used"] = "groq" if isinstance(self.backend, GroqBackend) else "local"

        # Safe output path
        output_path = transcript_path.parent / "summary_script.json"
        save_model_as_json(SummaryScript(**summary_data), output_path)
        logger.info(f"Summary script generated at: {output_path}")
        
        return output_path

    def _generate_with_retry(self, system_prompt: str, user_prompt: str, retries: int = 3) -> Dict[str, Any]:
        last_error = None
        for attempt in range(retries):
            try:
                response = self.backend.generate(system_prompt, user_prompt)
                data = self._extract_json(response)
                # Minimal validation here, full validation at the end of run()
                if "sentences" not in data:
                    raise ValueError("JSON missing 'sentences' key")
                return data
            except Exception as e:
                logger.warning(f"Generation attempt {attempt+1} failed: {e}")
                last_error = e
        
        raise last_error
