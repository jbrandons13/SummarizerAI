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

SYSTEM_PROMPT = """You are a professional video narrator who writes engaging voiceover scripts from transcripts.

Your job: read a transcript, extract the most valuable insights, and rewrite them as a compelling short narration that makes people want to watch.

WRITING RULES:
1. Start with a HOOK — a bold claim or surprising fact. Never start with "In this video..."
2. Sentences: Write 10-16 words per sentence. Use active voice.
3. TTS optimization: Minimize commas. No dashes, semicolons, or ellipses. Use periods.
4. Spell out numbers below 100 as words: "forty percent" not "40%".
5. Metadata: source_timestamp_hint [start, end] must be accurate. Include 3-5 visual keywords.

Target: {target_duration} seconds of spoken audio.

{schema_json}"""

COMBINE_PROMPT = """You are a script editor. Combine partial summaries into one cohesive narrative.
RULES:
1. Remove duplicates. 
2. Write a strong hook and ending.
3. Smooth transitions between chunks.
4. Maintain all TTS optimization rules (no commas between clauses, spell out numbers).

Target: {target_duration} seconds total.

{schema_json}"""

FEW_SHOT_EXAMPLE = """EXAMPLE INPUT (partial transcript):
[00:15] Okay so here's the thing. Most people think they need eight hours of sleep. [00:22] But our research at Stanford shows that sleep quality matters more than quantity.

EXAMPLE OUTPUT:
{
  "sentences": [
    {
      "id": 0,
      "text": "Most people believe eight hours of sleep is the magic number. New research says they are wrong.",
      "estimated_duration_seconds": 4.5,
      "source_timestamp_hint": [15.0, 22.0],
      "keywords": ["speaker on stage", "sleep diagram", "audience"]
    }
  ]
}"""

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
        sys_prompt = SYSTEM_PROMPT.format(target_duration=target_duration, schema_json=json.dumps(schema_json))

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
            final_sys_prompt = COMBINE_PROMPT.format(target_duration=target_duration, schema_json=json.dumps(schema_json))
            summary_data = self._generate_with_retry(final_sys_prompt, final_user_prompt)

        # Ensure metadata is correct
        summary_data["video_id"] = video_id
        summary_data["target_duration"] = target_duration
        summary_data["style"] = style
        summary_data["backend_used"] = "groq" if isinstance(self.backend, GroqBackend) else "local"

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
