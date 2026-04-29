import os
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple

from src.models.llm_wrapper import GroqBackend

logger = logging.getLogger(__name__)

class LLMJudge:
    """Orchestrator for LLM-as-judge evaluation using Groq."""
    
    PROMPT_TEMPLATE = """You are evaluating an AI-generated video summary.
ORIGINAL TRANSCRIPT (excerpt):
{transcript}
GENERATED SUMMARY SCRIPT:
{summary_script}
MATCHED VISUAL SCENES (described):
{matched_captions}
Rate on scale 1-5 (1=worst, 5=best):

INFORMATION_RETENTION: Does the summary capture the most important points?
FACTUAL_FAITHFULNESS: Are there any invented facts, numbers, or quotes?
VISUAL_RELEVANCE: Do the visuals match the narration semantically?

Output JSON: {{"information_retention": int, "factual_faithfulness": int, "visual_relevance": int, "reasoning": "string"}}
"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama-3.3-70b-versatile", backend=None):
        if backend is not None:
            self.backend = backend
            self.model_name = getattr(backend, "model_name", "Unknown")
        else:
            if not api_key:
                api_key = os.getenv("GROQ_API_KEY")
                
            if not api_key:
                logger.warning("GROQ_API_KEY not found in environment. LLM Judge will return placeholder scores.")
                self.backend = None
            else:
                from src.models.llm_wrapper import GroqBackend
                self.backend = GroqBackend(api_key=api_key, model_name=model_name)
                
            self.model_name = model_name

    def evaluate_video(self, transcript: str, summary_script: str, matched_captions: str) -> Dict[str, Any]:
        """Runs the evaluation with 3x retry logic."""
        system_prompt = "You are a helpful assistant that evaluates AI-generated video summaries. Respond ONLY with valid JSON."
        user_prompt = self.PROMPT_TEMPLATE.format(
            transcript=transcript[:8000], # Limit transcript size
            summary_script=summary_script,
            matched_captions=matched_captions
        )

        for attempt in range(3):
            try:
                if not self.backend:
                    return {
                        "information_retention": 3,
                        "factual_faithfulness": 3,
                        "visual_relevance": 3,
                        "reasoning": "No LLM backend available for evaluation."
                    }
                # Reuse GroqBackend.generate which already has retry logic for 429s
                # Use generic generate method which handles its own retries/exceptions
                raw_response = self.backend.generate(system_prompt, user_prompt)
                
                # Parse JSON
                # Some LLMs might wrap in markdown blocks
                if "```json" in raw_response:
                    raw_response = raw_response.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_response:
                    raw_response = raw_response.split("```")[1].split("```")[0].strip()
                
                data = json.loads(raw_response)
                # Validate keys
                required = ["information_retention", "factual_faithfulness", "visual_relevance"]
                if all(k in data for k in required):
                    return data
                else:
                    logger.warning(f"Attempt {attempt+1}: Missing keys in JSON response: {data}")
            except Exception as e:
                logger.error(f"Attempt {attempt+1}: LLM Judge failed: {e}")
                time.sleep(2)
        
        # Fallback if all retries fail
        return {
            "information_retention": 0,
            "factual_faithfulness": 0,
            "visual_relevance": 0,
            "reasoning": "Evaluation failed after 3 retries."
        }

    def get_cost_estimate(self, transcript: str, summary_script: str, matched_captions: str) -> float:
        # Groq is effectively free or very low cost for these quantities
        return 0.0
