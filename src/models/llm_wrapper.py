from abc import ABC, abstractmethod
import time
import json
import logging
from typing import Dict, Any, List, Optional
import os

try:
    import groq
except ImportError:
    groq = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.vram import VRAMManager

logger = logging.getLogger(__name__)

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response from LLM."""
        pass

class LocalBackend(LLMBackend):
    def __init__(self, vram_manager: VRAMManager, model_name: str):
        self.vram = vram_manager
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        def loader():
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            # AWQ model through transformers often needs specific config if not auto-detected
            # We'll try auto first since 'device_map' is used.
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
            return model, tokenizer

        model, tokenizer = self.vram.load_model(f"LocalLLM ({self.model_name})", loader)
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.model is None:
            self._load_model()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=False
        )
        
        # Extract only the new tokens
        input_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        prompt_tokens = input_len
        completion_tokens = len(response_ids)
        logger.info(f"LocalLLM ({self.model_name}) usage: {prompt_tokens} prompt, {completion_tokens} completion tokens.")
        
        # Unload after generation as per requirements
        self.vram.load_model("None (Cleanup)", lambda: None)
        self.model = None
        self.tokenizer = None
        
        return response

class GroqBackend(LLMBackend):
    def __init__(self, api_key: str, model_name: str, local_fallback: Optional[LocalBackend] = None):
        if groq is None:
            raise ImportError("Groq library not installed. Please run 'pip install groq'.")
        self.client = groq.Groq(api_key=api_key)
        self.model_name = model_name
        self.local_fallback = local_fallback

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        retries = 3
        backoff = 2
        for i in range(retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model_name,
                    temperature=0.1,
                    max_tokens=2048,
                )
                usage = chat_completion.usage
                logger.info(f"Groq ({self.model_name}) usage: {usage.prompt_tokens} prompt, {usage.completion_tokens} completion tokens.")
                return chat_completion.choices[0].message.content
            except Exception as e:
                # Handle 429 specifically if possible (groq.RateLimitError)
                if "rate_limit" in str(e).lower() or (hasattr(e, "status_code") and e.status_code == 429):
                    logger.warning(f"Groq rate limit hit (429): {e}")
                    if i < retries - 1:
                        wait_time = backoff ** (i + 1)
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    elif self.local_fallback:
                        logger.warning("Falling back to LocalBackend due to Groq rate limits.")
                        return self.local_fallback.generate(system_prompt, user_prompt)
                
                logger.error(f"Groq API error (Attempt {i+1}/{retries}): {e}")
                if i == retries - 1:
                    raise e
                time.sleep(backoff)
