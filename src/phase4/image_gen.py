import hashlib
import json
import logging
import os
import gc
from typing import Tuple, Dict, Any, Optional
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline

logger = logging.getLogger(__name__)

def get_deterministic_seed(shot_id: str) -> int:
    """Generate a deterministic seed based on shot_id."""
    return int(hashlib.sha256(shot_id.encode()).hexdigest()[:8], 16)

def _prep_reference(img: Image.Image, size: int = 224, mode: str = "crop") -> Image.Image:
    """Square-ify a (possibly non-square) reference before IP-Adapter encoding.
    The CLIP image processor center-crops its input, so a raw 832x480 reference
    loses its sides. Three modes:
      'crop'   -> center-crop to square (undistorted, NO black, drops periphery) [default]
      'resize' -> squash to square      (all content, distorted aspect, NO black)
      'pad'    -> letterbox to square    (all content, ~42% BLACK bars -> dilutes signal)
    'pad' was the earlier default and likely WEAKENED conditioning on 16:9 refs."""
    img = img.convert("RGB")
    w, h = img.size
    if mode == "resize":
        return img.resize((size, size), Image.LANCZOS)
    if mode == "pad":
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas.resize((size, size), Image.LANCZOS)
    # default: center-crop to square
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)

def load_pipeline(config: Dict[str, Any]) -> Tuple[Any, bool]:
    """
    Load SDXL pipeline, LoRA, and optionally IP-Adapter based on config.
    Returns (pipeline, is_ip_adapter_loaded).
    """
    image_gen_config = config.get("phase4", {}).get("image_gen", {})
    base_model = image_gen_config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")

    logger.info(f"Loading SDXL base model: {base_model}")

    # Load specific image encoder for ViT-H
    from transformers import CLIPVisionModelWithProjection
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        image_encoder=image_encoder
    )

    # Load LoRA
    lora_config = image_gen_config.get("cartoon_lora", {})
    lora_path = lora_config.get("repo_or_path")
    if lora_path:
        logger.info(f"Loading LoRA from: {lora_path}")
        if os.path.isfile(lora_path):
            pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path), adapter_name="cartoon")
        else:
            pipe.load_lora_weights(lora_path, adapter_name="cartoon")

        lora_weight = lora_config.get("weight", 0.8)
        pipe.set_adapters(["cartoon"], adapter_weights=[lora_weight])

    # Load IP-Adapter
    ip_config = image_gen_config.get("ip_adapter", {})
    repo = ip_config.get("repo", "h94/IP-Adapter")
    subfolder = ip_config.get("subfolder", "sdxl_models")
    weight_name = ip_config.get("weight_name", "ip-adapter-plus_sdxl_vit-h.safetensors")

    logger.info(f"Loading IP-Adapter: {repo} / {weight_name}")
    pipe.load_ip_adapter(repo, subfolder=subfolder, weight_name=weight_name)

    pipe.to("cuda")
    return pipe, True

def generate_image(
    pipe: Any,
    shot: Dict[str, Any],
    decision: str,
    config: Dict[str, Any],
    ref_image: Optional[Image.Image] = None,
    seed: Optional[int] = None
) -> Image.Image:
    """
    Generate an image for a shot based on the anchor decision.

    Decisions that consume a reference image (via IP-Adapter):
      - SOFT_CHAIN     -> ref = previous shot's image      (scale = soft_chain_ref_weight)
      - CONCEPT_ANCHOR -> ref = concept-canonical's image  (scale = concept_anchor_ref_weight)
    RESET / CHAIN -> no reference (IP-Adapter muted with a black dummy + scale 0).
    The caller is responsible for loading and passing the correct `ref_image`.
    """
    if seed is None:
        seed = get_deterministic_seed(shot["id"])

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image_gen_config = config.get("phase4", {}).get("image_gen", {})
    trigger_words = image_gen_config.get("cartoon_lora", {}).get("trigger_words", "fca style")

    # Inject trigger words
    original_prompt = shot.get("image_prompt", "")
    prompt = f"{trigger_words}, {original_prompt}" if trigger_words else original_prompt

    gen_resolution = image_gen_config.get("generation_resolution", [1344, 768])
    out_resolution = image_gen_config.get("output_resolution", [832, 480])

    gen_width, gen_height = gen_resolution[0], gen_resolution[1]
    out_width, out_height = out_resolution[0], out_resolution[1]

    num_inference_steps = image_gen_config.get("num_inference_steps", 30)
    guidance_scale = image_gen_config.get("guidance_scale", 7.0)

    # Prepare IP-Adapter reference for any decision that uses one.
    if decision in ("SOFT_CHAIN", "CONCEPT_ANCHOR") and ref_image is not None:
        prep_mode = image_gen_config.get("ip_reference_prep", "crop")
        ip_adapter_image = _prep_reference(ref_image, mode=prep_mode)  # square-ify (no black bars by default)
        if decision == "CONCEPT_ANCHOR":
            ref_weight = image_gen_config.get("concept_anchor_ref_weight", 0.5)
        else:
            ref_weight = image_gen_config.get("soft_chain_ref_weight", 0.4)
        pipe.set_ip_adapter_scale(ref_weight)
        logger.info(f"  IP-Adapter ON: scale={ref_weight} prep={image_gen_config.get('ip_reference_prep','crop')}")
    else:
        ip_adapter_image = Image.new("RGB", (224, 224), "black")
        pipe.set_ip_adapter_scale(0.0)

    logger.info(f"Generating image for shot {shot['id']} (Decision: {decision}) at {gen_width}x{gen_height}")

    negative_prompt = image_gen_config.get("negative_prompt", "")
    # Always suppress garbled text/labels: SDXL cannot spell, so any text it draws is gibberish.
    _text_neg = ("text, letters, words, numbers, captions, labels, infographic, diagram, "
                 "panels, charts, table, watermark, signature, gibberish text")
    negative_prompt = (negative_prompt.rstrip(", ") + ", " + _text_neg) if negative_prompt else _text_neg

    # Generate
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": gen_width,
        "height": gen_height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "ip_adapter_image": ip_adapter_image
    }

    result = pipe(**kwargs).images[0]

    # Resize output
    result = result.resize((out_width, out_height), Image.LANCZOS)

    return result

def unload_pipeline(pipe):
    """Unload pipeline and free VRAM."""
    logger.info("Unloading pipeline and freeing VRAM...")
    if pipe is not None:
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"VRAM Peak: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")