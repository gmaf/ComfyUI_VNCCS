"""Wizard text generation through ComfyUI's native TextGenerate node.

The Character Creator, Clothes Designer and Character Cloner wizards used a
llama-cpp-python Qwen3.5 GGUF. This fork runs them on the Qwen3-VL 8B text
encoder that the Qwen Image 2.1 profile already uses, through ComfyUI's
TextGenerate node, so no llama-cpp-python build or GGUF files are needed.
"""
import base64
import io
import json
import os
import random
import re

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import server
    from aiohttp import web
except ImportError:
    server = web = None

from ..utils import get_full_path_agnostic, privileged_route

WIZARD_CLIP_NAME = "qwen3vl_8b_int8_convrot.safetensors"
WIZARD_CLIP_TYPE = "qwen_image"
# Kept for callers and the frontend, which report this name in model messages.
QWEN_VL_MODEL_FILENAME = WIZARD_CLIP_NAME

WIZARD_SAMPLING = {
    "sampling_mode": "on",
    "top_k": 64,
    "top_p": 0.95,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
    "presence_penalty": 0.0,
}

_QWEN_VL_DOWNLOAD_STATUS = {
    "status": "idle",
    "progress": 0,
    "current_file": "",
    "error": "",
}

def parse_wizard_json(content):
    data = None
    try:
        import json_repair
        data = json_repair.loads(content)
    except Exception:
        data = None

    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        try:
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in json_str:
                json_str = json_str.split("```", 1)[1].split("```", 1)[0]
            else:
                match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
            data = json.loads(json_str.strip())
        except Exception:
            data = None

    if not isinstance(data, dict):
        return None

    return data


def normalize_character_wizard_traits(data):
    result = {}
    for key in ["race", "skin_color", "hair", "eyes", "face", "body", "additional_details"]:
        value = data.get(key, "")
        if isinstance(value, list):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        elif not isinstance(value, str):
            value = str(value) if value is not None else ""
        result[key] = value.strip()

    sex = str(data.get("sex", "female")).strip().lower()
    result["sex"] = "male" if sex.startswith("m") else "female"
    try:
        age = int(float(data.get("age", 18)))
    except Exception:
        age = 18
    result["age"] = max(1, min(100, age))
    return result


def _wizard_clip_path():
    if folder_paths is None:
        return None
    return get_full_path_agnostic(folder_paths, "text_encoders", WIZARD_CLIP_NAME)


def _ensure_qwen_vl_assets(allow_download=True, require_mmproj=True):
    """Return the wizard text encoder path; arguments are kept for old callers."""
    path = _wizard_clip_path()
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"{WIZARD_CLIP_NAME} is missing from models/text_encoders. "
            "It belongs to the Qwen Image 2.1 profile (Comfy-Org/Qwen-Image-2.1)."
        )
    return path, path


def _load_wizard_clip():
    # A fresh encoder per request: Qwen3-VL generation keeps state inside
    # cond_stage_model, so the Creator never reuses one across requests either.
    import comfy.sd

    path, _ = _ensure_qwen_vl_assets()
    clip_type = getattr(comfy.sd.CLIPType, WIZARD_CLIP_TYPE.upper())
    return comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
    )


def _image_from_data_url(url):
    import numpy as np
    import torch
    from PIL import Image

    payload = url.split(",", 1)[1] if url.startswith("data:") else url
    with Image.open(io.BytesIO(base64.b64decode(payload))) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]


def generate_wizard_text(messages, max_tokens=900, temperature=0.3):
    """Answer OpenAI-style chat messages with TextGenerate and return the text.

    System messages become the system prompt, user text parts are joined, and
    the first data-URL image in the user turn is passed as the image input.
    """
    from .character_generator import _call_comfy_node
    from .runtime_cleanup import inference_stage

    system_parts, user_parts, image = [], [], None
    for message in messages:
        content = message.get("content", "")
        items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for item in items:
            if item.get("type") == "text":
                (system_parts if message.get("role") == "system" else user_parts).append(item.get("text", ""))
            elif item.get("type") == "image_url" and image is None:
                image = _image_from_data_url(item["image_url"]["url"])

    with inference_stage():
        clip = _load_wizard_clip()
        return _call_comfy_node(
            "TextGenerate",
            clip=clip,
            prompt="\n\n".join(part for part in user_parts if part),
            image=image,
            max_length=int(max_tokens),
            sampling_mode={**WIZARD_SAMPLING, "temperature": float(temperature), "seed": random.randint(0, 2**32 - 1)},
            thinking=False,
            system_prompt="\n\n".join(part for part in system_parts if part),
        )[0]


def _start_qwen_vl_download(require_mmproj=True):
    # Models come from the deployment's model manifest; nothing to download here.
    try:
        _ensure_qwen_vl_assets()
    except FileNotFoundError as exc:
        _QWEN_VL_DOWNLOAD_STATUS.update(status="error", error=str(exc))
        return web.json_response({"error": "MODEL_MISSING", "message": str(exc),
                                  "model_name": WIZARD_CLIP_NAME}, status=404)
    _QWEN_VL_DOWNLOAD_STATUS.update(status="completed", progress=100,
                                    current_file="Wizard text encoder ready", error="")
    return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS))


if server is not None and web is not None:
    @server.PromptServer.instance.routes.get("/vnccs/qwen_vl_model_status")
    async def qwen_vl_model_status(request):
        try:
            _ensure_qwen_vl_assets()
            return web.json_response({"ready": True, "model_name": WIZARD_CLIP_NAME})
        except FileNotFoundError as exc:
            return web.json_response({"ready": False, "model_name": WIZARD_CLIP_NAME, "message": str(exc)})

    @server.PromptServer.instance.routes.get("/vnccs/qwen_vl_download_status")
    async def qwen_vl_download_status(request):
        return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS))

    @server.PromptServer.instance.routes.post("/vnccs/qwen_vl_download_model")
    @privileged_route
    async def qwen_vl_download_model(request):
        return _start_qwen_vl_download()
