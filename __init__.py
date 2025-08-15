# ComfyUI-OpenAI-Transcribe
# Minimal OpenAI Audio Transcription node for ComfyUI
# Inputs: audio_path (string), api_key (string, optional if OPENAI_API_KEY env var is set)
# Output: text (string) - transcription
#
# Model default: gpt-4o-mini-transcribe
# Endpoint: POST https://api.openai.com/v1/audio/transcriptions
#
# Place this folder under ComfyUI/custom_nodes/ and restart ComfyUI.

import os
import requests
import mimetypes

class OpenAITranscribe:
    """
    Simple ComfyUI node that sends an audio file to OpenAI's transcription endpoint
    and returns the transcription text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"multiline": False, "default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"multiline": False, "default": "gpt-4o-mini-transcribe"}),
                "language": ("STRING", {"multiline": False, "default": ""}),  # e.g. "es", leave blank for auto
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "OpenAI/Audio"

    def run(self, audio_path: str, api_key: str, model: str = "gpt-4o-mini-transcribe",
            language: str = "", temperature: float = 0.0):
        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY no seteada. Pasá la API Key en el input 'api_key' o seteala como variable de entorno.")

        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError(f"Audio no encontrado: {audio_path!r}")

        url = "https://api.openai.com/v1/audio/transcriptions"
        mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"

        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, mime)}
            data = {
                "model": model,
                "temperature": str(temperature),
                "response_format": "text",  # pedimos string directo
            }
            if language:
                data["language"] = language

            headers = {"Authorization": f"Bearer {key}"}
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=600)

        # Manejo de errores explícito
        if resp.status_code // 100 != 2:
            # mostrar un recorte del cuerpo para debug
            snippet = resp.text[:500].strip().replace("\\n", " ")
            raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {snippet}")

        text = resp.text or ""
        return (text,)


NODE_CLASS_MAPPINGS = {
    "OpenAITranscribe": OpenAITranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAITranscribe": "OpenAI Transcribe (4o mini)",
}
