# ComfyUI-OpenAI-Transcribe
# Robust AUDIO -> file resolver (handles torch Tensors without boolean eval)
# OpenAI transcription caller. Returns plain text.

import os
import mimetypes
import tempfile
import wave
import requests

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


def _to_float(val, default=0.0):
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _first_present(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


class OpenAITranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"multiline": False, "default": "gpt-4o-mini-transcribe"}),
                "language": ("STRING", {"multiline": False, "default": ""}),  # e.g. 'es'
                "temperature": ("STRING", {"multiline": False, "default": "0"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "OpenAI/Audio"

    # --- helpers ---
    def _audio_to_file(self, audio) -> tuple[str, bool]:
        # If it's a direct path string
        if isinstance(audio, str) and os.path.exists(audio):
            return audio, False

        # If dict with a path-like entry
        if isinstance(audio, dict):
            for k in ("path", "filepath", "file", "filename", "temp_path"):
                p = audio.get(k)
                if isinstance(p, str) and os.path.exists(p):
                    return p, False

        # Else try to resolve raw samples
        samples, sr = None, None
        if isinstance(audio, dict):
            sr = _first_present(audio, ("sample_rate", "sampling_rate", "sr"))
            samples = _first_present(audio, ("samples", "waveform", "audio", "data"))
        elif isinstance(audio, (list, tuple)) and len(audio) == 2:
            samples, sr = audio[0], audio[1]

        if samples is None:
            raise RuntimeError("AUDIO no contiene path ni samples. Pasá un AUDIO válido (con path o samples).")

        # to numpy
        if torch is not None and isinstance(samples, torch.Tensor):
            arr = samples.detach().cpu().numpy()
        elif np is not None:
            arr = np.asarray(samples)
        else:
            raise RuntimeError("Este nodo requiere numpy (o torch) para serializar AUDIO→WAV. Instalá numpy en el venv de ComfyUI.")

        # ensure float or int array
        if arr.ndim == 1:
            channels = 1
            arr = arr[None, :]  # (C=1, T)
        elif arr.ndim == 2:
            # Accept (C, T) or (T, C). Assume channels <= 8
            c_first = arr.shape[0] <= 8
            channels = arr.shape[0] if c_first else arr.shape[1]
            if not c_first:
                arr = arr.T
        else:
            raise RuntimeError(f"Formato de samples inesperado: {arr.shape}")

        # normalize to int16 PCM
        if arr.dtype.kind == "f":
            if np is None:
                raise RuntimeError("numpy es requerido para normalizar audio float.")
            arr = np.clip(arr, -1.0, 1.0)
            pcm = (arr * 32767.0).astype(np.int16)
        elif np is not None and arr.dtype != np.int16:
            pcm = arr.astype(np.int16)
        else:
            pcm = arr  # already int16

        interleaved = pcm.T.reshape(-1)

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(int(channels))
            w.setsampwidth(2)  # 16-bit
            w.setframerate(int(sr or 44100))
            w.writeframes(interleaved.tobytes())

        return tmp_path, True

    # --- main ---
    def run(self, audio, api_key: str, model: str = "gpt-4o-mini-transcribe",
            language: str = "", temperature: str = "0"):
        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY no seteada. Pasá la API Key en 'api_key' o seteala como variable de entorno.")

        path, should_cleanup = self._audio_to_file(audio)
        url = "https://api.openai.com/v1/audio/transcriptions"
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        temp_val = _to_float(temperature, 0.0)

        try:
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f, mime)}
                data = {
                    "model": model,
                    "temperature": str(temp_val),
                    "response_format": "text",
                }
                if language:
                    data["language"] = language

                headers = {"Authorization": f"Bearer {key}"}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=600)

            if resp.status_code // 100 != 2:
                snippet = resp.text[:600].strip().replace("\\n", " ")
                raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {snippet}")

            text = resp.text or ""
            return (text,)
        finally:
            if should_cleanup:
                try:
                    os.remove(path)
                except Exception:
                    pass


NODE_CLASS_MAPPINGS = {"OpenAITranscribe": OpenAITranscribe}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAITranscribe": "OpenAI Transcribe (4o mini)"}
