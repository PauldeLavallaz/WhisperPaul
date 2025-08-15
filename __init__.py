# ComfyUI-OpenAI-Transcribe
# OpenAI Audio Transcription node for ComfyUI
# Inputs: audio (AUDIO), api_key (STRING), optional: model/language/temperature(STRING)
# Output: text (STRING) - transcription

import os
import requests
import mimetypes
import tempfile
import wave

try:
    import numpy as np
except Exception:
    np = None


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
                "language": ("STRING", {"multiline": False, "default": ""}),  # e.g. "es"; blank = auto
                # STRING to avoid Comfy Deploy sending '' for FLOAT
                "temperature": ("STRING", {"multiline": False, "default": "0"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "OpenAI/Audio"

    def _audio_to_file(self, audio) -> tuple[str, bool]:
        if isinstance(audio, str) and os.path.exists(audio):
            return audio, False
        if isinstance(audio, dict):
            for k in ("path", "filepath", "file", "filename", "temp_path"):
                p = audio.get(k)
                if isinstance(p, str) and os.path.exists(p):
                    return p, False

        samples, sr = None, None
        if isinstance(audio, dict):
            sr = audio.get("sample_rate") or audio.get("sampling_rate") or audio.get("sr") or 44100
            samples = audio.get("samples") or audio.get("waveform") or audio.get("audio") or audio.get("data")
        elif isinstance(audio, (list, tuple)) and len(audio) == 2:
            samples, sr = audio[0], audio[1]

        if samples is None:
            raise RuntimeError("AUDIO no contiene path ni samples reconocibles. Pasá un AUDIO válido (con path o samples).")
        if np is None:
            raise RuntimeError("Este nodo requiere numpy para serializar AUDIO→WAV. Instalá numpy en el venv de ComfyUI.")

        arr = np.asarray(samples)
        if arr.ndim == 1:
            channels = 1
            arr = arr[None, :]
        elif arr.ndim == 2:
            c_first = arr.shape[0] <= arr.shape[1]
            channels = arr.shape[0] if c_first else arr.shape[1]
            if not c_first:
                arr = arr.T
        else:
            raise RuntimeError(f"Formato de samples inesperado: {arr.shape}")

        if arr.dtype.kind == "f":
            arr = np.clip(arr, -1.0, 1.0)
            pcm = (arr * 32767.0).astype(np.int16)
        elif arr.dtype == np.int16:
            pcm = arr
        else:
            pcm = arr.astype(np.int16)

        interleaved = pcm.T.reshape(-1)
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(int(sr or 44100))
            w.writeframes(interleaved.tobytes())
        return tmp_path, True

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
