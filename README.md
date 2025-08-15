# ComfyUI-OpenAI-Transcribe

Nodo **mínimo y directo** para usar la API de OpenAI y transcribir audio dentro de **ComfyUI**.

- **Endpoint**: `POST /v1/audio/transcriptions`
- **Modelo por defecto**: `gpt-4o-mini-transcribe`
- **Salida**: `text` (STRING con la transcripción)

## Instalación

**Opción A — Git**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tuusuario/ComfyUI-OpenAI-Transcribe.git
# (reiniciar ComfyUI)
```

**Opción B — Copiar carpeta**
- Descargá/extraé esta carpeta en `ComfyUI/custom_nodes/ComfyUI-OpenAI-Transcribe`
- Reiniciá ComfyUI

> Si falta `requests`, instalalo en el venv de ComfyUI:
```bash
pip install requests
```

## Uso

1. Aportá un **path** de audio válido (`.wav`, `.mp3`, `.m4a`, etc.).
2. Pasá tu **API Key** de OpenAI en el input `api_key` o seteá `OPENAI_API_KEY` como variable de entorno.
3. Opcional: `language="es"`, `temperature=0.0..1.0`, `model="gpt-4o-mini-transcribe"`.
4. La salida `text` es un **STRING** con la transcripción.

### Flujo mínimo sugerido
- **String** (ruta del audio) → **OpenAI Transcribe (4o mini)** → **Save Text** (WAS o ComfyRoll) para guardar `.txt`.

## Parámetros

- `audio_path` *(STRING, requerido)*: Ruta absoluta o relativa al archivo de audio.
- `api_key` *(STRING, requerido si no usás env var)*: API key de OpenAI. Si está vacío, usa `OPENAI_API_KEY`.
- `model` *(STRING, opcional; default `gpt-4o-mini-transcribe`)*.
- `language` *(STRING, opcional)*: Código ISO (ej. `es`). Vacío = autodetección.
- `temperature` *(FLOAT, opcional; default `0.0`)*.

## Licencia
MIT
