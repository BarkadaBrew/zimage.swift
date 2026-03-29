# SPEC: Warm Server Mode (--serve)

## Goal
Keep model loaded in memory between renders. Eliminate 14-22s cold-start. Target: ~70-80s per render (denoising + VAE only).

## CLI
```bash
ZImageCLI serve \
  -m "/path/to/model" \
  --text-encoder-path "/path/to/encoder" \
  --port 7862 \
  --lora "/path/to/lora.safetensors=0.8"
```

## HTTP API

### POST /v1/generate
```json
{
  "prompt": "...",
  "negative_prompt": "...",
  "width": 1024, "height": 1024,
  "steps": 8, "guidance": 1.0,
  "seed": 42,
  "output_path": "/tmp/output.png"
}
```
Response: `{ "success": true, "output_path": "...", "duration_ms": 72000 }`

### POST /v1/lora/swap
Hot-swap LoRAs without reloading base model.
```json
{
  "loras": [
    {"path": "/path/to/lora.safetensors", "scale": 0.8}
  ]
}
```

### GET /health
Returns model info, LoRA state, uptime, render count, memory usage.

### POST /v1/shutdown
Graceful shutdown.

## Architecture
- Use Swift NIOCore/NIOHTTP1 or Network.framework for HTTP
- Single-threaded request processing (one render at a time, queue extras, max 10)
- Hold reference to loaded ZImagePipeline
- Keep text encoder + transformer + VAE all in memory (skip unload/reload)
- ~21GB for bf16 full model (fits M3 Max 128GB)

## Implementation
- New file: Sources/ZImage/Server/WarmServer.swift (~200 LOC)
- Changes to main.swift: add `serve` subcommand (~50 LOC)
- Pipeline changes: expose generateFromRequest() + swapLoRAs() (~30 LOC)

## Estimated: ~280 LOC Swift
