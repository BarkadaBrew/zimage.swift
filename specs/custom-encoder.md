# Custom Text Encoder — bf16 Checkpoint

## The bf16 model has 3 encoder variants:
```
z-image-turbo-bf16/
├── text_encoder/              ← Standard encoder
├── text_encoder Original/     ← Stock backup
├── text_encoder QWen Large/   ← CUSTOM (the good one)
```

## Selection Priority
1. CLI --text-encoder-path flag
2. ZIMAGE_ENCODER_PATH environment variable
3. Auto-detect: prefer "text_encoder QWen Large" if present
4. Fallback: "text_encoder" directory

## Why QWen Large Matters
The custom encoder is a larger Qwen text encoder (likely Qwen 3 4B+). Produces richer text embeddings → better prompt adherence → higher quality images. This is the single biggest quality differentiator in our pipeline.

## Implementation Status
✅ CLI flag (--text-encoder-path) — implemented in main.swift
✅ Env variable (ZIMAGE_ENCODER_PATH) — implemented in ModelPaths.swift  
✅ Auto-detection — implemented in ModelPaths.swift
✅ Logging which encoder was selected — implemented
