# SPEC: Bake Subcommand

## Goal
Pre-merge N LoRAs at specified scales into a base model checkpoint. Produces a standalone model directory that loads without any LoRA overhead.

## CLI
```bash
ZImageCLI bake \
  -m "/path/to/base-model" \
  --text-encoder-path "/path/to/encoder" \
  --lora "/path/to/moodyPornMix.safetensors=0.8" \
  --lora "/path/to/detail_slider.safetensors=0.3" \
  --lora "/path/to/BreastSize_Slider.safetensors=-2" \
  -o "/path/to/CoffeeShop-Baked/"
  --symlink    # symlink encoder/tokenizer/vae instead of copying
  --quantize 8 # optional: quantize merged weights
```

## Output Structure
```
CoffeeShop-Baked/
├── transformer/
│   ├── config.json          (generated)
│   ├── *.safetensors        (merged weights)
│   └── model.safetensors.index.json
├── text_encoder/            (copied or symlinked)
├── tokenizer/               (copied or symlinked)
├── vae/                     (copied or symlinked)
└── bake_manifest.json       (metadata: base model, LoRAs used, scales)
```

## Algorithm
```
For each LoRA:
  1. Load weights (down/up pairs)
  2. Map keys via LoRAKeyMapper
  3. For each matched layer: W' = W + scale * (up @ down)
After all merged:
  4. Write sharded SafeTensors
  5. Copy/symlink encoder, tokenizer, VAE
  6. Generate config.json + bake_manifest.json
```

## Implementation
- New file: Sources/ZImage/Bake/ModelBaker.swift (~150 LOC)
- main.swift: add `bake` subcommand (~40 LOC)

## Estimated: ~190 LOC Swift
