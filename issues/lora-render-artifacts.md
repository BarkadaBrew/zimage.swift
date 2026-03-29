# Bug: LoRA renders produce artifacts on stock model

## Status: OPEN

## Reproduction

```bash
# Clean (works):
ZImageCLI -p "portrait" -m Tongyi-MAI/Z-Image-Turbo -s 8 -W 1024 -H 1024 --seed 42 -o clean.png

# With LoRA (artifacts):
ZImageCLI -p "portrait" -m Tongyi-MAI/Z-Image-Turbo \
  --lora /path/to/moodyPornMix_v10DPO_zimage_turbo_lora_r128.safetensors=0.8 \
  -s 8 -W 1024 -H 1024 --seed 42 -o lora.png
```

**Result:** Face recognizable but background noise/tiling, skin corruption
**Expected:** Clean render with LoRA style applied

## Key Finding
Happens on STOCK model (not just bf16). LoRA reports 240 layers matched, correct scales. The application "works" but render quality degrades.

## Tested
| Config | Layers | Result |
|--------|--------|--------|
| Stock, no LoRA | 0 | Clean |
| Stock + moodyPornMix @0.8 | 240 | Artifacts |
| Stock + moodyPornMix + detail @0.3 | 240+240 | Artifacts |
| Stock + 3 LoRAs | 240+240+240 | Artifacts |

## Possible Causes
1. Scale math in LoRALinear — double-applied or wrong dimension
2. Weight dtype mismatch (float32 LoRA vs bfloat16 base)
3. Numerical instability from r128 large-rank adapters

## Debug Steps
1. Render at very low scale (0.05) — do artifacts scale proportionally?
2. Try r4 LoRA (detail_slider) solo — rank-specific?
3. Dump intermediate tensor stats in LoRALinear during inference
