# LoRA Render Artifacts — Debug Handoff for Codex

## Summary
LoRA renders produce tiling/noise artifacts on both stock and bf16 models. The LoRA loads correctly (240 layers, correct shapes) but the merged output is corrupted.

## What Bree Tested
1. Scale 0.0 → ARTIFACTS (zero LoRA contribution, modules still replaced)
2. Scale 0.05 → ARTIFACTS (not scale-proportional)
3. applyDynamically (module swap) → ARTIFACTS
4. applyToTransformer (weight baking) → ARTIFACTS
5. Key/shape analysis: all 240 pairs have correct shapes matching weights

## Root Cause: computeDelta() + alignShape()

The `computeDelta` function in LoRAApplicator.swift tries multiple matmul orientations:
```swift
if up.dim(1) == down.dim(0) { return MLX.matmul(up, down) }       // Branch 1
else if up.dim(0) == down.dim(1) { return MLX.matmul(up.T, down.T) } // Branch 2
else if up.dim(1) == down.dim(1) { return MLX.matmul(up, down.T) }   // Branch 3
```

For attention weights [3840, 3840], the delta is ALSO [3840, 3840] — square. The `alignShape` function then checks:
```swift
if delta.shape == targetShape { return delta }
else if delta.T.shape == targetShape { return delta.T }  // ← THIS
```

For square matrices, BOTH branches match. The function may transpose a correct delta or pass through an incorrect one.

## Suggested Fix
1. Remove ambiguity — `computeDelta` should ALWAYS compute `matmul(up, down)` (B @ A, the LoRA convention) and never try alternative orientations
2. Remove the transpose fallback in `alignShape` — if the delta doesn't match the target shape without transposing, it's an error
3. Add verbose logging: print the first branch that matched in computeDelta for each layer

## LoRA Convention
- A = down = lora_A = [rank, in_features]
- B = up = lora_B = [out_features, rank]  
- delta = B @ A = matmul(up, down) = [out_features, in_features]

## Test Command
```bash
# Stock model + 1 LoRA — should render cleanly
.build/debug/ZImageCLI -p portrait of a woman -m Tongyi-MAI/Z-Image-Turbo \
  --lora /Users/toddwalderman/.coffeeshop/image-service/loras/moodyPornMix_v10DPO_zimage_turbo_lora_r128.safetensors=0.8 \
  -s 8 -W 512 -H 512 --seed 42 -o /tmp/lora-test.png

# Compare: no LoRA (should be clean)
.build/debug/ZImageCLI -p portrait of a woman -m Tongyi-MAI/Z-Image-Turbo \
  -s 8 -W 512 -H 512 --seed 42 -o /tmp/nolora-test.png
```
