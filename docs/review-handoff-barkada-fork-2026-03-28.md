# Barkada Fork Review Handoff

Date: 2026-03-28

Audience: Claude review of the BarkadaBrew `zimage.swift` fork enhancements.

## Executive Summary

This branch implements the core Barkada fork work needed to make `zimage.swift` usable in the CoffeeShop pipeline:

- custom text encoder selection for the bf16 checkpoint
- config inference for configless local model directories
- multi-LoRA loading and application
- mflux-style LoRA key mapping coverage
- SwiftPM metallib staging for local CLI runs
- runtime fixes discovered during end-to-end validation:
  - prompt encoding now uses plain tokenization rather than chat-template tokenization
  - VAE weight loading now uses shape-aware layout alignment instead of unconditional 4D transposition
  - weight application failures are now fatal instead of log-and-continue

The branch was validated with both focused Swift tests and a real smoke render against:

- base model: `/Users/toddwalderman/.cache/huggingface/hub/z-image-turbo-bf16`
- custom encoder: `/Users/toddwalderman/.cache/huggingface/hub/z-image-turbo-bf16/text_encoder QWen Large`
- LoRA: `/Users/toddwalderman/Projects/moodyPornMix_v10DPO_zimage_turbo_lora_r64.safetensors`

The latest smoke render succeeded and wrote `/tmp/zimage-smoke.png`.

## Scope Completed

### P0

1. LoRA key mapping for mflux-style keys
2. `config.json` inference fallback for configless local checkpoints
3. custom text encoder selection

### P1

1. multiple LoRA support
2. CLI support for multi-LoRA argument forms

### Stability fixes discovered during implementation

1. fixed prompt encoding path for Z-Image generation
2. fixed VAE tensor layout handling for bf16 checkpoints
3. made weight-application failures fatal
4. staged MLX metallib next to the executable for SwiftPM CLI runs

## Not Implemented

The following requested items are still not implemented in this branch:

- warm server mode / `--serve`
- bake subcommand
- batch generation
- character injection
- CoffeeShop image service integration changes
- prompt enhancer integration with local Qwen endpoint beyond the existing prompt enhancement support already in-tree
- new img2img functionality beyond the project’s existing control/inpaint paths

## Change Map

### CLI and user-facing behavior

- `Sources/ZImageCLI/main.swift`
  - adds `--text-encoder-path`
  - supports repeatable `--lora`
  - prefers `path=scale` on `--lora`; legacy `path:scale` is still accepted when unambiguous
  - supports comma-separated `--lora-paths` and `--lora-scales`
  - documents that quoted commas are unsupported in comma-separated LoRA lists
  - updates help text for normal and control modes
  - stages `mlx.metallib` at startup using `MLXMetalLibraryLocator`

### Model path and config resolution

- `Sources/ZImage/Weights/ModelPaths.swift`
  - adds text encoder selection priority:
    - CLI `--text-encoder-path`
    - `ZIMAGE_ENCODER_PATH`
    - auto-detect `text_encoder QWen Large`
    - fallback `text_encoder`
  - recognizes configless local model directories as valid base models
  - dynamically resolves transformer, text encoder, and VAE shard files

- `Sources/ZImage/Weights/ModelConfigs.swift`
  - loads configs from explicit files when present
  - infers transformer config from safetensors tensor shapes when `transformer/config.json` is missing
  - infers text encoder config from tensor shapes when encoder `config.json` is missing
  - falls back to default scheduler and VAE config when needed

- `Sources/ZImage/Weights/ZImageWeightsMapper.swift`
  - threads the selected text encoder directory through text encoder loading
  - resolves dynamic VAE weight paths

### LoRA

- `Sources/ZImage/LoRA/LoRAKeyMapper.swift`
  - extends mapping coverage for mflux checkpoint-style names
  - adds `adaLN_modulation.0` targets

- `Sources/ZImage/LoRA/LoRAConfiguration.swift`
  - allows negative and unclamped scales

- `Sources/ZImage/LoRA/LoRALinear.swift`
  - supports multiple additive adapters rather than a single active adapter

- `Sources/ZImage/LoRA/LoRAApplicator.swift`
  - applies dynamic LoRAs additively for stacked runtime adapters

### Base pipeline

- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
  - request now carries `textEncoderPath` and `loras: [LoRAConfiguration]`
  - preserves backwards compatibility with `lora`
  - reloads the model when encoder selection changes
  - threads selected encoder directory into config loading and weight loading
  - loads and unloads multiple LoRAs
  - uses shared pipeline helpers for encoder selection and negative-embedding alignment
  - documents that cached pipeline state is not thread-safe
  - adds an early return fast-path when the requested model, AIO checkpoint, and encoder selection are already loaded
  - uses plain prompt tokenization for Z-Image generation
  - shape-audits AIO VAE weights using the same alignment logic as runtime loading
  - now propagates weight-application failures instead of silently continuing

### Control pipeline

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - request now carries `textEncoderPath` and `loras: [LoRAConfiguration]`
  - preserves backwards compatibility with `lora`
  - reloads when encoder selection changes
  - includes selected encoder directory in prompt-cache invalidation
  - reuses multi-LoRA logic
  - uses shared pipeline helpers for encoder selection and negative-embedding alignment
  - documents that cached pipeline state is not thread-safe
  - uses the same plain prompt tokenization path
  - now propagates control transformer and controlnet weight-application failures

### Weight mapping and runtime stability

- `Sources/ZImage/Weights/WeightsMapping.swift`
  - introduces `WeightApplicationError`
  - changes apply functions to throw on update failure
  - replaces unconditional VAE 4D transposition with shape-aware alignment

- `Sources/ZImage/Pipeline/PipelineUtilities.swift`
  - switches prompt encoding from `encodeChat` to `encodePlain`
  - centralizes text-encoder selection logging and negative-embedding alignment shared by both pipelines

- `Sources/ZImage/Support/MLXMetalLibraryLocator.swift`
  - locates a usable MLX metallib from build outputs
  - copies it next to the active executable as `mlx.metallib`
  - handles the colocated-copy race without a separate pre-copy existence check

## Post-Review Cleanup

Claude review findings raised after the initial handoff have already been addressed in the branch:

1. `parseLoRAEntry` ambiguity:
   - `Sources/ZImageCLI/main.swift` now prefers `path=scale`
   - legacy `path:scale` parsing only applies when the value does not already look like a local path
   - help text and README examples were updated to steer users toward the unambiguous form
2. `@unchecked Sendable` usage in cached LoRA state:
   - both pipelines now explicitly document that instances are not thread-safe
   - the `AppliedLoRA` comments explain that access is intended to stay pipeline-local and serialized
3. duplicated helpers across the two pipelines:
   - shared text-encoder selection and negative-embedding alignment logic now lives in `Sources/ZImage/Pipeline/PipelineUtilities.swift`
4. dead `inferVAEConfig` branch:
   - `Sources/ZImage/Weights/ModelConfigs.swift` now clearly returns the known Z-Image-Turbo default VAE config with an explanatory comment
5. repeated snapshot resolution before cache checks:
   - `Sources/ZImage/Pipeline/ZImagePipeline.swift` now performs an early cached-model fast-path before `PipelineSnapshot.prepare(...)`
6. metallib staging race:
   - `Sources/ZImage/Support/MLXMetalLibraryLocator.swift` now handles the colocated-copy race without a separate `fileExists` re-check before `copyItem`

## Why the Runtime Fixes Were Needed

These three issues were discovered only after the core spec work had already landed.

### 1. Prompt encoding

The pipeline was using `QwenTokenizer.encodeChat(...)` for image prompts. The tokenizer file already documented `encodePlain(...)` as the Z-Image path. On the bf16/custom-encoder setup, chat-template tokenization caused prompt encoding to fail at runtime. Switching to `encodePlain(...)` fixed this and the smoke render proceeded into denoising.

### 2. VAE tensor layout

The bf16 checkpoint stores VAE conv tensors in MLX layout already. The previous runtime path transposed every 4D VAE tensor, which corrupted correct tensors and caused shape mismatches during VAE load. The fix now transposes only when the transposed shape is what the module actually expects.

### 3. Silent partial loads

The previous code caught `module.update(...)` failures, logged them, and continued. That allowed the model to report as loaded successfully even after a failed weight application. The branch now throws `WeightApplicationError.updateFailed(...)`, and all base/control call sites propagate that failure.

## Verification

### Stable automated suite

Command:

```bash
swift test --filter 'VAEWeightLayoutTests|MLXMetalLibraryLocatorTests|LoRALoaderTests|ModelPathResolutionTests|ModelConfigFallbackTests|LoRALinearTests|ZImagePipelineSelectionTests'
```

Result:

- passed
- 30 tests executed
- 2 skipped
- 0 failures

Notes:

- the skipped tests are `LoRALinearTests`, which are intentionally skipped because the SwiftPM test runner on this machine does not reliably provide the MLX metallib to MLX-backed unit tests

### Real smoke render

Command:

```bash
./.build/debug/ZImageCLI \
  -p "a woman in warm light" \
  -m /Users/toddwalderman/.cache/huggingface/hub/z-image-turbo-bf16 \
  --text-encoder-path "/Users/toddwalderman/.cache/huggingface/hub/z-image-turbo-bf16/text_encoder QWen Large" \
  -s 1 -W 256 -H 256 \
  --lora /Users/toddwalderman/Projects/moodyPornMix_v10DPO_zimage_turbo_lora_r64.safetensors \
  --lora-scale 0.1 \
  --no-progress \
  -o /tmp/zimage-smoke.png
```

Result:

- model loaded
- custom encoder selected
- LoRA loaded and applied
- prompt encoding completed
- denoising completed
- image written to `/tmp/zimage-smoke.png`

Image check:

- verified as `256x256`

## Review Guidance

### High-value files to review first

1. `Sources/ZImage/Weights/ModelPaths.swift`
2. `Sources/ZImage/Weights/ModelConfigs.swift`
3. `Sources/ZImage/Pipeline/ZImagePipeline.swift`
4. `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
5. `Sources/ZImage/Weights/WeightsMapping.swift`
6. `Sources/ZImageCLI/main.swift`

### What to verify in review

1. The text encoder selection precedence is correct and deterministic.
2. Config inference uses safe assumptions for the known Z-Image bf16 checkpoint.
3. Prompt tokenization change to `encodePlain` is appropriate for image generation and does not break existing expectations.
4. Fatal weight-application behavior is preferable to the previous silent partial-load behavior.
5. Multi-LoRA request parsing and stacking behavior are coherent in both base and control pipelines.

## Known Limitations

### SwiftPM + MLX metallib

There is still an environment limitation in the SwiftPM test runner:

- some MLX-backed unit tests cannot run reliably because the test runner does not consistently provide the MLX metallib
- attempts to force metallib staging inside `xctest` caused hangs in some MLX-heavy suites

Practical result:

- stable non-MLX-heavy tests are automated
- the most important MLX-backed path was validated with the real CLI smoke render instead

### Remaining warning noise

`Sources/ZImage/Weights/AIOCheckpoint.swift` still emits Swift 6 sendability warnings because `MLXArray` is not `Sendable`. This is not new runtime breakage, but it remains unresolved.

## Test-First Methodology Notes

The work followed test-first where feasible:

- new pure-shape and config/path regressions were added before or alongside the corresponding fixes
- MLX-backed runtime issues could not always be locked down in SwiftPM unit tests due the metallib limitation on this machine
- where reliable unit tests were not feasible, validation used real end-to-end CLI smoke renders against the target bf16 checkpoint

## Pre-Existing Local Changes

Before this Barkada work started, the local checkout already had unrelated or user-owned changes in:

- `Package.swift`
- `Package.resolved`
- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `release-artifacts/`

What that means for review:

- the two pipeline files contain a mix of pre-existing local work plus the Barkada changes described here
- `release-artifacts/` was left untouched
- `Package.swift` and `Package.resolved` were not reverted

## Files Added for This Work

- `Sources/ZImage/Support/MLXMetalLibraryLocator.swift`
- `Tests/ZImageTests/Config/ModelConfigFallbackTests.swift`
- `Tests/ZImageTests/Weights/LoRALinearTests.swift`
- `Tests/ZImageTests/Weights/MLXMetalLibraryLocatorTests.swift`
- `Tests/ZImageTests/Weights/ModelPathResolutionTests.swift`
- `Tests/ZImageTests/Weights/VAEWeightLayoutTests.swift`
- `Tests/ZImageTests/Weights/ZImagePipelineSelectionTests.swift`

## Files Modified for This Work

- `Sources/ZImage/LoRA/LoRAApplicator.swift`
- `Sources/ZImage/LoRA/LoRAConfiguration.swift`
- `Sources/ZImage/LoRA/LoRAKeyMapper.swift`
- `Sources/ZImage/LoRA/LoRALinear.swift`
- `Sources/ZImage/Pipeline/PipelineUtilities.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Weights/ModelConfigs.swift`
- `Sources/ZImage/Weights/ModelPaths.swift`
- `Sources/ZImage/Weights/WeightsMapping.swift`
- `Sources/ZImage/Weights/ZImageWeightsMapper.swift`
- `Sources/ZImageCLI/main.swift`
- `Tests/ZImageIntegrationTests/PipelineIntegrationTests.swift`
- `Tests/ZImageTests/Weights/LoRALoaderTests.swift`

## Reviewer Bottom Line

This branch is ready for review and merge.

The only material risk that remained near the end of implementation was silent partial model loads after weight-application failures. That is now fixed. The remaining caveats are test-runner limitations and warning cleanup, not known product-level regressions in the validated bf16 path.
