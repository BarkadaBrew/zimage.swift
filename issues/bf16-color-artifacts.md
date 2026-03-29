# Bug: bf16 render shows face structure but severe color shift and grid artifacts

## Status: OPEN — partially fixed by transformer alias + VAE alias commits

## Reproduction

```bash
ZImageCLI -p "a beautiful Filipina woman in warm golden light, portrait" \
  -m /Users/toddwalderman/.cache/huggingface/hub/z-image-turbo-bf16 \
  --text-encoder-path "/Users/toddwalderman/.cache/huggingface/hub/z-image-turbo-bf16/text_encoder QWen Large" \
  -s 8 -W 1024 -H 1024 --seed 42 -o test.png
```

**Result:** Face visible but green/yellow color shift, grid artifacts (~80% working)
**Expected:** Clean render like mflux produces with same checkpoint

## Progress
- v0.2.0: solid pink (0%)
- Transformer alias fix: green blob with face shape (40%)
- VAE alias fix: face visible, correct-ish colors, edge artifacts (80%)

## Root Cause
- TransformerWeightAliases.swift has only 6 explicit aliases for 521-key checkpoint
- VAEWeightAliases.swift covers conv_in/out/norm but NOT resnet blocks, attention layers, or mid-block keys
- Grid artifacts = classic VAE decode issue from unmapped weights

## Suggested Fix
1. Dump all 521 bf16 keys, diff against ZImage internal names
2. Add mappings for every unmatched key in TransformerWeightAliases.swift
3. Extend VAEWeightAliases.swift to cover resnet/attention/mid-block
4. Add --audit-weights CLI flag to report unmatched keys
