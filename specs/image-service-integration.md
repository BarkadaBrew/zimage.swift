# SPEC: CoffeeShop Image Service Integration

## Goal
Make ZImageCLI a first-class engine in the CoffeeShop image service alongside mflux.

## Changes (TypeScript, in coffeeshop-image-service repo)

### 1. Update buildZImageArgs() — generator.ts
- Pass --text-encoder-path from request.encoderPath
- Pass ALL LoRAs (currently only passes first one)
- Use --lora path=scale syntax
- Add --no-progress flag

### 2. New zimage-warm-client.ts
HTTP client for the ZImageCLI warm server:
- isAvailable() — check /health
- generate(request) — POST /v1/generate
- swapLoRAs(loras) — POST /v1/lora/swap

### 3. Generator routing
When engine === 'zimage':
  1. Try warm server (POST /v1/generate)
  2. Fallback to cold CLI spawn

### 4. Config additions
- zimage.warmServer.enabled + baseUrl
- Preset `engine` field: 'mflux' | 'zimage'
- Per-preset encoderPath

### 5. LoRA sync on preset change
When preset changes to a zimage engine preset, call /v1/lora/swap on the warm server.

## Estimated: ~220 LOC TypeScript
