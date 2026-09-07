# ComfyBox Server — API Notes (hand-maintained)

Operational notes and body schemas that complement the generated
[`api-reference.md`](api-reference.md). That file is regenerated wholesale by
`comfybox docs generate` (and byte-checked in CI); THIS file is hand-maintained
and never touched by the generator — put prose, schemas and examples here.
Content below carried verbatim from the pre-Phase-4 hand-maintained reference
(`docs/api-reference.md` @ 3654f8b4).

## Video generation (LTX-2 / Replicate)

`POST /v1/video/generate`: Video generation. **Local LTX-2** (T2V + I2V) when
the server is started with `--ltx2-weights` + `--ltx2-gemma` (runs through the
render queue, returns 200 with the MP4 path); otherwise the Replicate cloud
proxy (202, job-based). `GET /v1/video/status/{id}` reports video job status
(Replicate proxy).

Local LTX-2 body (snake_case): `{prompt, negative_prompt?, image_path?, width?, height?, frames? (1+8k), steps?, seed?, strength?, extend_to_seconds?, fps?, output_path?}`
→ `{success, output_path, frame_count, duration_seconds, elapsed_seconds, backend: "ltx2-local"}`.
`image_path` present = image-to-video; absent = text-to-video. Output is
contained to the server's allowed output directory.

### Generation record — MP4 parity with the PNG XMP record (comfybox#401)

Every local LTX-2 output (t2v, i2v, extend, `/v1/video/rerender`, and each
per-shot clip AND the final assembled clip of a `/v1/storyboard/render`) gets
the same provenance record the PNG side embeds in EXIF `UserComment`:
`prompt`, `negative_prompt`, `seed`, `steps`, `model`, `width`/`height`
(requested budget), `frames`, `fps`, `resolved_width`/`resolved_height`
(actual encoded size — differs from the budget when two-stage refine ran),
`dimension_reason` (currently always `null` — populated once
`VideoDimensionResolver`, #405/#408, lands), `two_pass`/`refine`
(`refine` is `false` when `two_pass` was requested but skipped — see
`refine_skipped_reason`), `audio`, `kind` (`t2v`\|`i2v`\|`extend`\|`storyboard`),
and `loras[]` (`{name, scale}`).

Two sinks, one mandatory:

1. **`.json` sidecar** next to the output file — `<basename>.json`, same
   directory, same convention the desktop editor's image sidecars and the
   DAM ingestor's video reader already use (`EditSidecar.sidecarPath`,
   `AssetIngestor.readSidecar`). Always written; a write failure never fails
   the render (the clip is the primary artifact — logged, not thrown).
2. **MP4 metadata atom** (best-effort, header-only, no re-encode) — the
   record's JSON in the container's standard "common description" field
   (`AVMetadataKeySpace.common` / `commonKeyDescription`; readable via
   `AVAsset.metadata` or `exiftool`'s `Description` tag). AVAssetWriter's
   QuickTime-style keyed (`mdta`) metadata and userdata comment atoms are
   both silently dropped for `fileType: .mp4` — verified empirically — so
   this is the one keyspace that actually persists into an ISO-brand `.mp4`.

`GET /v1/video/status/{id}` carries the same record as the additive
`generation_record` field once the render succeeds (local LTX-2 backend
only; `null` while queued/processing, on failure, and on the Replicate cloud
path).

## Prompt enhancement

`POST /v1/enhance` body: `{prompt, character?, character_description?, content_mode?}`
→ `{success, prompt, enhanced, note?}`.

## LoRA roles (`POST /v1/lora/swap`)

`POST /v1/lora/swap` accepts an optional semantic role on every entry:

```json
{
  "loras": [
    {
      "path": "krea2_turbo_distill_r256.safetensors",
      "scale": 0.6,
      "role": "accel"
    }
  ]
}
```

Valid roles are `kroma`, `accel`, `bypass`, and `control`; omit `role` for an
ordinary style/character LoRA. Roles are declarations, not filename guesses.
In particular, Krea-2 distillation files such as
`krea2_turbo_distill_r256.safetensors` must declare `"role":"accel"` when
they fill the accelerator slot. Auto-staging may change `path`, but preserves
`role`.

## Per-request LoRA stacks and the warm default (#282)

**Every render carries its own stack.** A job's adapters are resolved once, at
submit, and applied at dequeue. There are three sources, in strict precedence:

| # | Source | When it wins |
|---|---|---|
| 1 | the request's own `loras` | whenever the key is present — including `"loras": []`, which means *no adapters* |
| 2 | the named `preset`'s expanded stack | when the request sent no `loras` and the preset was expandable (#286) |
| 3 | the **warm default** | only when the request named neither `preset` nor `loras` |

The generate response says which, in the additive `lora_stack_origin` field:
`"request"`, `"preset"` or `"warm_default"` (absent on the ControlNet arm,
which has always rendered its own request's stack). `GET /v1/generate/status/{id}`
carries the same field once the job has dequeued.

**The warm default is only valid for the base it was published under.** A swap
records the family and model spec it applied to. A bare request that dequeues
onto a *different* checkpoint does **not** take that default: it renders with
**no adapters** and the response carries `warm_default_skipped`:

| code | meaning |
|---|---|
| `family_mismatch` | the default was published under another model family |
| `model_mismatch` | same family, different checkpoint |

This is never a 4xx or 5xx — forcing another base's adapters is what could
throw, and a request that always rendered must not start failing. An **empty**
default (clear the adapters) and the engine's launch-time `--lora` stack are
admitted everywhere; an unknown spec on either side is not a mismatch.

**`lora_reload`** is set on the response when a job actually cleared the
resident adapters and bound a different stack, rather than taking the
same-stack shortcut. Alternating bare and preset renders (Krita + Kira at the
same time) legitimately pay a full clear+reload per job; this field and a
matching warning log are how that cost is measured rather than merely assumed.

**What `POST /v1/lora/swap` now means.** The route, its payload and its
response JSON are unchanged. What changed is its scope: a swap publishes the
**warm default** — the stack a request carrying neither `preset` nor `loras`
renders with — instead of mutating a shared resident stack that later jobs
silently inherit. A swap still applies its stack to the resident pipeline
(a swap-first client expects that, and `SwapResidencyRestore` exists so it can
happen against an evicted pipeline), but no later job picks it up except
through the default.

Read the current warm default from `GET /v1/model/pool`, field
`warm_default_stack` (additive, same per-entry shape as `/health.loras`):

```json
{
  "active": "krea2-raw",
  "warm_default_stack": [
    {"name": "kroma.safetensors", "path": "/…/kroma.safetensors", "scale": 0.6, "role": "kroma"}
  ]
}
```

`/health.loras` answers a different question — what is **resident**, i.e. the
last job's stack. Since #282 that is a consequence of the last render, not a
prediction of the next one.

**Consequences for daemon owners.**

- A bare `/v1/generate` no longer inherits the previous job's adapters. On an
  engine launched without `--lora` and with no swap yet, a bare request renders
  with **no adapters** — deterministically, instead of on whatever happened to
  be resident.
- To keep a stack across bare requests, swap it (it becomes the warm default)
  or name it on every request.
- FIBO and Chroma have **no LoRA application path at all**: `ChromaPipeline
  .generate` takes no adapters, the model pool never forwards `initialLoRAs` to
  either family, and `/v1/lora/swap` refuses them. They render with no adapters,
  always. Nothing is applied for them now — a warm default *or* a stack the
  request named — where the request-named case previously loaded into the
  Flux-1 pipeline they do not render through. Nothing 4xx's that did not
  before; the render is the same bare render it always was, and the Chroma PNG
  now records the empty stack it actually used instead of the coordinator's
  unrelated resident list.
- The video path (`/v1/video/generate`) has always resolved its own stack per
  request (`loras` → the video preset's `loras` → `--ltx2-lora`) and is
  unchanged.

## Preset LoRA references (Krea-2)

Preset LoRA references use `filename` rather than `path` and preserve the
same optional `role`:

```json
{
  "loras": [
    {
      "filename": "krea2_turbo_distill_r256.safetensors",
      "scale": 0.6,
      "role": "accel"
    }
  ]
}
```

### Kroma is a regular LoRA (structured `kroma` is DEPRECATED, 2026-09-04)

Todd reversed the #276/#350-era design: kroma has no special engine
semantics anywhere. `PresetLoRAStack.decide` (the `POST /v1/generate
{"preset": id}` expansion) applies a preset's `loras[]` exactly as declared,
in order — no prepend, no strip, no reordering for a `role: "kroma"` entry.
Declare kroma the same way as any other adapter:

```json
{
  "loras": [
    {
      "filename": "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors",
      "scale": 0.6,
      "role": "kroma"
    }
  ]
}
```

**One-release compatibility shim.** The structured `kroma` object
(`{"kroma": {"strength": <number>, "file": <optional>}}`) still decodes on
`PUT /v1/presets`, but `PresetStore` migrates it on every load and save
(`ImagePreset.migratingKromaDeprecation`): a declared `kroma` with an
explicit, non-empty `file` folds into `loras[]` as a `role: "kroma"` entry
(idempotently — a matching entry already present is never duplicated), and
the structured field itself becomes a DERIVED, read-only echo of that
`loras[]` entry, never an independent value a client can set. A `kroma`
with no `file` (the old "engine-default file" case) has nothing concrete to
become a LoRA of — it migrates to nothing, and the echo is `nil`. Every
response that carries a non-nil `kroma` also carries an additive
`"kroma_deprecated": true` marker. The krea2-family "a preset must declare
kroma" validation rule (O4a) is retired along with this — its absence is as
legal as any other adapter's.

Existing consumers that read `.kroma` (the daemon, the desktop app) keep
working unmodified during the compatibility window; new code should read
`loras[]` and stop relying on the structured field. See
[Krea-2 Raw + r256 preset stack](methods/krea2-r256-preset-stack.md) (some
of that document's structured-kroma framing predates this reversal).

## Model detection — `GET /v1/model/family` (comfybox#359)

`GET /v1/model/family?model=<spec>` answers, for one model spec, what the
engine thinks it is and whether it would load it. Read-only: file-existence
checks plus a `model_index.json` read inside a model root. It never loads
weights, never mutates the model pool, and never returns file contents, so it
is safe to call once per preset in a batch.

```json
{
  "model":   "~/LocalModels/krea2-raw",   // echoed verbatim
  "family":  "krea2",                     // "krea2" | "z-image" | null
  "variant": "raw",                       // krea2: turbo|raw; z-image: turbo|base; null
  "spec":    "krea2-raw",                 // canonical engine spec — write this into a preset's `model`
  "loadable": true,                       // would POST /v1/generate {"model": spec} be accepted?
  "reason":  null                         // why not, when loadable is false
}
```

**`spec` is the load-bearing field.** `PresetLoRAStack.decide` returns
`no_model` whenever a preset's `model` is empty and the request names none —
*before* it ever reads `checkpoint_family`. So a preset that carries only
`custom_model_path` cannot be made expandable by writing a
`checkpoint_family` label; the fix is to write `model`. `spec` is the
declared krea2 alias when the probed path matches one (`krea2-raw`,
`kroma-v0.2-turbo`, or a `config.json` `krea2Models` entry), otherwise the
tilde-expanded, standardized absolute path — `ModelResolution.resolve` does
not expand `~`, so an unexpanded tilde path in `model` would fail to load.

`loadable` mirrors the real acceptance path (`WarmServer.parseModelSpec` →
`ModelPool.detectFamily` → `Krea2ModelDetection.resolve` /
`ModelResolution.resolve`) by file existence only. `loadable: true` means the
engine would accept the spec, not that the render will succeed — a truncated
or incomplete checkpoint still reads as loadable here.

`family`/`variant` are the broad half, and `variant` is **never guessed from
text**. Only a declared alias (after the `-q4`/`-q8`/`-bf16` suffixes
`parseModelSpec` strips) or a readable Krea-2 model root yields a variant; a
checkpoint that merely *names* z-image comes back `family: "z-image",
variant: null`. `cyberrealisticZImage_v50.safetensors` is served as BASE and
its filename says neither "base" nor "turbo" — reading turbo off the spelling
put the wrong recipe under the right name, which is what F3 and #286 exist to
prevent.

A null `variant` costs nothing that matters: the five `checkpoint_family`
policy labels are a CLIENT decision (the `raw-accel` vs `raw-stock` split
within Krea-2 "raw" depends on whether the preset's own `loras[]` declares
`role: "accel"`, which the engine does not see here), and
`PresetLoRAStack.declaredFamily` maps `raw-accel` and `raw-stock` to the same
`"krea2"`. The label is a record; `model` is what makes a preset expandable,
and it is written either way.

**Scope note (deliberate):** `model` may be any local path, and the route will
stat it. The warm server is a localhost-trusted process on the Mac; the probe
reveals only existence and Krea-2-shape, never contents.

## VAE selection (Krea-2) — external decoder swap (#285, WP-E9)

Krea 2 can decode (and encode) through a VAE file other than the one bundled
in the model directory — e.g. the Wan 2.1 FP32 decoder the RAW bf16 reference
stack was validated with (`docs/FDD-krea2-raw-recipe.md` §3.9), or a
Krea2-specific fine-tune such as `krea2RealVae_v10.safetensors`. Krea 2 only;
any other family refuses the field outright (400, naming the field and the
family) rather than silently ignoring it.

**Precedence, the same three-source shape LoRAs use (`RequestStackResolver`,
above), collapsed onto one field since a VAE is a single file, not a stack:**

| # | Source | When it wins |
|---|---|---|
| 1 | the request's own `vae` | whenever the key is present |
| 2 | the named `preset`'s declared `vae` | when the request sent no `vae` and the preset was expandable (#286's `PresetLoRAStack`) |
| 3 | the model directory's own VAE | `model_index.json`'s `"vae_file"`, else `vae/diffusion_pytorch_model.safetensors` — the no-regression default |

```json
{"prompt": "…", "vae": "~/LocalModels/vae/Wan2_1_VAE_fp32.safetensors"}
```

`vae` is a path (tilde allowed). **A named file that does not exist FAILS the
render** — there is no fallback to the model directory's VAE; a bad name is a
bad request, not "use the default". The layout (Qwen-Image `diffusers` vs Wan
2.1 native module names) is sniffed from the file's own tensor keys, never
from its filename, and an unrecognised layout is the same kind of hard
failure. A file that is short a parameter the module needs (a truncated
download, or a genuinely different architecture) is refused before any
weight is written — never a decoder left half-swapped.

**The decoder is swapped IN PLACE, never pool-keyed.** A resident Krea 2
pipeline is ~22.5 GB against a 40 GB pool budget; keying the ~500 MB VAE into
the pool would turn every VAE change into a full pipeline evict+reload
(measured ~67 s) for a decoder swap that itself takes a fraction of a second.
Instead the one resident `Krea2VAE` instance reloads its weights in place —
the same object serves both encode and decode before and after, so an
img2img request can never have its encoder and decoder disagree — and a
reload counter tracks how often it actually happened.

**The response always says what decoded**, in the `applied` recipe echo
(both the sync `/v1/generate` response and `/health`'s `last_recipe`):

```json
{"applied": {"vae": "/Users/todd/LocalModels/vae/Wan2_1_VAE_fp32.safetensors",
             "vae_layout": "wanNative", "vae_source": "preset"}}
```

`vae_source` is `"payload"`, `"preset"` or `"model_dir"` — never guessed, and
never `"preset"` unless the render actually took the preset's declared value
(a request `vae` always wins, even against a preset that also declares one).
A crash-recovery replay of a persisted job carries its accepted `vae` as an
explicit request field (the same #286-era rewrite `model`/`loras`/`steps`
already go through), so a replay reports `vae_source: "payload"` even for an
originally preset-sourced render — the price of replaying a frozen body
instead of re-resolving a preset that may have changed since.

## Schedule shift — `shift` (comfybox#154, and Krea 2's D3)

`shift` is one request field with a **family-dependent meaning**, because
ComfyUI has two different nodes for it and the engine reproduces both faithfully
rather than averaging them. It is validated as a positive finite number and
REFUSED (400, naming the field and the family) on a family that does not read
it — never silently ignored.

| Family | ComfyUI node | What `shift` is | Neutral value |
|---|---|---|---|
| Z-Image / Flux 1 (`flux1`) | `ModelSamplingAuraFlow` | the LINEAR shift `σ' = shift·σ / (1 + (shift − 1)·σ)` | `1.0` (exact identity) |
| Krea 2 (`krea2`) | `ModelSamplingFlux` | `mu`, a LOG-shift: `σ' = e^shift / (e^shift + 1/σ − 1)` | `1.15` reproduces the published grid |
| `flux2`, `fibo`, `chroma` | — | not read; refused | — |

```json
{"prompt": "…", "shift": 3.0}
```

**Z-Image family — what the shift does.** It is ComfyUI's
`ModelSamplingAuraFlow` node, which patches the model's whole `model_sampling`
object; ComfyBox reproduces that as a value
(`ZImageSchedulerConfig.applyingExplicitShift`). Higher shift holds more noise
into the early steps, which is what recovers structural coherence on
flow-matching art models. Zeta Chroma's author (Lodestone) publishes
**shift 3.00, sampler Euler, schedule `simple`/`normal`, CFG 4.5–5.5**.

**Precedence — an explicit shift REPLACES the model's schedule, it never
composes with it.** That is what the node does upstream (there is no ComfyUI
graph where `ModelSamplingAuraFlow` and `ModelSamplingFlux` both apply), and
composing them would double-warp a grid whose `mu` the caller cannot see:

| # | Source | When it wins |
|---|---|---|
| 1 | the request's own `shift` | whenever the key is present |
| 2 | the named `preset`'s DECLARED `shift` | when the request sent none, the preset was expandable, **and the preset's declared family is Z-Image** |
| 3 | the model's own schedule | `use_dynamic_shifting` + the resolution-dependent `mu`, else `scheduler_config.json`'s `shift` — the no-regression default |

**Row 2 is Z-Image-only, on purpose.** A preset declares its family with
`checkpoint_family` (`zimage-base` / `zimage-turbo`) or with a `model` spec the
engine classifies; a preset that says neither has its `shift` ignored (fails
closed). The gate exists because four live krea2 presets — `krea-kira`,
`krea-kira-sfw`, `krea-kira-avocado`, `krea2-base` — already declare
`shift: 1.15`, which was desktop-display-only before #154. **Krea 2 takes its
shift from the REQUEST only, exactly as it did**, so those renders are unchanged.

Omitting `shift` on the Z-Image family is **byte-identical** to the pre-#154
engine: the same sigma grid, bit for bit (`ModelSamplingShiftTests`). **Krea 2
is unchanged either way** — this ticket adds nothing to its shift path. Because
the shift is applied to the scheduler CONFIG rather than to one schedule's
formula, the schedules that index the model's discrete sigma table — `simple`,
`beta`, `beta57`, and the `karras`/`exponential` bounds — pick it up too,
exactly as they read the patched `model_sampling` in ComfyUI.

**A shift a schedule would ignore is REFUSED (400), not accepted.** Three
schedules are defined by something other than the model's shift and would drop
it silently: `krea2` (which is `mu`), `bong_tangent` (model-free index
arithmetic), and — under Krea 2's `ModelSamplingFlux` sampling — the
table-backed ones. Asking for `shift` alongside one of those on the Z-Image
family returns a 400 naming the schedule. This is what keeps `applied_shift`
honest: it only ever reports a number that reached the sigma grid.

**The ComfyUI `multiplier` cancels out of the sigma grid**, which is why the
bridge reads `ModelSamplingSD3` (multiplier 1000) exactly as it reads
`ModelSamplingAuraFlow` (multiplier 1.0) — same `shift`, same sigmas.
`normal_scheduler` walks `linspace(timestep(σ_max) → timestep(σ_min))` and maps
each point back through `sigma(t) = time_snr_shift(shift, t / M)` while
`timestep(σ) = σ · M`, so the `M` introduced by the endpoints is divided straight
back out (pinned by
`ModelSamplingShiftTests.testMultiplierCancelsOutOfTheSigmaGrid`). What the
multiplier DOES scale is the timestep the model itself is fed — a property of how
the checkpoint was trained, **not changed by this field**: the Z-Image pipelines
feed `σ × num_train_timesteps` exactly as they always have.

**The response says what applied**, as a flat `applied_shift` on both
`POST /v1/generate` and `GET /v1/generate/status/{id}`:

```json
{"success": true, "output_path": "…", "applied_shift": 3.0}
```

The key is **absent** when the render used the model's own schedule — which is
every render that names no shift, so its absence is the unchanged contract. It
is set on the Z-Image and ControlNet arms only; **Krea 2 answers through its
full recipe instead** (`applied.shift`, `applied.shift_source`, and
`applied.stages[].shift_applied`, which is `false` for a schedule such as
`bong_tangent` that ignores it — `RenderRecipe` is Krea 2 only, D12). A flat
field beside that richer record would be a second, weaker claim about the same
render.

**Bridge (Krita / ComfyUI clients).** A workflow carrying a
`ModelSamplingAuraFlow` (or `ModelSamplingSD3`) node has its `shift` input read
and applied — but only when the resident family is Z-Image. On any other family
the workflow is **refused** with a 400 naming the node and the family, because
the node's linear shift and Krea 2's log-shift `mu` are different quantities
wearing the same number. (A ControlNet workflow takes an earlier branch that
already refuses every non-Z-Image family with `controlNetNotSupported`, so on
that path the client sees that error instead.) See
`bridge-developer-guide.md`.

## Gallery output filenames

Default render filenames (no `output_path` in the request) are built by
`ComfyBoxOutputNaming.defaultFilename` (`Sources/ZImage/Server/ComfyBoxOutputNaming.swift`):

```
comfybox-<model>[-<preset>]-<tier>[-<source>]-<yyyyMMdd-HHmmss>-<4-hex-salt>.<ext>
```

e.g. `comfybox-krea2-avocado-20260904-143022-a3f2.png`. `<model>` is the
short name of the active model spec (`krea2`, `kroma-v0.2`, `fibo`, …),
`<tier>` is the request's content mode (`manual` when absent). This
replaced the legacy `zimage-<uuid>.png` / `zimage-krea2-<uuid>.png` scheme
in 2026-08 (commits `3ed1996`, `1cb123e`) — the model segment already
carries the family, so **no code should reintroduce a hardcoded `zimage-`
prefix on a persisted gallery file** (issue #251).

Nothing in ComfyBox or the daemon (`coffeeshop-server`) parses this prefix
to classify a render — model family, mode, preset etc. all come from the
JSON metadata embedded in the PNG itself (`ComfyBoxCatalog/MetadataReader`)
or from the request/response body, never from filename text. Existing
`zimage-krea2-*` files on disk from before the 2026-08 change keep working
unmodified — nothing needs to read or migrate them. `WarmServer.swift`'s
`"zimage-…"` temp-file names (control image, mask, inpaint init, ESRGAN
scratch files) are unrelated: they're process-local scratch paths deleted
before the response returns, never a persisted gallery filename, and are
unreachable for Krea-2 (`ControlNet is not supported for Flux 2 or
Krea-2 models` — the route throws before any temp file is written).

## Queue lifecycle ledger (comfybox#283 / comfybox#217)

`GET /v1/queue/lifecycle?job_id=&limit=` — read-only, append-only record of
what actually happened to queue jobs: `enqueued`, `admitted`, `started`,
`progress` (bounded rate — see below), `checkpointed`, `resumed`, `abandoned`
(a checkpointed video was NOT resumed — an operator interrupt arrived during
the preemption episode, so the checkpoint was dropped instead; distinct from
`resumed` since PR #370 review round 1, which found the original version
recorded `resumed` even on this path), `interrupted`, `completed`, `failed`,
`replayed_after_restart`, `dropped`.
Built as the TELEMETRY that #283 (a restart re-enqueues the active job and
re-renders it from step 1, and nothing reported that accurately) and #217
(the Desktop queue/progress UI goes stale during a render) need before either
issue's proposed behavior changes can be evaluated safely — it changes
nothing about queue behavior itself.

Query params: `job_id` (optional — filter to one job) and `limit` (optional,
default 200, clamped to 1–2000). Response:

```json
{"boot_id": "…", "count": 3, "events": [
  {"sequence": 41, "boot_id": "…", "wall_time": "2026-09-04T…Z", "job_id": "…",
   "kind": "admitted", "job_kind": "generate", "source": "api"}
]}
```

`boot_id` is a fresh UUID generated once per process start — two events with
different `boot_id`s straddle a restart, which is the direct answer to
#283's "nothing distinguishes a recovered job from a new one." `sequence` is
a process-wide (not per-job) monotonic counter, reseeded from the JSONL tail
on restart so it never resets to 0.

Additive fields on existing routes, `null`/absent-safe for older clients:

- `GET /v1/queue`: each `pending[]` entry gains `last_event`; the response
  gains `active_last_event` when a job is active. Both are one
  `QueueLifecycleEvent` object (see above), or absent if the ledger has
  never seen that job (e.g. a snapshot recovered from before this instrument
  shipped).
- `GET /v1/generate/status/{id}`: gains `lifecycle_tail`, the last 5 events
  recorded for that job id, or absent if none.

**Diagnosing #283 from the ledger**: after a bounce, `GET
/v1/queue/lifecycle?job_id=<the id from queue-state.json>` shows the pre-crash
history ending mid-render (no `completed`/`failed`), followed by a
`replayed_after_restart` event under a NEW `boot_id` — `from_step1: true`
(image generate/LoRA swap never checkpoint today, so this is currently
always the answer) confirms the render actually restarted from step 1 rather
than resuming, and `original_job_id` names the job. This is the
operator-visible signal #283 finding 1 says is missing; it does not by
itself change whether a restart drops or resumes the job — that is #283's
open decision, not this instrument's.

Storage: an in-memory ring (default last 4000 events, actor-hop-free reads,
a real circular buffer — no per-insert array shift) plus
`~/.comfybox/queue-lifecycle.jsonl` (append-only, survives a restart; honors
`COMFYBOX_STATE_DIR` like every other engine state path). `progress` events
are throttled to at most one per job per second — a fast-ticking render
cannot flood either store.

**Disk footprint and write path** (PR #370 review round 1, C1/C2). Writes
are asynchronous: `record` appends to a bounded in-memory buffer (default
2000 events) and returns immediately; a dedicated background queue drains
that buffer to disk in batches, separately from the lock that guards the
ring — so a slow or near-full disk can never stall a caller, including
`GET /v1/queue`/`GET /v1/queue/lifecycle` answered by the sync control
plane specifically so they stay responsive during a render (#217). If the
writer ever falls behind that buffer's bound (a stalled disk), the OLDEST
queued writes are dropped — counted, never silent, and the in-memory ring
is completely unaffected (a dropped write only ever costs one event's
durability, never its visibility for the life of the process).

The file itself is bounded by rotation: it rotates once appending would
push it past 20 MB, keeping 2 generations (`queue-lifecycle.jsonl` +
`queue-lifecycle.jsonl.1`) — worst case **~40 MB** on disk. A fresh
`QueueLifecycleLedger()` never reads this file at construction time (that
would block `WarmServer.init`, i.e. engine startup, on however large the
file has grown); instead it reseeds LAZILY, on the first real `record`/
`events` call, and reads only the trailing 64 KB (not the whole file) —
bounded regardless of how large the file gets between rotations, and
tolerant of a truncated/malformed line at either end of that window (a
crash mid-write, or the window simply starting mid-line).

See `Sources/ZImage/Server/QueueLifecycleLedger.swift` for the full event
schema and `ReplayClassifier`'s pure from-step-1-vs-resumed logic.

## Startup imports

- Character + preset legacy imports also run **once at server startup**
  (idempotent), merging from `~/.coffeeshop/image-service/`.
