# Claude Handoff: Stock Render Regression Investigation

Date: March 28, 2026

## Current State

- Repo: `/Users/toddwalderman/Projects/zimage.swift`
- Current checkout: detached `HEAD` at commit `3472686` (`feat: add model weight audit command`)
- Worktree: clean
- Current branch tips nearby:
  - `3472686` `feat: add model weight audit command`
  - `a37583f` `fix: generalize final-layer adaln aliases`
  - `0dee852` `fix: normalize lora delta orientation`
- Stashes still present:
  - `stash@{0}` `codex: notes and specs after split commits`
  - `stash@{1}` `codex: clean worktree after 0dee852`

## User-Reported Problem

On March 28, 2026, the user reported that the stock model with **no LoRA** is now producing artifacts on the current build. Earlier that same day, stock model + no LoRA had rendered cleanly.

The initial suspicion was commit `0dee852` (`fix: normalize lora delta orientation`).

## What I Checked

### 1. Commit `0dee852` scope

`0dee852` only changes:

- `Sources/ZImage/LoRA/LoRAApplicator.swift`
- `Tests/ZImageTests/Weights/LoRAApplicatorTests.swift`

It does **not** touch:

- model loading
- transformer loading
- VAE loading
- tokenizer/text-encoder loading
- base inference path
- CLI argument handling

### 2. No-LoRA request path

In `Sources/ZImage/Pipeline/ZImagePipeline.swift`:

- `ZImageGenerationRequest` defaults to `loras: []`
- `generateCore(...)` only calls `loadLoRAs(...)` when `!request.loras.isEmpty`
- if `request.loras` is empty and `currentLoRAs` is also empty, `LoRAApplicator` is never reached

Relevant code path:

- `Sources/ZImage/Pipeline/ZImagePipeline.swift` around the `if !request.loras.isEmpty { ... } else if !currentLoRAs.isEmpty { unloadLoRA() }` block

Conclusion: on a **fresh process** with **no `--lora` argument**, commit `0dee852` should not affect the stock render path.

### 3. Other nearby commits

The commits after `0dee852` are:

- `a37583f` `fix: generalize final-layer adaln aliases`
- `3472686` `feat: add model weight audit command`

`3472686` adds:

- `--audit-weights` CLI plumbing
- `ZImageModelAudit.swift`
- public audit helpers in `WeightsAudit.swift`
- main CLI `do/catch` wrapper

That commit does not appear to change render math.

`a37583f` changes transformer weight alias normalization:

- generalizes alias creation for any `all_final_layer.*.adaLN_modulation.0.{weight,bias}`
- creates `.1.{weight,bias}` aliases for those keys

Of the three commits above, `a37583f` is the only one that plausibly touches a base no-LoRA stock render.

## Likely Suspect Ranking

1. `a37583f` `fix: generalize final-layer adaln aliases`
2. local runtime/build/cache state during the repro
3. `3472686` `feat: add model weight audit command` is unlikely
4. `0dee852` `fix: normalize lora delta orientation` is very unlikely for a fresh no-LoRA run

## Verification Already Done

I verified code-level consistency and ran the repo-approved validations:

- `xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests`
- `xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode build`

Both passed on the restored code stack.

I did **not** run a manual stock/no-LoRA render in this handoff because project instructions in `AGENTS.md` / `CLAUDE.md` say to limit verification to `-only-testing:ZImageTests` and leave e2e/integration/manual verification to the user.

## Recommended Next Step

Use temp worktrees instead of reverting in place. That keeps the current checkout intact and makes the comparison unambiguous.

### Suggested comparison worktrees

```bash
git worktree add /tmp/zimage-0dee852 0dee852
git worktree add /tmp/zimage-a37583f a37583f
git worktree add /tmp/zimage-3472686 3472686
```

### Suggested build command

```bash
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode build
```

### Suggested stock no-LoRA repro command

Run the same command in each worktree, from a fresh process each time:

```bash
.build/xcode/Build/Products/Release/ZImageCLI \
  -p "portrait of a woman" \
  -m Tongyi-MAI/Z-Image-Turbo \
  -s 8 -W 512 -H 512 --seed 42 \
  -o /tmp/zimage-stock-test.png
```

### Decision rule

- If `0dee852` is clean and `a37583f` is corrupted, revert `a37583f`
- If `a37583f` is clean and `3472686` is corrupted, inspect CLI/runtime startup changes
- If all three are clean, the reported regression is likely from local cache/state or a different checkout than expected

## Bottom Line

The user-reported regression is real enough to investigate, but the current code evidence does **not** support `0dee852` as the cause of a fresh stock no-LoRA corruption. The better first target is `a37583f`, with a controlled worktree-based render comparison.
