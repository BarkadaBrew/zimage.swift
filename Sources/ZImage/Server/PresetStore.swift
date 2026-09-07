// PresetStore.swift — Generation presets, persisted to ~/.comfybox/presets.json.
//
// Straight port of the Coffee Shop image service's ImagePreset concept and preset
// resolution (see coffeeshop-image-service: src/types.ts `ImagePreset`,
// src/service.ts `resolveJobRequest`/`validatePreset`, src/config.ts `DEFAULT_PRESETS`).
//
// A preset is a reusable bundle of generation parameters (prompt bits, engine/model/mode,
// steps, guidance, dimensions, LoRAs, …) keyed by id/name/description. `resolve(id)` merges
// a preset onto system defaults to yield a fully-populated parameter set — the behavior the
// Node service exposed at `/v1/presets/resolve`.
//
// Persistence mirrors ``ComfyBoxServerConfig``: JSON under ~/.comfybox, tolerant decode
// (older/partial files load with defaults), atomic writes.

import Foundation
import Logging

// MARK: - LoRA reference

/// One LoRA a preset applies, by filename + scale and an optional semantic
/// slot. `role` is declared metadata — never inferred from a filename. It is
/// what lets clients distinguish a Krea-2 accelerator such as
/// `krea2_turbo_distill_r256.safetensors` from an ordinary style LoRA.
public struct LoraReference: Codable, Equatable, Sendable {
  public var filename: String
  public var scale: Double
  public var role: String?

  public init(filename: String, scale: Double, role: String? = nil) {
    self.filename = filename
    self.scale = scale
    self.role = role
  }
}

// MARK: - Post-render upscale

/// Optional post-render upscale config. When enabled, callers auto-chain an upscale job
/// after the base render. Port of `ImagePreset.upscale` (types.ts).
public struct PresetUpscale: Codable, Equatable, Sendable {
  public var enabled: Bool
  public var mode: String?   // "seedvr2" | "controlnet"
  public var scale: Double?  // default 2

  public init(enabled: Bool, mode: String? = nil, scale: Double? = nil) {
    self.enabled = enabled
    self.mode = mode
    self.scale = scale
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
    mode = try c.decodeIfPresent(String.self, forKey: .mode)
    scale = try c.decodeIfPresent(Double.self, forKey: .scale)
  }
}

// MARK: - Kroma policy (WP-E20, D14)

/// DEPRECATED (Todd 2026-09-04): kroma is a regular LoRA now, not a
/// first-class field — see ``ImagePreset/kroma`` and
/// ``ImagePreset/migratingKromaDeprecation(_:)``. This type only still
/// exists as the shape of the one-release compatibility shim: a client may
/// still PUT `{"kroma": {...}}`, which migrates into `loras[]`; every GET
/// echoes it back as a DERIVED, read-only view (never an independent
/// value), alongside `kromaDeprecated: true`. O4a (a krea2-family image
/// preset must declare `kroma`) is retired along with the rest of this.
public struct KromaPolicy: Codable, Equatable, Sendable {
  public var strength: Double
  public var file: String?

  public init(strength: Double, file: String? = nil) {
    self.strength = strength
    self.file = file
  }
}

// MARK: - Bypass policy (WP-E8, D10, FDD §3.8, ledger ruling 17:35)

/// The censorship-bypass `.diff` LoRA as a FIRST-CLASS preset dial, mirroring
/// ``KromaPolicy`` exactly: `strength: 0` is a declaration ("no bypass"), and
/// `file` nil means the workflow's own artifact
/// (``Krea2BypassPolicy/workflowFile``) rather than "whatever is around".
///
/// Unlike `kroma`, an ABSENT `bypass` is not a configuration error: the
/// family default is DERIVED from kroma (17:35 — kroma already unlocks, so
/// the bypass is wanted only on a kroma-free preset). See
/// ``Krea2BypassPolicy/resolve(bypass:kroma:requestStrength:)``.
public struct BypassPolicy: Codable, Equatable, Sendable {
  public var strength: Double
  public var file: String?

  public init(strength: Double, file: String? = nil) {
    self.strength = strength
    self.file = file
  }

  /// Whether anything is applied at all. `strength == 0` is a declaration,
  /// not a request to load a file.
  public var isActive: Bool { strength > 0 }
}

/// The bypass strength policy (D10 + the 17:35 ledger ruling), in one place —
/// the engine half of the dial whose family table lives on the daemon (C8).
///
/// **The engine invents no strength.** The only number here is the one the
/// reference workflow declares, and it is quoted, not chosen: FDD §3.8
/// ("the preset default is **1.0** — the workflow author's figure, and the
/// figure the reference recipe is defined by") and §3.15's `krea2-reference`
/// stack line, `{ "filename": "krea2filterbypass_2vector.safetensors",
/// "scale": 1.0 }`. The Fedor artifact's own `__metadata__` recommends
/// 3.0–5.0; that is RECORDED (``fedorRecommendedStrength``) and not adopted
/// (§9 Q4).
public enum Krea2BypassPolicy {

  /// The workflow's declared strength (FDD §3.8 / §3.15). Not a guess.
  public static let workflowStrength: Double = 1.0

  /// The workflow's own artifact — what `krea2-reference` names.
  public static let workflowFile = "krea2_filter_bypass_2vector.safetensors"

  /// The substitute (civitai 2746817). Selectable by naming it in
  /// `bypass.file`; never the default.
  public static let fedorFile = "krea2_filter_bypass_fedor.safetensors"

  /// Fedor's author's divergent guidance, recorded so the 5× gap stays
  /// visible in the code that does not adopt it (§9 Q4).
  public static let fedorRecommendedStrength: ClosedRange<Double> = 3.0...5.0

  /// The one rule, in precedence order:
  ///
  /// 1. a per-render override (`bypass_strength` on the request) wins;
  /// 2. an explicit preset `bypass` wins next;
  /// 3. otherwise the DERIVED default is **0** — always off (Todd 2026-08-31:
  ///    projector_scale + adherence make the auto-loaded bypass unnecessary;
  ///    this retires the 17:35 "kroma off ⇒ workflow strength" branch).
  ///
  /// The returned policy always names the effective `file`, so provenance can
  /// never be ambiguous about which of the two artifacts applied — even when
  /// the preset left it to the default. `strength == 0` means nothing is
  /// loaded and the file is not consulted.
  public static func resolve(
    bypass: BypassPolicy?, kroma: KromaPolicy?, requestStrength: Double? = nil
  ) -> BypassPolicy {
    let file = bypass?.file ?? workflowFile
    if let requestStrength {
      return BypassPolicy(strength: requestStrength, file: file)
    }
    if let bypass {
      return BypassPolicy(strength: bypass.strength, file: file)
    }
    // Derived. `nil` here means a preset with no kroma dial at all —
    // treated as "no kroma", the same as `strength: 0`. (O4a, the rule that
    // used to make this case rare for krea2-family presets, is retired —
    // `nil` is now exactly as common as any other absent adapter.)
    // Todd 2026-08-31: kroma=0 must NOT auto-load the bypass LoRA — projector_scale
    // + adherence make it unnecessary. The DERIVED bypass is now always OFF; an
    // explicit per-render bypass_strength or preset bypass (precedence 1 & 2 above)
    // still turns it on when genuinely wanted. _ = kroma keeps the signature stable.
    _ = kroma
    return BypassPolicy(strength: 0, file: file)
  }

  /// The preset-level entry point. Fail-closed for anything that is not a
  /// krea2-family image preset: only an explicit `bypass` or a per-render
  /// override ever turns the adapter on (since 2026-08-31 the derived default
  /// is 0 everywhere, so the historical "no kroma ⇒ bypass on" hazard for
  /// `zimage-*` presets no longer exists — the guard stays as belt and
  /// braces against any future derivation).
  public static func resolve(
    for preset: ImagePreset, requestStrength: Double? = nil
  ) -> BypassPolicy {
    let isKrea2 = PresetStore.isImagePreset(preset) && PresetStore.resolvesToKrea2Family(preset)
    guard isKrea2 else {
      let file = preset.bypass?.file ?? workflowFile
      if let requestStrength { return BypassPolicy(strength: requestStrength, file: file) }
      if let declared = preset.bypass { return BypassPolicy(strength: declared.strength, file: file) }
      return BypassPolicy(strength: 0, file: file)
    }
    return resolve(bypass: preset.bypass, kroma: preset.kroma, requestStrength: requestStrength)
  }
}

// MARK: - Second-stage recipe (WP-E20, D4)

/// The optional second stage of a two-stage recipe (O5), as a preset declares
/// it: every field optional so a preset can state only what it pins. The
/// request-side shape (`stage2` on `/v1/generate`) is WP-E17's; this is the
/// stored declaration that feeds it.
public struct PresetStage: Codable, Equatable, Sendable {
  public var sampler: String?
  public var sigmaSchedule: String?
  public var steps: Int?
  public var denoise: Double?
  public var eta: Double?
  public var bongmath: Bool?

  public init(
    sampler: String? = nil,
    sigmaSchedule: String? = nil,
    steps: Int? = nil,
    denoise: Double? = nil,
    eta: Double? = nil,
    bongmath: Bool? = nil
  ) {
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
    self.steps = steps
    self.denoise = denoise
    self.eta = eta
    self.bongmath = bongmath
  }
}

// MARK: - ImagePreset

/// A named, reusable set of generation parameters. Port of `ImagePreset` (types.ts).
///
/// Decoding is tolerant: only `id`/`name` are truly needed to round-trip; every other field
/// falls back to nil / empty so partial or older files load cleanly. Enum-like fields
/// (`mediaKind`, `provider`, `engine`, `mode`, `model`) are kept as free-form strings for the
/// same forward-compatibility reason the Node config used string unions.
public struct ImagePreset: Codable, Equatable, Sendable, Identifiable {
  public var id: String
  public var name: String
  public var description: String

  // Routing / engine selection.
  public var mediaKind: String?   // "image" | "video"
  public var provider: String?    // "local" | "replicate" | "auto"
  public var engine: String?      // "mflux" | "zimage"
  public var mode: String?        // MfluxExecutable (e.g. "z-image-turbo", "generate")
  public var model: String?       // SupportedModel or custom
  public var customModelPath: String?
  public var baseModel: String?

  // Prompt shaping.
  public var prompt: String?
  public var negativePrompt: String?
  public var promptPrefix: String?
  public var promptSuffix: String?
  public var injectedKeywords: [String]?

  // Numeric generation params.
  public var steps: Int?
  /// Tier A video tuning block (task #9 Phase 2) — preset-level overrides
  /// resolved between request fields and config.json/env.
  public var videoTuning: LTX2VideoTuning?
  public var guidance: Double?
  /// Krea2 projector-scale gain (CFG-free prompt adherence). nil = neutral (1.0).
  public var projectorScale: Double?
  /// RES4LYF spatial-noise recipe. All four fields are opt-in; nil preserves
  /// the engine's established gaussian/alpha-0/explicit-RK/c2-0.5 defaults.
  public var noiseType: String?
  public var noiseAlpha: Double?
  public var implicitSteps: Int?
  public var c2: Double?
  public var seed: Int?
  public var width: Int?
  public var height: Int?

  // Adapters + scheduler + post-processing.
  public var loras: [LoraReference]
  public var scheduler: String?
  public var upscale: PresetUpscale?
  /// WP-E9 (FDD §3.9, D16): path of the VAE file this preset decodes through.
  /// nil = the model directory's VAE (the no-regression default). Wan is
  /// never ambient — it is a named field that appears in every record.
  public var vae: String?

  // WP-E20 (FDD §3.15): the recipe as a preset declares it. Every one of these
  // must appear at ALL FIVE sites (stored property, CodingKeys, init(from:),
  // memberwise init, ResolvedPreset) — the `videoTuning` lesson.
  /// Client policy label (D7): "turbo" | "raw-accel" | "raw-stock" |
  /// "zimage-turbo" | "zimage-base". Never a physical fact — that is
  /// `Krea2Variant`, which is reported, not requested.
  public var checkpointFamily: String?
  /// DEPRECATED (Todd 2026-09-04 — kroma has no special engine semantics;
  /// it is a regular LoRA, applied via `loras[]` like any other). Kept for
  /// one release as a compatibility shim: no longer an independent
  /// declaration a client can set — `PresetStore` recomputes it as a
  /// DERIVED, read-only mirror of whichever `loras[]` entry carries
  /// `role: "kroma"` (nil when none does). See
  /// `ImagePreset.migratingKromaDeprecation` and ``kromaDeprecated``.
  public var kroma: KromaPolicy?
  /// Present and `true` only when ``kroma`` is non-nil — omitted otherwise,
  /// so an ordinary (non-krea2) preset's JSON is not littered with
  /// `"kroma_deprecated": false`. Client input is ignored; the engine always
  /// recomputes it.
  public var kromaDeprecated: Bool?
  /// Additive, engine-generated notes about what the kroma-deprecation
  /// migration could not preserve — currently only `"kroma_dropped_no_file"`
  /// (a structured `kroma` declared active with no `file`: there is no
  /// engine-side family→default-file table, FDD §3.17, so nothing concrete
  /// exists to become a LoRA of). nil when migration had nothing to report.
  public var migrationNotes: [String]?
  /// WP-E8 (§3.8, D10): the bypass `.diff` LoRA as a declared dial. ABSENT is
  /// legal and means the kroma-derived default —
  /// ``Krea2BypassPolicy/resolve(for:requestStrength:)``.
  public var bypass: BypassPolicy?
  /// Sampler name, as `/v1/generate` accepts it (`res_2s`, `dpmpp_2m`, …).
  public var sampler: String?
  /// Sigma-schedule name (`beta`, `karras`, `flow`, …).
  public var sigmaSchedule: String?
  /// Explicit flow shift (D3: the reference preset states 1.15).
  public var shift: Double?
  /// SDE eta (T2). 0 = deterministic.
  public var eta: Double?
  /// RES4LYF bongmath fixed point (T3).
  public var bongmath: Bool?
  /// Optional second-stage detail pass (O5, D4).
  public var stage2: PresetStage?

  public init(
    id: String,
    name: String,
    description: String = "",
    mediaKind: String? = nil,
    provider: String? = nil,
    engine: String? = nil,
    mode: String? = nil,
    model: String? = nil,
    customModelPath: String? = nil,
    baseModel: String? = nil,
    prompt: String? = nil,
    negativePrompt: String? = nil,
    promptPrefix: String? = nil,
    promptSuffix: String? = nil,
    injectedKeywords: [String]? = nil,
    steps: Int? = nil,
    guidance: Double? = nil,
    projectorScale: Double? = nil,
    noiseType: String? = nil,
    noiseAlpha: Double? = nil,
    implicitSteps: Int? = nil,
    c2: Double? = nil,
    seed: Int? = nil,
    width: Int? = nil,
    height: Int? = nil,
    loras: [LoraReference] = [],
    scheduler: String? = nil,
    upscale: PresetUpscale? = nil,
    vae: String? = nil,
    checkpointFamily: String? = nil,
    kroma: KromaPolicy? = nil,
    kromaDeprecated: Bool? = nil,
    migrationNotes: [String]? = nil,
    bypass: BypassPolicy? = nil,
    sampler: String? = nil,
    sigmaSchedule: String? = nil,
    shift: Double? = nil,
    eta: Double? = nil,
    bongmath: Bool? = nil,
    stage2: PresetStage? = nil
  ) {
    self.id = id
    self.name = name
    self.description = description
    self.mediaKind = mediaKind
    self.provider = provider
    self.engine = engine
    self.mode = mode
    self.model = model
    self.customModelPath = customModelPath
    self.baseModel = baseModel
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.promptPrefix = promptPrefix
    self.promptSuffix = promptSuffix
    self.injectedKeywords = injectedKeywords
    self.steps = steps
    self.guidance = guidance
    self.projectorScale = projectorScale
    self.noiseType = noiseType
    self.noiseAlpha = noiseAlpha
    self.implicitSteps = implicitSteps
    self.c2 = c2
    self.seed = seed
    self.width = width
    self.height = height
    self.loras = loras
    self.scheduler = scheduler
    self.upscale = upscale
    self.vae = vae
    self.checkpointFamily = checkpointFamily
    self.kroma = kroma
    self.kromaDeprecated = kromaDeprecated
    self.migrationNotes = migrationNotes
    self.bypass = bypass
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
    self.shift = shift
    self.eta = eta
    self.bongmath = bongmath
    self.stage2 = stage2
  }

  private enum CodingKeys: String, CodingKey {
    case id, name, description
    case mediaKind, provider, engine, mode, model, customModelPath, baseModel
    case prompt, negativePrompt, promptPrefix, promptSuffix, injectedKeywords
    case steps, guidance, projectorScale, noiseType, noiseAlpha, implicitSteps, c2, seed, width, height
    case loras, scheduler, upscale
    // Missing until 2026-08-07: with it absent, BOTH the custom decoder and
    // the synthesized encoder dropped videoTuning — every preset-level Tier-A
    // tuning write since task #9 Phase 2 silently vanished on the JSON/API
    // path. The desktop tuning UI was writing values nothing ever read.
    case videoTuning
    // WP-E9: same regression class — a field must be listed here AND in the
    // custom decoder, or both directions silently drop it.
    case vae
    // WP-E20: the nine recipe/policy fields (AC-58 round-trips every one).
    case checkpointFamily, kroma, sampler, sigmaSchedule, shift, eta, bongmath, stage2
    // Todd 2026-09-04: additive deprecation marker for `kroma` — see its doc comment.
    case kromaDeprecated, migrationNotes
    // WP-E8: the tenth. Same regression class — listed here AND decoded below.
    case bypass
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decodeIfPresent(String.self, forKey: .id) ?? ""
    name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
    description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
    mediaKind = try c.decodeIfPresent(String.self, forKey: .mediaKind)
    provider = try c.decodeIfPresent(String.self, forKey: .provider)
    engine = try c.decodeIfPresent(String.self, forKey: .engine)
    mode = try c.decodeIfPresent(String.self, forKey: .mode)
    model = try c.decodeIfPresent(String.self, forKey: .model)
    customModelPath = try c.decodeIfPresent(String.self, forKey: .customModelPath)
    baseModel = try c.decodeIfPresent(String.self, forKey: .baseModel)
    prompt = try c.decodeIfPresent(String.self, forKey: .prompt)
    negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
    promptPrefix = try c.decodeIfPresent(String.self, forKey: .promptPrefix)
    promptSuffix = try c.decodeIfPresent(String.self, forKey: .promptSuffix)
    injectedKeywords = try c.decodeIfPresent([String].self, forKey: .injectedKeywords)
    steps = try c.decodeIfPresent(Int.self, forKey: .steps)
    guidance = try c.decodeIfPresent(Double.self, forKey: .guidance)
    projectorScale = try c.decodeIfPresent(Double.self, forKey: .projectorScale)
    noiseType = try c.decodeIfPresent(String.self, forKey: .noiseType)
    noiseAlpha = try c.decodeIfPresent(Double.self, forKey: .noiseAlpha)
    implicitSteps = try c.decodeIfPresent(Int.self, forKey: .implicitSteps)
    c2 = try c.decodeIfPresent(Double.self, forKey: .c2)
    seed = try c.decodeIfPresent(Int.self, forKey: .seed)
    width = try c.decodeIfPresent(Int.self, forKey: .width)
    height = try c.decodeIfPresent(Int.self, forKey: .height)
    // Tolerate a malformed `loras` value by treating it as empty rather than failing the decode.
    loras = ((try? c.decodeIfPresent([LoraReference].self, forKey: .loras)) ?? nil) ?? []
    scheduler = try c.decodeIfPresent(String.self, forKey: .scheduler)
    upscale = try c.decodeIfPresent(PresetUpscale.self, forKey: .upscale)
    videoTuning = try c.decodeIfPresent(LTX2VideoTuning.self, forKey: .videoTuning)
    vae = try c.decodeIfPresent(String.self, forKey: .vae)
    checkpointFamily = try c.decodeIfPresent(String.self, forKey: .checkpointFamily)
    kroma = try c.decodeIfPresent(KromaPolicy.self, forKey: .kroma)
    // Decoded but never trusted from input — `migratingKromaDeprecation`
    // always recomputes it. A benign placeholder keeps the decode tolerant.
    kromaDeprecated = try c.decodeIfPresent(Bool.self, forKey: .kromaDeprecated)
    migrationNotes = try c.decodeIfPresent([String].self, forKey: .migrationNotes)
    bypass = try c.decodeIfPresent(BypassPolicy.self, forKey: .bypass)
    sampler = try c.decodeIfPresent(String.self, forKey: .sampler)
    sigmaSchedule = try c.decodeIfPresent(String.self, forKey: .sigmaSchedule)
    shift = try c.decodeIfPresent(Double.self, forKey: .shift)
    eta = try c.decodeIfPresent(Double.self, forKey: .eta)
    bongmath = try c.decodeIfPresent(Bool.self, forKey: .bongmath)
    stage2 = try c.decodeIfPresent(PresetStage.self, forKey: .stage2)
  }
}

// MARK: - Kroma deprecation shim (Todd 2026-09-04)
//
// Reverses the #350/#276-era structured-kroma special-casing:
// `PresetLoRAStack.decide` no longer prepends, strips, or otherwise treats
// `kroma` specially — it applies `loras[]` exactly as declared. This is the
// one-release compatibility shim (intent.md: "version or shim, never
// silently change") so a daemon/desktop client still reading `.kroma` does
// not break outright while it migrates off the structured field.
extension ImagePreset {
  /// Fold an active structured `kroma` declaration into `loras[]`, then
  /// replace `kroma`/`kromaDeprecated`/`migrationNotes` with the DERIVED,
  /// read-only view of whatever `role: "kroma"` entry now exists. Called on
  /// load and on every `upsert`, so both an already-migrated preset and a
  /// legacy one still carrying only the structured field converge on the
  /// same canonical form.
  ///
  /// - A `loras[]` entry already tagged `role: "kroma"` — by ANY filename,
  ///   not just the structured field's — means the fold already happened
  ///   (or the caller declared it directly, the now-canonical way): nothing
  ///   is appended, so swapping which file a preset tags `kroma` and
  ///   re-saving never leaves the OLD file behind as a second entry.
  ///   Filenames are compared by `lastPathComponent`, matching
  ///   `PresetLoRAStack.isSameStack`.
  /// - Otherwise the migrated entry is inserted at the FRONT of `loras[]`,
  ///   not appended — this is a render-parity concern, not a style choice:
  ///   before this deprecation, the engine's expanding sender always
  ///   PREPENDED kroma, so a preset migrating for the first time renders
  ///   with the same LoRA application order it always had. A `loras[]`
  ///   entry a caller declares directly (the canonical path going forward)
  ///   is never reordered — only THIS one-time fold inserts at the front.
  /// - A `kroma` declared active (`strength > 0`) with no `file` has
  ///   nothing concrete to become a LoRA of — there is no engine-side
  ///   family→default-file table (FDD §3.17, client policy only). It
  ///   migrates to nothing, the preset stays loadable, and
  ///   `"kroma_dropped_no_file"` is recorded in `migrationNotes` (and
  ///   logged via `log`) so the loss is visible instead of silent.
  static func migratingKromaDeprecation(_ preset: ImagePreset, log: (String) -> Void = { _ in }) -> ImagePreset {
    var out = preset
    out.migrationNotes = nil
    if let kroma = preset.kroma, kroma.strength > 0 {
      let file = kroma.file?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      let hasKromaRole = preset.loras.contains { ($0.role ?? "").lowercased() == "kroma" }
      let hasMatchingFile = !file.isEmpty && preset.loras.contains {
        ($0.filename as NSString).lastPathComponent == (file as NSString).lastPathComponent
      }
      if file.isEmpty {
        let note = "kroma_dropped_no_file"
        out.migrationNotes = [note]
        log("WARNING: preset '\(preset.id)' declares kroma.strength \(kroma.strength) with no "
          + "kroma.file — the engine has no family-default kroma table, so this migrates to "
          + "nothing (\(note))")
      } else if !hasKromaRole && !hasMatchingFile {
        out.loras.insert(LoraReference(filename: file, scale: kroma.strength, role: "kroma"), at: 0)
      }
    }
    if let entry = out.loras.first(where: { ($0.role ?? "").lowercased() == "kroma" }) {
      out.kroma = KromaPolicy(strength: entry.scale, file: entry.filename)
      out.kromaDeprecated = true
    } else {
      out.kroma = nil
      out.kromaDeprecated = nil
    }
    return out
  }
}

// MARK: - Resolution defaults + result

/// System defaults a preset is merged onto in ``PresetStore/resolve(_:)``.
///
/// Mirrors the fallback ladder in the Node `resolveJobRequest`: steps→4, width/height→512,
/// provider→"local", engine→"mflux", mediaKind→"image".
public struct PresetDefaults: Equatable, Sendable {
  public var mediaKind: String
  public var provider: String
  public var engine: String
  public var steps: Int
  public var width: Int
  public var height: Int
  public var guidance: Double?
  public var projectorScale: Double?
  public var noiseType: String?
  public var noiseAlpha: Double?
  public var implicitSteps: Int?
  public var c2: Double?

  public init(
    mediaKind: String = "image",
    provider: String = "local",
    engine: String = "mflux",
    steps: Int = 4,
    width: Int = 512,
    height: Int = 512,
    guidance: Double? = nil,
    projectorScale: Double? = nil,
    noiseType: String? = nil,
    noiseAlpha: Double? = nil,
    implicitSteps: Int? = nil,
    c2: Double? = nil
  ) {
    self.mediaKind = mediaKind
    self.provider = provider
    self.engine = engine
    self.steps = steps
    self.width = width
    self.height = height
    self.guidance = guidance
    self.projectorScale = projectorScale
    self.noiseType = noiseType
    self.noiseAlpha = noiseAlpha
    self.implicitSteps = implicitSteps
    self.c2 = c2
  }

  public static let standard = PresetDefaults()
}

/// A fully-resolved parameter set: a preset merged onto ``PresetDefaults``. Every routing and
/// numeric field callers need to launch a render is populated (nil only where genuinely
/// optional, e.g. `seed`, `model`, `negativePrompt`).
public struct ResolvedPreset: Codable, Equatable, Sendable {
  public var id: String
  public var name: String
  public var description: String

  public var mediaKind: String
  public var provider: String
  public var engine: String
  public var mode: String?
  public var model: String?
  public var customModelPath: String?
  public var baseModel: String?

  public var prompt: String?
  public var negativePrompt: String?
  public var promptPrefix: String?
  public var promptSuffix: String?
  public var injectedKeywords: [String]

  public var steps: Int
  public var guidance: Double?
  /// Krea2 projector-scale gain (CFG-free prompt adherence). nil = neutral (1.0).
  public var projectorScale: Double?
  public var noiseType: String?
  public var noiseAlpha: Double?
  public var implicitSteps: Int?
  public var c2: Double?
  public var seed: Int?
  public var width: Int
  public var height: Int

  public var loras: [LoraReference]
  public var scheduler: String?
  public var upscale: PresetUpscale?
  /// WP-E9: nil = the model directory's VAE.
  public var vae: String?
  // WP-E20: carried through verbatim — a preset's recipe fields have no
  // system default; absent means "the engine's default for the variant",
  // which the record (RenderRecipe, WP-E10) then names.
  public var checkpointFamily: String?
  /// DEPRECATED — see ``ImagePreset/kroma``. A derived, read-only view.
  public var kroma: KromaPolicy?
  /// Optional: present and `true` only when ``kroma`` is non-nil — see
  /// ``ImagePreset/kromaDeprecated``. A non-optional `Bool` here would force
  /// every response to carry `"kroma_deprecated": false` and would fail to
  /// decode an older engine's response that predates this field entirely.
  public var kromaDeprecated: Bool?
  /// See ``ImagePreset/migrationNotes``.
  public var migrationNotes: [String]?
  /// WP-E8: nil = the kroma-derived default (§3.8), which the record then
  /// names as applied.
  public var bypass: BypassPolicy?
  public var sampler: String?
  public var sigmaSchedule: String?
  public var shift: Double?
  public var eta: Double?
  public var bongmath: Bool?
  public var stage2: PresetStage?

  public init(preset: ImagePreset, defaults: PresetDefaults = .standard) {
    id = preset.id
    name = preset.name
    description = preset.description
    mediaKind = preset.mediaKind ?? defaults.mediaKind
    provider = preset.provider ?? defaults.provider
    engine = preset.engine ?? defaults.engine
    mode = preset.mode
    model = preset.model
    customModelPath = preset.customModelPath
    baseModel = preset.baseModel
    prompt = preset.prompt
    negativePrompt = preset.negativePrompt
    promptPrefix = preset.promptPrefix
    promptSuffix = preset.promptSuffix
    injectedKeywords = preset.injectedKeywords ?? []
    steps = preset.steps ?? defaults.steps
    guidance = preset.guidance ?? defaults.guidance
    projectorScale = preset.projectorScale ?? defaults.projectorScale
    noiseType = preset.noiseType ?? defaults.noiseType
    noiseAlpha = preset.noiseAlpha ?? defaults.noiseAlpha
    implicitSteps = preset.implicitSteps ?? defaults.implicitSteps
    c2 = preset.c2 ?? defaults.c2
    seed = preset.seed
    width = preset.width ?? defaults.width
    height = preset.height ?? defaults.height
    loras = preset.loras
    scheduler = preset.scheduler
    upscale = preset.upscale
    vae = preset.vae
    checkpointFamily = preset.checkpointFamily
    kroma = preset.kroma
    kromaDeprecated = preset.kromaDeprecated
    migrationNotes = preset.migrationNotes
    bypass = preset.bypass
    sampler = preset.sampler
    sigmaSchedule = preset.sigmaSchedule
    shift = preset.shift
    eta = preset.eta
    bongmath = preset.bongmath
    stage2 = preset.stage2
  }
}

// MARK: - Errors

public enum PresetStoreError: Error, Equatable, CustomStringConvertible {
  case validation(String)
  case notFound(String)
  /// WP-E20 (AC-44c): a preset that is on disk but failed validation at load.
  /// It stays listed (flagged) so it can be fixed, and it can never resolve.
  case invalid(id: String, reason: String)

  public var description: String {
    switch self {
    case .validation(let m): return "Preset validation failed: \(m)"
    case .notFound(let id): return "Preset not found: \(id)"
    case .invalid(let id, let reason): return "Preset \"\(id)\" is invalid and cannot be selected: \(reason)"
    }
  }
}

// MARK: - PresetStore

/// Persists generation presets to `~/.comfybox/presets.json` and resolves them against
/// system defaults. Thread-safe: all access is guarded by an internal lock.
public final class PresetStore: @unchecked Sendable {

  private let path: URL
  private let fileManager: FileManager
  private let defaults: PresetDefaults
  private let lock = NSLock()
  private var presets: [ImagePreset]
  /// WP-E20 (AC-44c): id → reason for every preset that is on disk but fails
  /// validation. Populated at load, cleared by a successful `upsert`/`delete`.
  private var invalidReasons: [String: String] = [:]
  private let logger: Logger

  /// On-disk envelope. Wrapping the array in an object leaves room for schema growth
  /// (e.g. a future `version` or `presetMap`) without breaking older readers.
  private struct PresetFile: Codable {
    var presets: [ImagePreset]
  }

  /// `~/.comfybox/presets.json`.
  public static func defaultPath() -> URL {
    ComfyBoxServerConfig.homeDirectory().appendingPathComponent(".comfybox/presets.json")
  }

  /// Seed presets written on first run (file absent). Port of `DEFAULT_PRESETS` (config.ts).
  public static let defaultPresets: [ImagePreset] = [
    ImagePreset(
      id: "zimage-chat",
      name: "Z-Image Chat",
      description: "Fast chat lane preset",
      mediaKind: "image",
      provider: "local",
      engine: "zimage",
      mode: "z-image-turbo",
      model: "z-image-turbo",
      steps: 8,
      guidance: 1,
      width: 512,
      height: 512,
      loras: []
    ),
    ImagePreset(
      id: "schnell-hq",
      name: "Schnell HQ",
      description: "Higher-quality mflux preset",
      mediaKind: "image",
      provider: "local",
      engine: "mflux",
      mode: "generate",
      model: "schnell",
      steps: 4,
      guidance: 3.5,
      width: 1024,
      height: 1024,
      loras: []
    ),
  ]

  /// Load presets from `path`. If the file is absent, seed ``defaultPresets`` and persist them.
  /// A malformed/partial file loads tolerantly (recoverable entries survive; the rest default).
  public init(
    path: URL = PresetStore.defaultPath(),
    defaults: PresetDefaults = .standard,
    seedDefaults: Bool = true,
    fileManager: FileManager = .default,
    logger: Logger = Logger(label: "comfybox.presets")
  ) {
    self.path = path
    self.defaults = defaults
    self.fileManager = fileManager
    self.logger = logger

    if fileManager.fileExists(atPath: path.path), let data = try? Data(contentsOf: path) {
      let loaded = PresetStore.decodeEntries(data, log: { logger.warning("\($0)") })
      self.presets = loaded.presets
      self.invalidReasons = loaded.undecodable
    } else if seedDefaults {
      self.presets = PresetStore.defaultPresets
      try? PresetStore.persist(self.presets, to: path, fileManager: fileManager)
    } else {
      self.presets = []
    }
    revalidate()
  }

  /// WP-E20 (AC-44c): run ``validate(_:)`` over every loaded preset. An entry
  /// that fails is logged at error and flagged — it stays in ``list()`` so the
  /// desktop app can show and fix it, but ``resolve(_:)`` refuses it and
  /// ``listing()`` serves it with `invalid: true`. Entries that failed to
  /// decode keep their decode reason.
  public func revalidate() {
    lock.lock(); defer { lock.unlock() }
    var reasons: [String: String] = [:]
    for preset in presets {
      if let decodeReason = invalidReasons[preset.id], decodeReason.hasPrefix(PresetStore.undecodablePrefix) {
        reasons[preset.id] = decodeReason
        continue
      }
      do {
        _ = try PresetStore.validate(preset)
      } catch let error as PresetStoreError {
        guard case .validation(let message) = error else { continue }
        reasons[preset.id] = message
      } catch {
        reasons[preset.id] = error.localizedDescription
      }
    }
    invalidReasons = reasons
    for (id, reason) in reasons.sorted(by: { $0.key < $1.key }) {
      logger.error("Preset \"\(id)\" is invalid and cannot be selected: \(reason)")
    }
  }

  // MARK: Reads

  /// All presets, in insertion order.
  public func list() -> [ImagePreset] {
    lock.lock(); defer { lock.unlock() }
    return presets
  }

  /// The preset with `id`, or nil. Returns flagged presets too — editing
  /// (`upsert`) is how they get fixed; selection goes through ``resolve(_:)``.
  public func get(_ id: String) -> ImagePreset? {
    lock.lock(); defer { lock.unlock() }
    return presets.first { $0.id == id }
  }

  /// WP-E20 (AC-44c): why `id` is invalid, or nil when it is valid/unknown.
  public func validationError(for id: String) -> String? {
    lock.lock(); defer { lock.unlock() }
    return invalidReasons[id]
  }

  /// Ids of every flagged preset, in store order.
  public var invalidPresetIds: [String] {
    lock.lock(); defer { lock.unlock() }
    return presets.map(\.id).filter { invalidReasons[$0] != nil }
  }

  /// One entry of `GET /v1/presets`: the preset's own fields, flat, plus the
  /// validity flag (`invalid`, `invalid_reason`) so nothing downstream can
  /// select a flagged preset without seeing why.
  public struct PresetListing: Encodable, Equatable, Sendable {
    public let preset: ImagePreset
    public let invalid: Bool
    public let invalidReason: String?

    private enum FlagKeys: String, CodingKey { case invalid, invalidReason }

    public func encode(to encoder: Encoder) throws {
      try preset.encode(to: encoder)
      var c = encoder.container(keyedBy: FlagKeys.self)
      try c.encode(invalid, forKey: .invalid)
      try c.encodeIfPresent(invalidReason, forKey: .invalidReason)
    }
  }

  /// All presets with their validity, in insertion order (what the API serves).
  public func listing() -> [PresetListing] {
    lock.lock(); defer { lock.unlock() }
    return presets.map {
      PresetListing(preset: $0, invalid: invalidReasons[$0.id] != nil, invalidReason: invalidReasons[$0.id])
    }
  }

  // MARK: Writes

  /// Insert `preset`, or replace the existing one with the same id. Validates first; on success
  /// the store is persisted atomically. Returns the (validated) stored preset.
  @discardableResult
  public func upsert(
    _ preset: ImagePreset,
    loraLookup: (String) -> LoRALibraryEntry? = { _ in nil }
  ) throws -> ImagePreset {
    let validated = try PresetStore.validate(
      preset, log: { [logger] in logger.warning("\($0)") }, loraLookup: loraLookup)
    lock.lock(); defer { lock.unlock() }
    if let idx = presets.firstIndex(where: { $0.id == validated.id }) {
      presets[idx] = validated
    } else {
      presets.append(validated)
    }
    try PresetStore.persist(presets, to: path, fileManager: fileManager)
    invalidReasons[validated.id] = nil
    return validated
  }

  /// Delete the preset with `id`. Returns true if one was removed. Persists on change.
  @discardableResult
  public func delete(_ id: String) throws -> Bool {
    lock.lock(); defer { lock.unlock() }
    let before = presets.count
    presets.removeAll { $0.id == id }
    let changed = presets.count != before
    if changed {
      try PresetStore.persist(presets, to: path, fileManager: fileManager)
      invalidReasons[id] = nil
    }
    return changed
  }

  // MARK: Legacy import

  /// Default location of the old Coffee Shop image-service presets (one
  /// JSON file per preset).
  public static func legacyImageServiceDirectory() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
      .appendingPathComponent(".coffeeshop/image-service/presets", isDirectory: true)
  }

  /// One legacy image-service preset file (per-file JSON shape).
  private struct LegacyPreset: Decodable {
    struct Lora: Decodable { let path: String?; let scale: Double? }
    let id: String?
    let name: String?
    let description: String?
    let model: String?
    let steps: Int?
    let guidance: Double?
    let width: Int?
    let height: Int?
    let loras: [Lora]?
    let injectedKeywords: String?   // legacy stored a single comma string
    let negativePrompt: String?
  }

  /// Import presets from the old image-service (one JSON per file), merging
  /// idempotently. Legacy ids are prefixed `imported-` so a built-in preset
  /// of the same name is never clobbered and a re-run is a no-op. Returns the
  /// number newly added.
  @discardableResult
  public func importLegacyImageService(
    from directory: URL = PresetStore.legacyImageServiceDirectory()
  ) -> Int {
    guard let files = try? fileManager.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: nil)
    else { return 0 }

    var added = 0
    for file in files where file.pathExtension == "json" {
      guard let data = try? Data(contentsOf: file),
            let legacy = try? JSONDecoder().decode(LegacyPreset.self, from: data),
            let legacyId = legacy.id ?? file.deletingPathExtension().lastPathComponent as String?
      else { continue }

      let importedId = "imported-\(legacyId)"
      if get(importedId) != nil { continue }  // already imported

      let keywords = (legacy.injectedKeywords ?? "")
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty }
      let negative = (legacy.negativePrompt?.isEmpty == false) ? legacy.negativePrompt : nil
      let loras = (legacy.loras ?? []).compactMap { lora -> LoraReference? in
        guard let path = lora.path, !path.isEmpty else { return nil }
        return LoraReference(filename: path, scale: lora.scale ?? 1.0)
      }

      let preset = ImagePreset(
        id: importedId,
        name: legacy.name ?? legacyId,
        description: legacy.description ?? "",
        mediaKind: "image",
        provider: "local",
        engine: "zimage",
        model: legacy.model,
        negativePrompt: negative,
        injectedKeywords: keywords.isEmpty ? nil : keywords,
        steps: legacy.steps,
        guidance: legacy.guidance,
        width: legacy.width,
        height: legacy.height,
        loras: loras
      )
      if (try? upsert(preset)) != nil { added += 1 }
    }
    return added
  }

  // MARK: Resolve

  /// The preset AS DECLARED plus its validation flag, read under ONE lock —
  /// so a concurrent `upsert`/`delete` cannot hand back a preset with another
  /// revision's flag. ``resolve(_:)`` and `/v1/generate`'s preset expansion
  /// both go through it, which is what makes them agree.
  ///
  /// `preset` nil = unknown id. `invalidReason` non-nil = flagged at load
  /// (WP-E20 AC-44c).
  public func lookup(_ id: String) -> (preset: ImagePreset?, invalidReason: String?) {
    lock.lock(); defer { lock.unlock() }
    return (presets.first { $0.id == id }, invalidReasons[id])
  }

  /// Merge the preset `id` onto the store's ``PresetDefaults`` and return the fully-populated
  /// parameter set. Port of the `/v1/presets/resolve` behavior. Throws ``PresetStoreError/notFound(_:)``.
  public func resolve(_ id: String) throws -> ResolvedPreset {
    let (found, invalidReason) = lookup(id)
    guard let preset = found else { throw PresetStoreError.notFound(id) }
    // WP-E20 (AC-44c): a flagged preset can never be selected.
    if let reason = invalidReason {
      throw PresetStoreError.invalid(id: id, reason: reason)
    }
    return ResolvedPreset(preset: preset, defaults: defaults)
  }

  /// Resolve an in-hand preset against the store's defaults without a lookup.
  public func resolve(preset: ImagePreset) -> ResolvedPreset {
    ResolvedPreset(preset: preset, defaults: defaults)
  }

  // MARK: - Validation

  /// Port of `validatePreset` (service.ts): required non-empty `id`/`name`; when present,
  /// `steps`/`width`/`height` must be positive integers, and every LoRA scale must be finite.
  /// Kept lenient on the string enum fields (like the tolerant decode) — the client owns them.
  static func validate(
    _ preset: ImagePreset, log: (String) -> Void = { _ in },
    loraLookup: (String) -> LoRALibraryEntry? = { _ in nil }
  ) throws -> ImagePreset {
    if preset.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      throw PresetStoreError.validation(#"required field "id" is missing or empty"#)
    }
    if preset.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      throw PresetStoreError.validation(#"required field "name" is missing or empty"#)
    }
    for (label, value) in [("steps", preset.steps), ("width", preset.width), ("height", preset.height)] {
      if let v = value, v <= 0 {
        throw PresetStoreError.validation("required field \"\(label)\" must be positive (got \(v))")
      }
    }
    for (i, lora) in preset.loras.enumerated() {
      if lora.filename.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        throw PresetStoreError.validation("loras[\(i)].filename is missing or empty")
      }
      if !lora.scale.isFinite {
        throw PresetStoreError.validation("loras[\(i)].scale must be a finite number")
      }
      if let role = lora.role, !LoRAEntry.roles.contains(role) {
        throw PresetStoreError.validation(
          "loras[\(i)].role \"\(role)\" is invalid — expected one of "
            + LoRAEntry.roles.joined(separator: ", "))
      }
    }
    try validateRecipeFields(preset)
    try validateKromaPolicy(preset)
    try validateLoRAFamilyCompatibility(preset, lookup: loraLookup, log: log)
    // Todd 2026-09-04: fold a structured `kroma` declaration into `loras[]`
    // and replace it with the derived, read-only view — see
    // `ImagePreset.migratingKromaDeprecation`.
    return ImagePreset.migratingKromaDeprecation(preset, log: log)
  }

  // MARK: - #402: cross-family LoRA guard at preset save

  /// The canonical `LoRACompatibility.checkFamily` group this preset
  /// targets, or nil when it declares none — resolved the SAME way the
  /// engine already resolves a preset's family elsewhere (comfybox#377/#393):
  /// `mediaKind` for video (LTX-2 is the only video path, `intent.md`),
  /// `resolvesToKrea2Family` for krea2 (checkpointFamily OR model spec, same
  /// as `validateKromaPolicy`'s neighbor), the z-image checkpoint labels, and
  /// otherwise `model` through the same `SamplingRecipeCatalog.canonicalFamily`
  /// + `WarmModelFamily` resolution `/v1/generate`'s
  /// `ImageMemoryPreflight.resolvedFamily` uses. nil (no model, no
  /// checkpointFamily, no video mediaKind) is a legitimate answer — the #402
  /// ruling requires it to warn, never refuse.
  static func resolvedLoRAFamily(for preset: ImagePreset) -> String? {
    if preset.mediaKind?.lowercased() == "video" { return "ltx" }
    if resolvesToKrea2Family(preset) { return "krea2" }
    if let family = preset.checkpointFamily, zimageCheckpointFamilies.contains(family) { return "z-image" }
    if let model = preset.model, !model.isEmpty,
       let canonical = SamplingRecipeCatalog.canonicalFamily(model),
       let warmFamily = WarmModelFamily(rawValue: canonical) {
      return warmFamily.loraCompatibilityFamily
    }
    return nil
  }

  /// #402 — `POST`/`PUT /v1/presets`: reject a LoRA whose declared
  /// `model_compatibility` confidently targets a DIFFERENT family than this
  /// preset resolves to. A preset that resolves to no family at all is never
  /// refused here (`resolvedLoRAFamily` returning nil) — only logged, since
  /// there is nothing to compare against.
  static func validateLoRAFamilyCompatibility(
    _ preset: ImagePreset,
    lookup: (String) -> LoRALibraryEntry?,
    log: (String) -> Void = { _ in }
  ) throws {
    guard !preset.loras.isEmpty else { return }
    guard let targetFamily = resolvedLoRAFamily(for: preset) else {
      log("preset \"\(preset.id)\": no resolvable model family (mediaKind/checkpointFamily/model) "
        + "— LoRA family compatibility not enforced")
      return
    }
    for lora in preset.loras {
      let name = (lora.filename as NSString).lastPathComponent
      let declared = lookup(lora.filename) ?? lookup(name)
      let decision = LoRACompatibility.checkFamily(
        modelCompatibility: declared?.modelCompatibility ?? [], targetFamily: targetFamily)
      if let warning = decision.warning {
        log("preset \"\(preset.id)\" loras[\(name)]: \(warning)")
      }
      guard decision.allowed else {
        throw PresetStoreError.validation(
          "preset \"\(preset.id)\": LoRA \"\(name)\" is compatible with "
            + "\(decision.loraFamilies.joined(separator: "/")), not \"\(targetFamily)\" "
            + "— remove it or target a compatible family")
      }
    }
  }

  // MARK: Checkpoint family + kroma (WP-E20, D7, D14, O4a)

  /// The five client policy labels (D7). `turbo`/`raw-accel`/`raw-stock` are
  /// the krea2 families; `zimage-*` keep today's path and need no kroma.
  public static let krea2CheckpointFamilies: Set<String> = ["turbo", "raw-accel", "raw-stock"]
  public static let zimageCheckpointFamilies: Set<String> = ["zimage-turbo", "zimage-base"]
  public static var checkpointFamilies: [String] {
    (krea2CheckpointFamilies.sorted() + zimageCheckpointFamilies.sorted())
  }

  /// Does this preset resolve to a krea2 family? A declared `checkpointFamily`
  /// answers outright; otherwise the `model` spec decides — the four Turbo
  /// aliases, the declared spec→directory table (`krea2-raw`,
  /// `kroma-v0.2-turbo`, config `krea2Models`), or an existing directory that
  /// `Krea2ModelDetection.detect` recognises. Never a filename guess (F3).
  public static func resolvesToKrea2Family(_ preset: ImagePreset) -> Bool {
    if let family = preset.checkpointFamily {
      return krea2CheckpointFamilies.contains(family)
    }
    guard let model = preset.model, !model.isEmpty else { return false }
    if Krea2ModelDetection.isKnownKrea2Model(model) { return true }
    let expanded = (model as NSString).expandingTildeInPath
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue {
      return Krea2ModelDetection.isKrea2ModelDirectory(URL(fileURLWithPath: expanded, isDirectory: true))
    }
    return false
  }

  /// Image/video discriminator for the kroma rule. The live store's eight Krea
  /// entries carry `mediaKind: null` (FDD §3.16 — they are image presets by
  /// `engine: "zimage"`), and a krea2 checkpoint is only ever an image
  /// checkpoint — so anything not declared `"video"` is an image preset here.
  static func isImagePreset(_ preset: ImagePreset) -> Bool {
    preset.mediaKind?.lowercased() != "video"
  }

  /// Range/shape checks on the deprecated `kroma`/`bypass` dials, WHEN a
  /// client sends one. O4a (FDD §3.15, AC-44b/44c — a krea2-family image
  /// preset must declare `kroma`) is RETIRED (Todd 2026-09-04): kroma has
  /// no special semantics, so its absence gates nothing here any more.
  static func validateKromaPolicy(_ preset: ImagePreset) throws {
    if let family = preset.checkpointFamily,
       !krea2CheckpointFamilies.contains(family), !zimageCheckpointFamilies.contains(family) {
      throw PresetStoreError.validation(
        "preset \"\(preset.id)\": unknown checkpointFamily \"\(family)\" — expected one of "
          + checkpointFamilies.joined(separator: ", "))
    }
    if let kroma = preset.kroma {
      if !kroma.strength.isFinite || kroma.strength < 0 {
        throw PresetStoreError.validation(
          "preset \"\(preset.id)\": kroma.strength must be a finite number >= 0 (got \(kroma.strength))")
      }
      if let file = kroma.file, file.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        throw PresetStoreError.validation("preset \"\(preset.id)\": kroma.file is empty — omit it for the family default")
      }
    }
    // WP-E8 (§3.8, D10): the bypass dial's own ranges. An absent `bypass` is
    // legal and means the kroma-derived default (ledger 17:35) — that
    // derivation is itself always-off since 2026-08-31 (see
    // `Krea2BypassPolicy.resolve`), independent of whether `kroma` is
    // present at all.
    if let bypass = preset.bypass {
      if !bypass.strength.isFinite || bypass.strength < 0 {
        throw PresetStoreError.validation(
          "preset \"\(preset.id)\": bypass.strength must be a finite number >= 0 (got \(bypass.strength))")
      }
      if let file = bypass.file, file.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        throw PresetStoreError.validation(
          "preset \"\(preset.id)\": bypass.file is empty — omit it for the default "
            + "(\(Krea2BypassPolicy.workflowFile))")
      }
    }
    // O4a (the krea2-family-must-declare-kroma requirement) is RETIRED (Todd
    // 2026-09-04): kroma is a regular LoRA now, not an independent
    // declaration, so a krea2-family preset with none is exactly as legal as
    // one missing any other specific adapter.
  }

  /// The recipe fields go through the SAME resolver `/v1/generate` uses
  /// (WP-E4), so a preset can never name a sampler or schedule the engine
  /// does not have, and the numeric knobs are range-checked here rather than
  /// at render time.
  static func validateRecipeFields(_ preset: ImagePreset) throws {
    func named(_ error: Error) -> PresetStoreError {
      .validation("preset \"\(preset.id)\": " + ((error as? LocalizedError)?.errorDescription ?? "\(error)"))
    }
    do {
      _ = try RecipeNameResolver.resolveSchedulerKind(preset.sampler)
      _ = try RecipeNameResolver.resolveSigmaScheduleKind(preset.sigmaSchedule)
      _ = try RecipeNameResolver.resolveSchedulerKind(preset.stage2?.sampler)
      _ = try RecipeNameResolver.resolveSigmaScheduleKind(preset.stage2?.sigmaSchedule)
    } catch {
      throw named(error)
    }
    if let shift = preset.shift, !(shift.isFinite && shift > 0) {
      throw PresetStoreError.validation("preset \"\(preset.id)\": shift must be a finite number > 0 (got \(shift))")
    }
    if let eta = preset.eta, !(eta.isFinite && eta >= 0) {
      throw PresetStoreError.validation("preset \"\(preset.id)\": eta must be a finite number >= 0 (got \(eta))")
    }
    if let noiseType = preset.noiseType,
       RES4LYFNoiseType(rawValue: noiseType) == nil {
      throw PresetStoreError.validation(
        "preset \"\(preset.id)\": noise_type must be one of gaussian, fractal, pyramid (got \(noiseType))")
    }
    if let noiseAlpha = preset.noiseAlpha, !noiseAlpha.isFinite {
      throw PresetStoreError.validation(
        "preset \"\(preset.id)\": noise_alpha must be a finite number (got \(noiseAlpha))")
    }
    if let implicitSteps = preset.implicitSteps, !(0...8).contains(implicitSteps) {
      throw PresetStoreError.validation(
        "preset \"\(preset.id)\": implicit_steps must be an integer in 0...8 (got \(implicitSteps))")
    }
    if let c2 = preset.c2,
       !(c2.isFinite && c2 > 0 && c2 <= 1 && abs(c2 - (2.0 / 3.0)) >= 1e-6) {
      throw PresetStoreError.validation(
        "preset \"\(preset.id)\": c2 must be a finite number in (0, 1] other than 2/3 (got \(c2))")
    }
    if let vae = preset.vae, vae.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      throw PresetStoreError.validation("preset \"\(preset.id)\": vae is empty — omit it for the model directory's VAE")
    }
    if let stage2 = preset.stage2 {
      if let steps = stage2.steps, steps <= 0 {
        throw PresetStoreError.validation("preset \"\(preset.id)\": stage2.steps must be positive (got \(steps))")
      }
      if let denoise = stage2.denoise, !(denoise.isFinite && denoise > 0 && denoise <= 1) {
        throw PresetStoreError.validation("preset \"\(preset.id)\": stage2.denoise must be in (0, 1] (got \(denoise))")
      }
      if let eta = stage2.eta, !(eta.isFinite && eta >= 0) {
        throw PresetStoreError.validation("preset \"\(preset.id)\": stage2.eta must be a finite number >= 0 (got \(eta))")
      }
    }
  }

  // MARK: - Persistence helpers

  /// Tolerant decode: accepts the `{ "presets": [...] }` envelope or a bare `[...]` array,
  /// and falls back to empty on unrecoverable input. Undecodable entries are
  /// kept as flagged placeholders — see ``decodeEntries(_:)``.
  static func decode(_ data: Data) -> [ImagePreset] {
    decodeEntries(data).presets
  }

  static let undecodablePrefix = "could not decode preset"

  /// Per-entry decode (WP-E20). Before this, ONE malformed entry failed the
  /// whole-file decode and the store silently loaded EMPTY — every preset
  /// gone until the file was hand-fixed. Now each entry decodes on its own;
  /// one that fails is kept as an `id`/`name` placeholder with its reason in
  /// `undecodable`, so it is listed, flagged and never silently dropped.
  static func decodeEntries(
    _ data: Data, log: (String) -> Void = { _ in }
  ) -> (presets: [ImagePreset], undecodable: [String: String]) {
    guard let root = try? JSONSerialization.jsonObject(with: data) else { return ([], [:]) }
    let rawEntries: [Any]
    if let envelope = root as? [String: Any], let array = envelope["presets"] as? [Any] {
      rawEntries = array
    } else if let array = root as? [Any] {
      rawEntries = array
    } else {
      return ([], [:])
    }
    let decoder = JSONDecoder()
    var presets: [ImagePreset] = []
    var undecodable: [String: String] = [:]
    for (index, raw) in rawEntries.enumerated() {
      guard let object = raw as? [String: Any],
            let entryData = try? JSONSerialization.data(withJSONObject: object)
      else { continue }
      do {
        // Todd 2026-09-04: migrate a structured `kroma` declaration on
        // EVERY load, not just the next save — a preset a client never
        // re-saves must still converge on the derived, single-source form.
        let decoded = try decoder.decode(ImagePreset.self, from: entryData)
        presets.append(ImagePreset.migratingKromaDeprecation(decoded, log: log))
      } catch {
        let id = (object["id"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "#\(index)"
        let name = (object["name"] as? String) ?? id
        presets.append(ImagePreset(id: id, name: name))
        undecodable[id] = "\(undecodablePrefix) \"\(id)\": \(Self.describeDecodingError(error))"
      }
    }
    return (presets, undecodable)
  }

  private static func describeDecodingError(_ error: Error) -> String {
    guard let decodingError = error as? DecodingError else { return "\(error)" }
    func path(_ context: DecodingError.Context) -> String {
      context.codingPath.map(\.stringValue).joined(separator: ".")
    }
    switch decodingError {
    case .keyNotFound(let key, let context):
      let prefix = path(context)
      return "missing key \"\(prefix.isEmpty ? "" : prefix + ".")\(key.stringValue)\""
    case .typeMismatch(_, let context), .valueNotFound(_, let context), .dataCorrupted(let context):
      return "\(context.debugDescription) at \"\(path(context))\""
    @unknown default:
      return "\(decodingError)"
    }
  }

  static func persist(_ presets: [ImagePreset], to path: URL, fileManager: FileManager) throws {
    let dir = path.deletingLastPathComponent()
    try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(PresetFile(presets: presets))
    try data.write(to: path, options: .atomic)
  }
}
