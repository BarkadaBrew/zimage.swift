// LoRACompatibility.swift — Model compatibility matrix for LoRA adapters
//
// Part of the LoRA Library Manager (#73). Maps compatibility strings to
// ComfyBoxModelFamily and performs trial key mapping to verify LoRA/model
// compatibility with match ratio reporting.

import Foundation

// MARK: - Compatibility Result

/// Result of checking a LoRA's compatibility with a specific model family.
public struct LoRACompatibilityResult: Sendable {
  /// Whether the LoRA is considered compatible (matchRatio > 0.5).
  public let isCompatible: Bool
  /// Number of LoRA keys that successfully mapped to model targets.
  public let matchedKeys: Int
  /// Total number of LoRA base keys tested.
  public let totalKeys: Int
  /// Ratio of matched keys to total keys (0.0 - 1.0).
  public let matchRatio: Float
  /// Warnings about potential issues (e.g. "LoKr format not supported for this model").
  public let warnings: [String]
}

// MARK: - Compatibility Checker

/// Checks LoRA compatibility against model families using key mappers.
public enum LoRACompatibility {

  /// Map a compatibility string from library.json to a ComfyBoxModelFamily.
  ///
  /// - Parameter compat: A compatibility tag (e.g. "z-image", "klein-9b").
  /// - Returns: The matching model family, or nil if unknown.
  public static func familyMapping(_ compat: String) -> ComfyBoxModelFamily? {
    switch compat.lowercased() {
    case "z-image", "zimage":
      return .zImage
    case "klein-9b", "klein-4b", "flux2-klein":
      return .flux2Klein
    case "chroma":
      return .chroma
    case "krea2", "krea-2", "krea-2-turbo":
      return .krea2
    default:
      return nil
    }
  }

  /// Map a ComfyBoxModelFamily to its compatibility string(s).
  ///
  /// - Parameter family: The model family to look up.
  /// - Returns: Compatibility strings that match this family.
  public static func compatibilityStrings(for family: ComfyBoxModelFamily) -> [String] {
    switch family {
    case .zImage:
      return ["z-image"]
    case .flux2Klein:
      return ["klein-9b", "klein-4b"]
    case .chroma:
      return ["chroma"]
    case .krea2:
      return ["krea2"]
    case .fibo, .seedvr2, .esrgan:
      return []
    }
  }

  /// Check if a LoRA entry is compatible with a given model family.
  ///
  /// First checks the declared `model_compatibility` tags. If the entry
  /// declares compatibility, returns a positive result without trial mapping.
  /// If not declared, performs trial key mapping through the family's key mapper.
  ///
  /// - Parameters:
  ///   - entry: The LoRA library entry to check.
  ///   - modelFamily: The target model family.
  /// - Returns: A `LoRACompatibilityResult` with match details.
  public static func check(
    entry: LoRALibraryEntry,
    modelFamily: ComfyBoxModelFamily
  ) -> LoRACompatibilityResult {
    let familyCompats = compatibilityStrings(for: modelFamily)
    var warnings: [String] = []

    // Quick check: does the entry declare compatibility?
    let declaredMatch = entry.modelCompatibility.contains { compat in
      familyCompats.contains(compat.lowercased())
    }

    if declaredMatch {
      // Check format compatibility
      if entry.format == .lokr {
        switch modelFamily {
        case .flux2Klein:
          // Only Flux2LoRALoader supports LoKr, not the standard loadForFlux2
          warnings.append("LoKr format — requires Flux2LoRALoader (not all Klein loaders support LoKr)")
        case .chroma:
          warnings.append("LoKr format not supported for Chroma")
        default:
          break
        }
      }

      return LoRACompatibilityResult(
        isCompatible: true,
        matchedKeys: entry.keyCount,
        totalKeys: entry.keyCount,
        matchRatio: 1.0,
        warnings: warnings
      )
    }

    // Trial key mapping: test sample keys against the model's key mapper
    return trialKeyMapping(entry: entry, modelFamily: modelFamily)
  }

  // MARK: - Trial Key Mapping

  /// Perform trial key mapping to estimate compatibility.
  ///
  /// Generates sample LoRA keys based on the entry's detected compatibility
  /// and tests them against the target model family's key mapper.
  private static func trialKeyMapping(
    entry: LoRALibraryEntry,
    modelFamily: ComfyBoxModelFamily
  ) -> LoRACompatibilityResult {
    // We can't do actual key mapping without reading the file,
    // so use the declared compatibility and key count for estimation.
    let familyCompats = compatibilityStrings(for: modelFamily)
    let entryCompats = Set(entry.modelCompatibility.map { $0.lowercased() })
    let familySet = Set(familyCompats)

    // No overlap in declared compatibility = incompatible
    if entryCompats.isDisjoint(with: familySet) && !entryCompats.contains("unknown") {
      return LoRACompatibilityResult(
        isCompatible: false,
        matchedKeys: 0,
        totalKeys: entry.keyCount,
        matchRatio: 0.0,
        warnings: ["LoRA targets \(entry.modelCompatibility.joined(separator: ", ")), not \(modelFamily.displayName)"]
      )
    }

    // Unknown compatibility — report as uncertain
    return LoRACompatibilityResult(
      isCompatible: false,
      matchedKeys: 0,
      totalKeys: entry.keyCount,
      matchRatio: 0.0,
      warnings: ["Unknown compatibility — run `lora scan` to detect"]
    )
  }

  /// Perform a live compatibility check by reading actual keys from a file.
  ///
  /// This reads the safetensors header and maps each LoRA base key through
  /// the target model's key mapper to count successful matches.
  ///
  /// - Parameters:
  ///   - url: Path to the safetensors file.
  ///   - modelFamily: The target model family.
  /// - Returns: A detailed compatibility result with actual match counts.
  public static func checkFile(
    _ url: URL,
    modelFamily: ComfyBoxModelFamily
  ) throws -> LoRACompatibilityResult {
    let reader = try SafeTensorsReader(fileURL: url)
    let allKeys = reader.tensorNames
    var warnings: [String] = []

    // Extract base keys (strip lora_down/up, lokr_w1/w2 suffixes)
    let baseKeys = extractBaseKeys(from: allKeys)
    let totalBaseKeys = baseKeys.count

    guard totalBaseKeys > 0 else {
      return LoRACompatibilityResult(
        isCompatible: false,
        matchedKeys: 0,
        totalKeys: allKeys.count,
        matchRatio: 0.0,
        warnings: ["No LoRA keys found in file"]
      )
    }

    // Map each base key through the target model's key mapper
    var matchedCount = 0

    switch modelFamily {
    case .zImage:
      let validTargets = Set(LoRAKeyMapper.supportedTargetPaths)
      for baseKey in baseKeys {
        let mapped = LoRAKeyMapper.mapToZImageKey(baseKey)
        if validTargets.contains(mapped) {
          matchedCount += 1
        }
      }

    case .flux2Klein:
      for baseKey in baseKeys {
        let stripped = Flux2LoRAMapping.stripPrefix(baseKey)
        let result = Flux2LoRAMapping.map(stripped)
        switch result {
        case .direct, .qkvSplit:
          matchedCount += 1
        case .unmapped:
          break
        }
      }

    case .chroma:
      for baseKey in baseKeys {
        let mapped = ChromaLoRAKeyMapper.map(baseKey)
        // ChromaLoRAKeyMapper always returns something — check if it's a valid target
        // Valid Chroma targets start with double_blocks., single_blocks., txt_in, img_in
        if mapped.hasPrefix("double_blocks.") || mapped.hasPrefix("single_blocks.") ||
           mapped.hasPrefix("txt_in") || mapped.hasPrefix("img_in") {
          matchedCount += 1
        }
      }

    case .krea2:
      // Krea2SingleStreamDiT keys match 1:1 (no remapping) once the
      // diffusion_model. prefix is stripped — valid targets are
      // blocks.<n>.attn.* or blocks.<n>.mlp.*.
      for baseKey in baseKeys {
        var key = baseKey
        if key.hasPrefix("diffusion_model.") {
          key = String(key.dropFirst("diffusion_model.".count))
        }
        if key.hasPrefix("blocks.") && (key.contains(".attn.") || key.contains(".mlp.")) {
          matchedCount += 1
        }
      }

    default:
      warnings.append("\(modelFamily.displayName) does not support LoRA")
      return LoRACompatibilityResult(
        isCompatible: false,
        matchedKeys: 0,
        totalKeys: totalBaseKeys,
        matchRatio: 0.0,
        warnings: warnings
      )
    }

    let ratio = totalBaseKeys > 0 ? Float(matchedCount) / Float(totalBaseKeys) : 0.0
    let isCompatible = ratio > 0.5

    // Check LoKr format compatibility
    let hasLoKr = allKeys.contains { $0.contains(".lokr_w1") }
    if hasLoKr {
      switch modelFamily {
      case .chroma:
        warnings.append("LoKr format not supported for Chroma")
      default:
        break
      }
    }

    return LoRACompatibilityResult(
      isCompatible: isCompatible,
      matchedKeys: matchedCount,
      totalKeys: totalBaseKeys,
      matchRatio: ratio,
      warnings: warnings
    )
  }

  // MARK: - #402: cross-family enforcement guard

  /// Canonical compatibility "family group" for one tag on a LoRA's
  /// `model_compatibility` list (`LoRALibraryEntry.modelCompatibility`,
  /// `LoRAScanner.knownCompatibilityTags`). Groups the aliases the scanner
  /// and library writers use (`klein-9b`/`klein-4b` → one Flux 2 Klein group,
  /// `krea2`/`krea-2`/`krea-2-turbo` → one group, …) onto ONE token per
  /// architecture, reusing `familyMapping` for the families it already
  /// normalizes.
  ///
  /// `"flux1"` is its OWN group — the real Flux.1-dev/schnell architecture a
  /// LoRA's metadata can declare — and is DELIBERATELY NOT folded into
  /// `"z-image"`, even though this engine's `WarmModelFamily.flux1` case is
  /// the Z-Image family under an internal name (comfybox#154: the DiT
  /// `.flux1` boots is Z-Image/Lumina2, not Flux.1; comfybox#393 named this
  /// exact ambiguity). Conflating the two would let a genuine Flux.1 LoRA
  /// silently pass this guard for a `WarmModelFamily.flux1` (Z-Image) request
  /// purely because both happen to spell "flux1" — callers must map
  /// `WarmModelFamily.flux1` to `"z-image"` explicitly (see
  /// `WarmModelFamily.loraCompatibilityFamily` in `WarmServer.swift`) rather
  /// than passing its raw value through here.
  ///
  /// `"ltx"` is LTX-2's group (video) — the only video path this engine has
  /// (`intent.md`). `"unknown"` (a legal declared value,
  /// `LoRAScanner.knownCompatibilityTags`) and any tag this function does not
  /// recognize both return nil — "no confident family", never a family of
  /// their own.
  public static func familyGroup(forCompatibilityTag tag: String) -> String? {
    let lower = tag.lowercased()
    if lower == "flux1" { return "flux1" }
    if ["ltx", "ltx2", "ltx-2", "ltxv", "ltx-video", "ltx_video"].contains(lower) { return "ltx" }
    if lower == "unknown" { return nil }
    if let family = familyMapping(lower) { return canonicalGroup(for: family) }
    return nil
  }

  private static func canonicalGroup(for family: ComfyBoxModelFamily) -> String {
    switch family {
    case .zImage: return "z-image"
    case .flux2Klein: return "flux2-klein"
    case .chroma: return "chroma"
    case .krea2: return "krea2"
    case .fibo: return "fibo"
    case .seedvr2: return "seedvr2"
    case .esrgan: return "esrgan"
    }
  }

  /// Result of the #402 cross-family guard. Distinct from
  /// `LoRACompatibilityResult` above (the #73 library UI's trial-key-mapping
  /// estimate, which reads a file) — this is the enforcement decision at
  /// swap/generate/preset-save time, using ONLY the entry's DECLARED
  /// `model_compatibility`, never a file read.
  public struct GuardDecision: Sendable, Equatable {
    /// Whether the request/swap/preset may proceed.
    public let allowed: Bool
    /// Present when `allowed` is true but the answer is not confident
    /// (no declared tags, or none recognized) — #402 ruling 2: unknown
    /// compatibility is allowed, with a warning, never a refusal.
    public let warning: String?
    /// The LoRA's own recognized family group(s), for the caller's 400
    /// message. Empty when `warning` is set (nothing confident was found).
    public let loraFamilies: [String]

    public init(allowed: Bool, warning: String? = nil, loraFamilies: [String] = []) {
      self.allowed = allowed
      self.warning = warning
      self.loraFamilies = loraFamilies
    }
  }

  /// #402 — the pure cross-family validator: `LoRACompatibility.check`'s
  /// counterpart for enforcement rather than the library UI's match-ratio
  /// estimate. Compares a LoRA's DECLARED `model_compatibility` tags against
  /// the canonical family group the request/swap/preset targets.
  ///
  /// - No recognized tag at all (empty list, every tag unrecognized, or the
  ///   explicit `"unknown"` value) → `.allowed = true` with a warning. Per
  ///   the #402 ruling, unknown compatibility is never a refusal.
  /// - At least one tag resolves to the SAME group as `targetFamily` →
  ///   allowed, no warning — an entry may legitimately declare more than one
  ///   compatible family.
  /// - Every recognized tag resolves to a DIFFERENT, known group → rejected,
  ///   naming the LoRA's family/families so the caller can build a 400
  ///   naming both.
  public static func checkFamily(modelCompatibility: [String], targetFamily: String) -> GuardDecision {
    let groups = modelCompatibility.compactMap(familyGroup(forCompatibilityTag:))
    if groups.isEmpty {
      let detail = modelCompatibility.isEmpty
        ? "no model_compatibility declared"
        : "model_compatibility \(modelCompatibility) is not a recognized family"
      return GuardDecision(allowed: true, warning: "\(detail) — allowing with a warning")
    }
    if groups.contains(targetFamily) {
      return GuardDecision(allowed: true)
    }
    return GuardDecision(allowed: false, loraFamilies: Array(Set(groups)).sorted())
  }

  // MARK: - Key Extraction Helpers

  /// Extract unique base keys from LoRA tensor names.
  ///
  /// Strips lora_down/lora_up, lora_A/lora_B, lokr_w1/lokr_w2, and alpha suffixes
  /// to get the underlying model path each adapter targets.
  private static func extractBaseKeys(from keys: [String]) -> Set<String> {
    var baseKeys = Set<String>()
    let suffixes = [
      ".lora_down.weight", ".lora_up.weight",
      ".lora_A.weight", ".lora_B.weight",
      ".lokr_w1", ".lokr_w2",
    ]

    for key in keys {
      for suffix in suffixes {
        if key.hasSuffix(suffix) {
          let base = String(key.dropLast(suffix.count))
          baseKeys.insert(base)
          break
        }
      }
    }

    return baseKeys
  }
}
