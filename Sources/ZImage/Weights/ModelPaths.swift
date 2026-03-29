import Foundation

/// Canonical locations for the Hugging Face snapshot. Kept small and explicit
/// so the rest of the pipeline can assemble download/caching steps.
public enum ZImageRepository {
  public static let id = "Tongyi-MAI/Z-Image-Turbo"
  public static let revision = "main"

  public static func defaultCacheDirectory(base: URL = URL(fileURLWithPath: "models")) -> URL {
    base.appendingPathComponent("z-image-turbo")
  }
}

public enum TextEncoderSelectionSource: Equatable {
  case overridePath
  case environment
  case autoDetectedPreferred
  case defaultDirectory
}

public struct TextEncoderSelection: Equatable {
  public let directory: URL
  public let source: TextEncoderSelectionSource

  public init(directory: URL, source: TextEncoderSelectionSource) {
    self.directory = directory
    self.source = source
  }
}

public enum PromptEncodingMode: Equatable {
  case chatTemplate
  case plain
}

public enum ZImageFiles {
  public static let modelIndex = "model_index.json"
  public static let schedulerConfig = "scheduler/scheduler_config.json"
  public static let transformerConfig = "transformer/config.json"
  public static let defaultTextEncoderDirectory = "text_encoder"
  public static let preferredTextEncoderDirectory = "text_encoder QWen Large"
  public static let textEncoderEnvironmentVariable = "ZIMAGE_ENCODER_PATH"
  // Legacy defaults for current snapshot; dynamic resolvers should be preferred.
  public static let transformerWeights = [
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
  ]
  public static let transformerIndex = "transformer/diffusion_pytorch_model.safetensors.index.json"
  public static let transformerAlternateIndex = "transformer/model.safetensors.index.json"

  public static let textEncoderConfig = "text_encoder/config.json"
  // Legacy defaults for current snapshot; dynamic resolvers should be preferred.
  public static let textEncoderWeights = [
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors"
  ]
  public static let textEncoderIndex = "text_encoder/model.safetensors.index.json"

  public static let tokenizerFiles = [
    "tokenizer/merges.txt",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json"
  ]

  public static let vaeConfig = "vae/config.json"
  public static let vaeWeights = ["vae/diffusion_pytorch_model.safetensors"]
  public static let vaeIndex = "vae/model.safetensors.index.json"

  public static func hasRecognizableModelDirectory(at directory: URL) -> Bool {
    let transformerDir = directory.appendingPathComponent("transformer", isDirectory: true)
    let vaeDir = directory.appendingPathComponent("vae", isDirectory: true)
    let textEncoderSelection = resolveTextEncoderSelection(at: directory, overridePath: nil, environment: [:])

    return directoryHasWeights(transformerDir)
      && directoryHasWeights(vaeDir)
      && directoryHasWeights(textEncoderSelection.directory)
  }

  public static func resolveTextEncoderSelection(
    at snapshot: URL,
    overridePath: String?,
    environment: [String: String] = ProcessInfo.processInfo.environment
  ) -> TextEncoderSelection {
    if let overridePath,
       let overrideDirectory = resolvedDirectory(for: overridePath, relativeTo: snapshot),
       directoryHasWeights(overrideDirectory) {
      return TextEncoderSelection(directory: overrideDirectory, source: .overridePath)
    }

    if let environmentPath = environment[textEncoderEnvironmentVariable],
       let environmentDirectory = resolvedDirectory(for: environmentPath, relativeTo: snapshot),
       directoryHasWeights(environmentDirectory) {
      return TextEncoderSelection(directory: environmentDirectory, source: .environment)
    }

    let preferredDirectory = snapshot.appendingPathComponent(preferredTextEncoderDirectory, isDirectory: true)
    if directoryHasWeights(preferredDirectory) {
      return TextEncoderSelection(directory: preferredDirectory, source: .autoDetectedPreferred)
    }

    let defaultDirectory = snapshot.appendingPathComponent(defaultTextEncoderDirectory, isDirectory: true)
    return TextEncoderSelection(directory: defaultDirectory, source: .defaultDirectory)
  }

  public static func resolvePromptEncodingMode(
    at snapshot: URL?,
    selection: TextEncoderSelection?
  ) -> PromptEncodingMode {
    guard let snapshot, let selection else {
      return .chatTemplate
    }

    let defaultDirectory = snapshot
      .appendingPathComponent(defaultTextEncoderDirectory, isDirectory: true)
      .standardizedFileURL
    let selectedDirectory = selection.directory.standardizedFileURL

    return selectedDirectory == defaultDirectory ? .chatTemplate : .plain
  }

  // MARK: - Dynamic weight resolution

  /// Resolve text encoder shard paths relative to the snapshot root.
  /// Prefers index.json when present, otherwise discovers shards by filename patterns.
  public static func resolveTextEncoderWeights(at snapshot: URL) -> [String] {
    resolveWeights(
      at: snapshot,
      componentDir: "text_encoder",
      indexRelativePaths: [textEncoderIndex],
      preferredPrefixes: ["model-"],
      singleFileCandidates: ["model.safetensors"]
    )
  }

  /// Resolve transformer shard paths relative to the snapshot root.
  /// Prefers index.json when present, otherwise discovers shards by filename patterns.
  public static func resolveTransformerWeights(at snapshot: URL) -> [String] {
    resolveWeights(
      at: snapshot,
      componentDir: "transformer",
      indexRelativePaths: [transformerIndex, transformerAlternateIndex],
      preferredPrefixes: ["diffusion_pytorch_model-"],
      singleFileCandidates: ["diffusion_pytorch_model.safetensors", "model.safetensors", "0.safetensors"]
    )
  }

  public static func resolveVAEWeights(at snapshot: URL) -> [String] {
    resolveWeights(
      at: snapshot,
      componentDir: "vae",
      indexRelativePaths: [vaeIndex],
      preferredPrefixes: ["diffusion_pytorch_model-", "model-"],
      singleFileCandidates: ["diffusion_pytorch_model.safetensors", "model.safetensors", "0.safetensors"]
    )
  }

  // MARK: - Helpers

  private struct SafetensorsIndex: Decodable {
    let weight_map: [String: String]?
  }

  private static func resolveWeights(
    at snapshot: URL,
    componentDir: String,
    indexRelativePaths: [String],
    preferredPrefixes: [String],
    singleFileCandidates: [String]
  ) -> [String] {
    let fm = FileManager.default

    // 1) Try reading the safetensors index to enumerate shard files
    for indexRelativePath in indexRelativePaths {
      let indexURL = snapshot.appending(path: indexRelativePath)
      if fm.fileExists(atPath: indexURL.path),
         let data = try? Data(contentsOf: indexURL),
         let idx = try? JSONDecoder().decode(SafetensorsIndex.self, from: data),
         let weightMap = idx.weight_map {
        let uniqueFiles = Array(Set(weightMap.values))
        let relative = uniqueFiles
          .map { file in file.contains("/") ? file : "\(componentDir)/\(file)" }
          .sorted(by: shardAwareLess)
          .filter { fm.fileExists(atPath: snapshot.appending(path: $0).path) }
        if !relative.isEmpty { return relative }
      }
    }

    // 2) Discover shards via directory scan with preferred filename prefixes
    let dirURL = snapshot.appending(path: componentDir)
    if let contents = try? fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil) {
      return resolveWeightURLs(
        in: contents,
        componentDir: componentDir,
        preferredPrefixes: preferredPrefixes,
        singleFileCandidates: singleFileCandidates,
        snapshot: snapshot
      )
    }

    // 3) Fallback to legacy lists (may be stale for newer snapshots)
    if componentDir == "text_encoder" { return textEncoderWeights }
    if componentDir == "transformer" { return transformerWeights }
    if componentDir == "vae" { return vaeWeights }
    return []
  }

  static func resolveWeightFiles(in directory: URL, componentName: String) -> [URL] {
    let fm = FileManager.default
    let contents = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []

    let resolvedRelative: [String]
    switch componentName {
    case "text_encoder":
      resolvedRelative = resolveWeightURLs(
        in: contents,
        componentDir: nil,
        preferredPrefixes: ["model-"],
        singleFileCandidates: ["model.safetensors"],
        snapshot: directory
      )
    case "vae":
      resolvedRelative = resolveWeightURLs(
        in: contents,
        componentDir: nil,
        preferredPrefixes: ["diffusion_pytorch_model-", "model-"],
        singleFileCandidates: ["diffusion_pytorch_model.safetensors", "model.safetensors", "0.safetensors"],
        snapshot: directory
      )
    default:
      resolvedRelative = resolveWeightURLs(
        in: contents,
        componentDir: nil,
        preferredPrefixes: ["diffusion_pytorch_model-", "model-"],
        singleFileCandidates: ["diffusion_pytorch_model.safetensors", "model.safetensors", "0.safetensors"],
        snapshot: directory
      )
    }

    return resolvedRelative.map { directory.appendingPathComponent($0) }
  }

  private static func resolveWeightURLs(
    in contents: [URL],
    componentDir: String?,
    preferredPrefixes: [String],
    singleFileCandidates: [String],
    snapshot: URL
  ) -> [String] {
    let fm = FileManager.default
    let safetensors = contents.filter { $0.pathExtension == "safetensors" }
    let preferred = safetensors.filter { url in
      let name = url.lastPathComponent
      return preferredPrefixes.contains(where: { name.hasPrefix($0) })
    }

    let candidates = preferred.isEmpty ? safetensors : preferred
    let prefix = componentDir.map { "\($0)/" } ?? ""

    var relative = candidates
      .map { "\(prefix)\($0.lastPathComponent)" }
      .sorted(by: shardAwareLess)

    if relative.isEmpty {
      for single in singleFileCandidates {
        let path = "\(prefix)\(single)"
        if fm.fileExists(atPath: snapshot.appending(path: path).path) {
          relative = [path]
          break
        }
      }
    }

    return relative
  }

  private static func resolvedDirectory(for path: String, relativeTo snapshot: URL) -> URL? {
    let candidate: URL
    if path.hasPrefix("/") || path.hasPrefix("~") {
      candidate = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    } else {
      candidate = snapshot.appendingPathComponent(path, isDirectory: true)
    }

    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      return nil
    }
    return candidate
  }

  private static func directoryHasWeights(_ directory: URL) -> Bool {
    let fm = FileManager.default
    guard let contents = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
      return false
    }
    return contents.contains { $0.pathExtension == "safetensors" }
  }

  /// Comparator that sorts shard files numerically when the filename contains
  /// the pattern "-00001-of-000NN.safetensors", otherwise falls back to lexicographic.
  private static func shardAwareLess(_ a: String, _ b: String) -> Bool {
    func shardIndex(_ name: String) -> Int? {
      // Extract the integer between the last '-' before "-of-" and the "-of-" marker.
      guard let ofRange = name.range(of: "-of-") else { return nil }
      if let lastDash = name[..<ofRange.lowerBound].lastIndex(of: "-") {
        let start = name.index(after: lastDash)
        let idxStr = String(name[start..<ofRange.lowerBound])
        return Int(idxStr)
      }
      return nil
    }
    let ia = shardIndex((a as NSString).lastPathComponent)
    let ib = shardIndex((b as NSString).lastPathComponent)
    switch (ia, ib) {
    case let (xa?, xb?):
      return xa < xb
    case (nil, nil):
      return a.localizedCompare(b) == .orderedAscending
    case (_?, nil):
      // Prefer shard-numbered files to non-numbered ones
      return true
    case (nil, _?):
      return false
    }
  }
}
