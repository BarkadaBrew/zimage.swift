import Foundation
import Logging
import MLX


public struct ZImageWeightsMapper {
  private let snapshot: URL
  private let logger: Logger
  private let textEncoderDirectory: URL?

  public init(snapshot: URL, logger: Logger, textEncoderDirectory: URL? = nil) {
    self.snapshot = snapshot
    self.logger = logger
    self.textEncoderDirectory = textEncoderDirectory
  }

  public func hasQuantization() -> Bool {
    ZImageQuantizer.hasQuantization(at: snapshot)
  }

  public func loadQuantizationManifest() -> ZImageQuantizationManifest? {
    let manifestURL = snapshot.appendingPathComponent("quantization.json")
    return try? ZImageQuantizationManifest.load(from: manifestURL)
  }


  public func loadAll(dtype: DType? = .bfloat16) throws -> [String: MLXArray] {
    if hasQuantization() {
      logger.info("Detected quantized model, loading from quantized safetensors")
      return try loadQuantizedAll()
    }
    return try loadStandardAll(dtype: dtype)
  }

  public func loadTextEncoder(dtype: DType? = .bfloat16) throws -> [String: MLXArray] {
    if let textEncoderDirectory {
      return try loadStandardComponent(urls: ZImageFiles.resolveWeightFiles(in: textEncoderDirectory, componentName: "text_encoder"), dtype: dtype)
    }
    if hasQuantization() {
      return try loadQuantizedComponent("text_encoder")
    }
    return try loadStandardComponent(files: ZImageFiles.resolveTextEncoderWeights(at: snapshot), dtype: dtype)
  }

  public func loadTransformer(dtype: DType? = .bfloat16) throws -> [String: MLXArray] {
    if hasQuantization() {
      return ZImageTransformerWeightAliases.normalized(try loadQuantizedComponent("transformer"))
    }
    return ZImageTransformerWeightAliases.normalized(
      try loadStandardComponent(files: ZImageFiles.resolveTransformerWeights(at: snapshot), dtype: dtype)
    )
  }

  /// Load transformer weights from a standalone safetensors file (override file)
  public func loadTransformer(fromFile url: URL, dtype: DType? = .bfloat16) throws -> [String: MLXArray] {
    var tensors: [String: MLXArray] = [:]
    let reader = try SafeTensorsReader(fileURL: url)
    for meta in reader.allMetadata() {
      var tensor = try reader.tensor(named: meta.name)
      if let targetDtype = dtype, tensor.dtype != targetDtype {
        tensor = tensor.asType(targetDtype)
      }
      tensors[meta.name] = tensor
    }
    logger.info("Loaded \(tensors.count) transformer tensors from override file \(url.lastPathComponent)")
    return ZImageTransformerWeightAliases.normalized(tensors)
  }

  public func loadVAE(dtype: DType? = .bfloat16) throws -> [String: MLXArray] {
    if hasQuantization() {
      var tensors = try loadQuantizedComponent("vae")
      if let targetDtype = dtype {
        // Honor requested dtype even for quantized snapshots
        for (k, v) in tensors {
          if v.dtype != targetDtype {
            tensors[k] = v.asType(targetDtype)
          }
        }
      }
      return tensors
    }
    return try loadStandardComponent(files: ZImageFiles.resolveVAEWeights(at: snapshot), dtype: dtype)
  }

  /// Load controlnet weights from a standalone safetensors file
  public func loadControlnetWeights(from path: String, dtype: DType? = .bfloat16) throws -> [String: MLXArray] {
    let url: URL
    if path.hasPrefix("/") {
      url = URL(fileURLWithPath: path)
    } else {
      url = snapshot.appending(path: path)
    }

    guard FileManager.default.fileExists(atPath: url.path) else {
      throw NSError(domain: "ZImageWeightsMapper", code: 1, userInfo: [
        NSLocalizedDescriptionKey: "Controlnet weights file not found: \(url.path)"
      ])
    }

    var tensors: [String: MLXArray] = [:]
    let reader = try SafeTensorsReader(fileURL: url)
    for meta in reader.allMetadata() {
      var tensor = try reader.tensor(named: meta.name)
      if let targetDtype = dtype, tensor.dtype != targetDtype {
        tensor = tensor.asType(targetDtype)
      }
      tensors[meta.name] = tensor
    }

    logger.info("Loaded \(tensors.count) controlnet tensors from \(url.lastPathComponent)")
    return tensors
  }

  private func loadStandardComponent(files: [String], dtype: DType?) throws -> [String: MLXArray] {
    try loadStandardComponent(urls: files.map { snapshot.appending(path: $0) }, dtype: dtype)
  }

  private func loadStandardComponent(urls: [URL], dtype: DType?) throws -> [String: MLXArray] {
    var tensors: [String: MLXArray] = [:]
    for url in urls {
      guard FileManager.default.fileExists(atPath: url.path) else {
        logger.warning("Weight shard missing: \(url.path)")
        continue
      }
      let reader = try SafeTensorsReader(fileURL: url)
      for meta in reader.allMetadata() {
        var tensor = try reader.tensor(named: meta.name)
        if let targetDtype = dtype, tensor.dtype != targetDtype {
          tensor = tensor.asType(targetDtype)
        }
        tensors[meta.name] = tensor
      }
    }
    return tensors
  }

  private func loadQuantizedComponent(_ componentName: String) throws -> [String: MLXArray] {
    let fm = FileManager.default
    let resolvedSnapshot = snapshot.resolvingSymlinksInPath()
    var tensors: [String: MLXArray] = [:]

    let componentDir = resolvedSnapshot.appendingPathComponent(componentName)
    guard fm.fileExists(atPath: componentDir.path) else {
      logger.warning("Component directory not found: \(componentName)")
      return tensors
    }

    let contents = try fm.contentsOfDirectory(at: componentDir, includingPropertiesForKeys: nil)
    let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

    for file in safetensorFiles {
      let weights = try MLX.loadArrays(url: file)
      for (key, value) in weights {
        tensors[key] = value
      }
    }
    return tensors
  }

  private func loadStandardAll(dtype: DType?) throws -> [String: MLXArray] {
    var tensors: [String: MLXArray] = [:]

    for (key, value) in try loadStandardComponent(files: ZImageFiles.resolveTransformerWeights(at: snapshot), dtype: dtype) {
      tensors["transformer.\(key)"] = value
    }
    for (key, value) in try loadTextEncoder(dtype: dtype) {
      tensors["text_encoder.\(key)"] = value
    }
    for (key, value) in try loadStandardComponent(files: ZImageFiles.resolveVAEWeights(at: snapshot), dtype: dtype) {
      tensors["vae.\(key)"] = value
    }

    if let targetDtype = dtype {
      logger.info("Converted weights to \(targetDtype)")
    }
    logger.info("Aggregated \(tensors.count) tensors from safetensors shards")
    return tensors
  }

  private func loadQuantizedAll() throws -> [String: MLXArray] {
    var tensors: [String: MLXArray] = [:]

    for (key, value) in try loadQuantizedComponent("transformer") {
      tensors["transformer.\(key)"] = value
    }
    for (key, value) in try loadQuantizedComponent("text_encoder") {
      tensors["text_encoder.\(key)"] = value
    }
    for (key, value) in try loadQuantizedComponent("vae") {
      tensors["vae.\(key)"] = value
    }

    logger.info("Aggregated \(tensors.count) tensors from quantized safetensors")
    return tensors
  }
}
