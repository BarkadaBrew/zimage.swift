import Foundation
import Logging
import MLX
import MLXNN

public struct ZImageWeightsMapping {
  public enum WeightApplicationError: Error, LocalizedError {
    case updateFailed(prefix: String, underlying: Error)

    public var errorDescription: String? {
      switch self {
      case .updateFailed(let prefix, let underlying):
        return "Failed to apply weights to \(prefix): \(underlying)"
      }
    }
  }

  public struct Partition {
    public let transformer: [String: MLXArray]
    public let textEncoder: [String: MLXArray]
    public let vae: [String: MLXArray]
    public let unassigned: [String: MLXArray]
  }

  public static func partition(weights: [String: MLXArray], logger: Logger? = nil) -> Partition {
    var transformer: [String: MLXArray] = [:]
    var textEncoder: [String: MLXArray] = [:]
    var vae: [String: MLXArray] = [:]
    var unassigned: [String: MLXArray] = [:]

    for (key, tensor) in weights {
      if key.hasPrefix("transformer.") {
        transformer[String(key.dropFirst("transformer.".count))] = tensor
      } else if key.hasPrefix("text_encoder.") {
        textEncoder[String(key.dropFirst("text_encoder.".count))] = tensor
      } else if key.hasPrefix("vae.") {
        vae[String(key.dropFirst("vae.".count))] = tensor
      } else {
        unassigned[key] = tensor
      }
    }

    return Partition(
      transformer: transformer,
      textEncoder: textEncoder,
      vae: vae,
      unassigned: unassigned
    )
  }

  private static func transformerMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      mapped["transformer.\(k)"] = v
    }
    return mapped
  }

  private static func textEncoderMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      if k.hasPrefix("model.") {
        let remainder = String(k.dropFirst("model.".count))
        mapped["text_encoder.encoder.\(remainder)"] = v
      } else {
        mapped["text_encoder.\(k)"] = v
      }
    }
    return mapped
  }

  private static func vaeMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      mapped["vae.\(k)"] = v
    }
    return mapped
  }

  static func alignTensorShape(_ tensor: MLXArray, to expectedShape: [Int]) -> MLXArray {
    guard tensor.shape != expectedShape else { return tensor }
    guard needsTransposeToMatchShape(tensor.shape, expectedShape: expectedShape) else { return tensor }
    return tensor.transposed(0, 2, 3, 1)
  }

  static func needsTransposeToMatchShape(_ tensorShape: [Int], expectedShape: [Int]) -> Bool {
    guard tensorShape.count == 4, expectedShape.count == 4 else { return false }
    guard tensorShape != expectedShape else { return false }
    return [tensorShape[0], tensorShape[2], tensorShape[3], tensorShape[1]] == expectedShape
  }

  public static func applyTransformer(
    weights: [String: MLXArray],
    to model: ZImageTransformer2DModel,
    manifest: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) throws {
    if weights.isEmpty {
      logger.warning("Transformer weights empty; nothing to apply.")
      return
    }

    if let manifest = manifest {
      let availableKeys = Set(weights.keys)
      ZImageQuantizer.applyQuantization(
        to: model,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.transformerTensorName
      )
    }

    let mapped = transformerMapping(weights)
    try applyToModule(model, weights: mapped, prefix: "transformer", logger: logger)

    let groupSize = manifest?.groupSize ?? 32
    let bits = manifest?.bits ?? 8
    model.loadCapEmbedderWeights(from: weights)
    model.loadXEmbedderWeights(from: weights, groupSize: groupSize, bits: bits)
    model.loadFinalLayerWeights(from: weights, groupSize: groupSize, bits: bits)

    model.setPadTokens(xPad: weights["x_pad_token"], capPad: weights["cap_pad_token"])
  }

  public static func applyTextEncoder(
    weights: [String: MLXArray],
    to model: QwenTextEncoder,
    manifest: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) throws {
    if weights.isEmpty {
      logger.warning("Text encoder weights empty; nothing to apply.")
      return
    }

    if let manifest = manifest {
      let availableKeys = Set(weights.keys)
      ZImageQuantizer.applyQuantization(
        to: model,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.textEncoderTensorName
      )
    }

    let mapped = textEncoderMapping(weights)
    try applyToModule(model, weights: mapped, prefix: "text_encoder", logger: logger)
  }

  public static func applyVAE(
    weights: [String: MLXArray],
    to model: Module,
    manifest: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) throws {
    if weights.isEmpty {
      logger.warning("VAE weights empty; nothing to apply.")
      return
    }

    let mapped = vaeMapping(weights)
    try applyToModule(model, weights: mapped, prefix: "vae", logger: logger)
  }

  private static func applyToModule(_ module: Module, weights: [String: MLXArray], prefix: String, logger: Logger) throws {
    let params = module.parameters().flattened()
    var updates: [(String, MLXArray)] = []

    for (key, param) in params {
      let candidates = [key, "\(prefix).\(key)"]
      if let found = candidates.compactMap({ weights[$0] }).first {
        let aligned = alignTensorShape(found, to: param.shape)
        updates.append((key, aligned))
      }
    }

    for (weightKey, tensor) in weights {
      var paramKey = weightKey
      if weightKey.hasPrefix("\(prefix).") {
        paramKey = String(weightKey.dropFirst("\(prefix).".count))
      }

      if (paramKey.hasSuffix(".scales") || paramKey.hasSuffix(".biases")) {
        if !updates.contains(where: { $0.0 == paramKey }) {
          updates.append((paramKey, tensor))
        }
      }
    }

    if updates.isEmpty {
      logger.warning("\(prefix) received no matching weights; skipping apply.")
      return
    }

    do {
      let nd = ModuleParameters.unflattened(updates)
      try module.update(parameters: nd, verify: [.shapeMismatch])
    } catch {
      logger.error("Failed to apply weights to \(prefix): \(error)")
      throw WeightApplicationError.updateFailed(prefix: prefix, underlying: error)
    }
  }
}
