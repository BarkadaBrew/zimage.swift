import Foundation
import Logging
import MLXNN
import MLX

public struct WeightsAudit {
  public struct Summary: Sendable {
    public let matched: Int
    public let missing: [String]
    public let extra: [String]
  }

  public static func audit(module: Module, weights: [String: MLXArray], prefix: String = "", logger: Logger, sample: Int = 5) -> Summary {
    let params = module.parameters().flattened()
    var matched = 0
    var missingKeys: [String] = []
    var remaining = Set(weights.keys)

    for (key, _) in params {
      let candidate1 = key
      let candidate2 = prefix.isEmpty ? key : "\(prefix).\(key)"
      if weights[candidate2] != nil {
        matched += 1
        remaining.remove(candidate2)
      } else if weights[candidate1] != nil {
        matched += 1
        remaining.remove(candidate1)
      } else {
        missingKeys.append(candidate2)
      }
    }

    let missingSample = Array(missingKeys.prefix(max(0, sample)))
    let extraSample = Array(Array(remaining).sorted().prefix(max(0, sample)))

    logger.info("\(prefix.isEmpty ? "module" : prefix) weights audit -> matched: \(matched), missing: \(missingKeys.count), extra: \(remaining.count)")
    if !missingKeys.isEmpty {
      let suffix = missingKeys.count > missingSample.count ? ", ..." : ""
      let sampleText = missingSample.isEmpty ? "" : " (sample: \(missingSample.joined(separator: ", "))\(suffix))"
      logger.warning("Missing weights: \(missingKeys.count)\(sampleText)")
    }
    if !remaining.isEmpty {
      let suffix = remaining.count > extraSample.count ? ", ..." : ""
      let sampleText = extraSample.isEmpty ? "" : " (sample: \(extraSample.joined(separator: ", "))\(suffix))"
      logger.info("Extra weights: \(remaining.count)\(sampleText)")
    }

    return Summary(matched: matched, missing: missingKeys, extra: Array(remaining))
  }

  public static func shapeMismatches(
    module: Module,
    weights: [String: MLXArray],
    prefix: String = "",
    logger: Logger,
    transpose4DTensors: Bool = false,
    sample: Int = 5
  ) -> [String] {
    let params = module.parameters().flattened()
    var mismatches: [String] = []

    for (key, param) in params {
      let candidate1 = key
      let candidate2 = prefix.isEmpty ? key : "\(prefix).\(key)"
      guard var tensor = weights[candidate1] ?? weights[candidate2] else { continue }

      if transpose4DTensors && tensor.ndim == 4 {
        tensor = ZImageWeightsMapping.alignTensorShape(tensor, to: param.shape)
      }

      if tensor.shape != param.shape {
        mismatches.append("\(candidate2) expected \(param.shape) got \(tensor.shape)")
      }
    }

    if !mismatches.isEmpty {
      let sampleText = mismatches.prefix(max(0, sample)).joined(separator: "; ")
      let suffix = mismatches.count > sample ? "; ..." : ""
      logger.warning("\(prefix.isEmpty ? "module" : prefix) shape mismatches: \(mismatches.count) (sample: \(sampleText)\(suffix))")
    }

    return mismatches
  }
}
