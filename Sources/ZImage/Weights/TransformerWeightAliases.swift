import Foundation
import MLX

enum ZImageTransformerWeightAliases {
  static func normalized(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var normalized = weights

    addAlias(
      from: "t_embedder.linear1.weight",
      to: "t_embedder.mlp.0.weight",
      in: &normalized
    )
    addAlias(
      from: "t_embedder.linear1.bias",
      to: "t_embedder.mlp.0.bias",
      in: &normalized
    )
    addAlias(
      from: "t_embedder.linear2.weight",
      to: "t_embedder.mlp.2.weight",
      in: &normalized
    )
    addAlias(
      from: "t_embedder.linear2.bias",
      to: "t_embedder.mlp.2.bias",
      in: &normalized
    )

    addFinalLayerAdaLNAliases(in: &normalized)

    return normalized
  }

  private static func addFinalLayerAdaLNAliases(in weights: inout [String: MLXArray]) {
    for key in Array(weights.keys) {
      guard key.hasPrefix("all_final_layer.") else { continue }
      let aliasKey: String
      if key.contains(".adaLN_modulation.0.weight") {
        aliasKey = key.replacingOccurrences(of: ".adaLN_modulation.0.weight", with: ".adaLN_modulation.1.weight")
      } else if key.contains(".adaLN_modulation.0.bias") {
        aliasKey = key.replacingOccurrences(of: ".adaLN_modulation.0.bias", with: ".adaLN_modulation.1.bias")
      } else {
        continue
      }

      addAlias(from: key, to: aliasKey, in: &weights)
    }
  }

  private static func addAlias(from sourceKey: String, to aliasKey: String, in weights: inout [String: MLXArray]) {
    guard weights[aliasKey] == nil, let value = weights[sourceKey] else { return }
    weights[aliasKey] = value
  }
}
