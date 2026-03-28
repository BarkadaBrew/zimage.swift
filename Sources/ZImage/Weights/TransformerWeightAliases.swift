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

    addAlias(
      from: "all_final_layer.2-1.adaLN_modulation.0.weight",
      to: "all_final_layer.2-1.adaLN_modulation.1.weight",
      in: &normalized
    )
    addAlias(
      from: "all_final_layer.2-1.adaLN_modulation.0.bias",
      to: "all_final_layer.2-1.adaLN_modulation.1.bias",
      in: &normalized
    )

    return normalized
  }

  private static func addAlias(from sourceKey: String, to aliasKey: String, in weights: inout [String: MLXArray]) {
    guard weights[aliasKey] == nil, let value = weights[sourceKey] else { return }
    weights[aliasKey] = value
  }
}
