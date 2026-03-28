import Foundation
import MLX

enum ZImageVAEWeightAliases {
  static func normalized(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var normalized = weights

    addAlias(
      from: "encoder.conv_in.conv2d.weight",
      to: "encoder.conv_in.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_in.conv2d.bias",
      to: "encoder.conv_in.bias",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_out.conv2d.weight",
      to: "encoder.conv_out.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_out.conv2d.bias",
      to: "encoder.conv_out.bias",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_in.conv.weight",
      to: "encoder.conv_in.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_in.conv.bias",
      to: "encoder.conv_in.bias",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_out.conv.weight",
      to: "encoder.conv_out.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_out.conv.bias",
      to: "encoder.conv_out.bias",
      in: &normalized
    )

    addAlias(
      from: "encoder.conv_norm_out.norm.weight",
      to: "encoder.conv_norm_out.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_norm_out.norm.bias",
      to: "encoder.conv_norm_out.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_in.conv.weight",
      to: "decoder.conv_in.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_in.conv.bias",
      to: "decoder.conv_in.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_out.conv.weight",
      to: "decoder.conv_out.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_out.conv.bias",
      to: "decoder.conv_out.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_in.conv2d.weight",
      to: "decoder.conv_in.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_in.conv2d.bias",
      to: "decoder.conv_in.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_out.conv2d.weight",
      to: "decoder.conv_out.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_out.conv2d.bias",
      to: "decoder.conv_out.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_norm_out.norm.weight",
      to: "decoder.conv_norm_out.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_norm_out.norm.bias",
      to: "decoder.conv_norm_out.bias",
      in: &normalized
    )

    return normalized
  }

  private static func addAlias(from sourceKey: String, to aliasKey: String, in weights: inout [String: MLXArray]) {
    guard weights[aliasKey] == nil, let value = weights[sourceKey] else { return }
    weights[aliasKey] = value
  }
}
