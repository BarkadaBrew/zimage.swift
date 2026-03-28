import Foundation
import Logging
import MLX
import MLXNN

public struct ZImageModelAuditReport {
  public struct Component: Sendable {
    public let name: String
    public let matched: Int
    public let missing: [String]
    public let extra: [String]
    public let shapeMismatches: [String]

    public var totalExpected: Int {
      matched + missing.count
    }

    public var coverage: Double {
      guard totalExpected > 0 else { return 1.0 }
      return Double(matched) / Double(totalExpected)
    }
  }

  public let snapshot: URL
  public let textEncoderSelection: TextEncoderSelection
  public let transformer: Component
  public let textEncoder: Component
  public let vaeDecoder: Component

  public func formattedDescription(sample: Int = 10) -> String {
    [
      "Model audit for \(snapshot.path)",
      "Text encoder: \(textEncoderSelection.directory.path) [\(String(describing: textEncoderSelection.source))]",
      format(component: transformer, sample: sample),
      format(component: textEncoder, sample: sample),
      format(component: vaeDecoder, sample: sample),
    ].joined(separator: "\n")
  }

  private func format(component: Component, sample: Int) -> String {
    let coveragePercent = Int((component.coverage * 100.0).rounded())
    var lines = [
      "",
      "[\(component.name)] matched \(component.matched)/\(component.totalExpected) (\(coveragePercent)%)",
      "missing: \(component.missing.count), extra: \(component.extra.count), shape mismatches: \(component.shapeMismatches.count)",
    ]

    if !component.missing.isEmpty {
      lines.append("missing sample: \(component.missing.prefix(max(0, sample)).joined(separator: ", "))")
    }
    if !component.extra.isEmpty {
      lines.append("extra sample: \(component.extra.sorted().prefix(max(0, sample)).joined(separator: ", "))")
    }
    if !component.shapeMismatches.isEmpty {
      lines.append("shape mismatch sample: \(component.shapeMismatches.prefix(max(0, sample)).joined(separator: "; "))")
    }

    return lines.joined(separator: "\n")
  }
}

public enum ZImageModelAuditor {
  public static func audit(
    modelSpec: String?,
    textEncoderPath: String?,
    logger: Logger,
    dtype: DType? = .bfloat16
  ) async throws -> ZImageModelAuditReport {
    let snapshot = try await ModelResolution.resolveOrDefault(modelSpec: modelSpec)
    let textEncoderSelection = ZImageFiles.resolveTextEncoderSelection(at: snapshot, overridePath: textEncoderPath)
    logger.info("Auditing model snapshot at \(snapshot.path)")
    logger.info("Using text encoder directory \(textEncoderSelection.directory.path)")

    let configs = try ZImageModelConfigs.load(from: snapshot, textEncoderDirectory: textEncoderSelection.directory)
    let mapper = ZImageWeightsMapper(
      snapshot: snapshot,
      logger: logger,
      textEncoderDirectory: textEncoderSelection.directory
    )

    let transformerWeights = try mapper.loadTransformer(dtype: dtype)
    let textEncoderWeights = try mapper.loadTextEncoder(dtype: dtype)
    let vaeWeights = try mapper.loadVAE(dtype: dtype).filter { $0.key.hasPrefix("decoder.") }

    let transformer = ZImageTransformer2DModel(configuration: configs.transformer)
    let textEncoder = QwenTextEncoder(
      configuration: .init(
        vocabSize: configs.textEncoder.vocabSize,
        hiddenSize: configs.textEncoder.hiddenSize,
        numHiddenLayers: configs.textEncoder.numHiddenLayers,
        numAttentionHeads: configs.textEncoder.numAttentionHeads,
        numKeyValueHeads: configs.textEncoder.numKeyValueHeads,
        intermediateSize: configs.textEncoder.intermediateSize,
        ropeTheta: configs.textEncoder.ropeTheta,
        maxPositionEmbeddings: configs.textEncoder.maxPositionEmbeddings,
        rmsNormEps: configs.textEncoder.rmsNormEps,
        headDim: configs.textEncoder.headDim
      )
    )
    let vaeDecoder = AutoencoderDecoderOnly(configuration: .init(
      inChannels: configs.vae.inChannels,
      outChannels: configs.vae.outChannels,
      latentChannels: configs.vae.latentChannels,
      scalingFactor: configs.vae.scalingFactor,
      shiftFactor: configs.vae.shiftFactor,
      blockOutChannels: configs.vae.blockOutChannels,
      layersPerBlock: configs.vae.layersPerBlock,
      normNumGroups: configs.vae.normNumGroups,
      sampleSize: configs.vae.sampleSize,
      midBlockAddAttention: configs.vae.midBlockAddAttention
    ))

    var transformerAuditWeights = transformerWeights
    if let weight = transformerWeights["cap_embedder.0.weight"] {
      transformerAuditWeights["capEmbedNorm.weight"] = weight
    }
    if let weight = transformerWeights["cap_embedder.1.weight"] {
      transformerAuditWeights["capEmbedLinear.weight"] = weight
    }
    if let bias = transformerWeights["cap_embedder.1.bias"] {
      transformerAuditWeights["capEmbedLinear.bias"] = bias
    }

    return ZImageModelAuditReport(
      snapshot: snapshot,
      textEncoderSelection: textEncoderSelection,
      transformer: summarize(
        name: "transformer",
        module: transformer,
        weights: transformerAuditWeights,
        logger: logger,
        transpose4DTensors: false
      ),
      textEncoder: summarize(
        name: "text_encoder",
        module: textEncoder,
        weights: textEncoderWeights,
        logger: logger,
        transpose4DTensors: false
      ),
      vaeDecoder: summarize(
        name: "vae_decoder",
        module: vaeDecoder,
        weights: vaeWeights,
        logger: logger,
        transpose4DTensors: true
      )
    )
  }

  private static func summarize(
    name: String,
    module: Module,
    weights: [String: MLXArray],
    logger: Logger,
    transpose4DTensors: Bool
  ) -> ZImageModelAuditReport.Component {
    let summary = WeightsAudit.audit(module: module, weights: weights, prefix: name, logger: logger, sample: 10)
    let shapeMismatches = WeightsAudit.shapeMismatches(
      module: module,
      weights: weights,
      prefix: name,
      logger: logger,
      transpose4DTensors: transpose4DTensors,
      sample: 10
    )

    return .init(
      name: name,
      matched: summary.matched,
      missing: summary.missing,
      extra: summary.extra,
      shapeMismatches: shapeMismatches
    )
  }
}
