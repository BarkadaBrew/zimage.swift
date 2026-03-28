import Foundation

public struct ZImageTransformerConfig: Decodable {
  public let inChannels: Int
  public let dim: Int
  public let nLayers: Int
  public let nRefinerLayers: Int
  public let nHeads: Int
  public let nKVHeads: Int
  public let normEps: Float
  public let qkNorm: Bool
  public let capFeatDim: Int
  public let ropeTheta: Float
  public let tScale: Float
  public let axesDims: [Int]
  public let axesLens: [Int]

  enum CodingKeys: String, CodingKey {
    case inChannels = "in_channels"
    case dim
    case nLayers = "n_layers"
    case nRefinerLayers = "n_refiner_layers"
    case nHeads = "n_heads"
    case nKVHeads = "n_kv_heads"
    case normEps = "norm_eps"
    case qkNorm = "qk_norm"
    case capFeatDim = "cap_feat_dim"
    case ropeTheta = "rope_theta"
    case tScale = "t_scale"
    case axesDims = "axes_dims"
    case axesLens = "axes_lens"
  }
}

public struct ZImageVAEConfig: Decodable {
  public let blockOutChannels: [Int]
  public let latentChannels: Int
  public let scalingFactor: Float
  public let shiftFactor: Float
  public let sampleSize: Int
  public let inChannels: Int
  public let outChannels: Int
  public let layersPerBlock: Int
  public let normNumGroups: Int
  public let midBlockAddAttention: Bool
  public let usePostQuantConv: Bool?
  public let useQuantConv: Bool?

  enum CodingKeys: String, CodingKey {
    case blockOutChannels = "block_out_channels"
    case latentChannels = "latent_channels"
    case scalingFactor = "scaling_factor"
    case shiftFactor = "shift_factor"
    case sampleSize = "sample_size"
    case inChannels = "in_channels"
    case outChannels = "out_channels"
    case layersPerBlock = "layers_per_block"
    case normNumGroups = "norm_num_groups"
    case midBlockAddAttention = "mid_block_add_attention"
    case usePostQuantConv = "use_post_quant_conv"
    case useQuantConv = "use_quant_conv"
  }

  public var vaeScaleFactor: Int {
    max(1, 1 << max(0, blockOutChannels.count - 1))
  }

  public var latentDivisor: Int {
    vaeScaleFactor  // 8 for Z-Image-Turbo (4 downsampling stages with factor 2 each)
  }
}

public struct ZImageSchedulerConfig: Decodable {
  public let numTrainTimesteps: Int
  public let shift: Float
  public let useDynamicShifting: Bool
  public let baseShift: Float?
  public let maxShift: Float?
  public let baseImageSeqLen: Int?
  public let maxImageSeqLen: Int?

  enum CodingKeys: String, CodingKey {
    case numTrainTimesteps = "num_train_timesteps"
    case shift
    case useDynamicShifting = "use_dynamic_shifting"
    case baseShift = "base_shift"
    case maxShift = "max_shift"
    case baseImageSeqLen = "base_image_seq_len"
    case maxImageSeqLen = "max_image_seq_len"
  }
}

public struct ZImageTextEncoderConfig: Decodable {
  public let hiddenSize: Int
  public let numHiddenLayers: Int
  public let numAttentionHeads: Int
  public let numKeyValueHeads: Int
  public let intermediateSize: Int
  public let maxPositionEmbeddings: Int
  public let ropeTheta: Float
  public let vocabSize: Int
  public let rmsNormEps: Float
  public let headDim: Int

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case intermediateSize = "intermediate_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case ropeTheta = "rope_theta"
    case vocabSize = "vocab_size"
    case rmsNormEps = "rms_norm_eps"
    case headDim = "head_dim"
  }
}

public struct ZImageModelConfigs {
  public let transformer: ZImageTransformerConfig
  public let vae: ZImageVAEConfig
  public let scheduler: ZImageSchedulerConfig
  public let textEncoder: ZImageTextEncoderConfig

  public static func load(from snapshot: URL, textEncoderDirectory: URL? = nil) throws -> ZImageModelConfigs {
    let decoder = JSONDecoder()
    func loadJSON<T: Decodable>(_ relativePath: String, as type: T.Type) throws -> T {
      let url = snapshot.appending(path: relativePath)
      let data = try Data(contentsOf: url)
      return try decoder.decode(T.self, from: data)
    }

    let selectedTextEncoderDirectory = textEncoderDirectory
      ?? ZImageFiles.resolveTextEncoderSelection(at: snapshot, overridePath: nil, environment: [:]).directory

    let transformer: ZImageTransformerConfig
    if FileManager.default.fileExists(atPath: snapshot.appending(path: ZImageFiles.transformerConfig).path) {
      transformer = try loadJSON(ZImageFiles.transformerConfig, as: ZImageTransformerConfig.self)
    } else {
      transformer = try inferTransformerConfig(from: snapshot)
    }

    let vae: ZImageVAEConfig
    if FileManager.default.fileExists(atPath: snapshot.appending(path: ZImageFiles.vaeConfig).path) {
      vae = try loadJSON(ZImageFiles.vaeConfig, as: ZImageVAEConfig.self)
    } else {
      vae = try inferVAEConfig(from: snapshot)
    }

    let scheduler: ZImageSchedulerConfig
    if FileManager.default.fileExists(atPath: snapshot.appending(path: ZImageFiles.schedulerConfig).path) {
      scheduler = try loadJSON(ZImageFiles.schedulerConfig, as: ZImageSchedulerConfig.self)
    } else {
      scheduler = defaultSchedulerConfig
    }

    let textEncoderConfigURL = selectedTextEncoderDirectory.appendingPathComponent("config.json")
    let textEncoder: ZImageTextEncoderConfig
    if FileManager.default.fileExists(atPath: textEncoderConfigURL.path) {
      let data = try Data(contentsOf: textEncoderConfigURL)
      textEncoder = try decoder.decode(ZImageTextEncoderConfig.self, from: data)
    } else {
      textEncoder = try inferTextEncoderConfig(from: selectedTextEncoderDirectory)
    }

    return ZImageModelConfigs(transformer: transformer, vae: vae, scheduler: scheduler, textEncoder: textEncoder)
  }

  static var defaultSchedulerConfig: ZImageSchedulerConfig {
    ZImageSchedulerConfig(
      numTrainTimesteps: 1000,
      shift: 3.0,
      useDynamicShifting: false,
      baseShift: nil,
      maxShift: nil,
      baseImageSeqLen: nil,
      maxImageSeqLen: nil
    )
  }

  static var defaultVAEConfig: ZImageVAEConfig {
    ZImageVAEConfig(
      blockOutChannels: [128, 256, 512, 512],
      latentChannels: 16,
      scalingFactor: 0.3611,
      shiftFactor: 0.1159,
      sampleSize: 1024,
      inChannels: 3,
      outChannels: 3,
      layersPerBlock: 2,
      normNumGroups: 32,
      midBlockAddAttention: true,
      usePostQuantConv: false,
      useQuantConv: false
    )
  }

  static func inferTransformerConfig(from snapshot: URL) throws -> ZImageTransformerConfig {
    let shapes = try loadTensorShapes(from: ZImageFiles.resolveWeightFiles(in: snapshot.appendingPathComponent("transformer"), componentName: "transformer"))
    guard let config = inferTransformerConfig(fromTensorShapes: shapes) else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return config
  }

  static func inferTransformerConfig(fromTensorShapes shapes: [String: [Int]]) -> ZImageTransformerConfig? {
    let layerCount = maxIndex(prefix: "layers", in: shapes.keys).map { $0 + 1 } ?? 0
    let noiseRefinerCount = maxIndex(prefix: "noise_refiner", in: shapes.keys).map { $0 + 1 } ?? 0
    let contextRefinerCount = maxIndex(prefix: "context_refiner", in: shapes.keys).map { $0 + 1 } ?? 0
    let refinerCount = max(noiseRefinerCount, contextRefinerCount)

    guard let qShape = shapes["layers.0.attention.to_q.weight"] ?? shapes["noise_refiner.0.attention.to_q.weight"],
          qShape.count == 2,
          let normShape = shapes["layers.0.attention.norm_q.weight"] ?? shapes["noise_refiner.0.attention.norm_q.weight"],
          let headDim = normShape.first,
          headDim > 0 else {
      return nil
    }

    let dim = qShape[0]
    let nHeads = max(1, dim / headDim)
    let nKVHeads: Int
    if let kShape = shapes["layers.0.attention.to_k.weight"] ?? shapes["noise_refiner.0.attention.to_k.weight"], kShape.count == 2 {
      nKVHeads = max(1, kShape[0] / headDim)
    } else {
      nKVHeads = nHeads
    }

    let capFeatDim = (shapes["cap_embedder.1.weight"]?.count == 2 ? shapes["cap_embedder.1.weight"]?[1] : nil) ?? 2560
    let patchVolume = (shapes["all_x_embedder.2-1.weight"]?.count == 2 ? shapes["all_x_embedder.2-1.weight"]?[1] : nil) ?? 64
    let inChannels = max(1, patchVolume / 4)

    return ZImageTransformerConfig(
      inChannels: inChannels,
      dim: dim,
      nLayers: layerCount,
      nRefinerLayers: refinerCount,
      nHeads: nHeads,
      nKVHeads: nKVHeads,
      normEps: 1e-5,
      qkNorm: shapes.keys.contains("layers.0.attention.norm_q.weight") || shapes.keys.contains("noise_refiner.0.attention.norm_q.weight"),
      capFeatDim: capFeatDim,
      ropeTheta: 256.0,
      tScale: 1000.0,
      axesDims: [32, 48, 48],
      axesLens: [1536, 512, 512]
    )
  }

  static func inferTextEncoderConfig(from directory: URL) throws -> ZImageTextEncoderConfig {
    let shapes = try loadTensorShapes(from: ZImageFiles.resolveWeightFiles(in: directory, componentName: "text_encoder"))
    guard let config = inferTextEncoderConfig(fromTensorShapes: shapes) else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return config
  }

  static func inferTextEncoderConfig(fromTensorShapes shapes: [String: [Int]]) -> ZImageTextEncoderConfig? {
    guard let embedShape = shapes["model.embed_tokens.weight"], embedShape.count == 2 else {
      return nil
    }
    let hiddenSize = embedShape[1]
    let vocabSize = embedShape[0]
    let numHiddenLayers = (maxIndex(prefix: "model.layers", in: shapes.keys).map { $0 + 1 }) ?? 0

    guard let qShape = shapes["model.layers.0.self_attn.q_proj.weight"], qShape.count == 2,
          let kShape = shapes["model.layers.0.self_attn.k_proj.weight"], kShape.count == 2 else {
      return nil
    }

    let headDim = (shapes["model.layers.0.self_attn.q_norm.weight"]?.first)
      ?? (shapes["model.layers.0.self_attn.k_norm.weight"]?.first)
      ?? 128
    let numAttentionHeads = max(1, qShape[0] / headDim)
    let numKeyValueHeads = max(1, kShape[0] / headDim)
    let intermediateSize = (shapes["model.layers.0.mlp.gate_proj.weight"]?.first)
      ?? (shapes["model.layers.0.mlp.up_proj.weight"]?.first)
      ?? hiddenSize * 4

    return ZImageTextEncoderConfig(
      hiddenSize: hiddenSize,
      numHiddenLayers: numHiddenLayers,
      numAttentionHeads: numAttentionHeads,
      numKeyValueHeads: numKeyValueHeads,
      intermediateSize: intermediateSize,
      maxPositionEmbeddings: 40960,
      ropeTheta: 1_000_000,
      vocabSize: vocabSize,
      rmsNormEps: 1e-6,
      headDim: headDim
    )
  }

  static func inferVAEConfig(from snapshot: URL) throws -> ZImageVAEConfig {
    _ = snapshot
    // We do not have a reliable VAE config inference path yet. Keep local checkpoints
    // working by using the known Z-Image-Turbo defaults until a real shape-based
    // inference implementation is added.
    return defaultVAEConfig
  }

  private static func loadTensorShapes(from files: [URL]) throws -> [String: [Int]] {
    var shapes: [String: [Int]] = [:]
    for file in files {
      let reader = try SafeTensorsReader(fileURL: file)
      for metadata in reader.allMetadata() {
        shapes[metadata.name] = metadata.shape
      }
    }
    return shapes
  }

  private static func maxIndex(prefix: String, in keys: Dictionary<String, [Int]>.Keys) -> Int? {
    let prefixComponents = prefix.split(separator: ".").map(String.init)
    return keys.compactMap { key in
      let components = key.split(separator: ".").map(String.init)
      guard components.count > prefixComponents.count else { return nil }
      guard Array(components.prefix(prefixComponents.count)) == prefixComponents else { return nil }
      return Int(components[prefixComponents.count])
    }.max()
  }
}
