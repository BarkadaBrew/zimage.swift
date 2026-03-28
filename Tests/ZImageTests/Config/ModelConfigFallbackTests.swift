import XCTest
@testable import ZImage

final class ModelConfigFallbackTests: XCTestCase {

  func testLoadFallsBackToInferredConfigsAndPreferredCustomEncoderDirectory() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let transformerDir = modelDir.appendingPathComponent("transformer")
    let standardEncoderDir = modelDir.appendingPathComponent("text_encoder")
    let customEncoderDir = modelDir.appendingPathComponent("text_encoder QWen Large")
    let vaeDir = modelDir.appendingPathComponent("vae")

    try makeDirectory(transformerDir)
    try makeDirectory(standardEncoderDir)
    try makeDirectory(customEncoderDir)
    try makeDirectory(vaeDir)

    try writeFakeSafeTensors([
      "all_x_embedder.2-1.weight": [12, 16],
      "cap_embedder.1.weight": [12, 16],
      "noise_refiner.0.attention.to_q.weight": [12, 12],
      "noise_refiner.0.attention.to_k.weight": [12, 12],
      "noise_refiner.0.attention.norm_q.weight": [4],
      "noise_refiner.0.attention.norm_k.weight": [4],
      "context_refiner.0.attention.to_q.weight": [12, 12],
      "context_refiner.0.attention.to_k.weight": [12, 12],
      "context_refiner.0.attention.norm_q.weight": [4],
      "context_refiner.0.attention.norm_k.weight": [4],
      "layers.0.attention.to_q.weight": [12, 12],
      "layers.0.attention.to_k.weight": [12, 12],
      "layers.0.attention.to_v.weight": [12, 12],
      "layers.0.attention.norm_q.weight": [4],
      "layers.0.attention.norm_k.weight": [4],
      "layers.0.adaLN_modulation.0.weight": [48, 8],
    ], to: transformerDir.appendingPathComponent("0.safetensors"))

    try writeFakeSafeTensors([
      "model.embed_tokens.weight": [32, 8],
      "model.layers.0.self_attn.q_proj.weight": [16, 8],
      "model.layers.0.self_attn.k_proj.weight": [8, 8],
      "model.layers.0.self_attn.o_proj.weight": [8, 16],
      "model.layers.0.self_attn.q_norm.weight": [4],
      "model.layers.0.mlp.gate_proj.weight": [24, 8],
      "model.norm.weight": [8],
    ], to: standardEncoderDir.appendingPathComponent("model.safetensors"))

    try writeFakeSafeTensors([
      "model.embed_tokens.weight": [64, 16],
      "model.layers.0.self_attn.q_proj.weight": [24, 16],
      "model.layers.0.self_attn.k_proj.weight": [8, 16],
      "model.layers.0.self_attn.o_proj.weight": [16, 24],
      "model.layers.0.self_attn.q_norm.weight": [4],
      "model.layers.0.mlp.gate_proj.weight": [32, 16],
      "model.norm.weight": [16],
    ], to: customEncoderDir.appendingPathComponent("model.safetensors"))

    try writeFakeSafeTensors([
      "decoder.conv_in.conv.weight": [8, 1, 1, 4],
      "decoder.conv_out.conv.weight": [3, 1, 1, 8],
    ], to: vaeDir.appendingPathComponent("0.safetensors"))

    let selection = ZImageFiles.resolveTextEncoderSelection(at: modelDir, overridePath: nil, environment: [:])
    XCTAssertEqual(selection.directory.standardizedFileURL.path, customEncoderDir.standardizedFileURL.path)

    let configs = try ZImageModelConfigs.load(from: modelDir, textEncoderDirectory: selection.directory)

    XCTAssertEqual(configs.transformer.dim, 12)
    XCTAssertEqual(configs.transformer.nLayers, 1)
    XCTAssertEqual(configs.transformer.nRefinerLayers, 1)
    XCTAssertEqual(configs.transformer.capFeatDim, 16)

    XCTAssertEqual(configs.textEncoder.hiddenSize, 16)
    XCTAssertEqual(configs.textEncoder.numHiddenLayers, 1)
    XCTAssertEqual(configs.textEncoder.numAttentionHeads, 6)
    XCTAssertEqual(configs.textEncoder.numKeyValueHeads, 2)
    XCTAssertEqual(configs.textEncoder.intermediateSize, 32)

    XCTAssertEqual(configs.scheduler.numTrainTimesteps, 1000)
    XCTAssertEqual(configs.scheduler.shift, 3.0)
  }

  private func makeTempDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try makeDirectory(directory)
    return directory
  }

  private func makeDirectory(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  }

  private func writeFakeSafeTensors(_ shapes: [String: [Int]], to url: URL) throws {
    var offset = 0
    var header: [String: [String: Any]] = [:]

    for key in shapes.keys.sorted() {
      let shape = shapes[key] ?? []
      let elementCount = max(1, shape.reduce(1, *))
      let byteCount = elementCount * 4
      header[key] = [
        "dtype": "F32",
        "shape": shape,
        "data_offsets": [offset, offset + byteCount]
      ]
      offset += byteCount
    }

    let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var length = UInt64(headerData.count).littleEndian
    var data = Data(bytes: &length, count: MemoryLayout<UInt64>.size)
    data.append(headerData)
    data.append(Data(count: offset))
    try data.write(to: url)
  }
}
