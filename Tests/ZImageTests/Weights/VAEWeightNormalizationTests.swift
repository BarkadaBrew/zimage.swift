import XCTest
import MLX
import Logging
@testable import ZImage

final class VAEWeightNormalizationTests: XCTestCase {

  func testLoadVAEAddsAliasesForMfluxWrapperKeys() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let vaeDir = modelDir.appendingPathComponent("vae")
    try makeDirectory(vaeDir)

    try writeFakeSafeTensors([
      "encoder.conv_in.conv2d.weight": [8, 3, 3, 3],
      "encoder.conv_in.conv2d.bias": [8],
      "encoder.conv_out.conv2d.weight": [16, 3, 3, 8],
      "encoder.conv_out.conv2d.bias": [16],
      "encoder.conv_norm_out.norm.weight": [16],
      "encoder.conv_norm_out.norm.bias": [16],
      "decoder.conv_in.conv.weight": [16, 3, 3, 4],
      "decoder.conv_in.conv.bias": [16],
      "decoder.conv_out.conv.weight": [3, 3, 3, 16],
      "decoder.conv_out.conv.bias": [3],
      "decoder.conv_norm_out.norm.weight": [16],
      "decoder.conv_norm_out.norm.bias": [16],
    ], to: vaeDir.appendingPathComponent("0.safetensors"))

    let mapper = ZImageWeightsMapper(snapshot: modelDir, logger: Logger(label: "test"))
    let weights = try mapper.loadVAE(dtype: nil)

    XCTAssertEqual(weights["encoder.conv_in.weight"]?.shape, [8, 3, 3, 3])
    XCTAssertEqual(weights["encoder.conv_out.weight"]?.shape, [16, 3, 3, 8])
    XCTAssertEqual(weights["encoder.conv_norm_out.weight"]?.shape, [16])
    XCTAssertEqual(weights["encoder.conv_norm_out.bias"]?.shape, [16])
    XCTAssertEqual(weights["decoder.conv_in.weight"]?.shape, [16, 3, 3, 4])
    XCTAssertEqual(weights["decoder.conv_out.weight"]?.shape, [3, 3, 3, 16])
    XCTAssertEqual(weights["decoder.conv_norm_out.weight"]?.shape, [16])
    XCTAssertEqual(weights["decoder.conv_norm_out.bias"]?.shape, [16])
  }

  func testRuntimeVAEParameterNamesMatchNormalizedAliasTargets() {
    let config = VAEConfig(
      inChannels: 3,
      outChannels: 3,
      latentChannels: 4,
      scalingFactor: 0.3611,
      shiftFactor: 0.1159,
      blockOutChannels: [8, 16],
      layersPerBlock: 1,
      normNumGroups: 4,
      sampleSize: 32,
      midBlockAddAttention: true
    )

    let autoencoder = AutoencoderKL(configuration: config)
    let decoderOnly = AutoencoderDecoderOnly(configuration: config)
    let autoencoderKeys = Set(autoencoder.parameters().flattened().map { $0.0 })
    let decoderKeys = Set(decoderOnly.parameters().flattened().map { $0.0 })
    let encoderConvKeys = autoencoderKeys.filter {
      $0.hasPrefix("encoder.conv_in") || $0.hasPrefix("encoder.conv_out") || $0.hasPrefix("encoder.conv_norm_out")
    }.sorted()
    let decoderConvKeys = decoderKeys.filter {
      $0.hasPrefix("decoder.conv_in") || $0.hasPrefix("decoder.conv_out") || $0.hasPrefix("decoder.conv_norm_out")
    }.sorted()

    XCTAssertTrue(autoencoderKeys.contains("encoder.conv_in.weight"), "encoder keys: \(encoderConvKeys)")
    XCTAssertTrue(autoencoderKeys.contains("encoder.conv_out.weight"), "encoder keys: \(encoderConvKeys)")
    XCTAssertTrue(autoencoderKeys.contains("encoder.conv_norm_out.weight"), "encoder keys: \(encoderConvKeys)")
    XCTAssertFalse(autoencoderKeys.contains("encoder.conv_in.conv2d.weight"))
    XCTAssertFalse(autoencoderKeys.contains("encoder.conv_in.conv.weight"))
    XCTAssertFalse(autoencoderKeys.contains("encoder.conv_norm_out.norm.weight"))

    XCTAssertTrue(decoderKeys.contains("decoder.conv_in.weight"), "decoder keys: \(decoderConvKeys)")
    XCTAssertTrue(decoderKeys.contains("decoder.conv_out.weight"), "decoder keys: \(decoderConvKeys)")
    XCTAssertTrue(decoderKeys.contains("decoder.conv_norm_out.weight"), "decoder keys: \(decoderConvKeys)")
    XCTAssertFalse(decoderKeys.contains("decoder.conv_in.conv.weight"))
    XCTAssertFalse(decoderKeys.contains("decoder.conv_norm_out.norm.weight"))

    let normalized = ZImageVAEWeightAliases.normalized([
      "encoder.conv_in.conv2d.weight": MLXArray([Float(0.0)]),
      "encoder.conv_out.conv2d.weight": MLXArray([Float(0.0)]),
      "decoder.conv_in.conv.weight": MLXArray([Float(0.0)]),
      "decoder.conv_out.conv.weight": MLXArray([Float(0.0)]),
      "encoder.conv_norm_out.norm.weight": MLXArray([Float(0.0)]),
      "decoder.conv_norm_out.norm.weight": MLXArray([Float(0.0)]),
    ])

    XCTAssertNotNil(normalized["encoder.conv_in.weight"])
    XCTAssertNotNil(normalized["encoder.conv_out.weight"])
    XCTAssertNotNil(normalized["decoder.conv_in.weight"])
    XCTAssertNotNil(normalized["decoder.conv_out.weight"])
    XCTAssertNotNil(normalized["encoder.conv_norm_out.weight"])
    XCTAssertNotNil(normalized["decoder.conv_norm_out.weight"])
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
