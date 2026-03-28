import XCTest
import MLX
import Logging
@testable import ZImage

final class TransformerWeightNormalizationTests: XCTestCase {

  func testLoadTransformerAddsAliasesForMfluxTimestepAndFinalLayerWeights() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let transformerDir = modelDir.appendingPathComponent("transformer")
    try makeDirectory(transformerDir)

    try writeFakeSafeTensors([
      "t_embedder.linear1.weight": [4, 2],
      "t_embedder.linear1.bias": [4],
      "t_embedder.linear2.weight": [2, 4],
      "t_embedder.linear2.bias": [2],
      "all_final_layer.2-1.adaLN_modulation.0.weight": [6, 2],
      "all_final_layer.2-1.adaLN_modulation.0.bias": [6],
      "all_final_layer.2-1.linear.weight": [3, 6],
      "all_final_layer.2-1.linear.bias": [3],
    ], to: transformerDir.appendingPathComponent("0.safetensors"))

    let mapper = ZImageWeightsMapper(snapshot: modelDir, logger: Logger(label: "test"))
    let weights = try mapper.loadTransformer(dtype: nil)

    XCTAssertEqual(weights["t_embedder.linear1.weight"]?.shape, [4, 2])
    XCTAssertEqual(weights["t_embedder.mlp.0.weight"]?.shape, [4, 2])
    XCTAssertEqual(weights["t_embedder.linear2.bias"]?.shape, [2])
    XCTAssertEqual(weights["t_embedder.mlp.2.bias"]?.shape, [2])
    XCTAssertEqual(weights["all_final_layer.2-1.adaLN_modulation.0.weight"]?.shape, [6, 2])
    XCTAssertEqual(weights["all_final_layer.2-1.adaLN_modulation.1.weight"]?.shape, [6, 2])
    XCTAssertEqual(weights["all_final_layer.2-1.adaLN_modulation.1.bias"]?.shape, [6])
  }

  func testFinalLayerAdaLNAliasAppliesToAnyFinalLayerKey() {
    let value = MLXArray([Float(0.0)])
    let normalized = ZImageTransformerWeightAliases.normalized([
      "all_final_layer.custom-key.adaLN_modulation.0.weight": value,
      "all_final_layer.custom-key.adaLN_modulation.0.bias": value,
    ])

    XCTAssertNotNil(normalized["all_final_layer.custom-key.adaLN_modulation.1.weight"])
    XCTAssertNotNil(normalized["all_final_layer.custom-key.adaLN_modulation.1.bias"])
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
