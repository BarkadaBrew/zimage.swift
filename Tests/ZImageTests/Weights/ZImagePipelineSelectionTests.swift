import XCTest
import Logging
@testable import ZImage

final class ZImagePipelineSelectionTests: XCTestCase {

  func testResolveModelSelectionKeepsConfiglessLocalModelDirectoryAsBaseModel() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    try makeDirectory(modelDir.appendingPathComponent("tokenizer"))
    try makeDirectory(modelDir.appendingPathComponent("transformer"))
    try makeDirectory(modelDir.appendingPathComponent("text_encoder"))
    try makeDirectory(modelDir.appendingPathComponent("vae"))

    FileManager.default.createFile(
      atPath: modelDir.appendingPathComponent("transformer/0.safetensors").path,
      contents: Data(),
      attributes: nil
    )
    FileManager.default.createFile(
      atPath: modelDir.appendingPathComponent("text_encoder/model.safetensors").path,
      contents: Data(),
      attributes: nil
    )
    FileManager.default.createFile(
      atPath: modelDir.appendingPathComponent("vae/0.safetensors").path,
      contents: Data(),
      attributes: nil
    )

    let pipeline = ZImagePipeline(logger: Logger(label: "ZImageTests.pipeline-selection"))
    let selection = pipeline.resolveModelSelection(modelDir.path, forceTransformerOverrideOnly: false)

    XCTAssertEqual(selection.baseModelSpec, modelDir.path)
    XCTAssertNil(selection.transformerOverrideURL)
    XCTAssertNil(selection.aioCheckpointURL)
  }

  private func makeTempDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try makeDirectory(directory)
    return directory
  }

  private func makeDirectory(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  }
}
