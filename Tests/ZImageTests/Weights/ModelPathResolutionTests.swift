import XCTest
@testable import ZImage

final class ModelPathResolutionTests: XCTestCase {

  func testRecognizesModelDirectoryWithoutConfigFiles() throws {
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

    XCTAssertTrue(ZImageFiles.hasRecognizableModelDirectory(at: modelDir))
  }

  func testTextEncoderSelectionPriorityIsOverrideThenEnvThenAutoThenDefault() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let standardDir = modelDir.appendingPathComponent("text_encoder")
    let preferredDir = modelDir.appendingPathComponent("text_encoder QWen Large")
    let explicitDir = modelDir.appendingPathComponent("encoder-override")

    try makeDirectory(standardDir)
    try makeDirectory(preferredDir)
    try makeDirectory(explicitDir)

    FileManager.default.createFile(atPath: standardDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)
    FileManager.default.createFile(atPath: preferredDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)
    FileManager.default.createFile(atPath: explicitDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)

    let overrideSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: explicitDir.path,
      environment: ["ZIMAGE_ENCODER_PATH": standardDir.path]
    )
    XCTAssertEqual(overrideSelection.directory.standardizedFileURL.path, explicitDir.standardizedFileURL.path)
    XCTAssertEqual(overrideSelection.source, .overridePath)

    let envSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: ["ZIMAGE_ENCODER_PATH": standardDir.path]
    )
    XCTAssertEqual(envSelection.directory.standardizedFileURL.path, standardDir.standardizedFileURL.path)
    XCTAssertEqual(envSelection.source, .environment)

    let autoSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: [:]
    )
    XCTAssertEqual(autoSelection.directory.standardizedFileURL.path, preferredDir.standardizedFileURL.path)
    XCTAssertEqual(autoSelection.source, .autoDetectedPreferred)

    try FileManager.default.removeItem(at: preferredDir)

    let defaultSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: [:]
    )
    XCTAssertEqual(defaultSelection.directory.standardizedFileURL.path, standardDir.standardizedFileURL.path)
    XCTAssertEqual(defaultSelection.source, .defaultDirectory)
  }

  func testResolveVAEWeightsSupportsGenericShardNamesAndIndex() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let vaeDir = modelDir.appendingPathComponent("vae")
    try makeDirectory(vaeDir)

    let indexJSON = """
    {
      "weight_map": {
        "decoder.conv_in.conv.weight": "0.safetensors"
      }
    }
    """

    try indexJSON.write(to: vaeDir.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
    FileManager.default.createFile(atPath: vaeDir.appendingPathComponent("0.safetensors").path, contents: Data(), attributes: nil)

    XCTAssertEqual(ZImageFiles.resolveVAEWeights(at: modelDir), ["vae/0.safetensors"])
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
