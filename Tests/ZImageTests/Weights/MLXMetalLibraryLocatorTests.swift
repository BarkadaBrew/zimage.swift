import XCTest
@testable import ZImage

final class MLXMetalLibraryLocatorTests: XCTestCase {

  func testLocateMetalLibraryFindsSwiftPMBundleFallback() throws {
    let packageRoot = try makeTempPackageRoot()
    defer { try? FileManager.default.removeItem(at: packageRoot) }

    let executableURL = packageRoot.appendingPathComponent(".build/arm64-apple-macosx/debug/ZImageCLI")
    let bundleLibrary = packageRoot.appendingPathComponent(
      ".build/xcode-release/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
    )

    try writeFile(at: executableURL, contents: Data("cli".utf8))
    try writeFile(at: bundleLibrary, contents: Data("bundle".utf8))

    let located = MLXMetalLibraryLocator.locateMetalLibrary(executableURL: executableURL)

    XCTAssertEqual(located?.standardizedFileURL.path, bundleLibrary.standardizedFileURL.path)
  }

  func testPrepareColocatedMetalLibraryCopiesReleaseFallbackNextToExecutable() throws {
    let packageRoot = try makeTempPackageRoot()
    defer { try? FileManager.default.removeItem(at: packageRoot) }

    let executableURL = packageRoot.appendingPathComponent(".build/arm64-apple-macosx/debug/ZImageCLI")
    let releaseLibrary = packageRoot.appendingPathComponent(".build/arm64-apple-macosx/release/mlx.metallib")
    let releaseContents = Data("release-metallib".utf8)

    try writeFile(at: executableURL, contents: Data("cli".utf8))
    try writeFile(at: releaseLibrary, contents: releaseContents)

    let staged = try MLXMetalLibraryLocator.prepareColocatedMetalLibrary(executableURL: executableURL)

    XCTAssertEqual(
      staged?.standardizedFileURL.path,
      executableURL.deletingLastPathComponent().appendingPathComponent("mlx.metallib").standardizedFileURL.path
    )
    XCTAssertEqual(try Data(contentsOf: staged!), releaseContents)
  }

  func testPrepareColocatedMetalLibraryReplacesSwiftPMReleaseLibraryWithXcodeReleaseFallback() throws {
    let packageRoot = try makeTempPackageRoot()
    defer { try? FileManager.default.removeItem(at: packageRoot) }

    let executableURL = packageRoot.appendingPathComponent(".build/release/ZImageCLI")
    let colocatedLibrary = packageRoot.appendingPathComponent(".build/release/mlx.metallib")
    let xcodeReleaseLibrary = packageRoot.appendingPathComponent(".build/xcode/Build/Products/Release/mlx.metallib")

    try writeFile(at: executableURL, contents: Data("cli".utf8))
    try writeFile(at: colocatedLibrary, contents: Data("swiftpm-release".utf8))
    try writeFile(at: xcodeReleaseLibrary, contents: Data("xcode-release".utf8))

    let staged = try MLXMetalLibraryLocator.prepareColocatedMetalLibrary(executableURL: executableURL)

    XCTAssertEqual(staged?.standardizedFileURL.path, colocatedLibrary.standardizedFileURL.path)
    XCTAssertEqual(try Data(contentsOf: colocatedLibrary), Data("xcode-release".utf8))
  }

  private func makeTempPackageRoot() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data("// test package".utf8).write(to: directory.appendingPathComponent("Package.swift"))
    return directory
  }

  private func writeFile(at url: URL, contents: Data) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try contents.write(to: url)
  }
}
