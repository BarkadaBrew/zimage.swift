import Foundation

public enum MLXMetalLibraryLocator {
  public static let colocatedLibraryName = "mlx.metallib"
  public static let swiftPMBundleLibraryName = "default.metallib"
  public static let swiftPMBundleDirectoryName = "mlx-swift_Cmlx.bundle"

  public static func prepareColocatedMetalLibrary(
    executableURL: URL,
    fileManager: FileManager = .default
  ) throws -> URL? {
    let executableDirectory = executableURL.deletingLastPathComponent()
    let colocatedLibrary = executableDirectory.appendingPathComponent(colocatedLibraryName)
    if fileManager.fileExists(atPath: colocatedLibrary.path) {
      return colocatedLibrary
    }

    let resourcesLibrary = executableDirectory.appendingPathComponent("Resources/\(colocatedLibraryName)")
    if fileManager.fileExists(atPath: resourcesLibrary.path) {
      return resourcesLibrary
    }

    guard let sourceLibrary = locateMetalLibrary(executableURL: executableURL, fileManager: fileManager) else {
      return nil
    }

    try fileManager.createDirectory(at: executableDirectory, withIntermediateDirectories: true)
    do {
      try fileManager.copyItem(at: sourceLibrary, to: colocatedLibrary)
    } catch let error as CocoaError where error.code == .fileWriteFileExists {
      return colocatedLibrary
    } catch {
      if fileManager.fileExists(atPath: colocatedLibrary.path) {
        return colocatedLibrary
      }
      throw error
    }
    return colocatedLibrary
  }

  public static func locateMetalLibrary(
    executableURL: URL,
    fileManager: FileManager = .default
  ) -> URL? {
    let executableDirectory = executableURL.deletingLastPathComponent()
    let colocatedLibrary = executableDirectory.appendingPathComponent(colocatedLibraryName)
    if fileManager.fileExists(atPath: colocatedLibrary.path) {
      return colocatedLibrary
    }

    let resourcesLibrary = executableDirectory.appendingPathComponent("Resources/\(colocatedLibraryName)")
    if fileManager.fileExists(atPath: resourcesLibrary.path) {
      return resourcesLibrary
    }

    guard let packageRoot = packageRoot(containing: executableURL, fileManager: fileManager) else {
      return nil
    }

    let buildRoot = packageRoot.appendingPathComponent(".build", isDirectory: true)
    let executableBuildDirectory = executableDirectory.deletingLastPathComponent()
    let buildConfiguration = executableBuildDirectory.lastPathComponent
    let platformBuildRoot = executableBuildDirectory.deletingLastPathComponent()

    let preferredCandidates = [
      platformBuildRoot.appendingPathComponent("\(buildConfiguration)/\(colocatedLibraryName)"),
      platformBuildRoot.appendingPathComponent("release/\(colocatedLibraryName)"),
      buildRoot.appendingPathComponent("xcode-release/Build/Products/Release/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)"),
      buildRoot.appendingPathComponent("xcode-debug/Build/Products/Debug/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)")
    ]

    for candidate in preferredCandidates where fileManager.fileExists(atPath: candidate.path) {
      return candidate
    }

    guard let enumerator = fileManager.enumerator(
      at: buildRoot,
      includingPropertiesForKeys: [.isRegularFileKey],
      options: [.skipsHiddenFiles]
    ) else {
      return nil
    }

    var firstBundleCandidate: URL?
    var firstLibraryCandidate: URL?

    for case let candidate as URL in enumerator {
      guard fileManager.fileExists(atPath: candidate.path) else { continue }

      if candidate.lastPathComponent == colocatedLibraryName {
        firstLibraryCandidate = firstLibraryCandidate ?? candidate
      } else if candidate.lastPathComponent == swiftPMBundleLibraryName
        && candidate.path.contains("/\(swiftPMBundleDirectoryName)/")
      {
        firstBundleCandidate = firstBundleCandidate ?? candidate
      }
    }

    return firstLibraryCandidate ?? firstBundleCandidate
  }

  static func packageRoot(containing url: URL, fileManager: FileManager = .default) -> URL? {
    var current = url.standardizedFileURL.deletingLastPathComponent()
    while current.path != current.deletingLastPathComponent().path {
      if fileManager.fileExists(atPath: current.appendingPathComponent("Package.swift").path) {
        return current
      }
      current.deleteLastPathComponent()
    }
    return fileManager.fileExists(atPath: current.appendingPathComponent("Package.swift").path) ? current : nil
  }
}
