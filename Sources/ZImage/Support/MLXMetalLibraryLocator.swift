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

    let resourcesLibrary = executableDirectory.appendingPathComponent("Resources/\(colocatedLibraryName)")
    if fileManager.fileExists(atPath: resourcesLibrary.path) {
      return resourcesLibrary
    }

    guard let sourceLibrary = locateMetalLibrary(executableURL: executableURL, fileManager: fileManager) else {
      if fileManager.fileExists(atPath: colocatedLibrary.path) {
        return colocatedLibrary
      }
      return nil
    }

    if fileManager.fileExists(atPath: colocatedLibrary.path) {
      if colocatedLibrary.standardizedFileURL == sourceLibrary.standardizedFileURL {
        return colocatedLibrary
      }

      if shouldRestageBuildMetalLibrary(
        executableURL: executableURL,
        colocatedLibrary: colocatedLibrary,
        preferredSourceLibrary: sourceLibrary,
        fileManager: fileManager
      ) {
        try replaceItem(at: colocatedLibrary, with: sourceLibrary, fileManager: fileManager)
      }
      return colocatedLibrary
    }

    try fileManager.createDirectory(at: executableDirectory, withIntermediateDirectories: true)
    try replaceItem(at: colocatedLibrary, with: sourceLibrary, fileManager: fileManager)
    return colocatedLibrary
  }

  public static func locateMetalLibrary(
    executableURL: URL,
    fileManager: FileManager = .default
  ) -> URL? {
    let executableDirectory = executableURL.deletingLastPathComponent()
    let resourcesLibrary = executableDirectory.appendingPathComponent("Resources/\(colocatedLibraryName)")
    if fileManager.fileExists(atPath: resourcesLibrary.path) {
      return resourcesLibrary
    }

    if let preferredBuildLibrary = preferredBuildMetalLibrary(executableURL: executableURL, fileManager: fileManager) {
      return preferredBuildLibrary
    }

    let colocatedLibrary = executableDirectory.appendingPathComponent(colocatedLibraryName)
    if fileManager.fileExists(atPath: colocatedLibrary.path) {
      return colocatedLibrary
    }

    guard let packageRoot = packageRoot(containing: executableURL, fileManager: fileManager) else {
      return nil
    }

    let buildRoot = packageRoot.appendingPathComponent(".build", isDirectory: true)

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

  private static func preferredBuildMetalLibrary(
    executableURL: URL,
    fileManager: FileManager
  ) -> URL? {
    guard let packageRoot = packageRoot(containing: executableURL, fileManager: fileManager) else {
      return nil
    }

    let buildRoot = packageRoot.appendingPathComponent(".build", isDirectory: true)
    let executableDirectory = executableURL.deletingLastPathComponent()
    guard executableDirectory.path.hasPrefix(buildRoot.path) else {
      return nil
    }

    let buildConfiguration = executableDirectory.lastPathComponent.lowercased()
    let platformBuildRoot = executableDirectory.deletingLastPathComponent()

    let xcodeCandidates: [URL]
    if buildConfiguration == "release" {
      xcodeCandidates = [
        buildRoot.appendingPathComponent("xcode/Build/Products/Release/\(colocatedLibraryName)"),
        buildRoot.appendingPathComponent("xcode-release/Build/Products/Release/\(colocatedLibraryName)"),
        buildRoot.appendingPathComponent("xcode/Build/Products/Release/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)"),
        buildRoot.appendingPathComponent("xcode-release/Build/Products/Release/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)"),
        buildRoot.appendingPathComponent("xcode/Build/Products/Debug/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)"),
        buildRoot.appendingPathComponent("xcode-debug/Build/Products/Debug/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)")
      ]
    } else {
      xcodeCandidates = [
        buildRoot.appendingPathComponent("xcode/Build/Products/Debug/\(colocatedLibraryName)"),
        buildRoot.appendingPathComponent("xcode-debug/Build/Products/Debug/\(colocatedLibraryName)"),
        buildRoot.appendingPathComponent("xcode/Build/Products/Debug/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)"),
        buildRoot.appendingPathComponent("xcode-debug/Build/Products/Debug/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)"),
        buildRoot.appendingPathComponent("xcode/Build/Products/Release/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)"),
        buildRoot.appendingPathComponent("xcode-release/Build/Products/Release/\(swiftPMBundleDirectoryName)/Contents/Resources/\(swiftPMBundleLibraryName)")
      ]
    }

    for candidate in xcodeCandidates where fileManager.fileExists(atPath: candidate.path) {
      return candidate
    }

    let buildCandidates = [
      platformBuildRoot.appendingPathComponent("\(buildConfiguration)/\(colocatedLibraryName)"),
      platformBuildRoot.appendingPathComponent("release/\(colocatedLibraryName)")
    ]

    for candidate in buildCandidates where fileManager.fileExists(atPath: candidate.path) {
      return candidate
    }

    return nil
  }

  private static func shouldRestageBuildMetalLibrary(
    executableURL: URL,
    colocatedLibrary: URL,
    preferredSourceLibrary: URL,
    fileManager: FileManager
  ) -> Bool {
    guard let packageRoot = packageRoot(containing: executableURL, fileManager: fileManager) else {
      return false
    }

    let buildRoot = packageRoot.appendingPathComponent(".build", isDirectory: true).standardizedFileURL
    let colocatedPath = colocatedLibrary.standardizedFileURL.path
    let preferredSourcePath = preferredSourceLibrary.standardizedFileURL.path

    return colocatedPath.hasPrefix(buildRoot.path) && preferredSourcePath.hasPrefix(buildRoot.path)
  }

  private static func replaceItem(
    at destination: URL,
    with source: URL,
    fileManager: FileManager
  ) throws {
    do {
      if fileManager.fileExists(atPath: destination.path) {
        try fileManager.removeItem(at: destination)
      }
      try fileManager.copyItem(at: source, to: destination)
    } catch let error as CocoaError where error.code == .fileWriteFileExists {
      return
    }
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
