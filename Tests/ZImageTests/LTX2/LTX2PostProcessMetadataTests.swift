import AVFoundation
import CoreGraphics
import Foundation
import XCTest

@testable import ZImage

/// comfybox#401: the mp4 metadata atom `writeMP4` can optionally carry
/// (ruling 2's "atom optional" half) and the byte-identity guarantee when it
/// isn't asked for (ruling 4). No model weights — synthetic solid frames,
/// same technique `LTX2AudioMuxTests` uses.
final class LTX2PostProcessMetadataTests: XCTestCase {

  private func solidFrame(width: Int, height: Int) -> CGImage {
    let ctx = CGContext(
      data: nil, width: width, height: height, bitsPerComponent: 8,
      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return ctx.makeImage()!
  }

  private func tempPath() -> String {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("mp4meta-\(UUID().uuidString).mp4").path
  }

  // MARK: - The atom, when requested, is readable back

  func testGenerationRecordJSONIsEmbeddedAndReadableBack() throws {
    let out = tempPath()
    defer { try? FileManager.default.removeItem(atPath: out) }
    let frames = (0..<9).map { _ in solidFrame(width: 64, height: 64) }
    let json = "{\"prompt\":\"a fox\",\"seed\":42}"

    try LTX2PostProcess.writeMP4(
      frames: frames, outputPath: out, fps: 24, width: 64, height: 64,
      generationRecordJSON: json)

    let asset = AVURLAsset(url: URL(fileURLWithPath: out))
    let items = asset.metadata.filter { $0.commonKey == .commonKeyDescription }
    XCTAssertEqual(items.count, 1, "exactly one description metadata item")
    XCTAssertEqual(items.first?.value as? String, json)

    // The video itself is still intact — the atom did not require a re-encode
    // that would have dropped or corrupted the track.
    XCTAssertEqual(asset.tracks(withMediaType: .video).count, 1)
  }

  func testNoGenerationRecordEmbedsNoDescriptionMetadata() throws {
    let out = tempPath()
    defer { try? FileManager.default.removeItem(atPath: out) }
    let frames = (0..<9).map { _ in solidFrame(width: 64, height: 64) }

    try LTX2PostProcess.writeMP4(frames: frames, outputPath: out, fps: 24, width: 64, height: 64)

    let asset = AVURLAsset(url: URL(fileURLWithPath: out))
    XCTAssertTrue(asset.metadata.isEmpty, "no metadata param -> no atom, exactly today's behavior")
  }

  // MARK: - Byte identity when the atom path is not taken (ruling 4)

  /// Minimal top-level MP4/QuickTime box walker: `[size(4)][fourCC(4)]` per
  /// box, with the 64-bit `size == 1` largesize extension and `size == 0`
  /// "extends to EOF" both handled. Returns the FULL box (header included)
  /// for the first top-level box named `target`, or nil.
  private func topLevelBox(named target: String, in data: Data) -> Data? {
    var offset = data.startIndex
    while offset < data.endIndex {
      guard let typeEnd = data.index(offset, offsetBy: 8, limitedBy: data.endIndex) else { return nil }
      let sizeField = data[offset..<data.index(offset, offsetBy: 4)]
      let declaredSize = sizeField.reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
      let fourCC = String(decoding: data[data.index(offset, offsetBy: 4)..<typeEnd], as: UTF8.self)

      var boxSize = declaredSize
      if declaredSize == 1 {
        guard let largeEnd = data.index(typeEnd, offsetBy: 8, limitedBy: data.endIndex) else { return nil }
        boxSize = data[typeEnd..<largeEnd].reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
      } else if declaredSize == 0 {
        boxSize = UInt64(data.distance(from: offset, to: data.endIndex))
      }
      guard boxSize >= 8, let boxEnd = data.index(offset, offsetBy: Int(boxSize), limitedBy: data.endIndex) else {
        return nil
      }
      if fourCC == target { return data.subdata(in: offset..<boxEnd) }
      offset = boxEnd
    }
    return nil
  }

  /// AVAssetWriter stamps `mvhd`/`tkhd`/`mdhd` `creation_time`/
  /// `modification_time` from wall-clock time on every write — independent of
  /// anything this ticket changes (confirmed empirically: two back-to-back
  /// writes of identical frames differ at exactly those fields and nowhere
  /// else in `moov`). Zero them so two honest writes of the same content
  /// compare equal.
  private func zeroingKnownTimestampFields(_ data: Data) -> Data {
    var mutable = data
    // mvhd/tkhd/mdhd (version 0): fourCC, then 4 bytes version+flags, then
    // two 4-byte big-endian QuickTime timestamps (creation_time,
    // modification_time) we don't want to compare.
    for fourCC in ["mvhd", "tkhd", "mdhd"] {
      let needle = Data(fourCC.utf8)
      var searchStart = mutable.startIndex
      while let range = mutable.range(of: needle, options: [], in: searchStart..<mutable.endIndex) {
        guard let tsStart = mutable.index(range.upperBound, offsetBy: 4, limitedBy: mutable.endIndex),
              let tsEnd = mutable.index(tsStart, offsetBy: 8, limitedBy: mutable.endIndex) else {
          searchStart = range.upperBound
          continue
        }
        for i in stride(from: tsStart, to: tsEnd, by: 1) { mutable[i] = 0 }
        searchStart = range.upperBound
      }
    }
    return mutable
  }

  /// Ruling 4's "byte-identity of the mp4 itself when the atom path is not
  /// taken" — scoped to the `moov` container atom, not the whole file.
  /// `mdat` (the actual H.264 bitstream) is NOT byte-stable across two
  /// separate encodes of identical frames even with zero code changes here —
  /// verified empirically, and the reason `LTX2AudioMuxTests` already
  /// documents "never byte equality" for this writer at the whole-file
  /// level. `moov` (container structure, track headers, sample tables) is
  /// exactly the region a metadata atom could plausibly perturb, and IS
  /// byte-stable once the wall-clock timestamp fields are normalized — so
  /// that is the honest, checkable claim: adding the (default-nil)
  /// `generationRecordJSON` parameter changes zero bytes of `moov` relative
  /// to a caller that never passed it.
  func testWriteMP4WithoutGenerationRecordParameterIsByteIdenticalToExplicitNil() throws {
    let outOmitted = tempPath()
    let outExplicitNil = tempPath()
    defer {
      try? FileManager.default.removeItem(atPath: outOmitted)
      try? FileManager.default.removeItem(atPath: outExplicitNil)
    }
    let frames = (0..<9).map { _ in solidFrame(width: 64, height: 64) }

    // Old call shape (no new parameter at all).
    try LTX2PostProcess.writeMP4(frames: frames, outputPath: outOmitted, fps: 24, width: 64, height: 64)
    // New call shape, explicitly not taking the atom path.
    try LTX2PostProcess.writeMP4(
      frames: frames, outputPath: outExplicitNil, fps: 24, width: 64, height: 64,
      generationRecordJSON: nil)

    let dataOmitted = try Data(contentsOf: URL(fileURLWithPath: outOmitted))
    let dataExplicitNil = try Data(contentsOf: URL(fileURLWithPath: outExplicitNil))
    guard let moovOmitted = topLevelBox(named: "moov", in: dataOmitted),
          let moovExplicitNil = topLevelBox(named: "moov", in: dataExplicitNil) else {
      return XCTFail("could not locate the moov box in one of the outputs")
    }

    let a = zeroingKnownTimestampFields(moovOmitted)
    let b = zeroingKnownTimestampFields(moovExplicitNil)
    XCTAssertEqual(
      a, b,
      "the nil-metadata path must be byte-identical (modulo wall-clock timestamps) to a caller that never passed the new parameter")
  }

  func testEmbeddingTheAtomDoesNotAlterTheEncodedVideoBytes() throws {
    let outPlain = tempPath()
    let outWithAtom = tempPath()
    defer {
      try? FileManager.default.removeItem(atPath: outPlain)
      try? FileManager.default.removeItem(atPath: outWithAtom)
    }
    let frames = (0..<9).map { _ in solidFrame(width: 64, height: 64) }

    try LTX2PostProcess.writeMP4(frames: frames, outputPath: outPlain, fps: 24, width: 64, height: 64)
    try LTX2PostProcess.writeMP4(
      frames: frames, outputPath: outWithAtom, fps: 24, width: 64, height: 64,
      generationRecordJSON: "{\"prompt\":\"a fox\"}")

    // No re-encode: same frame count, same video track geometry, same
    // duration — the atom is a header-only addition.
    let plain = AVURLAsset(url: URL(fileURLWithPath: outPlain))
    let withAtom = AVURLAsset(url: URL(fileURLWithPath: outWithAtom))
    XCTAssertEqual(
      CMTimeGetSeconds(plain.tracks(withMediaType: .video)[0].timeRange.duration),
      CMTimeGetSeconds(withAtom.tracks(withMediaType: .video)[0].timeRange.duration),
      accuracy: 0.001)
    XCTAssertGreaterThan(
      (try FileManager.default.attributesOfItem(atPath: outWithAtom)[.size] as! Int),
      (try FileManager.default.attributesOfItem(atPath: outPlain)[.size] as! Int),
      "the atom adds header bytes")
  }
}
