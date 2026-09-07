import Foundation
import XCTest

@testable import ZImage

/// #402 (coffeeshop-server #1681, the Luxe_Sensual incident): the
/// cross-family LoRA guard at `POST`/`PUT /v1/presets`
/// (`PresetStore.validateLoRAFamilyCompatibility`, called from
/// `PresetStore.validate`). Mirrors `PresetStoreTests.testNewFieldsAreValidatedOnSave`'s
/// pattern: exercise `PresetStore.upsert` directly against a throwaway file,
/// no server, no weights.
final class PresetStoreLoRAFamilyGuardTests: XCTestCase {

  private func makeTempPath() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-preset-lora-guard-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return dir.appendingPathComponent("presets.json")
  }

  /// Minimal, valid `LoRALibraryEntry` declaring `compat`.
  private func makeEntry(id: String, compat: [String]) -> LoRALibraryEntry {
    let json = """
    {
      "id": "\(id)",
      "filename": "\(id).safetensors",
      "relative_path": "\(id).safetensors",
      "size_bytes": 1024,
      "model_compatibility": [\(compat.map { "\"\($0)\"" }.joined(separator: ","))],
      "format": "lora",
      "rank": 64,
      "key_count": 10,
      "layer_targets": ["attention"],
      "triggerwords": [],
      "recommended_scale": 1.0,
      "scale_range": [0.0, 2.0],
      "tags": [],
      "category": "uncategorized",
      "notes": "",
      "date_added": "2026-09-04",
      "quarantined": false
    }
    """
    return try! JSONDecoder().decode(LoRALibraryEntry.self, from: Data(json.utf8))
  }

  private func expectValidationRefusal(
    _ store: PresetStore, _ preset: ImagePreset, loraLookup: (String) -> LoRALibraryEntry?,
    naming needles: [String], file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertThrowsError(try store.upsert(preset, loraLookup: loraLookup), file: file, line: line) { error in
      guard case .validation(let message)? = error as? PresetStoreError else {
        return XCTFail("expected .validation, got \(error)", file: file, line: line)
      }
      for needle in needles {
        XCTAssertTrue(message.contains(needle), "\(message) should name \(needle)", file: file, line: line)
      }
    }
    XCTAssertNil(store.get(preset.id), file: file, line: line)
  }

  // MARK: - resolvedLoRAFamily

  func testResolvedFamilyFromMediaKindVideo() {
    let preset = ImagePreset(id: "v", name: "V", mediaKind: "video")
    XCTAssertEqual(PresetStore.resolvedLoRAFamily(for: preset), "ltx")
  }

  func testResolvedFamilyFromCheckpointFamilyKrea2() {
    let preset = ImagePreset(id: "k", name: "K", checkpointFamily: "raw-accel")
    XCTAssertEqual(PresetStore.resolvedLoRAFamily(for: preset), "krea2")
  }

  func testResolvedFamilyFromCheckpointFamilyZImage() {
    let preset = ImagePreset(id: "z", name: "Z", checkpointFamily: "zimage-turbo")
    XCTAssertEqual(PresetStore.resolvedLoRAFamily(for: preset), "z-image")
  }

  func testResolvedFamilyFromModelSpec() {
    let preset = ImagePreset(id: "m", name: "M", model: "krea2")
    XCTAssertEqual(PresetStore.resolvedLoRAFamily(for: preset), "krea2")
  }

  func testResolvedFamilyNilWhenNothingDeclared() {
    let preset = ImagePreset(id: "n", name: "N")
    XCTAssertNil(PresetStore.resolvedLoRAFamily(for: preset))
  }

  // MARK: - Save-time guard: the four required scenarios

  /// An image (z-image) LoRA on a VIDEO preset.
  func testImageLoRAOnVideoPresetIsRejected() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let entry = makeEntry(id: "portrait-style", compat: ["z-image"])
    let preset = ImagePreset(
      id: "video-preset", name: "Video", mediaKind: "video",
      loras: [LoraReference(filename: "portrait-style.safetensors", scale: 1.0)])
    expectValidationRefusal(
      store, preset, loraLookup: { _ in entry },
      naming: ["portrait-style.safetensors", "z-image", "ltx"])
  }

  /// A video (ltx) LoRA on an image (z-image) preset — the Luxe_Sensual
  /// incident.
  func testVideoLoRAOnImagePresetIsRejected() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let entry = makeEntry(id: "sulphur-video", compat: ["ltx"])
    let preset = ImagePreset(
      id: "image-preset", name: "Image",
      loras: [LoraReference(filename: "sulphur-video.safetensors", scale: 1.0)],
      checkpointFamily: "zimage-turbo")
    expectValidationRefusal(
      store, preset, loraLookup: { _ in entry },
      naming: ["sulphur-video.safetensors", "ltx", "z-image"])
  }

  /// comfybox#393: a real-Flux.1 LoRA on a krea2 preset — "flux1" the tag is
  /// NOT the same thing as this engine's `flux1`-named Z-Image family, and
  /// it is not krea2 either, so it must be rejected either way.
  func testRealFlux1LoRAOnKrea2PresetIsRejected() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let entry = makeEntry(id: "civitai-flux-style", compat: ["flux1"])
    let preset = ImagePreset(
      id: "krea2-preset", name: "Krea2",
      loras: [LoraReference(filename: "civitai-flux-style.safetensors", scale: 1.0)],
      checkpointFamily: "raw-accel")
    expectValidationRefusal(
      store, preset, loraLookup: { _ in entry },
      naming: ["civitai-flux-style.safetensors", "flux1", "krea2"])
  }

  /// Unknown/unscanned compatibility never refuses — only warns.
  func testUnknownLoRAOnDeclaredFamilyPresetIsAllowedWithWarning() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    var logged: [String] = []
    let preset = ImagePreset(
      id: "krea2-preset-2", name: "Krea2",
      loras: [LoraReference(filename: "never-scanned.safetensors", scale: 1.0)],
      checkpointFamily: "raw-accel")
    XCTAssertNoThrow(
      try store.upsert(preset, loraLookup: { _ in nil }))
    // Exercise the pure validator directly too, to assert the warning text.
    try? PresetStore.validateLoRAFamilyCompatibility(
      preset, lookup: { _ in nil }, log: { logged.append($0) })
    XCTAssertTrue(logged.contains { $0.contains("never-scanned.safetensors") })
    XCTAssertNotNil(store.get("krea2-preset-2"))
  }

  /// A preset that declares NO family at all (no model, no checkpointFamily,
  /// no video mediaKind) is never refused, whatever the LoRA's own tags say
  /// — there is nothing to compare against (ruling 2).
  func testPresetWithNoDeclaredFamilyIsNeverRefused() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let entry = makeEntry(id: "sulphur-video-3", compat: ["ltx"])
    let preset = ImagePreset(
      id: "bare-preset", name: "Bare",
      loras: [LoraReference(filename: "sulphur-video-3.safetensors", scale: 1.0)])
    XCTAssertNoThrow(try store.upsert(preset, loraLookup: { _ in entry }))
  }

  // MARK: - Additivity (#402 ruling 3)

  /// Pinning `PresetStore.upsert`'s output for a krea2 preset with its
  /// normal stack: byte-identical before and after this guard exists,
  /// whether the LoRA library has scanned the file (declares krea2) or not
  /// (the default `loraLookup`, "unknown"). Neither path may mutate the
  /// preset — only `validate`/`throw` or pass through.
  func testKrea2PresetWithNormalStackIsUnchangedByTheGuard() throws {
    let stackDeclaredCompatible = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let accelEntry = makeEntry(id: "krea2_turbo_distill_r256", compat: ["krea2"])
    let preset = ImagePreset(
      id: "krea2-normal", name: "Krea2 Normal", model: "krea2-raw",
      loras: [LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel")],
      checkpointFamily: "raw-accel")

    let savedWithLibrary = try stackDeclaredCompatible.upsert(preset, loraLookup: { _ in accelEntry })
    XCTAssertEqual(savedWithLibrary.loras, preset.loras)
    XCTAssertEqual(savedWithLibrary.checkpointFamily, "raw-accel")

    let stackUnscanned = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let savedWithoutLibrary = try stackUnscanned.upsert(preset)
    XCTAssertEqual(savedWithoutLibrary.loras, preset.loras)
    XCTAssertEqual(savedWithoutLibrary, savedWithLibrary, "declared-compatible and unscanned must save identically")
  }

  /// The default `loraLookup` on `PresetStore.upsert`/`validate` (no library
  /// wired) resolves every LoRA as unknown — every EXISTING call site that
  /// omits the parameter keeps behaving exactly as it did before #402.
  func testDefaultLoraLookupNeverRefuses() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let preset = ImagePreset(
      id: "default-lookup", name: "Default", mediaKind: "video",
      loras: [LoraReference(filename: "anything.safetensors", scale: 1.0)])
    XCTAssertNoThrow(try store.upsert(preset))
  }
}
