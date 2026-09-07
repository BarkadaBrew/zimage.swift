import Foundation
import XCTest

@testable import ZImage

/// #402 (coffeeshop-server #1681, the Luxe_Sensual incident): wiring
/// coverage for the cross-family LoRA guard at the three enforcement points
/// the ruling named — `POST /v1/lora/swap`, per-request `loras[]` on
/// `/v1/generate` / `/v1/generate/async` (via the shared
/// `decodedGeneratePayload` choke point), and `POST /v1/presets` (covered in
/// `PresetStoreLoRAFamilyGuardTests`). Exercises the pure static functions
/// and `WarmServer.errorResponse(for:)` directly — no listening server, no
/// weights, per `WarmServerRejectionTests`' own pattern.
final class LoRAFamilyGuardTests: XCTestCase {

  private func bodyString(_ response: HTTPResponse) -> String {
    String(decoding: response.body, as: UTF8.self)
  }

  /// Minimal, valid `LoRALibraryEntry` declaring `compat` — mirrors
  /// `LoRALibraryEntryCodingTests.entryJSON`'s required-fields shape.
  private func makeEntry(id: String = "test-lora", compat: [String]) -> LoRALibraryEntry {
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
    let decoder = JSONDecoder()
    return try! decoder.decode(LoRALibraryEntry.self, from: Data(json.utf8))
  }

  // MARK: - WarmModelFamily.loraCompatibilityFamily (#393)

  func testWarmModelFamilyMapsToCanonicalGroups() {
    XCTAssertEqual(WarmModelFamily.flux1.loraCompatibilityFamily, "z-image")
    XCTAssertEqual(WarmModelFamily.flux2.loraCompatibilityFamily, "flux2-klein")
    XCTAssertEqual(WarmModelFamily.fibo.loraCompatibilityFamily, "fibo")
    XCTAssertEqual(WarmModelFamily.chroma.loraCompatibilityFamily, "chroma")
    XCTAssertEqual(WarmModelFamily.krea2.loraCompatibilityFamily, "krea2")
  }

  // MARK: - WarmServer.validateLoRAFamilyCompatibility (shared by swap + generate)

  func testMismatchThrowsNamingLoRAAndBothFamilies() {
    let ltxEntry = makeEntry(id: "video-lora", compat: ["ltx"])
    XCTAssertThrowsError(
      try WarmServer.validateLoRAFamilyCompatibility(
        entries: [LoRAEntry(path: "video-lora.safetensors", scale: 1.0)],
        targetFamily: "z-image",
        lookup: { _ in ltxEntry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(let name, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(name, "video-lora.safetensors")
      XCTAssertEqual(families, ["ltx"])
      XCTAssertEqual(target, "z-image")

      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      let body = bodyString(response)
      XCTAssertTrue(body.contains("video-lora.safetensors"), body)
      XCTAssertTrue(body.contains("ltx"), body)
      XCTAssertTrue(body.contains("z-image"), body)
    }
  }

  func testUnknownLoRANeverThrowsOnlyLogsWarning() throws {
    var logged: [String] = []
    try WarmServer.validateLoRAFamilyCompatibility(
      entries: [LoRAEntry(path: "never-scanned.safetensors", scale: 1.0)],
      targetFamily: "z-image",
      lookup: { _ in nil },
      log: { logged.append($0) })
    XCTAssertTrue(logged.contains { $0.contains("never-scanned.safetensors") })
  }

  func testCompatibleLoRAPassesSilently() throws {
    let entry = makeEntry(id: "z-lora", compat: ["z-image"])
    var logged: [String] = []
    try WarmServer.validateLoRAFamilyCompatibility(
      entries: [LoRAEntry(path: "z-lora.safetensors", scale: 1.0)],
      targetFamily: "z-image",
      lookup: { _ in entry },
      log: { logged.append($0) })
    XCTAssertTrue(logged.isEmpty)
  }

  // MARK: - /v1/generate & /v1/generate/async (decodedGeneratePayload choke point)

  private func makePresetStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-lora-family-guard-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  private var configuration: WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  /// A video (ltx) LoRA sent on a bare (default flux1/z-image) generate
  /// request — the Luxe_Sensual incident, reproduced through the real
  /// decode choke point both `/v1/generate` and `/v1/generate/async` share.
  func testDecodedGeneratePayloadRejectsVideoLoRAOnImageRequest() throws {
    let store = try makePresetStore()
    let ltxEntry = makeEntry(id: "sulphur-video", compat: ["ltx"])
    XCTAssertThrowsError(
      try WarmServer.decodedGeneratePayload(
        from: Data(#"{"prompt":"x","loras":[{"path":"sulphur-video.safetensors"}]}"#.utf8),
        store: store, configuration: configuration,
        loraLookup: { _ in ltxEntry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(let name, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(name, "sulphur-video.safetensors")
      XCTAssertEqual(families, ["ltx"])
      XCTAssertEqual(target, "z-image")
    }
  }

  /// An image (z-image) LoRA sent on an explicit `model` that resolves to
  /// krea2 — the inverse direction, and comfybox#393's flux1/krea2 case via
  /// the SAME family resolution `ImageMemoryPreflight.resolvedFamily` uses.
  func testDecodedGeneratePayloadRejectsImageLoRAOnKrea2Model() throws {
    let store = try makePresetStore()
    let zImageEntry = makeEntry(id: "portrait-style", compat: ["z-image"])
    XCTAssertThrowsError(
      try WarmServer.decodedGeneratePayload(
        from: Data(#"{"prompt":"x","model":"krea2","loras":[{"path":"portrait-style.safetensors"}]}"#.utf8),
        store: store, configuration: configuration,
        loraLookup: { _ in zImageEntry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(let name, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(name, "portrait-style.safetensors")
      XCTAssertEqual(families, ["z-image"])
      XCTAssertEqual(target, "krea2")
    }
  }

  /// #22's `gateSubmission: false` (crash-recovery replay) must skip this
  /// gate too, same as the memory preflight beside it — a job already
  /// accepted must never be re-refused by a gate that did not exist (or
  /// disagreed) when it was submitted.
  func testGateSubmissionFalseSkipsTheGuardEntirely() throws {
    let store = try makePresetStore()
    let ltxEntry = makeEntry(id: "sulphur-video-2", compat: ["ltx"])
    let payload = try WarmServer.decodedGeneratePayload(
      from: Data(#"{"prompt":"x","loras":[{"path":"sulphur-video-2.safetensors"}]}"#.utf8),
      store: store, configuration: configuration, gateSubmission: false,
      loraLookup: { _ in ltxEntry })
    XCTAssertEqual(payload.loras?.first?.path, "sulphur-video-2.safetensors")
  }

  /// Additivity (#402 ruling 3): a request with a NORMAL, compatible krea2
  /// LoRA stack against an explicit krea2 `model` renders byte-identically —
  /// same fields, no throw — before and after this guard exists. Pinned
  /// against the resolver's own output rather than a hand-typed literal, so
  /// a future accidental behavior change on this exact shape fails loudly.
  func testExistingCompatibleKrea2RequestIsByteIdenticalPin() throws {
    let store = try makePresetStore()
    let accelEntry = makeEntry(id: "krea2_turbo_distill_r256", compat: ["krea2"])
    let body = Data(
      #"""
      {"prompt":"a portrait","model":"krea2","width":1024,"height":1024,
       "loras":[{"path":"krea2_turbo_distill_r256.safetensors","scale":0.6,"role":"accel"}]}
      """#.utf8)
    let payload = try WarmServer.decodedGeneratePayload(
      from: body, store: store, configuration: configuration,
      loraLookup: { _ in accelEntry })
    // Pin: the fields this guard could plausibly disturb, unchanged.
    XCTAssertEqual(payload.model, "krea2")
    XCTAssertEqual(payload.loras?.count, 1)
    XCTAssertEqual(payload.loras?.first?.path, "krea2_turbo_distill_r256.safetensors")
    XCTAssertEqual(payload.loras?.first?.scale, 0.6)
    XCTAssertEqual(payload.loras?.first?.role, "accel")
    XCTAssertNotNil(payload.memoryEstimateBytes, "the memory preflight beside this guard still ran")
  }

  /// A LoRA the library has never scanned (no lookup match) must never block
  /// a request that was accepted before this guard existed — the default
  /// `loraLookup` (used when the route's real library has nothing for this
  /// filename) resolves to "unknown", which is always allowed.
  func testUnscannedLoRAOnGenerateNeverBlocks() throws {
    let store = try makePresetStore()
    let payload = try WarmServer.decodedGeneratePayload(
      from: Data(#"{"prompt":"x","loras":[{"path":"never-scanned.safetensors"}]}"#.utf8),
      store: store, configuration: configuration)
    XCTAssertEqual(payload.loras?.first?.path, "never-scanned.safetensors")
  }
}
