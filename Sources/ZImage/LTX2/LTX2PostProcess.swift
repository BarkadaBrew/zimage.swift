// LTX2PostProcess.swift -- Frame extraction and MP4 video encoding
// Phase 4 of the LTX-2 Swift/MLX port
//
// Converts decoded VAE output (MLXArray frames) to CGImage frames and
// encodes them to an H.264 MP4 file using AVFoundation.
//
// The VAE outputs (B, 3, F, H, W) in float32, range [0, 1].
// This module clamps, converts to uint8, creates CGImages, and writes MP4.

#if canImport(AVFoundation)
import AVFoundation
#endif

#if canImport(CoreGraphics)
import CoreGraphics
import CoreImage
#endif

import Foundation
import MLX

/// Post-processing utilities for LTX-2 video output.
public enum LTX2PostProcess {

  // MARK: - Frame Extraction

  /// Convert decoded VAE output to an array of pixel data buffers.
  ///
  /// Takes the VAE output `(B, 3, F, H, W)` float32 [0, 1] and converts
  /// each frame to RGBA pixel data (uint8).
  ///
  /// - Parameters:
  ///   - decoded: VAE decoded output `(B, 3, F, H, W)` in float32.
  ///   - batchIndex: Which batch element to extract. Default 0.
  /// - Returns: Array of `(width, height, pixelData)` tuples, one per frame.
  /// Temporal color anchor. i2v conditioning only pins frame 0's lighting; as
  /// temporal distance grows, frames drift toward the LoRA's darker/muddier
  /// training prior — a monotonic saturation/brightness decay confirmed
  /// universal across seeds and prompts (QA-CAMPAIGN-2026-07-26). This
  /// renormalizes each frame's per-channel mean/std back to frame 0, blended by
  /// `strength` (LTX2_COLOR_ANCHOR, default 0.0). Frame 0 is preserved exactly.
  /// Operates on the full decoded clip `(B, 3, F, H, W)`.
  public static func stabilizeColor(_ decoded: MLXArray, strength: Float) -> MLXArray {
    guard strength > 0, decoded.ndim == 5, decoded.dim(2) > 1 else { return decoded }
    let ref = decoded[0..., 0..., 0..<1]                                  // (B,3,1,H,W)
    let refMean = MLX.mean(ref, axes: [3, 4], keepDims: true)             // (B,3,1,1,1)
    let refVar = MLX.mean(MLX.square(ref - refMean), axes: [3, 4], keepDims: true)
    let refStd = MLX.sqrt(refVar) + 1e-3
    let mean = MLX.mean(decoded, axes: [3, 4], keepDims: true)            // (B,3,F,1,1)
    let variance = MLX.mean(MLX.square(decoded - mean), axes: [3, 4], keepDims: true)
    let std = MLX.sqrt(variance) + 1e-3
    let matched = (decoded - mean) / std * refStd + refMean
    let s = MLXArray(strength)
    return s * matched + (1.0 - s) * decoded
  }

  private static func colorAnchorStrength() -> Float {
    // Default OFF. This is a POST-HOC cosmetic mask (renormalize each frame to
    // frame 0) — it hides in-clip tone drift instead of fixing its root cause
    // (denoise-side appearance drift from the free-running frames / conditioning
    // decay). Kept only as a QA/export escape hatch. The real fix is a root-side
    // appearance anchor + a sharp (low-CRF) conditioning frame. See codex review
    // 2026-07-27 and qa/video/QA-CAMPAIGN-2026-07-26.md.
    Float(ProcessInfo.processInfo.environment["LTX2_COLOR_ANCHOR"] ?? "") ?? 0.0
  }

  public static func extractFrames(
    from decoded: MLXArray,
    batchIndex: Int = 0,
    colorAnchor: Float? = nil
  ) -> [(width: Int, height: Int, pixels: [UInt8])] {
    let decoded = stabilizeColor(decoded, strength: colorAnchor ?? colorAnchorStrength())
    // decoded shape: (B, 3, F, H, W)
    let numFrames = decoded.dim(2)
    let height = decoded.dim(3)
    let width = decoded.dim(4)

    var frames: [(width: Int, height: Int, pixels: [UInt8])] = []

    for f in 0..<numFrames {
      // Extract frame: (3, H, W)
      let frame = decoded[batchIndex, 0..., f]  // (3, H, W)

      // Clamp to [0, 1]
      let clamped = MLX.clip(frame, min: 0, max: 1)

      // Convert to uint8: (3, H, W) -> scale by 255
      let scaled = (clamped * 255.0).asType(.uint8)
      eval(scaled)

      // Use contiguous() before transposing to ensure correct memory layout
      let hwc = scaled.transposed(1, 2, 0).contiguous()  // (H, W, 3) contiguous
      eval(hwc)

      // Bulk copy pixel data as RGB
      let rgbData: [UInt8]
      let flatArray = hwc.reshaped(-1)
      eval(flatArray)
      rgbData = flatArray.asArray(UInt8.self)

      // Convert RGB to RGBA (add alpha = 255)
      var rgbaData = [UInt8](repeating: 255, count: height * width * 4)
      for i in 0..<(height * width) {
        rgbaData[i * 4 + 0] = rgbData[i * 3 + 0]  // R
        rgbaData[i * 4 + 1] = rgbData[i * 3 + 1]  // G
        rgbaData[i * 4 + 2] = rgbData[i * 3 + 2]  // B
        // Alpha already 255
      }

      frames.append((width: width, height: height, pixels: rgbaData))
    }

    return frames
  }

  #if canImport(CoreGraphics)
  /// Convert decoded VAE output to CGImage frames.
  ///
  /// - Parameters:
  ///   - decoded: VAE decoded output `(B, 3, F, H, W)` in float32.
  ///   - batchIndex: Which batch element to extract. Default 0.
  /// - Returns: Array of CGImages, one per frame.
  public static func framesToImages(
    from decoded: MLXArray,
    batchIndex: Int = 0,
    colorAnchor: Float? = nil
  ) -> [CGImage] {
    let rawFrames = extractFrames(from: decoded, batchIndex: batchIndex, colorAnchor: colorAnchor)
    var images: [CGImage] = []

    for frame in rawFrames {
      if let image = createCGImage(
        pixels: frame.pixels,
        width: frame.width,
        height: frame.height
      ) {
        images.append(image)
      }
    }

    return images
  }

  /// Create a CGImage from RGBA pixel data.
  private static func createCGImage(
    pixels: [UInt8],
    width: Int,
    height: Int
  ) -> CGImage? {
    let bitsPerComponent = 8
    let bitsPerPixel = 32
    let bytesPerRow = width * 4
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

    guard let provider = CGDataProvider(
      data: Data(pixels) as CFData
    ) else { return nil }

    return CGImage(
      width: width,
      height: height,
      bitsPerComponent: bitsPerComponent,
      bitsPerPixel: bitsPerPixel,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: provider,
      decode: nil,
      shouldInterpolate: false,
      intent: .defaultIntent
    )
  }
  #endif

  #if canImport(AVFoundation) && canImport(CoreGraphics)
  /// Write video frames to an MP4 file using AVFoundation.
  ///
  /// Uses H.264 encoding with AVAssetWriter for broad compatibility.
  ///
  /// - Parameters:
  ///   - frames: Array of CGImages to encode.
  ///   - outputPath: Path for the output MP4 file.
  ///   - fps: Frames per second. Default 24.
  ///   - width: Video width in pixels.
  ///   - height: Video height in pixels.
  /// - Throws: If video writing fails.
  /// Chunk interleaved PCM into 0.5 s CMSampleBuffers (built eagerly — the
  /// whole clip is small; appending happens in the writer's ready callback).
  private static func makeAudioSampleBuffers(_ audio: AudioTrack) throws -> [CMSampleBuffer] {
    let channels = audio.samples.dim(0)
    let n = audio.samples.dim(1)
    // Interleave [C, N] -> frame-major [n0c0, n0c1, n1c0, ...] Float32.
    let interleaved = audio.samples.asType(.float32).transposed(1, 0).asArray(Float.self)

    var asbd = AudioStreamBasicDescription(
      mSampleRate: Float64(audio.sampleRate),
      mFormatID: kAudioFormatLinearPCM,
      mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
      mBytesPerPacket: UInt32(4 * channels),
      mFramesPerPacket: 1,
      mBytesPerFrame: UInt32(4 * channels),
      mChannelsPerFrame: UInt32(channels),
      mBitsPerChannel: 32,
      mReserved: 0)
    var format: CMAudioFormatDescription?
    CMAudioFormatDescriptionCreate(
      allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil,
      magicCookieSize: 0, magicCookie: nil, extensions: nil,
      formatDescriptionOut: &format)
    guard let format else { throw LTX2PostProcessError.audioFormatCreationFailed }

    let chunkFrames = audio.sampleRate / 2  // 0.5 s per buffer
    var buffers: [CMSampleBuffer] = []
    var offset = 0
    while offset < n {
      let count = min(chunkFrames, n - offset)
      let bytes = interleaved.withUnsafeBufferPointer { buf -> Data in
        Data(bytes: buf.baseAddress! + offset * channels, count: count * channels * 4)
      }
      var blockBuffer: CMBlockBuffer?
      CMBlockBufferCreateWithMemoryBlock(
        allocator: nil, memoryBlock: nil, blockLength: bytes.count,
        blockAllocator: nil, customBlockSource: nil, offsetToData: 0,
        dataLength: bytes.count, flags: 0, blockBufferOut: &blockBuffer)
      guard let blockBuffer else { throw LTX2PostProcessError.audioBufferCreationFailed }
      bytes.withUnsafeBytes { raw in
        _ = CMBlockBufferReplaceDataBytes(
          with: raw.baseAddress!, blockBuffer: blockBuffer,
          offsetIntoDestination: 0, dataLength: bytes.count)
      }
      var sampleBuffer: CMSampleBuffer?
      CMAudioSampleBufferCreateWithPacketDescriptions(
        allocator: nil, dataBuffer: blockBuffer, dataReady: true,
        makeDataReadyCallback: nil, refcon: nil, formatDescription: format,
        sampleCount: count,
        presentationTimeStamp: CMTime(value: Int64(offset), timescale: Int32(audio.sampleRate)),
        packetDescriptions: nil, sampleBufferOut: &sampleBuffer)
      guard let sampleBuffer else { throw LTX2PostProcessError.audioBufferCreationFailed }
      buffers.append(sampleBuffer)
      offset += count
    }
    return buffers
  }

  /// PCM audio for muxing: `samples` is `[channels, N]` float in [-1, 1]
  /// (stereo = [2, N]) at `sampleRate` Hz. Encoded as AAC.
  public struct AudioTrack {
    public let samples: MLXArray
    public let sampleRate: Int
    public init(samples: MLXArray, sampleRate: Int) {
      self.samples = samples
      self.sampleRate = sampleRate
    }
  }

  /// Delivery dims for a target short edge (0 = off). Aspect preserved, both
  /// axes rounded to EVEN (h264 requirement); never upscales. The render keeps
  /// the full two-stage recipe — only the encoded output shrinks, so a 480p
  /// delivery is supersampled from the 2x refine rather than rendered soft
  /// (Todd 2026-08-07: "480p is enough for telegram", "2x scale is overkill
  /// for mobile").
  public static func deliveryDims(width: Int, height: Int, shortEdge: Int) -> (width: Int, height: Int) {
    guard shortEdge > 0 else { return (width, height) }
    let short = min(width, height)
    guard short > shortEdge else { return (width, height) }
    let scale = Double(shortEdge) / Double(short)
    func even(_ v: Double) -> Int { max(2, Int((v / 2.0).rounded()) * 2) }
    return width <= height
      ? (even(Double(width) * scale), even(Double(height) * scale))
      : (even(Double(width) * scale), even(Double(height) * scale))
  }

  public static func writeMP4(
    frames: [CGImage],
    outputPath: String,
    fps: Int = 24,
    width: Int,
    height: Int,
    bitsPerPixelOverride: Double? = nil,
    audio: AudioTrack? = nil,
    deliveryShortEdge: Int = 0,
    /// comfybox#401: the generation record's JSON, embedded as a metadata
    /// atom alongside the mandatory `.json` sidecar (ruling 2 — "atom
    /// optional"). `nil` (the default) touches NOTHING below — no atom is
    /// requested, so this parameter changes zero bytes of the written file
    /// relative to a caller that never passed it (see
    /// `LTX2PostProcessMetadataTests` for the byte-identity regression this
    /// guards).
    ///
    /// Carried in the container's standard "common" description field
    /// (`AVMetadataKeySpace.common` / `commonKeyDescription`) rather than a
    /// custom reverse-DNS key: AVAssetWriter's QuickTime-style keyed metadata
    /// (`.quickTimeMetadata`, `mdta` identifiers) and QuickTime userdata
    /// comment atoms are both silently DROPPED for `fileType: .mp4` — verified
    /// empirically (they write fine for `.mov`, add zero bytes for `.mp4`).
    /// `.common`/`commonKeyDescription` is the one keyspace AVAssetWriter
    /// actually persists into an ISO-brand `.mp4`, and it's a plain header
    /// field write — no video re-encode, same as the PNG side embedding its
    /// full JSON record in EXIF `UserComment` rather than a bespoke tag.
    generationRecordJSON: String? = nil
  ) throws {
    guard !frames.isEmpty else {
      throw LTX2PostProcessError.noFrames
    }

    let outputURL = URL(fileURLWithPath: outputPath)

    // Remove existing file
    try? FileManager.default.removeItem(at: outputURL)

    // Create asset writer
    let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
    if let generationRecordJSON {
      let item = AVMutableMetadataItem()
      item.keySpace = .common
      item.key = AVMetadataKey.commonKeyDescription as NSString
      item.value = generationRecordJSON as NSString
      writer.metadata = [item]
    }

    // Video settings. Bitrate: 0.5 bits/pixel (~12 Mbps @ 768x1280x24) is
    // visually equivalent to the old 4 bits/px for generated content but ~8x
    // smaller — the old setting produced 139MB 12s files that exceeded
    // Telegram's 50MB bot upload cap. Env-tunable via LTX2_VIDEO_BITS_PER_PX.
    let bitsPerPixel = bitsPerPixelOverride
      ?? Double(ProcessInfo.processInfo.environment["LTX2_VIDEO_BITS_PER_PX"] ?? "") ?? 0.5
    // Delivery downscale: the AVAssetWriterInput scales appended buffers to
    // the output dims, so shrinking here supersamples the encoded file from
    // the full-res render — no extra pass, smaller files, crisper 480p.
    let (outW, outH) = deliveryDims(width: width, height: height, shortEdge: deliveryShortEdge)
    let videoSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: outW,
      AVVideoHeightKey: outH,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: Int(Double(outW * outH * fps) * bitsPerPixel),
        AVVideoMaxKeyFrameIntervalKey: fps,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      ] as [String: Any],
    ]

    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: videoSettings
    )
    input.expectsMediaDataInRealTime = false

    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
      ]
    )

    writer.add(input)

    // Optional AAC audio input (task #21 wire 3). PCM is chunked into
    // sample buffers and appended AFTER video (file-based writer with
    // expectsMediaDataInRealTime=false tolerates sequential appends; the
    // muxer interleaves on finish).
    var audioInput: AVAssetWriterInput?
    if let audio {
      let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatMPEG4AAC,
        AVSampleRateKey: audio.sampleRate,
        AVNumberOfChannelsKey: audio.samples.dim(0),
        // AAC bitrate must scale with sample rate: 192k is valid for 48kHz
        // stereo but exceeds the encoder ceiling at the official vocoder's
        // 24kHz, which stalls the writer (comfybox#334). ~4 bits/sample keeps
        // 192k at 48k and drops to 96k at 24k.
        AVEncoderBitRateKey: min(192_000, audio.sampleRate * 4),
      ]
      let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: settings)
      ai.expectsMediaDataInRealTime = false
      writer.add(ai)
      audioInput = ai
    }

    // Pre-build audio sample buffers BEFORE starting the writer (cheap; the
    // whole clip fits in memory) so the ready callbacks only append.
    let audioBuffers: [CMSampleBuffer] = try audio.map(makeAudioSampleBuffers) ?? []

    guard writer.startWriting() else {
      throw LTX2PostProcessError.writingFailed(
        writer.error?.localizedDescription ?? "startWriting failed")
    }
    writer.startSession(atSourceTime: .zero)

    let frameDuration = CMTimeMake(value: 1, timescale: Int32(fps))

    // With two inputs on one writer, appends MUST be demand-driven: the
    // writer's interleaving window makes an input's isReadyForMoreMediaData
    // go false until the OTHER track catches up, so appending all video
    // before any audio deadlocks both spin-wait loops. Drive each input
    // with requestMediaDataWhenReady on its own queue (the canonical
    // multi-track pattern) and join on a group.
    let group = DispatchGroup()
    let errorLock = NSLock()
    var appendError: LTX2PostProcessError?

    group.enter()
    var frameIndex = 0
    var videoDone = false
    let videoQueue = DispatchQueue(label: "comfybox.mux.video")
    input.requestMediaDataWhenReady(on: videoQueue) {
      if videoDone { return }  // callback queued before markAsFinished landed
      func finish(_ error: LTX2PostProcessError?) {
        if let error {
          errorLock.lock(); appendError = appendError ?? error; errorLock.unlock()
        }
        videoDone = true
        input.markAsFinished()
        group.leave()
      }
      // A failed writer never flips isReadyForMoreMediaData back on — without
      // this check both callbacks stall and group.wait() hangs the render
      // queue forever (Codex 2026-08-04 #5).
      if writer.status != .writing {
        return finish(.writingFailed(writer.error?.localizedDescription ?? "writer left .writing during video append"))
      }
      while input.isReadyForMoreMediaData {
        if frameIndex >= frames.count { return finish(nil) }
        let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(frameIndex))
        guard let pixelBuffer = createPixelBuffer(
          from: frames[frameIndex], width: width, height: height) else {
          return finish(.pixelBufferCreationFailed(frameIndex: frameIndex))
        }
        guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
          return finish(.writingFailed(writer.error?.localizedDescription ?? "video append rejected at frame \(frameIndex)"))
        }
        frameIndex += 1
      }
    }

    if let audioInput {
      group.enter()
      var bufferIndex = 0
      var audioDone = false
      let audioQueue = DispatchQueue(label: "comfybox.mux.audio")
      audioInput.requestMediaDataWhenReady(on: audioQueue) {
        if audioDone { return }  // callback queued before markAsFinished landed
        func finish(_ error: LTX2PostProcessError?) {
          if let error {
            errorLock.lock(); appendError = appendError ?? error; errorLock.unlock()
          }
          audioDone = true
          audioInput.markAsFinished()
          group.leave()
        }
        if writer.status != .writing {
          return finish(.writingFailed(writer.error?.localizedDescription ?? "writer left .writing during audio append"))
        }
        while audioInput.isReadyForMoreMediaData {
          if bufferIndex >= audioBuffers.count { return finish(nil) }
          guard audioInput.append(audioBuffers[bufferIndex]) else {
            return finish(.writingFailed(writer.error?.localizedDescription ?? "audio append rejected at buffer \(bufferIndex)"))
          }
          bufferIndex += 1
        }
      }
    }

    // Bounded: a wedged writer surfaces as an error, never a hung render
    // queue. Generous ceiling — muxing a finished render is seconds of work.
    if group.wait(timeout: .now() + 600) == .timedOut {
      writer.cancelWriting()
      throw LTX2PostProcessError.writingFailed("mux timed out after 600s (writer status \(writer.status.rawValue))")
    }

    // Wait for writing to complete
    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting {
      semaphore.signal()
    }
    semaphore.wait()

    if let appendError { throw appendError }
    if writer.status == .failed {
      throw LTX2PostProcessError.writingFailed(writer.error?.localizedDescription ?? "unknown")
    }
  }

  /// Create a CVPixelBuffer from a CGImage.
  private static func createPixelBuffer(
    from image: CGImage,
    width: Int,
    height: Int
  ) -> CVPixelBuffer? {
    var pixelBuffer: CVPixelBuffer?
    let attrs: [String: Any] = [
      kCVPixelBufferCGImageCompatibilityKey as String: true,
      kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
    ]

    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width, height,
      kCVPixelFormatType_32ARGB,
      attrs as CFDictionary,
      &pixelBuffer
    )

    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      return nil
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    guard let context = CGContext(
      data: CVPixelBufferGetBaseAddress(buffer),
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
    ) else {
      return nil
    }

    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return buffer
  }
  #endif

  /// Write raw frame data to a sequence of PPM files (platform-independent fallback).
  ///
  /// - Parameters:
  ///   - decoded: VAE decoded output `(B, 3, F, H, W)`.
  ///   - outputDir: Directory to write PPM files.
  ///   - prefix: Filename prefix. Default "frame".
  ///   - batchIndex: Batch element index. Default 0.
  /// - Throws: If directory creation or file writing fails.
  public static func writeFramesPPM(
    from decoded: MLXArray,
    outputDir: String,
    prefix: String = "frame",
    batchIndex: Int = 0
  ) throws {
    let frames = extractFrames(from: decoded, batchIndex: batchIndex)

    let fm = FileManager.default
    try fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

    for (index, frame) in frames.enumerated() {
      let filename = String(format: "%@_%04d.ppm", prefix, index)
      let path = (outputDir as NSString).appendingPathComponent(filename)

      // PPM header
      var data = Data("P6\n\(frame.width) \(frame.height)\n255\n".utf8)

      // RGBA -> RGB
      for i in 0..<(frame.width * frame.height) {
        data.append(frame.pixels[i * 4])      // R
        data.append(frame.pixels[i * 4 + 1])  // G
        data.append(frame.pixels[i * 4 + 2])  // B
      }

      try data.write(to: URL(fileURLWithPath: path))
    }
  }

  /// Round-trip a single frame through a REAL H.264 encode/decode so it carries
  /// video-codec artifacts (deblocked DCT, chroma subsampling). Ports ComfyUI's
  /// LTXVPreprocess (PyAV libx264 CRF round-trip): LTX is trained on video
  /// frames, and a pristine still conditioning frame freezes i2v motion.
  /// `compression` follows the ComfyUI 0-100 scale (35 default ≈ CRF 35).
  public static func h264RoundTrip(_ image: CGImage, compression: Int) throws -> CGImage {
    // Per-frame bit budget mapped as CRF-equivalent: ComfyUI's LTXVPreprocess
    // passes img_compression DIRECTLY to libx264 as CRF, and x264 halves
    // bitrate per +6 CRF, so bpp ≈ a·2^(−crf/6). Calibrated at the proven
    // working point (35 → ~0.30 bpp): a = 0.3·2^(35/6) ≈ 17.1. The old
    // exponential (1.05·e^(−c/28)) matched at 33-35 but was 14x too LOSSY at
    // low values — GT-config's img_compression=2 means CRF 2 (visually
    // lossless, ~13 bpp), ours gave ~1 bpp, softening the conditioning frame
    // and with it the ENTIRE render (stage-1 sharp 14.5 vs ComfyUI-content
    // 41.4, 2026-07-25). Cap keeps single-frame encodes sane.
    let bpp = min(16.0, max(0.06, 17.1 * pow(2.0, -Double(compression) / 6.0)))
    let tmp = NSTemporaryDirectory() + "ltx2-cond-\(UUID().uuidString).mp4"
    defer { try? FileManager.default.removeItem(atPath: tmp) }
    try writeMP4(
      frames: [image], outputPath: tmp, fps: 24,
      width: image.width, height: image.height,
      bitsPerPixelOverride: bpp)
    let asset = AVURLAsset(url: URL(fileURLWithPath: tmp))
    let gen = AVAssetImageGenerator(asset: asset)
    gen.requestedTimeToleranceBefore = .zero
    gen.requestedTimeToleranceAfter = .positiveInfinity
    gen.appliesPreferredTrackTransform = false
    return try gen.copyCGImage(at: .zero, actualTime: nil)
  }
}

// MARK: - Errors

/// Errors from post-processing operations.
public enum LTX2PostProcessError: Error, CustomStringConvertible {
  case noFrames
  case pixelBufferCreationFailed(frameIndex: Int)
  case writingFailed(String)
  case audioFormatCreationFailed
  case audioBufferCreationFailed

  public var description: String {
    switch self {
    case .audioFormatCreationFailed: return "audio format description creation failed"
    case .audioBufferCreationFailed: return "audio sample buffer creation failed"
    case .noFrames:
      return "No frames to write"
    case .pixelBufferCreationFailed(let idx):
      return "Failed to create pixel buffer for frame \(idx)"
    case .writingFailed(let msg):
      return "Video writing failed: \(msg)"
    }
  }
}
