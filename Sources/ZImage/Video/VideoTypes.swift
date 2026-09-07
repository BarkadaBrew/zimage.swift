// VideoTypes.swift — Video generation types for ComfyBox MCP video tools.
//
// Defines the request/response types for video generation endpoints.
// Supports both Replicate proxy (Phase A) and future native MLX (Phase B/C).

import Foundation

// MARK: - VideoMode

/// Video generation mode.
public enum VideoMode: String, Codable, Sendable {
  /// Text-to-video: generate video from a text prompt.
  case t2v
  /// Image-to-video: animate a source image based on a motion prompt.
  case i2v
  /// Storyboard: an ordered shot list executed as chained i2v renders +
  /// assembly (comfybox#237).
  case storyboard
}

// MARK: - VideoJobState

/// State of a video generation job.
public enum VideoJobState: String, Codable, Sendable {
  case queued
  case processing
  case succeeded
  case failed
  /// #1479: checkpointed mid-render to let a `preempt: true` image job run;
  /// the render resumes automatically once that job finishes (success or
  /// failure) — NOT a terminal state, and polling clients that don't
  /// recognize it should keep polling rather than treat it as done.
  case pausedForPreemption
}

// MARK: - VideoGenerateRequest

/// Request payload for the `POST /v1/video/generate` endpoint.
public struct VideoGenerateRequest: Codable, Sendable {
  /// Text prompt describing the desired video content.
  public let prompt: String

  /// Absolute path to source image for I2V mode. Nil for T2V.
  /// var: may be filled in from imageBase64 (bytes upload) by the server.
  public var imagePath: String?

  /// I2V init image as base64 (image_base64) — for remote clients that can't
  /// place a file on the server's filesystem.
  public let imageBase64: String?

  /// Video duration in seconds. T2V only: 6, 8, 10, 12, 14, 16, 18, or 20 (default: 6). Ignored for I2V.
  public let duration: Int?

  /// Output resolution: "480p", "720p", or "1080p".
  public let resolution: String?

  /// Aspect ratio: "16:9" or "9:16" (default: "16:9").
  public let aspectRatio: String?

  /// Random seed for reproducibility.
  public let seed: Int?

  /// Output file path for the .mp4. Must be within the allowed output directory.
  public let outputPath: String?

  /// Explicit backend routing: "local"/"ltx" (on-device LTX-2) or
  /// "replicate"/"cloud" (paid Replicate). Nil = unspecified.
  public let backend: String?

  /// Requested model (e.g. "ltx"). Used only to infer local vs cloud intent
  /// when `backend` is absent; the actual model per backend is fixed.
  public let model: String?

  /// Derived mode based on whether image_path is present.
  public var mode: VideoMode {
    imagePath != nil ? .i2v : .t2v
  }

  public init(
    prompt: String,
    imagePath: String? = nil,
    duration: Int? = nil,
    resolution: String? = nil,
    aspectRatio: String? = nil,
    seed: Int? = nil,
    outputPath: String? = nil,
    backend: String? = nil,
    model: String? = nil,
    imageBase64: String? = nil
  ) {
    self.prompt = prompt
    self.imagePath = imagePath
    self.imageBase64 = imageBase64
    self.duration = duration
    self.resolution = resolution
    self.aspectRatio = aspectRatio
    self.seed = seed
    self.outputPath = outputPath
    self.backend = backend
    self.model = model
  }

  /// Classify the caller's backend intent from `backend`/`model`.
  public enum BackendIntent { case local, cloud, unspecified }
  public var backendIntent: BackendIntent {
    let b = (backend ?? "").lowercased()
    let m = (model ?? "").lowercased()
    if b == "replicate" || b == "cloud" { return .cloud }
    if b == "local" || b == "ltx" || b == "ltx2" || m.contains("ltx") { return .local }
    return .unspecified
  }

  // MARK: - Validation

  /// Valid T2V durations in seconds.
  public static let validT2VDurations = [6, 8, 10, 12, 14, 16, 18, 20]

  /// Valid resolution values.
  public static let validResolutions = ["480p", "720p", "1080p"]

  /// Valid aspect ratio values.
  public static let validAspectRatios = ["16:9", "9:16"]

  /// Validate duration for the given mode. Returns error string or nil.
  public static func validateDuration(_ duration: Int, mode: VideoMode) -> String? {
    // I2V duration is fixed (~5s), ignore the parameter
    guard mode == .t2v else { return nil }
    guard validT2VDurations.contains(duration) else {
      return "Invalid duration \(duration). T2V supports: \(validT2VDurations.map(String.init).joined(separator: ", "))"
    }
    return nil
  }

  /// Validate resolution string. Returns error string or nil.
  public static func validateResolution(_ resolution: String) -> String? {
    guard validResolutions.contains(resolution) else {
      return "Invalid resolution '\(resolution)'. Supported: \(validResolutions.joined(separator: ", "))"
    }
    return nil
  }

  /// Validate aspect ratio string. Returns error string or nil.
  public static func validateAspectRatio(_ aspectRatio: String) -> String? {
    guard validAspectRatios.contains(aspectRatio) else {
      return "Invalid aspect_ratio '\(aspectRatio)'. Supported: \(validAspectRatios.joined(separator: ", "))"
    }
    return nil
  }

  /// Validate the full request. Returns an error string or nil if valid.
  public func validate() -> String? {
    if prompt.trimmingCharacters(in: .whitespaces).isEmpty {
      return "'prompt' is required and cannot be empty"
    }
    if let duration = duration {
      if let error = Self.validateDuration(duration, mode: mode) {
        return error
      }
    }
    if let resolution = resolution {
      if let error = Self.validateResolution(resolution) {
        return error
      }
    }
    if let aspectRatio = aspectRatio {
      if let error = Self.validateAspectRatio(aspectRatio) {
        return error
      }
    }
    return nil
  }
}

// MARK: - VideoJobStatus

/// Status of a video generation job. Returned by both `generate_video` and `video_status`.
public struct VideoJobStatus: Codable, Sendable {
  /// Unique job identifier.
  public let jobId: String

  /// Current job state.
  public let status: VideoJobState

  /// Video mode (t2v or i2v).
  public let mode: VideoMode?

  /// Backend that produced the video.
  public let backend: String

  /// Model identifier used for generation.
  public let model: String?

  /// Output file path (non-nil on success).
  public let outputPath: String?

  /// Total wall-clock time in milliseconds (set on completion).
  public let durationMs: Int?

  /// Output file size in bytes (set on success).
  public let fileSizeBytes: Int?

  /// Duration of the generated video in seconds (set on success).
  public let videoDurationSeconds: Int?

  /// Error message (set on failure).
  public let error: String?

  /// Estimated time remaining in seconds (set when queued/processing).
  public let estimatedSeconds: Int?

  /// Elapsed time in milliseconds since job submission.
  public let elapsedMs: Int?

  /// Replicate prediction ID (for proxy mode debugging).
  public let replicatePredictionId: String?

  /// Live render progress (0-100) while queued/processing. Populated for the
  /// local LTX-2 backend (which streams a per-chunk/per-step callback); nil for
  /// the Replicate cloud path, which doesn't expose fine-grained progress.
  public let progressPercent: Int?
  /// Snapshot of the authoritative per-render config taken at SUBMIT time
  /// (task #9 / Codex finding #15) — what THIS render resolved, with
  /// provenance. Durable in the trace; jobs themselves prune after 1h.
  public let resolvedConfig: [LTX2ResolvedParam]?

  /// Number of frames written (set on success). Populated by the local LTX-2
  /// backend; nil for the cloud path (which reports duration, not frame count).
  public let frameCount: Int?

  /// comfybox#322: `true` when this render stopped because an operator called
  /// `/v1/queue/interrupt`, rather than because anything went wrong.
  ///
  /// ADDITIVE, and deliberately a separate field rather than a new
  /// `VideoJobState` case: `status` is the terminal-state signal every polling
  /// client already switches on (Desktop's `isTerminal` checks
  /// succeeded|failed; the Bree daemon decodes the same string), and a state
  /// name none of them know would read as "not finished yet" and poll forever.
  /// So an interrupted render still reports `status: failed` — terminal — with
  /// this flag and a plain-English `error` saying why. The render TRACE, which
  /// has no such compatibility constraint, records `status: interrupted`.
  /// Absent (nil, omitted from JSON) on every other outcome.
  public let interrupted: Bool?
  /// comfybox#307: non-nil only when `two_stage` was requested for this
  /// render and the refine pass could not run (upsampler unavailable, or the
  /// volume gate) — see `LTX2RefineGate`. nil on the cloud path.
  public let refineSkipped: String?
  /// comfybox#401: the same record written to the `.json` sidecar next to
  /// the output file (and, when encodable, the mp4's own metadata atom) —
  /// so a poller of `GET /v1/video/status/{id}` gets the full generation
  /// record without a second read of the filesystem. Set on success for the
  /// local LTX-2 backend; nil while queued/processing, on failure, and on
  /// the Replicate cloud path (no local render to record).
  public let generationRecord: VideoGenerationRecord?

  public init(
    jobId: String,
    status: VideoJobState,
    mode: VideoMode? = nil,
    backend: String = "replicate",
    model: String? = nil,
    outputPath: String? = nil,
    durationMs: Int? = nil,
    fileSizeBytes: Int? = nil,
    videoDurationSeconds: Int? = nil,
    error: String? = nil,
    estimatedSeconds: Int? = nil,
    elapsedMs: Int? = nil,
    replicatePredictionId: String? = nil,
    progressPercent: Int? = nil,
    resolvedConfig: [LTX2ResolvedParam]? = nil,
    frameCount: Int? = nil,
    interrupted: Bool? = nil,
    refineSkipped: String? = nil,
    generationRecord: VideoGenerationRecord? = nil
  ) {
    self.jobId = jobId
    self.status = status
    self.mode = mode
    self.backend = backend
    self.model = model
    self.outputPath = outputPath
    self.durationMs = durationMs
    self.fileSizeBytes = fileSizeBytes
    self.videoDurationSeconds = videoDurationSeconds
    self.error = error
    self.estimatedSeconds = estimatedSeconds
    self.elapsedMs = elapsedMs
    self.replicatePredictionId = replicatePredictionId
    self.progressPercent = progressPercent
    self.resolvedConfig = resolvedConfig
    self.frameCount = frameCount
    self.interrupted = interrupted
    self.refineSkipped = refineSkipped
    self.generationRecord = generationRecord
  }
}
