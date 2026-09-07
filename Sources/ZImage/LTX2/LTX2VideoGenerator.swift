// LTX2VideoGenerator.swift — Server-callable local LTX-2 video generation
//
// Lifts the proven `ComfyBox ltx2-i2v` CLI flow into a reusable service so the
// warm server can generate video locally (text-to-video and image-to-video)
// instead of only via the Replicate cloud proxy. Models are loaded lazily and
// cached; frame/chunk math and request validation are pure so they're testable
// without the 38 GB model.

import Foundation
import Logging
import MLX
import MLXNN

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

/// One LoRA to merge into the LTX-2 transformer for a render, applied in
/// the order given. Mirrors the image side's {path, scale} shape so LTX
/// video LoRAs are managed the same way as every other model's.
public struct LTX2LoRAReference: Sendable, Equatable {
    public var path: String
    public var scale: Float
    public init(path: String, scale: Float = 1.0) {
        self.path = path
        self.scale = scale
    }
}

/// Parameters for one local LTX-2 video generation.
public struct LTX2VideoRequest: Sendable {
    public var prompt: String
    public var negativePrompt: String?
    /// nil → text-to-video; a path → image-to-video conditioned on it.
    public var initImagePath: String?
    public var width: Int
    public var height: Int
    /// Frames per chunk; must be 1 + 8k (9, 17, 25, …, 97).
    public var framesPerChunk: Int
    public var steps: Int
    public var seed: UInt64
    /// I2V conditioning strength (0–1).
    public var strength: Float
    /// Conditioning compression (libx264 CRF) override; nil = env/default.
    /// Higher = more motion (frozen-still regime at 0-2), lower = more fidelity.
    public var imgCompression: Int?
    /// CFG guidance override; nil = env/config default. >1 amplifies the
    /// action-prompt direction (primary motion lever, Codex 2026-07-26).
    public var guidance: Float?
    /// Identity re-anchor strength for CONTINUATION chunks (0 = off). Each
    /// continuation chunk conditions on the previous chunk\u{27}s last frame at
    /// frame 0 (hard continuity) AND the ORIGINAL source image at the chunk\u{27}s
    /// last frame at this strength (soft identity pull) \u{2014} counters the
    /// cumulative subject/scene drift of tail-to-head chaining (#219).
    public var identityAnchorStrength: Float

    /// Frames between mid-pass identity re-anchors for a LONG single/first pass
    /// (0 = off). With only a frame-0 anchor, peripheral subjects (e.g. a
    /// partner's face) drift and melt over a long pass; re-splicing the source
    /// at this interval at `identityAnchorStrength` holds EVERY face. #partnered
    public var identityReAnchorInterval: Int
    /// Target duration; >0 generates continuation chunks (I2V only).
    public var extendToSeconds: Float
    public var fps: Int
    /// Deprecated single-LoRA fields — kept for wire/call-site back-compat.
    /// New callers should use `loras` instead. Folded into `effectiveLoRAs`.
    public var loraPath: String?
    public var loraStrength: Float
    /// LoRAs merged into the transformer for this render, applied in order.
    public var loras: [LTX2LoRAReference]
    public var outputPath: String
    /// Tier A tuning overrides (task #9 Phase 2) — nil fields defer to
    /// preset > configFile > env > builtin. `presetTuning` is the preset's
    /// block, resolved by the server and carried so the generator can build
    /// the full five-level resolution.
    public var tuning: LTX2VideoTuning?
    public var presetTuning: LTX2VideoTuning?
    /// Generate synchronized audio (task #21). T2V single-chunk only in v1;
    /// loads the audio branch (+~11GiB) into the transformer on first use.
    public var audio: Bool
    /// Temporal beat scheduling (comfybox#310): structured multi-beat
    /// content, carried as its own top-level field rather than a `tuning`
    /// key — tuning entries are scalar render knobs, this is structured
    /// content the engine locates inside the composed prompt. Each beat's
    /// `text` must be a verbatim substring of `prompt` (server-side
    /// contract, Phase 2); beats that can't be located are dropped
    /// fail-open at render time, never fail the render. nil/empty is
    /// byte-identical to today's flat (joined) behavior.
    public var beatSchedule: [BeatSegment]?

    /// `loras`, with the deprecated single `loraPath`/`loraStrength` (if set)
    /// prepended — the single field always applied first, matching the old
    /// single-LoRA behavior when only it is set.
    public var effectiveLoRAs: [LTX2LoRAReference] {
        var result: [LTX2LoRAReference] = []
        if let loraPath, !loraPath.isEmpty {
            result.append(LTX2LoRAReference(path: loraPath, scale: loraStrength))
        }
        result.append(contentsOf: loras)
        return result
    }

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        initImagePath: String? = nil,
        width: Int = 704,
        height: Int = 448,
        framesPerChunk: Int = 97,
        steps: Int = 8,
        seed: UInt64 = 42,
        strength: Float = 1.0,
        imgCompression: Int? = nil,
        guidance: Float? = nil,
        identityAnchorStrength: Float = 0,
        identityReAnchorInterval: Int = 0,
        extendToSeconds: Float = 0,
        fps: Int = 24,
        loraPath: String? = nil,
        loraStrength: Float = 1.0,
        loras: [LTX2LoRAReference] = [],
        outputPath: String,
        tuning: LTX2VideoTuning? = nil,
        presetTuning: LTX2VideoTuning? = nil,
        audio: Bool = false,
        beatSchedule: [BeatSegment]? = nil
    ) {
        self.audio = audio
        self.beatSchedule = beatSchedule
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImagePath = initImagePath
        self.width = width
        self.height = height
        self.framesPerChunk = framesPerChunk
        self.steps = steps
        self.seed = seed
        self.strength = strength
        self.imgCompression = imgCompression
        self.guidance = guidance
        self.identityAnchorStrength = identityAnchorStrength
        self.identityReAnchorInterval = identityReAnchorInterval
        self.extendToSeconds = extendToSeconds
        self.fps = fps
        self.loraPath = loraPath
        self.loraStrength = loraStrength
        self.loras = loras
        self.outputPath = outputPath
        self.tuning = tuning
        self.presetTuning = presetTuning
    }
}

public struct LTX2VideoResult: Sendable {
    public let outputPath: String
    public let frameCount: Int
    public let durationSeconds: Float
    public let elapsedSeconds: Double
    /// comfybox#307: why the two-stage refine did not run, when `two_stage`
    /// was requested for this render — nil when it ran, wasn't requested, or
    /// (montage/storyboard assembly) doesn't apply. See `LTX2RefineGate`.
    public let refineSkippedReason: String?
    /// comfybox#401: the record written to the sidecar (and, when encodable,
    /// the mp4 atom) for this render. `nil` for the montage/storyboard
    /// assembly result (`WarmServer.runStoryboard`), which composes
    /// already-recorded per-shot clips via a different writer and gets its
    /// own aggregate sidecar there — not this struct.
    public let generationRecord: VideoGenerationRecord?

    public init(
        outputPath: String, frameCount: Int, durationSeconds: Float, elapsedSeconds: Double,
        refineSkippedReason: String? = nil, generationRecord: VideoGenerationRecord? = nil
    ) {
        self.outputPath = outputPath
        self.frameCount = frameCount
        self.durationSeconds = durationSeconds
        self.elapsedSeconds = elapsedSeconds
        self.refineSkippedReason = refineSkippedReason
        self.generationRecord = generationRecord
    }
}

/// #1479: what one generator-level render produced. The completed payload is
/// `LTX2VideoResult` — the generator's own output (written MP4 + frame/duration
/// accounting), NOT the pipeline's per-chunk `LTX2PipelineOutput`, which never
/// reaches a caller of `generate`.
public enum LTX2RenderOutcome {
    case completed(LTX2VideoResult)
    case yielded(LTX2ResumeState)
}

#if canImport(CoreGraphics) && canImport(ImageIO)
/// #1479: the generator-level continuation a checkpoint needs — the request,
/// where the chunk loop got to, the frames already rendered, and the chained
/// seed frame that only exists as a function of the previous chunk's output.
///
/// This lives in a box travelling WITH the checkpoint rather than in a
/// `pendingResumeContext` on the generator, because `VideoGeneratorHolder
/// .release()` does `generator?.unload(); generator = nil` — the eviction
/// preemption performs deallocates the generator instance, taking any
/// instance-stored context with it. A checkpoint that outlives its generator
/// is the whole point (spec: evict weights, keep latents).
///
/// Everything else the resume needs is either in `LTX2ResumeState` or is
/// deterministically recomputed: text embeddings (the encoder has no RNG), the
/// i2v conditioning state (`MLXRandom.seed` precedes its noise draw), the
/// source/anchor/face-mask images (pure functions of the request), and
/// positions/PE (pure functions of the dims — spec).
public final class LTX2RenderContext: LTX2ResumeContext {
    /// The original request. `resume(from:)` needs no request parameter
    /// because of this.
    public let request: LTX2VideoRequest
    /// Chunk the checkpoint belongs to.
    var chunkIndex: Int = 0
    /// Frames from chunks that already finished.
    var frames: [CGImage] = []
    /// Audio latents from the chunk that produced them.
    var audioLatents: MLXArray?
    /// The chained conditioning frame for a CONTINUATION chunk (nil for chunk
    /// 0, whose seed image is re-derived from the request). Already
    /// compression-preprocessed and normalized, exactly as the chunk loop
    /// handed it on.
    var chunkSeedImage: MLXArray?
    /// GPU time already spent on this render, across all segments — wall clock
    /// would otherwise bill the preemptor's runtime to this render.
    var accumulatedSeconds: Double = 0
    /// comfybox#307 (review r1): why the two-stage refine did not run on some
    /// chunk of THIS render, if any — carried on the context (not
    /// `LTX2Pipeline.lastRefineSkipReason`) because a cold preemption resume
    /// can rebuild the pipeline/generator from scratch (`VideoGeneratorHolder
    /// .release()` deallocates them), which would otherwise silently drop a
    /// skip reason recorded on an earlier chunk before the eviction. See
    /// `LTX2VideoGenerator.render`.
    var refineSkippedReason: String? = nil

    init(request: LTX2VideoRequest) {
        self.request = request
    }

    /// comfybox#307 (review r2, item 2c): the snapshot construction
    /// `LTX2VideoGenerator.render`'s `checkpoint()` closure performs — pulled
    /// out so a test can call the SAME code that actually produces a
    /// checkpoint's context, not a hand re-implementation that could
    /// silently diverge from it (e.g. a field added to the snapshot later
    /// but never mirrored in a test's own copy). `elapsedThisSegment` is the
    /// raw (possibly negative) wall-clock delta for the CURRENT segment only
    /// — `ctx.accumulatedSeconds` from prior segments is added here, exactly
    /// as the original inline code did.
    static func checkpointSnapshot(
        from ctx: LTX2RenderContext, chunk: Int, frames: [CGImage],
        audio: MLXArray?, seedImage: MLXArray?, refineSkippedReason: String?,
        elapsedThisSegment: Double
    ) -> LTX2RenderContext {
        let snapshot = LTX2RenderContext(request: ctx.request)
        snapshot.chunkIndex = chunk
        snapshot.frames = frames
        snapshot.refineSkippedReason = refineSkippedReason
        // Materialize on capture, same contract as LTX2ResumeState's own
        // tensors — a cheap no-op when they are already evaluated, and the
        // guarantee stops depending on what upstream call sites happen to do.
        if let audio { eval(audio) }
        snapshot.audioLatents = audio
        let chained = chunk > 0 ? seedImage : nil
        if let chained { eval(chained) }
        snapshot.chunkSeedImage = chained
        snapshot.accumulatedSeconds = ctx.accumulatedSeconds + max(0, elapsedThisSegment)
        return snapshot
    }
}
#endif

public enum LTX2VideoError: Error, LocalizedError {
    case invalidFrameCount(Int)
    case invalidDimensions(Int, Int)
    case weightsMissing(String)
    case imageLoadFailed(String)
    case unsupportedPlatform
    case audioUnsupported(String)
    case resumeContextMissing

    public var errorDescription: String? {
        switch self {
        case .invalidFrameCount(let n):
            return "LTX-2 frames must be 1 + 8k (9, 17, 25, …, 97); got \(n)."
        case .invalidDimensions(let w, let h):
            return "LTX-2 width/height must be divisible by 32; got \(w)x\(h)."
        case .weightsMissing(let path):
            return "LTX-2 weights not found: \(path)"
        case .imageLoadFailed(let path):
            return "Failed to load init image: \(path)"
        case .unsupportedPlatform:
            return "LTX-2 video requires CoreGraphics/ImageIO (macOS)."
        case .audioUnsupported(let why):
            return "LTX-2 audio: \(why)"
        case .resumeContextMissing:
            return "LTX-2 resume: the checkpoint carries no render context — it cannot name the request to resume (#1479)."
        }
    }
}

public final class LTX2VideoGenerator {
    public struct Configuration: Sendable {
        /// Directory holding transformer / vae_{encoder,decoder} / connector.
        public var weightsDir: String
        /// Gemma-3 tokenizer + text-encoder snapshot directory.
        public var gemmaPath: String
        /// Transformer weights filename inside `weightsDir`.
        public var transformerFile: String

        public init(
            weightsDir: String,
            gemmaPath: String,
            transformerFile: String = "transformer-distilled.safetensors"
        ) {
            self.weightsDir = weightsDir
            self.gemmaPath = gemmaPath
            self.transformerFile = transformerFile
        }
    }

    public let config: Configuration
    private let logger: Logger

    private var pipeline: LTX2Pipeline?
    private var tokenizer: LTX2GemmaTokenizer?
    public private(set) var isLoaded = false

    /// Exposes the loaded pipeline + tokenizer for call sites that need a
    /// generation shape `generate(request:)` doesn't cover yet (e.g. the
    /// multi-keyframe path — see LTX2Pipeline.generateMultiKeyframe). nil
    /// until `load()` succeeds.
    public var loadedPipeline: LTX2Pipeline? { pipeline }
    public var loadedTokenizer: LTX2GemmaTokenizer? { tokenizer }
    /// "path@strength" of the LoRA merged into the loaded transformer (nil = base).
    private var loadedLoraKey: String?
    /// Whether the RESIDENT transformer carries the audio branch — admission
    /// uses this to spot audio-mode mismatches that force a full rebuild.
    public var isAudioLoaded: Bool { loadedLoraKey?.contains("+audio") == true }
    /// Audio codec (VAE + vocoder), lazily bound from the monolith on the
    /// first audio render; cheap (mmap subset) and kept for the process life.
    private var audioVAE: LTX2AudioVAE?

    /// #1479: raised by the coordinator to ask the in-flight render to
    /// check point and unwind. Read (never written) inside the render, with no
    /// actor hop — see `PreemptionSignal`.
    private var preemption: PreemptionSignal?
    /// #1479: per-phase timings feeding the refusal guard and /v1/queue.
    private var telemetry: LTX2PhaseTelemetry?

    public init(config: Configuration, logger: Logger = Logger(label: "ltx2.video")) {
        self.config = config
        self.logger = logger
    }

    /// Arm (or disarm) preemption. Honoured by `generatePreemptible` and
    /// `resume(from:)`; the legacy `generate` entry deliberately passes no
    /// signal so its behaviour is byte-for-byte what it always was.
    public func setPreemptionSignal(_ s: PreemptionSignal?) { preemption = s }

    /// Install (or clear) the phase-timing sink. nil = zero cost: every
    /// recording site is optional-chained.
    public func setTelemetry(_ t: LTX2PhaseTelemetry?) { telemetry = t }

    // MARK: - Pure planning helpers (testable without the model)

    /// A frame count is valid when it's 1 + 8k and ≥ 9.
    public static func isValidFrameCount(_ n: Int) -> Bool {
        n >= 9 && (n - 1) % 8 == 0
    }

    /// Resolve the Gemma tokenizer max length (LTX2_GEMMA_MAX_LENGTH).
    ///
    /// Default 1024 = the official Lightricks recipe (their tokenizer call and
    /// the ComfyUI Gemma loader both default to 1024). Our former hardcoded 128
    /// was a port artifact that silently truncated every long prompt.
    ///
    /// The connector tiles its 128 learnable registers with integer division
    /// (`numTiles = seqLen / 128`), so the value must be a positive multiple of
    /// 128 — anything else silently under-covers the sequence with registers.
    /// Invalid overrides fall back to the default rather than half-applying.
    public static func resolveGemmaMaxLength(env: String?) -> Int {
        let fallback = 1024
        guard let raw = env?.trimmingCharacters(in: .whitespaces), !raw.isEmpty else { return fallback }
        guard let n = Int(raw), n > 0, n % 128 == 0 else { return fallback }
        return n
    }

    /// Resolve the optional external 24 kHz HiFi-GAN override. A path alone is
    /// deliberately insufficient: production historically carried a stale
    /// `LTX2_VOCODER_PATH` that silently displaced the monolith's matched
    /// BigVGAN+BWE. Both variables make the mismatch an explicit experiment.
    public static func externalVocoderOverridePath(
        environment: [String: String]
    ) -> String? {
        guard environment["LTX2_USE_EXTERNAL_VOCODER"] == "1" else { return nil }
        guard let raw = environment["LTX2_VOCODER_PATH"] else { return nil }
        let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return path.isEmpty ? nil : path
    }

    /// fps must be positive and sane (chunk planning divides by it; the RoPE
    /// temporal coords divide by it too).
    public static func isValidFPS(_ fps: Int) -> Bool {
        fps >= 1 && fps <= 120
    }

    /// Dimensions must be positive multiples of 32.
    public static func areValidDimensions(width: Int, height: Int) -> Bool {
        width > 0 && height > 0 && width % 32 == 0 && height % 32 == 0
    }

    public struct ChunkPlan: Equatable, Sendable {
        public let totalChunks: Int
        public let totalFrames: Int
        public let durationSeconds: Float
    }

    /// How many chunks and frames a request produces. Each continuation chunk
    /// re-uses the previous chunk's last frame, so it adds `framesPerChunk - 1`
    /// new frames. `extendToSeconds == 0` → a single chunk.
    public static func chunkPlan(framesPerChunk: Int, extendToSeconds: Float, fps: Int) -> ChunkPlan {
        let totalChunks: Int
        if extendToSeconds > 0 {
            let targetFrames = Int(extendToSeconds * Float(fps))
            let continuations = max(0, Int(ceil(Float(targetFrames - framesPerChunk) / Float(framesPerChunk - 1))))
            totalChunks = 1 + continuations
        } else {
            totalChunks = 1
        }
        let totalFrames = framesPerChunk + (framesPerChunk - 1) * (totalChunks - 1)
        return ChunkPlan(
            totalChunks: totalChunks,
            totalFrames: totalFrames,
            durationSeconds: Float(totalFrames) / Float(fps)
        )
    }

    /// Validate a request without loading anything.
    public func validate(_ request: LTX2VideoRequest) throws {
        guard Self.isValidFrameCount(request.framesPerChunk) else {
            throw LTX2VideoError.invalidFrameCount(request.framesPerChunk)
        }
        if request.audio {
            // Audio scope: single-chunk T2V and I2V. Silently downgrading (the
            // first cut) shipped MP4s whose audio stopped after chunk 0 or
            // never existed — reject unsupported shapes loudly instead
            // (Codex 2026-08-04 #4).
            let plan = Self.chunkPlan(
                framesPerChunk: request.framesPerChunk,
                extendToSeconds: request.extendToSeconds, fps: request.fps)
            if plan.totalChunks > 1 {
                throw LTX2VideoError.audioUnsupported(
                    "audio is not yet supported for chunked renders (\(plan.totalChunks) chunks requested; keep duration within one \(request.framesPerChunk)-frame chunk)")
            }
            if request.identityAnchorStrength > 0, request.identityReAnchorInterval > 0,
               request.framesPerChunk > request.identityReAnchorInterval + 1 {
                throw LTX2VideoError.audioUnsupported(
                    "audio is not yet supported with mid-pass identity re-anchoring (multi-keyframe path)")
            }
        }
        guard Self.areValidDimensions(width: request.width, height: request.height) else {
            throw LTX2VideoError.invalidDimensions(request.width, request.height)
        }
        let weightsURL = resolveWeightsFileURL()
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTX2VideoError.weightsMissing(weightsURL.path)
        }
    }

    /// Resolve the checkpoint file to load. Prefers the configured per-component
    /// transformer file; if that's absent, falls back to a JoyAI-Echo monolith
    /// (`*.safetensors` in `weightsDir` that isn't one of the per-component
    /// VAE/connector files). Returns the configured path (possibly nonexistent)
    /// as a last resort so callers surface a clear `weightsMissing` error.
    private func resolveWeightsFileURL() -> URL {
        let configured = URL(fileURLWithPath:
            (config.weightsDir as NSString).appendingPathComponent(config.transformerFile))
        if FileManager.default.fileExists(atPath: configured.path) { return configured }

        let fm = FileManager.default
        if let entries = try? fm.contentsOfDirectory(
            at: URL(fileURLWithPath: config.weightsDir),
            includingPropertiesForKeys: nil
        ) {
            let perComponent: Set<String> = [
                "vae_encoder.safetensors", "vae_decoder.safetensors", "connector.safetensors",
            ]
            let candidates = entries
                .filter { $0.pathExtension == "safetensors" && !perComponent.contains($0.lastPathComponent) }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            if let monolith = candidates.first { return monolith }
        }
        return configured
    }

    // MARK: - Model loading (lazy, cached)

    /// Construct and load the transformer, VAE, text encoder, and pipeline,
    /// optionally merging one or more LoRAs into the transformer (applied in
    /// order). Idempotent for the same LoRA set; a different set reloads.
    public func load(loras: [LTX2LoRAReference] = [], audio: Bool = false) throws {
        // Audio joins the warm key: an audio render needs the dual-stream
        // transformer (audio branch +~11GiB); switching either way reloads.
        let loraPart = loras.isEmpty ? "" : loras.map { "\($0.path)@\($0.scale)" }.joined(separator: "|")
        let wantKey0 = loraPart + (audio ? "|+audio" : "")
        let wantKey: String? = wantKey0.isEmpty ? nil : wantKey0
        if isLoaded {
            if wantKey == loadedLoraKey { return }
            unload()   // LoRA set or audio mode changed — rebuild the transformer.
        }
        let modelDir = config.weightsDir

        logger.info("LTX-2: creating transformer…\(audio ? " (dual-stream A/V)" : "")")
        let transformer = LTX2Transformer(
            numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
            numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
            normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
            positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
            useMiddleIndicesGrid: true, ropeMode: .split, doublePrecisionRoPE: true,
            hasAudio: audio, audioInnerDim: 2048, audioInChannels: 128
        )

        let weightsURL = resolveWeightsFileURL()
        logger.info("LTX-2: loading transformer weights (\(weightsURL.lastPathComponent))…")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        // JoyAI-Echo ships one monolithic file (DiT + VAE + audio + vocoder). When
        // detected, the VAE and connector/projection weights come from this same
        // already-loaded dict via prefix-filtered subsets rather than separate
        // files — the audio/vocoder tensors stay lazy (never eval'd) on the
        // video-only path, so they don't hit RAM.
        let isMonolith = LTX2EchoCheckpoint.isMonolithLayout(rawWeights.keys)
        if isMonolith {
            logger.info("LTX-2: JoyAI-Echo monolithic checkpoint detected — prefix-filtered video-only load.")
        }
        var sanitized = audio && isMonolith
            ? LTX2Transformer.sanitizeWeightsWithAudio(rawWeights)
            : LTX2Transformer.sanitizeWeights(rawWeights)
        if audio && !isMonolith {
            throw LTX2VideoError.weightsMissing(
                "audio render requires a JoyAI-Echo monolithic checkpoint (audio branch tensors) — \(weightsURL.lastPathComponent) is per-component")
        }

        // Unsupported int8 checkpoints (comfybox#256): some exports (e.g.
        // PinkCherry v1.7) carry `int8`-dtype `.weight` tensors in a scheme
        // `LTX2Quantizer`'s own MLX affine `.scales`/`.biases` packed-uint32
        // format (and `applyQuantizedLayout`, which only recognises that
        // format) does not implement — PinkCherry specifically is ComfyUI's
        // `int8_tensorwise` with `convrot: true` (a rotated coordinate
        // basis; plain int8×scale reconstruction measured cosine similarity
        // ~0.008 against the real weights — not implementable as a scale
        // multiply). Left unchecked, those raw int8 bytes would load
        // straight into float weight parameters and render as noise. Fail
        // loudly here, before any weight is touched, naming the exact
        // unsupported format/tensor rather than silently producing noise.
        try LTX2Quantizer.rejectUnsupportedInt8Weights(sanitized)

        // Merge each LoRA into the base weights in order (skip audio branches),
        // as the CLI does — multiple LoRAs simply accumulate their deltas.
        for lora in loras {
            logger.info("LTX-2: merging LoRA \(lora.path) @ \(lora.scale)…")
            let loraWeights = try MLX.loadArrays(url: URL(fileURLWithPath: lora.path))
            var merged = 0
            for (key, loraA) in loraWeights {
                guard key.hasSuffix(".lora_A.weight") else { continue }
                var baseKey = String(key.dropLast(".lora_A.weight".count))
                if baseKey.hasPrefix("diffusion_model.") { baseKey = String(baseKey.dropFirst("diffusion_model.".count)) }
                if baseKey.contains("audio_") || baseKey.contains("av_ca_")
                    || baseKey.contains("video_to_audio_attn") || baseKey.contains("audio_to_video_attn") { continue }
                // LoRA keys use the RAW checkpoint naming, but `sanitized` keys have
                // been through sanitizeWeights' renames — without mirroring them here
                // the lookup below silently skips every renamed projection (to_out,
                // ff proj_in/out, adaln linear1/2): 196 of 584 pairs in the official
                // distil LoRA, i.e. runtime LoRAs applied at ~2/3 strength (2026-08-02).
                // baseKey is a module path (no trailing dot), so map suffixes.
                let suffixRenames: [(String, String)] = [
                    (".to_out.0", ".to_out"),
                    (".ff.net.0.proj", ".ff.proj_in"),
                    (".ff.net.2", ".ff.proj_out"),
                    (".linear_1", ".linear1"),
                    (".linear_2", ".linear2"),
                ]
                for (raw, renamed) in suffixRenames where baseKey.hasSuffix(raw) {
                    baseKey = String(baseKey.dropLast(raw.count)) + renamed
                    break
                }
                let bKey = key.replacingOccurrences(of: ".lora_A.weight", with: ".lora_B.weight")
                guard let loraB = loraWeights[bKey] else { continue }
                let targetKey = baseKey + ".weight"
                guard sanitized[targetKey] != nil else { continue }
                let delta = MLX.matmul(loraB.asType(.float32), loraA.asType(.float32)) * MLXArray(lora.scale)
                if sanitized["\(baseKey).scales"] != nil {
                    // int8/int4 base checkpoint (#230): dequantize -> merge -> requantize
                    // so LoRAs keep working against quantized weights.
                    guard let (dense, groupSize, bits) = LTX2Quantizer.dequantizeLayer(
                        base: baseKey, weights: sanitized) else { continue }
                    let mergedDense = dense.asType(.float32) + delta
                    let (wq, scales, biases) = MLX.quantized(
                        mergedDense, groupSize: groupSize, bits: bits, mode: .affine)
                    sanitized[targetKey] = wq
                    sanitized["\(baseKey).scales"] = scales.asType(.bfloat16)
                    if let b = biases { sanitized["\(baseKey).biases"] = b.asType(.bfloat16) }
                } else {
                    sanitized[targetKey] = sanitized[targetKey]!.asType(.float32) + delta
                }
                merged += 1
            }
            logger.info("LTX-2: merged \(merged) LoRA pairs from \(lora.path).")
        }

        // Quantized checkpoint support (#230): when the sanitized weights carry
        // `.scales` siblings (MLX affine int8/int4 from `ComfyBox quantize-ltx2`),
        // convert exactly those Linear layers to QuantizedLinear before the
        // update. Shapes are self-describing — no manifest needed at load.
        let quantizedLayerCount = LTX2Quantizer.applyQuantizedLayout(
            to: transformer, sanitizedWeights: sanitized)
        if quantizedLayerCount > 0 {
            logger.info("LTX-2: quantized checkpoint — \(quantizedLayerCount) block projections load as QuantizedLinear.")
        }
        // Anti-noise guard: update(verify:[.shapeMismatch]) silently DROPS any key
        // that doesn't match a module parameter — a mis-remapped checkpoint loads
        // 0 weights and renders pure noise while reporting success. Log how many of
        // the module's parameters the remap actually covers; a near-zero match on a
        // non-empty checkpoint means the key remap is wrong (e.g. leftover
        // `model.diffusion_model.` prefix), not that the load "succeeded".
        let moduleKeys = Set(transformer.parameters().flattened().map { $0.0 })
        let matched = sanitized.keys.filter { moduleKeys.contains($0) }.count
        let unmatchedSanitized = sanitized.count - matched
        logger.info("LTX-2: transformer remap matched \(matched)/\(moduleKeys.count) module params (\(sanitized.count) sanitized keys, \(unmatchedSanitized) unmatched/dropped).")
        if matched * 2 < moduleKeys.count {
            logger.error("LTX-2: transformer weight remap covered only \(matched)/\(moduleKeys.count) params — checkpoint key format likely unrecognized; output would be noise.")
            throw LTX2VideoError.weightsMissing(
                "transformer key remap matched only \(matched)/\(moduleKeys.count) module params from \(weightsURL.lastPathComponent) — unrecognized checkpoint key format")
        }
        if audio {
            // Audio-branch coverage guard (Codex #7): the 50% whole-model gate
            // can pass while the ENTIRE audio branch (2,729 tensors) is absent
            // or mis-keyed — silence/noise instead of a load error. Require
            // near-complete audio coverage explicitly.
            func isAudioKey(_ k: String) -> Bool {
                k.contains("audio_") || k.contains("av_ca_")
                    || k.contains("scale_shift_table_a2v")
                    || k.contains("audio_to_video_attn") || k.contains("video_to_audio_attn")
            }
            let audioModuleKeys = moduleKeys.filter(isAudioKey)
            let audioMatched = sanitized.keys.filter { isAudioKey($0) && moduleKeys.contains($0) }.count
            if audioMatched < audioModuleKeys.count * 95 / 100 {
                throw LTX2VideoError.audioUnsupported(
                    "audio branch weights incomplete: \(audioMatched)/\(audioModuleKeys.count) matched from \(weightsURL.lastPathComponent)")
            }
            logger.info("LTX-2 audio: branch coverage \(audioMatched)/\(audioModuleKeys.count).")
        }
        let params = ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) })
        try transformer.update(parameters: params, verify: [.shapeMismatch])
        MLX.eval(transformer.parameters())

        logger.info("LTX-2: loading VAE…")
        let vae = LTX2VAE(config: .v23)
        if isMonolith {
            // Echo carries the video VAE under the `vae.` prefix in the monolith,
            // exactly the layout loadVAEWeightsFromTensors expects (top-level
            // per-channel stats are mirrored into the decoder path by the adapter).
            let vaeTensors = LTX2EchoCheckpoint.videoVAETensors(from: rawWeights)
            try LTX2WeightLoader.loadVAEWeightsFromTensors(into: vae, tensors: vaeTensors, logger: logger)
        } else {
            var combinedVAEWeights: [String: MLXArray] = [:]
            let rawDecoderWeights = try MLX.loadArrays(url: URL(fileURLWithPath: (modelDir as NSString).appendingPathComponent("vae_decoder.safetensors")))
            for (key, value) in rawDecoderWeights where key.hasPrefix("vae_decoder.") {
                combinedVAEWeights["vae.decoder." + String(key.dropFirst("vae_decoder.".count))] = value
            }
            let rawEncoderWeights = try MLX.loadArrays(url: URL(fileURLWithPath: (modelDir as NSString).appendingPathComponent("vae_encoder.safetensors")))
            for (key, value) in rawEncoderWeights where key.hasPrefix("vae_encoder.") {
                combinedVAEWeights["vae.encoder." + String(key.dropFirst("vae_encoder.".count))] = value
            }
            if let m = combinedVAEWeights["vae.decoder.per_channel_statistics.mean"] {
                combinedVAEWeights["vae.per_channel_statistics.mean-of-means"] = m
            }
            if let s = combinedVAEWeights["vae.decoder.per_channel_statistics.std"] {
                combinedVAEWeights["vae.per_channel_statistics.std-of-means"] = s
            }
            try LTX2WeightLoader.loadVAEWeightsFromTensors(into: vae, tensors: combinedVAEWeights, logger: logger)
        }
        MLX.eval(vae.parameters())

        // comfybox#340: everything from here to "models ready." used to be a
        // single unnamed span. When a render wedged (production 2026-09-01
        // 00:16–08:26: six loads killed by the 15-minute watchdog, one that
        // finally completed after 14m30s) the only evidence was this one line
        // followed by silence — which stage was slow could not be told apart
        // from a deadlock. Each stage is now timed and named, and anything
        // over `slowLoadStageWarnSeconds` is called out explicitly, so the
        // next occurrence names itself in the log instead of needing a repro.
        logger.info("LTX-2: loading text encoder (Gemma 3 12B) from \(config.gemmaPath)…")
        let gemmaConfig = LTX2GemmaConfig(
            vocabSize: 262208, hiddenSize: 3840,
            numHiddenLayers: 48, numAttentionHeads: 16,
            numKeyValueHeads: 8, headDim: 256,
            intermediateSize: 15360,
            rmsNormEps: 1e-6, ropeTheta: 1_000_000.0,
            slidingWindow: 1024, slidingWindowPattern: 6,
            quantization: nil
        )
        let textEncoder = try timedLoadStage("text encoder construct") {
            LTX2TextEncoder(config: LTX2TextEncoderConfig(gemma: gemmaConfig, hasPromptAdaLN: true))
        }
        try timedLoadStage("text encoder bind weights (~14.5GB Gemma safetensors)") {
            if isMonolith {
                // Connectors (model.diffusion_model.*_embeddings_connector) and the
                // aggregate embeds (text_embedding_projection.*) live in the monolith;
                // Gemma still loads from its own directory.
                try textEncoder.loadWeightsFromMonolith(
                    gemmaPath: URL(fileURLWithPath: config.gemmaPath),
                    monolithTensors: rawWeights
                )
            } else {
                try textEncoder.loadWeights(
                    modelPath: URL(fileURLWithPath: modelDir),
                    textEncoderPath: URL(fileURLWithPath: config.gemmaPath)
                )
            }
        }
        // The mmap'd Gemma tensors are only actually paged in here — this eval,
        // not the bind above, is where a cold page cache or memory pressure is
        // paid for, so it gets its own name in the log.
        try timedLoadStage("text encoder materialize parameters (MLX.eval)") {
            MLX.eval(textEncoder.parameters())
        }

        // Reference ComfyUI-LTXVideo workflows always decode through
        // VAEDecodeTiled, never a plain single-pass decode — the decoder
        // likely relies on windowed/local processing that the (already
        // implemented, just never enabled) tiled path is designed for.
        // Running it as one giant pass is the leading suspect for the
        // uniform grid/mesh artifact seen in every local I2V test tonight.
        // Two-stage refine (Phase 3): load the spatial latent upsampler if enabled.
        // ltx-2.3-spatial-upscaler-x2-1.1 keys map 1:1 to LTX2LatentUpsampler.
        //
        // Codex r1: `loadUpsampler` does its own `MLX.loadArrays` + `MLX.eval`
        // (a second paged read), so a stall here produces the exact same
        // "after the text encoder, before models ready" symptom. It gets its
        // own named stage rather than hiding inside the gap.
        var upsampler: LTX2LatentUpsampler? = nil
        if ProcessInfo.processInfo.environment["LTX2_TWO_STAGE"] == "1",
           let upPath = ProcessInfo.processInfo.environment["LTX2_UPSAMPLER_PATH"],
           FileManager.default.fileExists(atPath: upPath) {
            upsampler = try timedLoadStage("two-stage upsampler load (loadArrays + eval)") {
                Self.loadUpsampler(path: upPath, logger: logger)
            }
        }
        // Tiled/chunked VAE decode is OOM-safe on long/large clips but seams on
        // fast motion (spatial-tile mosaic + temporal-window jitter). Plain
        // single-pass decode (as ComfyUI does) is clean but memory-heavier.
        // LTX2_TILED_DECODE=0 selects plain decode. Default stays tiled.
        let tiled = ProcessInfo.processInfo.environment["LTX2_TILED_DECODE"] != "0"
        // NOTE: the warm pipeline is built ONCE here, so per-request fps cannot
        // flow through config.fps (it would bake in the first request's value).
        // The temporal-RoPE conditioning fps (the motion dial) is therefore
        // controlled per-render via LTX2_COND_FPS (read fresh in createPositionGrid).
        let pipelineConfig = LTX2PipelineConfig(modelPath: modelDir, pipelineType: .distilled, hasPromptAdaLN: true, tiledDecode: tiled)
        // Codex r1: the pipeline and the tokenizer are published as ONE step,
        // after every throwing stage has succeeded. Assigning `self.pipeline`
        // before the (throwing) tokenizer load used to leave a failed attempt
        // retaining a complete ~54GB model stack that `isLoaded == false`
        // claimed was not there, so the retry built a second one beside it.
        try Self.atomicallyPublishLoad(
            build: { () throws -> (LTX2Pipeline, LTX2GemmaTokenizer) in
                let builtPipeline = try timedLoadStage("pipeline construct") {
                    LTX2Pipeline(
                        vae: vae, textEncoder: textEncoder, transformer: transformer,
                        config: pipelineConfig, upsampler: upsampler)
                }
                // 128 was a port artifact, NOT the trained recipe (discovered 2026-08-07):
                // the official Lightricks pipeline tokenizes at max_length 1024, the
                // ComfyUI Gemma loader defaults to 1024, and the reference PinkCherry
                // workflow feeds 256-token enhancer output through this same encoder.
                // The artifact silently truncated every long prompt for weeks — scene,
                // camera and identity fell off the tail. Mirror upstream's env knob.
                let builtTokenizer = try timedLoadStage("tokenizer load (tokenizer.json parse)") {
                    try LTX2GemmaTokenizer.load(
                      from: URL(fileURLWithPath: config.gemmaPath),
                      maxLength: Self.resolveGemmaMaxLength(env: ProcessInfo.processInfo.environment["LTX2_GEMMA_MAX_LENGTH"]))
                }
                return (builtPipeline, builtTokenizer)
            },
            publish: { stack in
                self.pipeline = stack.0
                self.tokenizer = stack.1
                self.isLoaded = true
                self.loadedLoraKey = wantKey
            },
            discard: { self.unload() })
        logger.info("LTX-2: models ready.")
    }


    // MARK: - Load-path stage timing (comfybox#340)

    /// A load stage slower than this is reported as abnormally slow, and the
    /// interval at which an in-flight stage repeats its heartbeat. A healthy
    /// warm load runs single-digit seconds per stage (production text-encoder
    /// loads: 3–17s end to end), so 30s is well clear of a cold start while
    /// still firing ~30 times before the 15-minute watchdog kill.
    static let slowLoadStageWarnSeconds: Double = 30

    /// The operational note shared by the heartbeat and the slow-completion
    /// warning. `/health` is called out explicitly as NOT a signal here: it
    /// reads a lock-backed snapshot precisely so it stays answerable while the
    /// coordinator is blocked (WarmServer, #217), so a green health check
    /// during this stall means nothing. Getting this wrong would send the next
    /// person reading the log to the wrong place.
    private static let stalledLoadNote =
        "The render queue is stalled for the whole stage (/health is snapshot-backed and stays "
        + "green, so it will NOT show this), and a watchdog restart here loses the in-flight "
        + "render (comfybox#340, #339). Suspect cold page cache, memory pressure, or a near-full "
        + "disk rather than a deadlock."

    /// Logged BEFORE a stage runs. Codex r1: a report emitted after `body()`
    /// returns is exactly the report a wedge never produces — the six
    /// 2026-09-01 loads were SIGTERM'd mid-stage. This line is the one that
    /// survives the kill and names the stage that was active.
    static func loadStageEntryMessage(stage: String) -> String {
        "LTX-2 load: \(stage) — started"
    }

    /// Repeated every `slowLoadStageWarnSeconds` while a stage is still
    /// running, so a wedge is visible WHILE it happens rather than only in
    /// hindsight (and at all, when the process is killed before completion).
    static func loadStageStillRunningMessage(
        stage: String,
        seconds: Double,
        warnAfter: Double = LTX2VideoGenerator.slowLoadStageWarnSeconds
    ) -> String {
        let secs = String(format: "%.0f", seconds)
        return "LTX-2 load: \(stage) STILL RUNNING after \(secs)s "
            + "(healthy is under \(Int(warnAfter))s). \(stalledLoadNote)"
    }

    /// One-line verdict for a COMPLETED load stage. Honest scope: this fires
    /// only once the stage returns — the entry line and the heartbeat above are
    /// what cover a stage that never returns.
    static func loadStageReport(
        stage: String,
        seconds: Double,
        warnAfter: Double = LTX2VideoGenerator.slowLoadStageWarnSeconds
    ) -> (message: String, isSlow: Bool) {
        let secs = String(format: "%.2f", seconds)
        guard seconds >= warnAfter else {
            return ("LTX-2 load: \(stage) — \(secs)s", false)
        }
        return (
            "LTX-2 load: \(stage) took \(secs)s — ABNORMALLY SLOW "
                + "(healthy is under \(Int(warnAfter))s). \(stalledLoadNote)",
            true
        )
    }

    /// Logged when a stage throws. Names the stage, how long it burned before
    /// failing, the cause, and that the partial load was dropped — so the next
    /// line in the log (a retry) is understood to start from nothing.
    static func loadStageFailureMessage(stage: String, seconds: Double, error: Error) -> String {
        "LTX-2 load: \(stage) FAILED after \(String(format: "%.2f", seconds))s — \(error). "
            + "Partial load discarded; nothing was published, so a retry starts clean."
    }

    /// Monotonic elapsed seconds. Wall-clock (`CFAbsoluteTimeGetCurrent`) can
    /// step under NTP correction, which would make an incident's timings lie.
    private static func elapsedSeconds(since start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds &- start.uptimeNanoseconds) / 1_000_000_000
    }

    /// Run one named load stage: entry line, in-flight heartbeat, then a
    /// completion or failure line. Splits what used to be a single unnamed span
    /// between "loading text encoder" and "models ready" into stages a log can
    /// point at — including a stage that never finishes.
    private func timedLoadStage<T>(_ stage: String, _ body: () throws -> T) throws -> T {
        let started = DispatchTime.now()
        logger.info("\(Self.loadStageEntryMessage(stage: stage))")

        // Detached on purpose: `body()` blocks this thread for the whole stage
        // (that IS the failure mode), so the heartbeat cannot live on it.
        let interval = Self.slowLoadStageWarnSeconds
        let heartbeat = Task.detached(priority: .utility) { [logger] in
            var waited = 0.0
            while !Task.isCancelled {
                do {
                    try await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                } catch {
                    return  // cancelled while sleeping — the stage finished
                }
                waited += interval
                logger.warning(
                    "\(Self.loadStageStillRunningMessage(stage: stage, seconds: waited))")
            }
        }
        defer { heartbeat.cancel() }

        do {
            let result = try body()
            let report = Self.loadStageReport(
                stage: stage, seconds: Self.elapsedSeconds(since: started))
            if report.isSlow {
                logger.warning("\(report.message)")
            } else {
                logger.info("\(report.message)")
            }
            return result
        } catch {
            let failure = Self.loadStageFailureMessage(
                stage: stage, seconds: Self.elapsedSeconds(since: started), error: error)
            logger.error("\(failure)")
            throw error
        }
    }

    /// Publish a freshly built model stack as ONE step.
    ///
    /// Codex review r1 (comfybox#340): `load()` used to assign `self.pipeline` —
    /// the complete transformer + VAE + text encoder — and only THEN run the
    /// throwing tokenizer stage. A tokenizer failure left `isLoaded == false`
    /// while the generator still retained that whole stack, and since `load()`
    /// unloads only when `isLoaded` is true, the retry built a SECOND stack
    /// beside the first. At ~54GB a stack, that is an OOM, not a leak.
    ///
    /// `build` runs every throwing stage; `publish` runs only if all of them
    /// succeed, and a throw runs `discard` instead.
    static func atomicallyPublishLoad<Stack>(
        build: () throws -> Stack,
        publish: (Stack) -> Void,
        discard: () -> Void
    ) throws {
        let stack: Stack
        do {
            stack = try build()
        } catch {
            discard()
            throw error
        }
        publish(stack)
    }

    /// Load + validate the spatial latent upsampler. Shared by the load-time
    /// path (env) and the per-request lazy path (finding #18). Returns nil —
    /// loudly — when the file's keys do not bind the module completely: a
    /// partially bound upsampler renders a periodic mesh (2026-08-01).
    static func loadUpsampler(path: String, logger: Logger) -> LTX2LatentUpsampler? {
        let up = LTX2LatentUpsampler()
        guard let w = try? MLX.loadArrays(url: URL(fileURLWithPath: path)) else {
            logger.error("LTX-2: upsampler file unreadable: \(path)")
            return nil
        }
        // Checkpoint stores conv weights in PyTorch layout (out, in, *spatial);
        // MLX conv layers are channels-last. Permute conv weights by ndim;
        // 1D params pass through untouched. Sequential index 0 renames to the
        // named `conv` child.
        let remapped: [(String, MLXArray)] = w.map { (rawKey, v) in
            let key = rawKey.hasPrefix("upsampler.0.")
                ? "upsampler.conv." + rawKey.dropFirst("upsampler.0.".count)
                : rawKey
            if key.hasSuffix(".weight") {
                if v.ndim == 5 { return (key, v.transposed(0, 2, 3, 4, 1)) }
                if v.ndim == 4 { return (key, v.transposed(0, 2, 3, 1)) }
            }
            return (key, v)
        }
        let expected = Set(up.parameters().flattened().map { $0.0 })
        let bound = remapped.filter { expected.contains($0.0) }.count
        guard bound == expected.count else {
            let sample = remapped.map(\.0).filter { !expected.contains($0) }.prefix(3)
            logger.error("""
                LTX-2: upsampler weights do NOT match the module — bound \(bound)/\(expected.count) \
                parameters from \(w.count) file tensors (unmatched e.g. \(Array(sample))). \
                Two-stage refine stays OFF; use the official Lightricks \
                ltx-2.3-spatial-upscaler-x2-1.1.safetensors (bare keys, PyTorch layout).
                """)
            return nil
        }
        do {
            try up.update(parameters: ModuleParameters.unflattened(remapped), verify: [.shapeMismatch])
        } catch {
            logger.error("LTX-2: upsampler update failed: \(error)")
            return nil
        }
        MLX.eval(up.parameters())
        logger.info("LTX-2: two-stage refine upsampler loaded (bound \(bound)/\(expected.count) parameters)")
        return up
    }

    /// Free the loaded models.
    public func unload() {
        pipeline = nil
        tokenizer = nil
        isLoaded = false
        loadedLoraKey = nil
    }

    // MARK: - Generate

    /// Render to completion. Non-preemptible by construction: it passes NO
    /// signal down, so behaviour is exactly what it was before #1479 even when
    /// a signal is armed on the generator. Callers that want preemption use
    /// `generatePreemptible`.
    public func generate(
        _ request: LTX2VideoRequest,
        progress: ((Int, Int, Int, Int) -> Void)? = nil   // (chunk, totalChunks, step, totalSteps)
    ) throws -> LTX2VideoResult {
        switch try render(request, progress: progress, preemption: nil, resume: nil) {
        case .completed(let result):
            return result
        case .yielded(let s):
            // Unreachable: `preemption: nil` above, and nothing yields without
            // a raised signal. There is no finished clip to return here, so
            // fail loudly rather than invent one.
            logger.error("LTX-2: non-preemptible generate() yielded at step \(s.stepIndex) — no signal was passed.")
            preconditionFailure("LTX2VideoGenerator.generate yielded with no preemption signal passed")
        }
    }

    /// #1479: render, honouring the armed preemption signal. Returns either the
    /// finished clip or a checkpoint the coordinator hands back to
    /// `resume(from:)` once the preempting job is done.
    public func generatePreemptible(
        _ request: LTX2VideoRequest,
        progress: ((Int, Int, Int, Int) -> Void)? = nil
    ) throws -> LTX2RenderOutcome {
        try render(request, progress: progress, preemption: preemption, resume: nil)
    }

    /// #1479: continue a checkpointed render.
    ///
    /// Takes no request parameter: the request travels in the checkpoint's
    /// `LTX2RenderContext`, because the generator that produced the checkpoint
    /// may already have been deallocated by the eviction (see
    /// `LTX2RenderContext`). THROWS on a config-fingerprint or sigma-schedule
    /// mismatch — never silently restarts from step 0 (spec, Error handling).
    public func resume(
        from state: LTX2ResumeState,
        progress: ((Int, Int, Int, Int) -> Void)? = nil
    ) throws -> LTX2RenderOutcome {
        #if canImport(CoreGraphics) && canImport(ImageIO)
        guard let ctx = state.context as? LTX2RenderContext else {
            throw LTX2VideoError.resumeContextMissing
        }
        logger.info("LTX-2 #1479: resuming render at chunk \(state.chunkIndex), phase \(state.phase.rawValue), step \(state.stepIndex).")
        return try render(ctx.request, progress: progress, preemption: preemption, resume: state)
        #else
        throw LTX2VideoError.unsupportedPlatform
        #endif
    }

    private func render(
        _ request: LTX2VideoRequest,
        progress: ((Int, Int, Int, Int) -> Void)?,   // (chunk, totalChunks, step, totalSteps)
        preemption: PreemptionSignal?,
        resume: LTX2ResumeState?
    ) throws -> LTX2RenderOutcome {
        #if canImport(CoreGraphics) && canImport(ImageIO)
        // Memory-leak fix (2026-07-18): the video render path never freed MLX
        // activation buffers, so idle mem climbed ~20GB -> 110GB+ across renders
        // until every render hit `Memory pressure` mid-flight and the shedding
        // corrupted the output into rainbow noise. Clear the MLX cache before
        // (drop leftover image-gen buffers, freeing headroom) and after (this
        // render's activations, via defer) EVERY render.
        GPU.clearCache()
        defer { GPU.clearCache() }
        // comfybox#322 (claim corrected in review r1): what actually guarantees
        // an interrupted render produces no clip is the `Task
        // .checkCancellation()` immediately BEFORE `writeMP4` — the write is
        // synchronous, so cancellation cannot preempt it part-way and there is
        // no "cancel lands mid-write" case to catch.
        //
        // This defer covers the narrower real one: `writeMP4` unlinks
        // `outputPath` and writes straight to it, so if it THROWS (an I/O
        // error, a full disk) after the task was already cancelled, it leaves a
        // truncated file that reads as finished output. The window is between
        // the two flags, so this never deletes a file the render did not
        // create: a cancel before the write leaves any pre-existing file alone,
        // and a completed write is kept. A #1479 yield is not a cancellation,
        // so a preempted render keeps everything and resumes.
        var startedWrite = false
        var wroteOutput = false
        defer {
            if Task.isCancelled, startedWrite, !wroteOutput,
               FileManager.default.fileExists(atPath: request.outputPath) {
                try? FileManager.default.removeItem(atPath: request.outputPath)
                logger.info("LTX-2 comfybox#322: interrupted mid-write — removed partial output at \(request.outputPath).")
            }
        }
        try validate(request)

        // #1479: the continuation the resume came in with, READ-ONLY from here
        // on. Each checkpoint gets its own fresh box (below), so a render that
        // yields twice never rewrites a checkpoint the coordinator still holds.
        let ctx = (resume?.context as? LTX2RenderContext) ?? LTX2RenderContext(request: request)
        // Start of THIS render segment. Re-stamped below at the point the
        // pre-#1479 code took its `start`, so a normal render's reported
        // `elapsedSeconds` keeps its old meaning exactly.
        var segmentStart = CFAbsoluteTimeGetCurrent()
        // comfybox#307 (review r1): seeded from the resumed context (nil on a
        // fresh render), then kept in sync with `pipeline.lastRefineSkipReason`
        // after every chunk that completes within THIS render() call (see the
        // chunk loop below) and snapshotted onto every checkpoint — so a skip
        // recorded before a preemption survives a cold resume even if the
        // pipeline/generator that recorded it was deallocated in between.
        var refineSkippedReason: String? = ctx.refineSkippedReason
        /// Close out this render segment and hand the checkpoint up with its
        /// own snapshot of the generator-level continuation.
        ///
        /// The box is COPIED, never mutated in place: `LTX2ResumeState` is a
        /// materialized snapshot by contract (Task 2), and a shared mutable
        /// context would quietly break that — a later yield would rewrite the
        /// chunk index, banked frames and chained seed frame of an earlier
        /// checkpoint the coordinator is still holding.
        func checkpoint(
            _ s: LTX2ResumeState, chunk: Int, frames: [CGImage],
            audio: MLXArray?, seedImage: MLXArray?
        ) -> LTX2RenderOutcome {
            // comfybox#307 (review r1 + r2 item 2c): the accumulated skip
            // reason (captured from the enclosing `refineSkippedReason`
            // local, kept in sync with `pipeline.lastRefineSkipReason` by
            // the chunk loop) rides the snapshot via the SAME static builder
            // a test calls directly — see `LTX2RenderContext.checkpointSnapshot`.
            let snapshot = LTX2RenderContext.checkpointSnapshot(
                from: ctx, chunk: chunk, frames: frames, audio: audio, seedImage: seedImage,
                refineSkippedReason: refineSkippedReason,
                elapsedThisSegment: CFAbsoluteTimeGetCurrent() - segmentStart)
            var stamped = s
            stamped.context = snapshot
            logger.info("LTX-2 #1479: yielded at chunk \(chunk), phase \(s.phase.rawValue), step \(s.stepIndex) (\(frames.count) frame(s) banked).")
            return .yielded(stamped)
        }
        /// A checkpoint taken at a free unwind point where no denoise loop has
        /// started — nothing to restore but the position (see
        /// `ltx2NotStartedFingerprint`).
        func notStartedCheckpoint(chunk: Int) -> LTX2ResumeState {
            LTX2ResumeState(
                videoLatents: MLXArray.zeros([1]), stepIndex: 0, sigmas: [],
                phase: .baseDenoise, chunkIndex: chunk,
                seed: request.seed &+ UInt64(chunk),
                audioLatents: nil, audioNoiseKey: nil,
                configFingerprint: ltx2NotStartedFingerprint)
        }

        // #1479 free unwind point: before the model load, the single most
        // expensive non-denoise phase. Nothing has been computed yet.
        if resume == nil, preemption?.isRaised == true {
            return checkpoint(notStartedCheckpoint(chunk: 0), chunk: 0,
                              frames: [], audio: nil, seedImage: nil)
        }

        // validate() has already rejected unsupported audio modes (i2v /
        // multi-chunk), so this is simply the request flag.
        let wantAudio = request.audio
        // #1479 (final review, minor 6): the end MUST be a `defer` scoped to
        // the load itself. With a plain trailing call, a throwing `load()`
        // (missing weights, OOM, a bad LoRA) left the `modelLoad` phase open
        // forever: `/v1/queue` then reported `phase: modelLoad` on an idle
        // server, and the preemption refusal guard projected remaining time
        // against a phantom phase. `LTX2PhaseTelemetry.end` removes the open
        // entry, so it is safe to call once per `begin` and a no-op after —
        // the `do` block scopes this one to exactly the load.
        // comfybox#322 (review r1): the model load is the single most expensive
        // non-denoise phase (tens of GB, tens of seconds, and uninterruptible
        // once it starts). An interrupt that arrived while this job waited its
        // turn, or during text encode, must not pay for it. #1479 already had a
        // free unwind point here for preemption; this is its cancellation twin.
        try Task.checkCancellation()
        telemetry?.begin(.modelLoad)
        do {
            defer { telemetry?.end(.modelLoad) }
            try load(loras: request.effectiveLoRAs, audio: wantAudio)
        }
        guard let pipeline, let tokenizer else { throw LTX2VideoError.weightsMissing(config.weightsDir) }

        // One greppable line per render: every Tier A/B param + provenance
        // (task #9 Phase 1). Invalid resolutions log separately and LOUDLY.
        // Phase 2: the TYPED resolution is authoritative — refresh it onto
        // the pipeline so render code reads it instead of raw env (Codex
        // finding #14). Request/preset tuning joins in the wire-format
        // increment; until then this resolves configFile > env > builtin.
        let typedConfig = LTX2ConfigResolver.resolveTyped(request: request.tuning, preset: request.presetTuning)
        pipeline.resolvedConfig = typedConfig
        // comfybox#307 (review r1): unconditionally reset — `pipeline
        // .lastRefineSkipReason` is now scoped to THIS render() invocation
        // only (it may be a brand-new pipeline instance after an eviction, or
        // the same instance carrying a stale value from an unrelated PRIOR
        // render). Cross-invocation persistence (surviving a resume) is the
        // `refineSkippedReason` local's job, seeded from `ctx` above and kept
        // in sync with this property after every chunk below.
        pipeline.lastRefineSkipReason = nil
        let resolved = typedConfig.params
        // Finding #18: two_stage was load-time only — a request could not turn
        // it on without a server restart. Lazy-load the upsampler on the first
        // request that resolves twoStage=true.
        if typedConfig.twoStage, pipeline.upsampler == nil,
           !typedConfig.upsamplerPath.isEmpty,
           FileManager.default.fileExists(atPath: typedConfig.upsamplerPath) {
            pipeline.upsampler = Self.loadUpsampler(path: typedConfig.upsamplerPath, logger: logger)
        }
        let summary = resolved.map { "\($0.name)=\($0.value)(\($0.source.rawValue))" }.joined(separator: " ")
        logger.info("[LTX2] effective-config: \(summary)")
        for p in resolved where !p.valid {
            logger.error("[LTX2] CONFIG REJECTED: \(p.name) — \(p.note ?? "invalid") — using \(p.value) (\(p.source.rawValue))")
        }

        let plan = Self.chunkPlan(
            framesPerChunk: request.framesPerChunk,
            extendToSeconds: request.extendToSeconds, fps: request.fps)

        // Prompt-conditioned audio used to disappear silently when callers
        // appended `audio:` after a long character/scene description: the
        // tokenizer keeps the head and drops the tail. (The 128 cap that made
        // this bite constantly was a port artifact, corrected to upstream's
        // 1024 on 2026-08-07 — the guard remains as the backstop for prompts
        // that exceed even the real cap, and for LTX2_GEMMA_MAX_LENGTH=128
        // rollback runs.)
        let guardedPrompt = try LTX2AudioPromptGuard.prepare(
            prompt: request.prompt,
            audio: wantAudio,
            maxLength: tokenizer.maxLength,
            tokenize: { tokenizer.untruncatedTokenIds(prompt: $0) })
        let audioMarkerIndex = guardedPrompt.audioMarkerTokenIndex.map(String.init) ?? "null"
        let promptFacts = [
            "[LTX2] prompt-truncation:",
            "pre_truncation_token_count=\(guardedPrompt.preTruncationTokenCount)",
            "audio_marker_token_index=\(audioMarkerIndex)",
            "quoted_line_present=\(guardedPrompt.quotedLinePresent)",
            "quoted_line_survived=\(guardedPrompt.quotedLineSurvived)",
            "effective_prompt_hash=\(guardedPrompt.effectivePromptHash)",
            "reordered=\(guardedPrompt.reordered)",
        ].joined(separator: " ")
        logger.info("\(promptFacts)")

        let batch = tokenizer.encode(prompt: guardedPrompt.effectivePrompt, maxLength: tokenizer.maxLength)
        MLX.eval(batch.inputIds, batch.attentionMask)

        // Negative prompt: tokenize when provided, or default to the PinkCherry
        // workflow negative when a CFG++ sampler is active (CFG++ requires a
        // negative pass every step even at cfg=1).
        let negText: String? = {
            if let n = request.negativePrompt, !n.isEmpty { return n }
            return pipeline.resolvedConfig.samplerIsCfgPP
                ? "subtitle, caption, text, text on screen, watermark, logo, timestamp, distorted sound, saturated sound, loud noises, static"
                : nil
        }()
        let negBatch = negText.map { tokenizer.encode(prompt: $0, maxLength: tokenizer.maxLength) }
        if let negBatch { MLX.eval(negBatch.inputIds, negBatch.attentionMask) }

        // Temporal beat scheduling (comfybox#310): locate each beat's text as
        // a token span in the SAME composed prompt just tokenized above, once
        // per render. Fail-open per beat (never per render) — a beat whose
        // span can't be located just contributes zero bias, logged once.
        // Absent field or the LTX2_BEAT_SCHEDULE=0 kill switch both resolve
        // to an empty list, which builds nil bias downstream (byte-identical
        // to before this feature existed).
        let resolvedBeats: [LTX2ResolvedBeat] = {
            guard typedConfig.beatScheduleEnabled,
                  let beats = request.beatSchedule, !beats.isEmpty else { return [] }
            let fullIds = tokenizer.untruncatedTokenIds(prompt: guardedPrompt.effectivePrompt)
            return LTX2BeatScheduleLocator.locate(
                beats: beats,
                fullPromptTokenIds: fullIds,
                maxLength: tokenizer.maxLength,
                onDrop: { beat, reason in
                    self.logger.warning("[LTX2] beat_schedule: dropping beat '\(beat.text.prefix(40))…' — \(reason)")
                },
                tokenize: { tokenizer.untruncatedTokenIds(prompt: $0) })
        }()
        if !resolvedBeats.isEmpty {
            logger.info("[LTX2] beat_schedule: \(resolvedBeats.count)/\(request.beatSchedule?.count ?? 0) beat(s) located.")
        }

        segmentStart = CFAbsoluteTimeGetCurrent()
        // #1479: frames and audio banked by chunks that finished before an
        // earlier preemption. Empty on a fresh render.
        var allFrames: [CGImage] = ctx.frames
        var audioLatents: MLXArray? = ctx.audioLatents

        // Center-crop a CGImage to the target aspect ratio, matching ComfyUI's
        // ImageScale crop="center" (workflow nodes 7 and 19). Our plain resize
        // STRETCHES on aspect mismatch — seed stills are often 9:16 (0.5625)
        // against 384x640 (0.6), a ~7% vertical squash that distorts the
        // conditioning content vs the workflow's crop.
        func centerCropped(_ cg: CGImage, targetW: Int, targetH: Int) -> CGImage {
            let srcW = Double(cg.width), srcH = Double(cg.height)
            let targetAspect = Double(targetW) / Double(targetH)
            let srcAspect = srcW / srcH
            var cropW = srcW, cropH = srcH
            if srcAspect > targetAspect {
                cropW = srcH * targetAspect
            } else {
                cropH = srcW / targetAspect
            }
            let rect = CGRect(
                x: ((srcW - cropW) / 2).rounded(.down),
                y: ((srcH - cropH) / 2).rounded(.down),
                width: cropW.rounded(), height: cropH.rounded())
            return cg.cropping(to: rect) ?? cg
        }

        // Conditioning compression (libx264 CRF), function-scoped so BOTH the
        // initial seed AND the chained continuation-chunk seeds get it. Chained
        // seeds that skip it condition on a PRISTINE generated frame = the
        // frozen-image regime → motion collapses across chunks (2026-07-26:
        // comp30 chunk1 action-zone 6.11, chunks 2-3 crash to 2.2 because the
        // chained frame was uncompressed). Higher = more motion.
        let conditioningCompression = request.imgCompression
            ?? pipeline.resolvedConfig.imgCompression

        // Seed image: the init image for I2V, else nil (T2V first chunk).
        var currentImage: MLXArray? = try request.initImagePath.map { path in
            let url = URL(fileURLWithPath: path)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let rawImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw LTX2VideoError.imageLoadFailed(path)
            }
            // Workflow order (nodes 7 -> 8): center-crop + scale to the render
            // size FIRST, then compression-preprocess at that size. Compressing
            // at native resolution and downscaling after (the old order)
            // changes the artifact character and stretches on aspect mismatch.
            var cgImage = try QwenImageIO.resizedCGImage(
                from: centerCropped(rawImage, targetW: request.width, targetH: request.height),
                width: request.width, height: request.height)
            // LTX conditioning preprocess (ComfyUI LTXVPreprocess, img_compression
            // = libx264 CRF): round-trip the still through lossy compression so
            // it carries codec-like artifacts. LTX is trained on VIDEO frames — a
            // pristine still is out-of-distribution and the model freezes it
            // (mannequin i2v, no locomotion). Measured on the same source/prompt/
            // seed: ComfyUI (with preprocess) motion 2.24 vs ours (raw PNG) 1.07.
            // LTX2_I2V_COMPRESSION=0 disables.
            let compression = conditioningCompression
            if compression > 0 {
                // Prefer a REAL H.264 round-trip (matches ComfyUI's libx264
                // preprocess artifact character); fall back to JPEG if the
                // encode fails for any reason.
                if let rt = try? LTX2PostProcess.h264RoundTrip(cgImage, compression: compression) {
                    cgImage = rt
                    logger.info("LTX-2 I2V: conditioning preprocess — H.264 round-trip (compression \(compression)).")
                } else {
                    let quality = max(0.05, 1.0 - Double(compression) / 100.0 * 1.4)
                    let jpeg = NSMutableData()
                    if let dest = CGImageDestinationCreateWithData(
                        jpeg as CFMutableData, "public.jpeg" as CFString, 1, nil) {
                        CGImageDestinationAddImage(dest, cgImage, [
                            kCGImageDestinationLossyCompressionQuality: quality
                        ] as CFDictionary)
                        if CGImageDestinationFinalize(dest),
                           let rtSource = CGImageSourceCreateWithData(jpeg as CFData, nil),
                           let rtImage = CGImageSourceCreateImageAtIndex(rtSource, 0, nil) {
                            cgImage = rtImage
                            logger.info("LTX-2 I2V: conditioning preprocess — JPEG fallback q=\(String(format: "%.2f", quality)) (compression \(compression)).")
                        }
                    }
                }
            }
            let pixels = try QwenImageIO.array(
                from: cgImage, addBatchDimension: true, dtype: .float32)
            return QwenImageIO.normalizeForEncoder(pixels)
        }

        // The ORIGINAL init image, kept for identity re-anchoring of
        // continuation chunks (currentImage is overwritten with each
        // chunk\u{27}s last frame).
        let sourceImage: MLXArray? = currentImage

        // Two-stage refine anchor: the RAW source (no compression preprocess)
        // at 2x the base resolution, mirroring workflow nodes 19/20 — the
        // refine re-anchors frame 0 to this for native high-res detail.
        let refineAnchorImage: MLXArray? = try {
            guard pipeline.resolvedConfig.twoStage,
                  let path = request.initImagePath,
                  let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
                  let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
            // Workflow node 19: center-crop + lanczos to 2x, RAW (no preprocess).
            let pixels = try QwenImageIO.resizedPixelArray(
                from: centerCropped(cgImage, targetW: request.width * 2, targetH: request.height * 2),
                width: request.width * 2, height: request.height * 2,
                addBatchDimension: true, dtype: .float32)
            return QwenImageIO.normalizeForEncoder(pixels)
        }()

        // Face-anchor (#partnered): detect faces on the source once, build a
        // latent-space mask so the denoise loop can hold EVERY face (esp. a
        // stationary partner) across a long pass. Env-gated: LTX2_FACE_ANCHOR_STRENGTH.
        // Face-region anchor defaults 0.5 for i2v — with IC-control it locks the
        // FACE across the render (IC-control alone holds body/scene but the face
        // drifts). Only engages when an init image is present. LTX2_FACE_ANCHOR_STRENGTH=0 disables.
        let faceAnchorStrength = pipeline.resolvedConfig.faceAnchorStrength
        var faceAnchorMask: MLXArray? = nil
        if faceAnchorStrength > 0, let path = request.initImagePath,
           let isrc = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
           let cg = CGImageSourceCreateImageAtIndex(isrc, 0, nil) {
            var rects = (try? RegionMaskUtilities.detectFaceRects(in: cg)) ?? []
            // Male-only anchor (LTX2_FACE_ANCHOR_MALE_ONLY=1, default on): the
            // FEMALE subject (Kira) is already identity-held by her seed + LoRA,
            // so anchoring her face is pure downside — it seams her skin at the
            // mask boundary and damps her action motion. Only the MALE partner
            // has no identity source and drifts. Heuristic: the largest face is
            // the foreground female subject; drop it, anchor the rest (the
            // peripheral male). Falls back to all-faces if only one detected.
            if (ProcessInfo.processInfo.environment["LTX2_FACE_ANCHOR_MALE_ONLY"] ?? "1") != "0" {
                if rects.count > 1 {
                    let largest = rects.max(by: { $0.width * $0.height < $1.width * $1.height })!
                    rects = rects.filter { $0 != largest }
                    logger.info("Face-anchor: male-only — anchoring \(rects.count) peripheral face(s), skipping primary subject.")
                } else if rects.count == 1 {
                    // Solo clip: the ONE detected face IS the primary subject. Anchoring
                    // it pins the seed frame's features as a static ghost overlay while
                    // the head moves (doubled nose/mouth — matched-seed A/B 2026-07-29:
                    // anchor-off arm was clean AND livelier) and damps facial animation.
                    // Identity is already held by the seed + LoRA; anchor nothing.
                    // Partnered clips (2+ faces) keep the peripheral-face anchor above.
                    rects = []
                    logger.info("Face-anchor: single face = primary subject; anchoring nothing (solo ghost fix 2026-07-29).")
                }
            }
            let latH = request.height / pipeline.spatialCompression
            let latW = request.width / pipeline.spatialCompression
            if !rects.isEmpty && latH > 0 && latW > 0 {
                var mask = [Float](repeating: 0, count: latH * latW)
                let pad: CGFloat = 0.35
                for r in rects {
                    let p = r.insetBy(dx: -r.width * pad, dy: -r.height * pad)
                    let x0 = max(0, Int(p.minX * CGFloat(latW)))
                    let x1 = min(latW, Int((p.minX + p.width) * CGFloat(latW) + 1))
                    // Vision rects are bottom-left origin; latent rows are top-origin -> flip Y.
                    let rowTop = max(0, Int((1.0 - (p.minY + p.height)) * CGFloat(latH)))
                    let rowBot = min(latH, Int((1.0 - p.minY) * CGFloat(latH) + 1))
                    if x1 > x0 && rowBot > rowTop {
                        for row in rowTop..<rowBot { for col in x0..<x1 { mask[row * latW + col] = 1 } }
                    }
                }
                // FEATHER the mask (LTX2_FACE_ANCHOR_FEATHER, default 2 cells):
                // the hard 0->1 binary edge created a visible skin-tone SEAM at
                // the mask boundary (the anchored region holds while the body
                // diverges, meeting at a line). Box-blur the mask to a smooth
                // falloff so anchoring fades gradually — no seam. Multiple 3-tap
                // passes approximate a Gaussian over the small latent grid.
                let feather = Int(ProcessInfo.processInfo.environment["LTX2_FACE_ANCHOR_FEATHER"] ?? "") ?? 2
                for _ in 0..<max(0, feather) {
                    var blurred = mask
                    for row in 0..<latH {
                        for col in 0..<latW {
                            var acc: Float = 0; var cnt: Float = 0
                            for dr in -1...1 { for dc in -1...1 {
                                let rr = row+dr, cc = col+dc
                                if rr>=0 && rr<latH && cc>=0 && cc<latW { acc += mask[rr*latW+cc]; cnt += 1 }
                            }}
                            blurred[row*latW+col] = acc/cnt
                        }
                    }
                    mask = blurred
                }
                faceAnchorMask = MLXArray(mask, [1, 1, 1, latH, latW])
                logger.info("Face-anchor: \(rects.count) face(s) detected, strength \(faceAnchorStrength)")
            }
        }

        // #1479: a continuation chunk's seed frame is a function of the PREVIOUS
        // chunk's decoded output, so it cannot be recomputed — it rides in the
        // context. Everything else above is a deterministic function of the
        // request and has just been rebuilt identically.
        if resume != nil, let chained = ctx.chunkSeedImage {
            currentImage = chained
        }
        let startChunk = resume != nil ? ctx.chunkIndex : 0

        for chunk in startChunk..<plan.totalChunks {
            // comfybox#322: chunk boundary. Cancellation is evaluated BEFORE
            // the #1479 unwind point below, so an interrupt arriving here
            // aborts instead of banking a checkpoint that would be resumed.
            try Task.checkCancellation()

            // The checkpoint being resumed belongs to `startChunk`; later chunks
            // start clean.
            let chunkResume: LTX2ResumeState? = (chunk == startChunk) ? resume : nil

            // #1479 free unwind point: between chunks. Skipped for the chunk we
            // are resuming INTO — re-checkpointing it here would throw away the
            // step progress the checkpoint we are holding already paid for.
            if chunkResume == nil, preemption?.isRaised == true {
                return checkpoint(notStartedCheckpoint(chunk: chunk), chunk: chunk,
                                  frames: allFrames, audio: audioLatents, seedImage: currentImage)
            }

            let chunkSeed = request.seed + UInt64(chunk)
            let outcome: LTX2PipelineOutcome
            if let image = currentImage {
                if chunk > 0, request.identityAnchorStrength > 0, let anchor = sourceImage {
                    // Continuation chunks drift chunk-by-chunk (each only sees
                    // the previous tail). Splice the original source in at the
                    // chunk\u{27}s last frame at reduced strength \u{2014} a soft pull
                    // back toward the subject \u{2014} while frame 0 stays the hard
                    // continuity anchor. First real caller of
                    // generateMultiKeyframe (see docs/ltx2-multi-keyframe-fdd.md).
                    outcome = try pipeline.generateMultiKeyframeResumable(
                        inputIds: batch.inputIds, attentionMask: batch.attentionMask,
                        keyframes: [
                            .init(image: image, videoFrameIndex: 0, strength: request.strength),
                            .init(image: anchor, videoFrameIndex: request.framesPerChunk - 1,
                                  strength: request.identityAnchorStrength),
                        ],
                        width: request.width, height: request.height,
                        numFrames: request.framesPerChunk, steps: request.steps, seed: chunkSeed,
                        guidance: request.guidance,
                        negativeInputIds: negBatch?.inputIds,
                        negativeAttentionMask: negBatch?.attentionMask,
                        preemption: preemption, telemetry: telemetry,
                        resume: chunkResume, chunkIndex: chunk,
                        progressCallback: { s, t in progress?(chunk, plan.totalChunks, s, t) })
                } else if request.identityAnchorStrength > 0,
                          request.identityReAnchorInterval > 0,
                          request.framesPerChunk > request.identityReAnchorInterval + 1,
                          let src = sourceImage {
                    // Single/long first pass: with only a frame-0 anchor, peripheral
                    // subjects (a partner's face) drift and melt over the pass. Re-splice
                    // the ORIGINAL source at fixed intervals at reduced strength — soft
                    // identity pulls that hold EVERY face across the whole pass without
                    // freezing motion. Same primitive as the continuation-chunk anchor.
                    var keyframes: [LTX2Pipeline.Keyframe] = [
                        .init(image: image, videoFrameIndex: 0, strength: request.strength)
                    ]
                    var f = request.identityReAnchorInterval
                    while f < request.framesPerChunk - 1 {
                        keyframes.append(.init(image: src, videoFrameIndex: f,
                                               strength: request.identityAnchorStrength))
                        f += request.identityReAnchorInterval
                    }
                    outcome = try pipeline.generateMultiKeyframeResumable(
                        inputIds: batch.inputIds, attentionMask: batch.attentionMask,
                        keyframes: keyframes,
                        width: request.width, height: request.height,
                        numFrames: request.framesPerChunk, steps: request.steps, seed: chunkSeed,
                        guidance: request.guidance,
                        negativeInputIds: negBatch?.inputIds,
                        negativeAttentionMask: negBatch?.attentionMask,
                        preemption: preemption, telemetry: telemetry,
                        resume: chunkResume, chunkIndex: chunk,
                        progressCallback: { s, t in progress?(chunk, plan.totalChunks, s, t) })
                } else {
                    outcome = try pipeline.generateI2VResumable(
                        inputIds: batch.inputIds, attentionMask: batch.attentionMask,
                        image: image, strength: request.strength,
                        width: request.width, height: request.height,
                        numFrames: request.framesPerChunk, steps: request.steps, seed: chunkSeed,
                        guidance: request.guidance,
                        negativeInputIds: negBatch?.inputIds,
                        negativeAttentionMask: negBatch?.attentionMask,
                        faceAnchorMask: chunk == 0 ? faceAnchorMask : nil,
                        faceAnchorStrength: faceAnchorStrength,
                        refineAnchorImage: chunk == 0 ? refineAnchorImage : nil,
                        audioSeconds: wantAudio && chunk == 0
                            ? Float(request.framesPerChunk) / Float(request.fps) : nil,
                        preemption: preemption, telemetry: telemetry,
                        resume: chunkResume, chunkIndex: chunk,
                        progressCallback: { s, t in progress?(chunk, plan.totalChunks, s, t) })
                }
            } else {
                outcome = try pipeline.generateT2VResumable(
                    inputIds: batch.inputIds, attentionMask: batch.attentionMask,
                    width: request.width, height: request.height,
                    numFrames: request.framesPerChunk, steps: request.steps, seed: chunkSeed,
                    // Every i2v variant above forwards request.guidance; this call
                    // omitted it, so the daemon's T2V_GUIDANCE=3.5 (Todd 2026-07-30)
                    // silently fell back to the distilled default 1.0 — t2v ran
                    // with CFG OFF in prod for three days (found 2026-08-02 while
                    // chasing extra limbs; cfg 1.0 under-drives t2v anatomy).
                    guidance: request.guidance,
                    negativeInputIds: negBatch?.inputIds,
                    negativeAttentionMask: negBatch?.attentionMask,
                    audioSeconds: wantAudio && chunk == 0
                        ? Float(request.framesPerChunk) / Float(request.fps) : nil,
                    preemption: preemption, telemetry: telemetry,
                    resume: chunkResume, chunkIndex: chunk,
                    beatSchedule: resolvedBeats,
                    progressCallback: { s, t in progress?(chunk, plan.totalChunks, s, t) })
            }

            let output: LTX2PipelineOutput
            switch outcome {
            case .completed(let o):
                output = o
            case .yielded(let s):
                // comfybox#307 (review r1): sync before checkpointing — this
                // chunk may have recorded a skip (the refine gate returns
                // `.completed` internally before any later phase yields) that
                // must ride the snapshot, not just live on `pipeline`.
                // (review r3, minor 2) Sync site 1 of 2 — reads whatever
                // `LTX2Pipeline.recordRefineSkip` last wrote to
                // `lastRefineSkipReason`; see that function's doc comment.
                refineSkippedReason = pipeline.lastRefineSkipReason ?? refineSkippedReason
                // #1479: propagate. Frames banked by EARLIER chunks ride in the
                // context; this chunk's own progress is in the checkpoint.
                return checkpoint(s, chunk: chunk, frames: allFrames,
                                  audio: audioLatents, seedImage: currentImage)
            }
            // comfybox#307 (review r1): this chunk finished cleanly — fold in
            // whatever it recorded so the NEXT chunk's between-chunk
            // checkpoint (above) and the final result (below) both see it.
            // (review r3, minor 2) Sync site 2 of 2 — same read as above; see
            // `LTX2Pipeline.recordRefineSkip`'s doc comment for both sites.
            refineSkippedReason = pipeline.lastRefineSkipReason ?? refineSkippedReason

            telemetry?.begin(.postProcess)
            let chunkFrames = LTX2PostProcess.framesToImages(from: output.decoded, colorAnchor: pipeline.resolvedConfig.colorAnchor)
            telemetry?.end(.postProcess)
            allFrames.append(contentsOf: chunk == 0 ? chunkFrames : Array(chunkFrames.dropFirst()))
            if let al = output.audioLatents { audioLatents = al }

            // Re-feed the last frame as the seed for the next continuation chunk.
            if chunk < plan.totalChunks - 1 {
                let t = output.decoded.dim(2)
                var lastFrame = output.decoded[0..., 0..., (t - 1)..<t, 0..., 0...].squeezed(axis: 2)
                // Two-stage refine decodes at 2x the base resolution. The
                // chained seed MUST be at base resolution: feeding the 2x frame
                // VAE-encodes to a conditioning latent at 2x latent dims and
                // applyConditioning fatalErrors on the shape mismatch —
                // instant process death, no log. EVERY multi-chunk crash of
                // 2026-07-25/26 (10:48, 01:28, 02:57, 03:16) died exactly here,
                // seconds after "Encoding N keyframe image(s)".
                if lastFrame.dim(2) != request.height || lastFrame.dim(3) != request.width {
                    let squeezed = lastFrame.squeezed(axis: 0)
                    let resized = try QwenImageIO.resize(
                        rgbArray: squeezed,
                        targetHeight: request.height, targetWidth: request.width)
                    // Lanczos overshoot can leave values outside [0,1]; the
                    // *2-1 normalization below would push them out of the
                    // VAE's expected range (Codex review 2026-07-26).
                    lastFrame = MLX.clip(resized.expandedDimensions(axis: 0), min: MLXArray(Float(0)), max: MLXArray(Float(1)))
                    logger.info("Chunk seed downscaled from refine resolution to \(request.width)x\(request.height) for chaining")
                }
                // Apply the SAME conditioning compression the initial seed got,
                // so continuation chunks read the chained frame as mid-motion
                // video (not a pristine still that freezes). Round-trip the
                // [0,1] frame through H.264; fall back to the raw frame if the
                // codec path fails (never block a render on the preprocess).
                if conditioningCompression > 0,
                   let cg = try? QwenImageIO.image(from: lastFrame),
                   let rt = try? LTX2PostProcess.h264RoundTrip(cg, compression: conditioningCompression),
                   let arr = try? QwenImageIO.array(from: rt, addBatchDimension: true, dtype: .float32) {
                    // array(from:) yields [1,3,H,W] in [0,1]; match lastFrame layout.
                    lastFrame = MLX.clip(arr, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
                    logger.info("Chunk seed: conditioning compression \(conditioningCompression) applied for chaining")
                }
                currentImage = lastFrame * 2.0 - 1.0
                MLX.eval(currentImage!)

                // Chunk-boundary drain (#34): the next chunk starts with the
                // previous chunk's decode intermediates (up to ~35GB at the
                // large formats) still in the MLX pool + lazily-reclaimed by
                // macOS. The per-JOB admission drain never sees this boundary.
                // A FIXED 3s settle proved insufficient at 12s/9:16 scale
                // (2026-07-26 01:28: chunk 1's 18,928-volume decode -> chunk 2
                // Metal-aborted 5s in). Drain adaptively like admission: drop
                // the pool and re-probe until real headroom exists, up to ~24s.
                // The free-memory probe LIES right after a large decode
                // (lazy reclaim): 2026-07-26 02:57 it reported 47GB free at
                // round 0, the threshold check passed with zero settling, and
                // chunk 2 Metal-aborted 7s later. Settle a minimum number of
                // rounds unconditionally, then keep going until real headroom.
                MLX.GPU.clearCache()
                var chunkFree = MemoryProbe.systemAvailableMemoryBytes()
                let chunkHeadroom: UInt64 = 40 * 1024 * 1024 * 1024
                var settleRounds = 0
                while settleRounds < 3 || (chunkFree < chunkHeadroom && settleRounds < 10) {
                    Thread.sleep(forTimeInterval: 3.0)
                    MLX.GPU.clearCache()
                    chunkFree = MemoryProbe.systemAvailableMemoryBytes()
                    settleRounds += 1
                }
                logger.info("Chunk-boundary drain: \(chunkFree >> 20)MB free after \(settleRounds) settle round(s) (#34)")
            }
        }

        // Use the ACTUAL decoded frame dimensions, not the request dims. With
        // two-stage refine on, frames come back at 2x (e.g. 448x768 -> 896x1536);
        // passing request.width/height here would downscale them and throw away
        // all the refine detail. framesToImages carries per-frame dims.
        let outW = allFrames.first?.width ?? request.width
        let outH = allFrames.first?.height ?? request.height

        // Audio decode (task #21): final audio latents -> 48kHz stereo via the
        // reference-parity codec chain, muxed as AAC. Decode failure degrades
        // to a video-only file rather than failing the render.
        var audioTrack: LTX2PostProcess.AudioTrack? = nil
        if let al = audioLatents {
            // comfybox#322: audio decode (VAE + BigVGAN/BWE vocoder) is a
            // single multi-second tensor pass with no inner loop, so this is
            // its boundary. Deliberately OUTSIDE the `do` below, whose `catch`
            // degrades to a video-only file — swallowing a cancel there would
            // let an interrupted render go on to write an MP4.
            try Task.checkCancellation()
            do {
                if audioVAE == nil {
                    logger.info("LTX-2 audio: binding audio VAE + vocoder from monolith…")
                    audioVAE = try LTX2AudioVAE.load(path: resolveWeightsFileURL().path, logger: logger)
                    // The monolith's `vocoder.*` BigVGAN+BWE is trained with its
                    // `audio_vae.*` and is therefore the 48 kHz default. The
                    // external Lightricks HiFi-GAN is a mismatched 24 kHz path;
                    // keep it available for experiments, but require explicit
                    // opt-in so a stale LTX2_VOCODER_PATH cannot replace the
                    // checkpoint's matched vocoder.
                    let env = ProcessInfo.processInfo.environment
                    if let vp = Self.externalVocoderOverridePath(environment: env),
                       let av = audioVAE {
                        av.externalVocoder = try LTX2HiFiGANVocoder.load(path: vp, logger: logger)
                        logger.warning("LTX-2 audio: external mismatched HiFi-GAN override bound from \(vp) (24 kHz, no BWE).")
                    } else if let vp = env["LTX2_VOCODER_PATH"], !vp.isEmpty {
                        logger.info("LTX-2 audio: ignoring LTX2_VOCODER_PATH=\(vp); bundled matched BigVGAN+BWE remains active. Set LTX2_USE_EXTERNAL_VOCODER=1 to opt in.")
                    }
                }
                if let av = audioVAE {
                    telemetry?.begin(.vocoder)
                    defer { telemetry?.end(.vocoder) }
                    // External HiFi-GAN outputs 24 kHz; bundled BigVGAN+BWE 48 kHz.
                    let audioSR = av.externalVocoder != nil ? 24000 : 48000
                    let wav = av.decodeToWaveform(al.asType(.float32))  // (1, 2, N)
                    var clamped = MLX.clip(wav[0], min: MLXArray(Float(-1)), max: MLXArray(Float(1)))
                    // Trim to the actual video duration (ceil(s*25) latent
                    // quantization overshoots; Codex #8). Shorter audio is
                    // left as-is — AAC tolerates a short tail.
                    let videoSamples = Int((Double(allFrames.count) / Double(request.fps) * Double(audioSR)).rounded(.up))
                    if clamped.dim(1) > videoSamples {
                        clamped = clamped[0..., 0..<videoSamples]
                    }
                    // In-engine mastering (task #26): rumble cut, BWE de-harsh,
                    // loudness raise with soft ceiling. The de-harsh dip targeted
                    // the BigVGAN path. LTX2_AUDIO_ENHANCE=0 disables it.
                    if ProcessInfo.processInfo.environment["LTX2_AUDIO_ENHANCE"] != "0" && av.externalVocoder == nil {
                        clamped = LTX2AudioEnhance.process(clamped, sampleRate: audioSR)
                        logger.info("LTX-2 audio: enhancement chain applied (hp50 + dip7.5k + loudnorm).")
                    }
                    eval(clamped)
                    audioTrack = LTX2PostProcess.AudioTrack(samples: clamped, sampleRate: audioSR)
                    logger.info("LTX-2 audio: decoded \(clamped.dim(1)) samples (\(String(format: "%.2f", Double(clamped.dim(1)) / Double(audioSR)))s stereo @\(audioSR)Hz).")
                }
            } catch {
                logger.error("LTX-2 audio: decode failed (\(error)) — writing video-only output.")
            }
        }

        // comfybox#401: the generation record — same schema as the PNG side's
        // EXIF `UserComment` JSON (see VideoGenerationRecord.swift). Built
        // BEFORE `writeMP4` so it can also ride as the mp4's own metadata
        // atom; the sidecar (mandatory) is written AFTER the mp4, so a render
        // that fails mid-write never leaves an orphaned sidecar next to a
        // file that doesn't exist. `deliveryDims` is what `writeMP4` actually
        // encodes — using `outW`/`outH` here would record the pre-delivery-
        // scale size on a downscaled delivery.
        let (deliveredW, deliveredH) = LTX2PostProcess.deliveryDims(
            width: outW, height: outH, shortEdge: pipeline.resolvedConfig.deliveryShortEdge)
        let generationRecord = VideoGenerationRecord.build(
            request: request,
            transformerFile: config.transformerFile,
            frameCount: allFrames.count,
            resolvedWidth: deliveredW,
            resolvedHeight: deliveredH,
            twoStageRequested: pipeline.resolvedConfig.twoStage,
            refineSkippedReason: refineSkippedReason,
            audioWritten: audioTrack != nil)
        let generationRecordJSON = try? generationRecord.encodeJSON()

        // comfybox#322: last boundary before anything is written to disk, so a
        // cancelled render never leaves a file at `outputPath` (the `defer`
        // above is the backstop for a cancel that lands mid-write).
        try Task.checkCancellation()
        telemetry?.begin(.postProcess)
        startedWrite = true
        try LTX2PostProcess.writeMP4(
            frames: allFrames, outputPath: request.outputPath,
            fps: request.fps, width: outW, height: outH,
            bitsPerPixelOverride: pipeline.resolvedConfig.videoBitsPerPx,
            audio: audioTrack,
            deliveryShortEdge: pipeline.resolvedConfig.deliveryShortEdge,
            generationRecordJSON: generationRecordJSON.flatMap { String(data: $0, encoding: .utf8) })
        wroteOutput = true
        telemetry?.end(.postProcess)

        // comfybox#401 ruling 2: the sidecar is MANDATORY — best-effort (never
        // fails a render that produced a real clip); `VideoSidecar.write`
        // logs its own failure.
        VideoSidecar.write(generationRecord, forMediaAt: request.outputPath)

        return .completed(LTX2VideoResult(
            outputPath: request.outputPath,
            frameCount: allFrames.count,
            durationSeconds: Float(allFrames.count) / Float(request.fps),
            // #1479: RENDER time, summed across segments — wall clock from a
            // single start would bill the preemptor's runtime to this render.
            elapsedSeconds: ctx.accumulatedSeconds + max(0, CFAbsoluteTimeGetCurrent() - segmentStart),
            // comfybox#307 (review r1): the local, accumulated across every
            // chunk (and any cold resume) this render() call processed — NOT
            // `pipeline.lastRefineSkipReason`, which only ever reflects THIS
            // invocation and would drop an earlier chunk's reason if a
            // preemption rebuilt the pipeline in between.
            refineSkippedReason: refineSkippedReason,
            generationRecord: generationRecord
        ))
        #else
        throw LTX2VideoError.unsupportedPlatform
        #endif
    }
}
