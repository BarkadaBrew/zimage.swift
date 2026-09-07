import Foundation
import Dispatch
import Logging
import Network
import Darwin
import MLX
import CoreGraphics
import ImageIO

/// #1479: refuse a preemption when finishing beats preempting. INERT until
/// telemetry has samples for both `meanStepSec` AND `evictReloadRoundTripSec`
/// — it never refuses on a guess (spec: "until telemetry has samples for a
/// family, the guard is inert rather than refusing"). `remainingPhaseMeansSec`
/// is the observed mean duration of each phase still ahead of the render's
/// current position (e.g. decode/vocoder/postProcess when mid-denoise) —
/// phases with no samples yet are simply omitted by the caller, not zeroed.
///
/// Returns nil to allow the preemption; a non-nil value is the projected
/// remaining seconds (the ETA a refused caller can report).
func preemptionRefusalETA(
  stepsRemaining: Int, meanStepSec: Double?,
  remainingPhaseMeansSec: [Double],
  evictReloadRoundTripSec: Double?
) -> Double? {
  guard let stepSec = meanStepSec, let roundTrip = evictReloadRoundTripSec else { return nil }
  let projected = Double(stepsRemaining) * stepSec + remainingPhaseMeansSec.reduce(0, +)
  return projected < roundTrip ? projected : nil
}

public struct WarmServerConfiguration: Sendable {
  public var port: UInt16
  public var modelSpec: String?
  public var textEncoderPath: String?
  public var initialLoRAs: [LoRAConfiguration]
  public var forceTransformerOverrideOnly: Bool
  public var maxSequenceLength: Int
  public var maxPendingRequests: Int
  /// Separate cap for MUTATING pool operations waiting in the FIFO
  /// (`/v1/model/load|activate|unload`), counted independently of
  /// `maxPendingRequests` (WP-E8 review, finding 1).
  ///
  /// Model ops deliberately do not sit under the render capacity gate — a
  /// paused engine full of parked renders must still accept the operator
  /// action that frees the GPU. "Bounded in number" was an assumption about
  /// the caller, though, and these routes are unauthenticated: repeated
  /// `wait: false` loads could grow the queue without limit. This is the
  /// bound that makes the sentence true.
  public var maxPendingModelOps: Int
  public var allowedOutputDirectory: String
  /// Path to SeedVR2 upscale model weights directory.
  /// When set, enables upscale via the ComfyUI bridge. The pipeline is lazy-loaded
  /// on first upscale request to avoid the ~6GB memory cost until needed.
  public var seedvr2WeightsPath: String?
  /// Path to the LTX-2 weights directory (transformer / VAE / connector).
  /// When set (with `ltx2GemmaPath`), enables LOCAL video generation on
  /// /v1/video/generate. Lazy-loaded on first request (~38GB), so it's off
  /// until a video is requested.
  public var ltx2WeightsPath: String?
  /// Gemma-3 tokenizer + text-encoder snapshot dir for LTX-2.
  public var ltx2GemmaPath: String?
  /// Default LoRA ("path" or "path@scale") merged into every LOCAL video
  /// render when the request carries none — lets preset-only callers (daemon
  /// MCP) get e.g. a distill LoRA required by a non-distilled checkpoint.
  public var ltx2DefaultLoRA: String?
  /// Explicit CivitAI API key — the top tier of `CivitAISecrets.resolve`'s
  /// resolution order (--civitai-key flag > CIVITAI_API_KEY env > Keychain).
  /// nil here just means "no explicit override"; the /v1/civitai/* routes
  /// still fall through to env/Keychain before giving up (#234).
  public var civitaiApiKey: String?

  public init(
    port: UInt16 = ComfyBoxServerConfig.canonicalPort,
    modelSpec: String? = nil,
    textEncoderPath: String? = nil,
    initialLoRAs: [LoRAConfiguration] = [],
    forceTransformerOverrideOnly: Bool = false,
    maxSequenceLength: Int = 512,
    maxPendingRequests: Int = 10,
    maxPendingModelOps: Int = 8,
    allowedOutputDirectory: String = FileManager.default.currentDirectoryPath,
    seedvr2WeightsPath: String? = nil,
    ltx2WeightsPath: String? = nil,
    ltx2GemmaPath: String? = nil,
    ltx2DefaultLoRA: String? = nil,
    civitaiApiKey: String? = nil
  ) {
    self.port = port
    self.modelSpec = modelSpec
    self.textEncoderPath = textEncoderPath
    self.initialLoRAs = initialLoRAs
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
    self.maxSequenceLength = maxSequenceLength
    self.maxPendingRequests = max(1, maxPendingRequests)
    self.maxPendingModelOps = max(1, maxPendingModelOps)
    self.allowedOutputDirectory = allowedOutputDirectory
    self.seedvr2WeightsPath = seedvr2WeightsPath
    self.ltx2WeightsPath = ltx2WeightsPath
    self.ltx2GemmaPath = ltx2GemmaPath
    self.ltx2DefaultLoRA = ltx2DefaultLoRA
    self.civitaiApiKey = civitaiApiKey
  }
}

/// Model family used by the warm server to route generation to the correct pipeline.
enum WarmModelFamily: String, Sendable, CaseIterable {
  case flux1
  case flux2
  case fibo
  case chroma
  case krea2
}

enum WarmServerOutputPathValidator {
  static func resolveOutputPath(_ outputPath: String, allowedOutputDirectory: String) throws -> URL {
    let allowedURL = canonicalFileURL(for: allowedOutputDirectory)
    let outputURL = canonicalFileURL(for: outputPath)

    guard outputURL.isContained(in: allowedURL) else {
      throw WarmServerError.invalidOutputPath(path: outputURL.path, allowedDirectory: allowedURL.path)
    }

    return outputURL
  }

  private static func canonicalFileURL(for path: String) -> URL {
    let expandedPath = (path as NSString).expandingTildeInPath
    let absolutePath: String
    if expandedPath.hasPrefix("/") {
      absolutePath = expandedPath
    } else {
      absolutePath = (FileManager.default.currentDirectoryPath as NSString)
        .appendingPathComponent(expandedPath)
    }

    return resolvePathComponents(in: absolutePath)
  }

  private static func resolvePathComponents(in path: String, symlinkDepth: Int = 0) -> URL {
    let fileManager = FileManager.default
    var currentURL = URL(fileURLWithPath: "/")

    for component in (path as NSString).pathComponents.dropFirst() {
      switch component {
      case "", ".":
        continue
      case "..":
        currentURL = currentURL.deletingLastPathComponent()
      default:
        let nextURL = currentURL.appendingPathComponent(component)
        if let destination = try? fileManager.destinationOfSymbolicLink(atPath: nextURL.path),
           symlinkDepth < 32 {
          let destinationPath: String
          if destination.hasPrefix("/") {
            destinationPath = destination
          } else {
            destinationPath = (currentURL.path as NSString).appendingPathComponent(destination)
          }
          currentURL = resolvePathComponents(in: destinationPath, symlinkDepth: symlinkDepth + 1)
        } else if fileManager.fileExists(atPath: nextURL.path) {
          currentURL = nextURL.resolvingSymlinksInPath()
        } else {
          currentURL = nextURL
        }
      }
    }

    return currentURL
  }
}

private extension URL {
  func isContained(in directory: URL) -> Bool {
    let pathComponents = standardizedFileURL.pathComponents
    let directoryComponents = directory.standardizedFileURL.pathComponents
    guard pathComponents.count >= directoryComponents.count else { return false }
    return Array(pathComponents.prefix(directoryComponents.count)) == directoryComponents
  }
}

public final class WarmServer {
  private static let pngSignature: [UInt8] = [137, 80, 78, 71, 13, 10, 26, 10]

  private let configuration: WarmServerConfiguration
  private let host: String
  private let logger: Logger
  private let coordinator: WarmServerCoordinator
  /// Submit/poll tracker for async image generation (GH: queue-submit —
  /// see ImageJobTracker's doc comment for the incident that motivated it).
  private let imageJobTracker = ImageJobTracker()
  /// Tracks async-submitted LOCAL LTX-2 video jobs (submit → 202 + jobId, poll
  /// GET /v1/video/status/{id}). Mirrors `imageJobTracker` so a multi-minute
  /// local render never holds an HTTP connection open. The Replicate cloud path
  /// keeps its own tracker inside `replicateVideoProxy`.
  private let videoJobTracker = VideoJobTracker()
  /// Cached, periodically-refreshed verdict on LOCAL LTX-2 disk readiness
  /// (#298: advertise it on `/health`). `/health` reads only
  /// `localVideoReadinessMonitor.current()` — filesystem work happens on a
  /// background task, never on the request path (review finding 3).
  private let localVideoReadinessMonitor: LocalVideoReadinessMonitor
  private let renderTraceStore = RenderTraceStore()
  /// #339: in-progress + remaining-count for `recoverPersistedQueue()`'s
  /// background replay. Gates submission of queue-job kinds that are never
  /// persisted (see QueueRecoveryGate.swift) so a job that cannot survive a
  /// second restart is refused with a retryable 503 (naming an estimated
  /// `retry_after_seconds`) instead of silently lost.
  private let queueRecoveryState = QueueRecoveryState()

  /// 0.B-1 (v2.3 rework, FDD-ui-api-parity.md §3.1.3, comfybox#300): lifts the
  /// async-internals route handlers (`/v1/enhance`, `/v1/civitai/search`,
  /// `/v1/civitai/harvest`) off the Swift cooperative pool, so they keep
  /// answering while a render saturates it (measured: 2964/2972 samples in
  /// `__psynch_cvwait` on the pool during a render). v2.2 tried this on the
  /// RENDER side instead and crashed LTX (native MLX mutex EINVAL from
  /// cross-thread eval migration) — deleted. These route handlers only ever
  /// `await` on network/disk/actor I/O (`PromptOptimizer.optimize`,
  /// `CivitAIClient.searchModels`, `CivitAIHarvestRunner.run`); they are
  /// verified MLX-free (§3.1.3), so thread migration is harmless and a plain
  /// concurrent `RouteTaskExecutor` is correct and safe. `nil` on macOS <15
  /// or with `COMFYBOX_RENDER_TASK_EXECUTOR=0`; `respondOnRouteExecutor`
  /// falls back to running the handler inline (today's pre-0.B-1 behavior,
  /// on whatever executor `respond(to:)` itself is running on) in that case.
  ///
  /// Typed `Any?` for the same reason the deleted `RenderTaskExecutor`
  /// property was: `RouteTaskExecutor` conforms to `TaskExecutor`, which is
  /// `@available(macOS 15.0, *)`, so a STORED property of that type would
  /// force this class's availability down with it, below the package's
  /// macOS 14 floor (`Package.swift:6`).
  private let routeTaskExecutor: Any? = {
    guard #available(macOS 15.0, *), RouteTaskExecutorFlag.isEnabled else { return nil }
    return RouteTaskExecutor()
  }()

  let comfyBridge: ComfyBridge

  /// Imported ComfyUI workflows (#238), file-backed at ~/.comfybox/workflows/.
  let workflowStore = WorkflowStore()
  private let listenerQueue = DispatchQueue(label: "z-image.warm-server.listener")
  private let lifecycleLock = NSLock()
  private var listener: NWListener?
  private var shutdownSignalled = false

  /// Lazy-loaded SeedVR2 upscale pipeline. Created on first upscale request
  /// to avoid the ~6GB memory cost until actually needed.
  private var seedvr2Pipeline: SeedVR2Pipeline?
  /// Resolved path to SeedVR2 weights directory.
  private let seedvr2WeightsPath: String?

  /// Lazy-loaded ESRGAN upscale pipeline. Created on first ESRGAN upscale request.
  private var esrganPipeline: ESRGANPipeline?

  /// Serializes lazy initialization of the upscale pipelines. WarmServer is a
  /// plain class reached from concurrent request tasks — without this lock,
  /// simultaneous first-use requests could double-load multi-GB pipelines.
  private let upscalePipelineLock = NSLock()

  /// Replicate video proxy — handles video generation via Replicate API.
  /// Initialized at startup if REPLICATE_API_TOKEN is available; nil otherwise.
  private var replicateVideoProxy: ReplicateVideoProxy?

  /// LoRA Library — indexes, queries, and manages LoRA adapter files.
  /// Initialized at startup; auto-scans if no library.json exists.
  private var loraLibrary: LoRALibrary?

  /// Default upscale models directory path — ESRGAN weights are stored here.
  private static let upscaleModelsDirectoryPath = ("~/bin/zimage/upscale_models" as NSString).expandingTildeInPath

  // MARK: - Creative-layer stores
  //
  // Feature parity with the Coffee Shop image service's creative subsystems. Each persists
  // to a JSON file under ~/.comfybox/ (characters.json, presets.json, content-modes.json,
  // audit-log.jsonl). Constructed eagerly so the first request has warm data; they are cheap
  // (small JSON loads) and thread-safe internally (CharacterStore is an actor; PresetStore /
  // AuditLog guard with a lock / serial queue; ContentModeStore is a value type).

  /// Character registry (~/.comfybox/characters.json).
  let characterStore = CharacterStore()
  /// Nearline model/LoRA catalog (attached storage staged on demand).
  let nearlineLibrary = NearlineLibrary()
  /// Local LTX-2 video generator, built lazily when the weights are configured.
  /// Held in a shared, lock-based box so the coordinator can evict it before an
  /// image load — image + video cannot co-reside in unified memory (#218).
  let videoHolder = VideoGeneratorHolder()

  // MARK: - #1479 preemption support
  //
  // Lock-based (see the block comment above `RollingMeanSec`), installed on
  // every LTX-2 generator instance (fresh or reused) in `prepareLocalVideo`,
  // so a video render is ALWAYS preemptible-capable but never actually
  // preempted unless a job raises `ltx2PreemptionSignal`.

  /// Per-phase render timings feeding the refusal guard and `/v1/queue`.
  let ltx2Telemetry = LTX2PhaseTelemetry()
  /// Raised by an image job's route handler to checkpoint the in-flight video
  /// render; read inside the render loop with no actor hop.
  let ltx2PreemptionSignal = PreemptionSignal()
  /// Live steps-remaining of the in-flight video render (fed from its
  /// progress callback) — the refusal guard's `stepsRemaining` input.
  let ltx2StepPosition = LTX2StepPosition()
  /// Observed evict/reload durations from past preemption episodes — the
  /// refusal guard's `evictReloadRoundTripSec` input (nil until both have a
  /// sample).
  let ltx2EvictMean = RollingMeanSec()
  let ltx2ReloadMean = RollingMeanSec()
  /// Exactly one preemption in flight at a time — a preemptor cannot itself
  /// be preempted (spec).
  let preemptionInFlight = LockedFlag()
  /// Single-slot mailbox bridging the route handler (raises the signal, then
  /// awaits a continuation) to the coordinator's `.localVideo` case (observes
  /// the yield, runs the image job, resumes the video) — see the mechanism
  /// note above `RollingMeanSec`.
  let pendingPreemptorBox = PendingPreemptorBox()

  /// Unified-memory pressure monitor (#218). On warning/critical it sheds the
  /// MLX buffer cache and any idle heavy model to stay clear of jetsam.
  private var memoryPressureSource: DispatchSourceMemoryPressure?
  /// Auto-rescans the LoRA library on any external filesystem change (CivitAI
  /// browser download, curl, cp, an MCP/Bree fetch) so new LoRAs are indexed
  /// without a manual `lora scan`. Started in `run()` once `loraLibrary` exists.
  private var loraLibraryWatcher: LoRALibraryWatcher?
  /// Lock-based health snapshot the coordinator publishes to, so GET /health is
  /// served without hopping onto the actor — stays responsive during a render (#217).
  private let liveHealth = LiveHealthState()
  /// comfybox#283/#217: read-only, append-only record of queue-job lifecycle
  /// transitions (enqueued/admitted/started/progress/checkpointed/resumed/
  /// interrupted/completed/failed/replayed-after-restart/dropped). Shared
  /// with the coordinator exactly like `liveHealth` — lock-based, no actor
  /// hop needed to read it from `GET /v1/queue/lifecycle`. See
  /// QueueLifecycleLedger.swift's file doc comment.
  let lifecycleLedger = QueueLifecycleLedger()
  /// Generation presets (~/.comfybox/presets.json). Seeds defaults on first run.
  let presetStore = PresetStore()
  /// Content-mode definitions (~/.comfybox/content-modes.json). Built-ins ship in-code.
  let contentModeStore = ContentModeStore.loadOrCreate()
  /// Append-only audit trail (~/.comfybox/audit-log.jsonl).
  let auditLog = AuditLog()
  /// Server stats + memory-pressure sampler (pure logic; live probes isolated).
  private let statsProvider = StatsProvider()
  /// Server start time, for the /v1/stats uptime figure.
  private let serverStartTime = Date()

  public init(
    configuration: WarmServerConfiguration,
    host: String = "127.0.0.1",
    logger: Logger = Logger(label: "z-image.warm-server")
  ) {
    self.configuration = configuration
    self.host = host
    self.logger = logger
    self.localVideoReadinessMonitor = LocalVideoReadinessMonitor(
      weightsPath: configuration.ltx2WeightsPath,
      gemmaPath: configuration.ltx2GemmaPath,
      upsamplerPath: ProcessInfo.processInfo.environment["LTX2_UPSAMPLER_PATH"])
    self.localVideoReadinessMonitor.start()
    self.coordinator = WarmServerCoordinator(
      configuration: configuration, logger: logger, videoHolder: self.videoHolder, liveHealth: self.liveHealth,
      videoJobTracker: self.videoJobTracker, ltx2Telemetry: self.ltx2Telemetry,
      ltx2PreemptionSignal: self.ltx2PreemptionSignal, ltx2StepPosition: self.ltx2StepPosition,
      ltx2EvictMean: self.ltx2EvictMean, ltx2ReloadMean: self.ltx2ReloadMean,
      preemptionInFlight: self.preemptionInFlight, pendingPreemptorBox: self.pendingPreemptorBox,
      lifecycleLedger: self.lifecycleLedger)
    self.seedvr2WeightsPath = configuration.seedvr2WeightsPath

    self.comfyBridge = ComfyBridge(logger: logger)

    // Initialize the LoRA Library. The library root defaults to ~/Models/loras/
    // (via COMFYBOX_MODELS env or LoRALibrary default). If no library.json exists,
    // the first API call to /v1/loras/scan will create it.
    do {
      let library = try LoRALibrary(logger: logger)
      self.loraLibrary = library
      Task { await coordinator.setLoraLibrary(library) }

      // Auto-scan if no library.json exists yet (first run).
      if library.count == 0 {
        logger.info("LoRA Library: no index found, running initial scan...")
        let result = try library.scan()
        logger.info("LoRA Library: initial scan complete — \(result.added) LoRAs indexed")
      } else {
        logger.info("LoRA Library: loaded \(library.count) entries from index")
      }

      // Wire the library into the ComfyBridge for LoRA discovery.
      comfyBridge.loraLibrary = library
    } catch {
      logger.warning("LoRA Library: failed to initialize — \(error.localizedDescription). LoRA API endpoints will return 503.")
    }

    // Initialize Replicate video proxy if API key is available.
    if let replicateKey = ProcessInfo.processInfo.environment["REPLICATE_API_TOKEN"], !replicateKey.isEmpty {
      self.replicateVideoProxy = ReplicateVideoProxy(
        apiKey: replicateKey,
        allowedOutputDirectory: configuration.allowedOutputDirectory,
        logger: logger
      )
      logger.info("Video proxy: enabled (Replicate)")
    } else {
      self.replicateVideoProxy = nil
      logger.info("Video proxy: disabled (no API key)")
    }

    // Wire up the upscale handler. ESRGAN models are always available (lazy-loaded from
    // ~/bin/zimage/upscale_models/); SeedVR2 additionally requires a configured weights path.
    let upscaleHandler: ComfyBridgeUpscaleHandler? = { [unowned self] (imageData: Data, modelName: String, progressCallback: ComfyBridgeProgressHandler?) async throws -> ComfyBridgeGenerateResult in
      try await self.bridgeUpscale(imageData: imageData, modelName: modelName, progressCallback: progressCallback)
    }

    self.comfyBridge.configureExecutor(
      generateHandler: { [unowned self] request, progressCallback, latentPreviewCallback in
        try await self.bridgeGenerate(request, progressCallback: progressCallback, latentPreviewCallback: latentPreviewCallback)
      },
      upscaleHandler: upscaleHandler
    )

    // Wire queue status provider and clear handler for ComfyUI /queue endpoint.
    self.comfyBridge.queueStatusProvider = { [unowned self] in
      await self.coordinator.queueStatus()
    }
    self.comfyBridge.queueClearHandler = { [unowned self] in
      let cleared = await self.coordinator.clearPending()
      self.logger.info("ComfyBridge: cleared \(cleared) pending job(s) from queue")
    }

    // Wire model switch handler for Krita checkpoint auto-detection.
    // When Krita sends a workflow with a different checkpoint, this handler
    // checks if the model is already in the pool (activate) or needs loading.
    // The switch runs through the coordinator's FIFO render queue so the pool
    // load/activate cannot mutate the active pipeline while a queued render
    // is mid-flight.
    self.comfyBridge.modelSwitchHandler = { [unowned self] (modelId: String) async throws -> Bool in
      return try await self.coordinator.enqueueModelSwitch { [unowned self] in
        // Check if this model is already active — no switch needed.
        let currentActive = await self.coordinator.modelPool.activeModelId()
        let requestedKey = ModelPool.poolKey(for: modelId)
        let isNoOpSwitch = currentActive == requestedKey
        // #339 review r3, item 2: the gate lives HERE, after the no-op
        // check — r2's version threw before this check ever ran, so every
        // Krita prompt carrying a checkpoint node hard-failed during
        // recovery even when the active model already matched and no
        // switch was needed at all. A model switch closes over live
        // in-memory state — never persisted (QueueRecoveryGate.swift) — so
        // an ACTUAL switch is still refused while a persisted-queue replay
        // is in flight, rather than risk a pool mutation a second restart
        // could leave half-applied. #339 review r4, item 2 (comment fix):
        // the caller (`handlePrompt`/`submitWorkflowGraph`) does NOT simply
        // log-and-continue on this specific error — `ModelSwitchFailurePolicy`
        // (review r2, item 1) recognizes `.queueRecoveryInProgress` and
        // FAILS the prompt outright via `ComfyBridgeExecutor.failPrompt`,
        // so a refused switch never silently renders on the wrong
        // checkpoint. Log-and-continue remains the behavior for every
        // OTHER model-switch failure.
        let switchRecovery = self.queueRecoveryState.snapshot()
        if ModelSwitchGate.shouldReject(isNoOpSwitch: isNoOpSwitch, recoveryInProgress: switchRecovery.inProgress) {
          throw WarmServerError.queueRecoveryInProgress(retryAfterSeconds:
            QueueRecoveryGate.retryAfterSeconds(remainingKinds: switchRecovery.remainingKinds))
        }
        if isNoOpSwitch {
          return false
        }

        // Check if the model is already in the pool — just activate it (instant).
        if let existing = await self.coordinator.modelPool.findEntry(for: modelId) {
          try await self.coordinator.poolActivate(modelId: existing.id)
          self.logger.info("ComfyBridge: activated pool model '\(existing.id)' for Krita checkpoint switch")
          return true
        }

        // Model not in pool — load and activate it.
        let quantization = Self.parseQuantization(from: modelId)
        let modelSpec = Self.parseModelSpec(from: modelId)
        let result = try await self.coordinator.poolLoad(modelSpec: modelSpec, quantization: quantization, activate: true)
        self.logger.info("ComfyBridge: loaded + activated '\(result.model)' (\(result.loadTimeMs)ms) for Krita checkpoint switch")
        return true
      }
    }

    // Wire interrupt handler so ComfyUI /interrupt cancels the in-flight render
    // task — the pipelines observe cancellation in their denoise loops.
    self.comfyBridge.interruptHandler = { [unowned self] in
      if case .cancelled = await self.coordinator.cancelActiveRender() { return true }
      return false
    }
  }

  public func run() throws {
    // Ignore SIGHUP before model loading — prevents SSH disconnect from
    // killing the daemon during the ~40s pipeline initialization phase.
    signal(SIGHUP, SIG_IGN)

    // Task #19: lifecycle traces. Recovery first — any trace left open by a
    // crash/kill is marked abandoned so failed renders are never invisible.
    videoJobTracker.traceStore = renderTraceStore
    renderTraceStore.markAbandonedOpenTraces()

    // Merge the legacy Coffee Shop image-service character registry (source of
    // truth for hand-written character text) before serving. Idempotent: only
    // missing or never-edited entries change, so user edits are never clawed back.
    let store = characterStore
    let migrationLogger = logger
    Task {
      let migrated = await store.importLegacyRegistry()
      if migrated > 0 {
        migrationLogger.info("Characters: merged \(migrated) entries from legacy image-service registry")
      }
    }

    // Same idempotent one-time merge for the old image-service presets.
    let importedPresets = presetStore.importLegacyImageService()
    if importedPresets > 0 {
      logger.info("Presets: imported \(importedPresets) from legacy image-service")
    }

    try preparePipeline()

    recoverPersistedQueue()

    guard let port = NWEndpoint.Port(rawValue: configuration.port) else {
      throw WarmServerError.invalidPort(configuration.port)
    }

    // Handle SIGTERM for clean launchd stop/restart.
    let sigTermSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: listenerQueue)
    signal(SIGTERM, SIG_IGN)
    sigTermSource.setEventHandler { [weak self] in
      self?.logger.info("Received SIGTERM, shutting down gracefully...")
      self?.initiateShutdown()
    }
    sigTermSource.resume()

    // Handle SIGINT for clean Ctrl-C during development.
    let sigIntSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: listenerQueue)
    signal(SIGINT, SIG_IGN)
    sigIntSource.setEventHandler { [weak self] in
      self?.logger.info("Received SIGINT, shutting down...")
      self?.initiateShutdown()
    }
    sigIntSource.resume()

    let endpoint = NWEndpoint.hostPort(host: NWEndpoint.Host(host), port: port)
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = endpoint
    let listener = try NWListener(using: parameters)
    self.listener = listener

    listener.stateUpdateHandler = { [weak self] state in
      self?.handleListenerState(state)
    }
    listener.newConnectionHandler = { [weak self] connection in
      self?.accept(connection: connection)
    }

    // Video job pruning timer — clean up completed jobs older than 1 hour.
    if replicateVideoProxy != nil {
      let pruneTimer = DispatchSource.makeTimerSource(queue: listenerQueue)
      pruneTimer.schedule(deadline: .now() + 600, repeating: 600)  // Every 10 minutes
      pruneTimer.setEventHandler { [weak self] in
        self?.replicateVideoProxy?.pruneCompletedJobs()
      }
      pruneTimer.resume()
    }

    // Same cleanup for async image generation jobs (queue-submit).
    let imageJobPruneTimer = DispatchSource.makeTimerSource(queue: listenerQueue)
    imageJobPruneTimer.schedule(deadline: .now() + 600, repeating: 600)
    imageJobPruneTimer.setEventHandler { [weak self] in
      self?.imageJobTracker.pruneCompleted()
      self?.videoJobTracker.pruneCompleted()
    }
    imageJobPruneTimer.resume()

    // Unified-memory pressure guard (#218). Loading LTX-2 outside the pool used
    // to co-reside with an image model and trip OS_REASON_JETSAM. As a
    // last-resort backstop to the single-heavy-model residency logic, when the
    // kernel signals memory pressure we drop the MLX buffer cache and release
    // any idle heavy model (a resident-but-not-rendering LTX-2 stack first,
    // then the LRU inactive image model) — never disturbing an in-flight render.
    let pressureSource = DispatchSource.makeMemoryPressureSource(
      eventMask: [.warning, .critical], queue: listenerQueue)
    pressureSource.setEventHandler { [weak self] in
      guard let self else { return }
      let event = pressureSource.data
      let level = event.contains(.critical) ? "critical" : "warning"
      self.logger.warning("Memory pressure: \(level) — shedding caches/idle heavy models (#218)")
      // Always drop the MLX buffer cache; cheap and often enough.
      GPU.clearCache()
      // Release an idle (not mid-render) LTX-2 stack — the single biggest chunk.
      if self.videoHolder.releaseIfIdle() {
        self.logger.warning("Memory pressure: released idle LTX-2 video stack")
      }
      // On critical, also shed the LRU inactive image model from the pool.
      if event.contains(.critical) {
        Task { [weak self] in
          guard let self else { return }
          let freed = await self.coordinator.shedInactivePoolModelUnderPressure()
          if freed > 0 {
            self.logger.warning("Memory pressure: evicted LRU inactive image model (~\(freed)MB)")
          }
        }
      }
    }
    pressureSource.resume()
    self.memoryPressureSource = pressureSource

    if let library = loraLibrary {
      self.loraLibraryWatcher = LoRALibraryWatcher(library: library, queue: listenerQueue, logger: logger)
    }

    listener.start(queue: listenerQueue)

    // Use dispatchMain() instead of semaphore.wait() for daemon reliability.
    // DispatchSemaphore.wait() blocks without processing GCD events, which
    // causes NWListener to enter .cancelled state when the process loses its
    // controlling terminal (SSH disconnect, launchd restart, nohup).
    // dispatchMain() keeps the main dispatch loop alive properly.
    dispatchMain()
  }

  private func preparePipeline() throws {
    let result = SyncResult<Void>()
    Task {
      do {
        try await coordinator.prepare()
        result.succeed(())
      } catch {
        result.fail(error)
      }
    }
    try result.wait()
  }

  private func handleListenerState(_ state: NWListener.State) {
    switch state {
    case .ready:
      logger.info("Warm server listening on http://\(self.host):\(self.configuration.port)")
    case .failed(let error):
      logger.error("Warm server listener failed: \(error.localizedDescription)")
      if case .posix(.EADDRINUSE) = error {
        // The most common cause on this machine: com.barkadabrew.comfybox is a
        // KeepAlive=true launchd agent, so killing a manually-started `serve`
        // process's port-holder — or even just a stray manual `serve` left
        // running — gets silently re-occupied within ~5s (ThrottleInterval).
        // Point at the actual fix instead of a bare "address in use" (GH #153).
        fputs("""
        Port \(self.configuration.port) is already in use.

        If you're trying to run a manual/dev server, com.barkadabrew.comfybox \
        (a launchd agent with KeepAlive) may have respawned onto this port. \
        Stop it first:
          launchctl bootout gui/\(getuid())/com.barkadabrew.comfybox
        Then restart it later with:
          launchctl bootstrap gui/\(getuid()) ~/Library/LaunchAgents/com.barkadabrew.comfybox.plist

        To restart the managed server normally (rebuild + relaunch in place),
        use scripts/deploy-server.sh instead of killing the process directly.

        """, stderr)
      }
      initiateShutdown(exitCode: 1)
    case .cancelled:
      // Only exit if we intentionally cancelled (via /v1/shutdown or signal).
      // NWListener can be cancelled by macOS when the process loses its
      // controlling terminal — we must NOT treat that as a shutdown request.
      lifecycleLock.lock()
      let wasIntentional = shutdownSignalled
      lifecycleLock.unlock()

      if wasIntentional {
        logger.info("Listener cancelled (intentional shutdown)")
        exit(0)
      } else {
        logger.warning("Listener cancelled unexpectedly — ignoring (daemon will continue)")
      }
    default:
      break
    }
  }

  private func accept(connection: NWConnection) {
    let handler = ConnectionHandler(
      connection: connection,
      // #300: pin QoS explicitly. Without it the queue runs at whatever QoS
      // is donated by whoever schedules onto it, which during a render is
      // the coordinator's `.utility` render work — demoting control/HTTP
      // responses on this connection right when they need to stay responsive.
      queue: DispatchQueue(label: "z-image.warm-server.connection.\(UUID().uuidString)", qos: .userInitiated),
      server: self
    )
    handler.start()
  }

  /// 0.B-1 (v2.3 rework): runs `operation` under `routeTaskExecutor`'s
  /// preference when available and enabled, otherwise runs it inline exactly
  /// as before this change (pre-0.B-1 behavior — no executor involved at
  /// all). Structured, not a new unstructured `Task {}`: `respond(to:)` is
  /// already running inside the connection's own task, so this only needs to
  /// redirect where ITS continuations resume, via `withTaskExecutorPreference`
  /// (SE-0417) — no child task, no `.value` await, no change to cancellation
  /// or the caller's control flow.
  private func respondOnRouteExecutor(
    _ operation: () async -> RoutedResponse
  ) async -> RoutedResponse {
    if #available(macOS 15.0, *), let executor = routeTaskExecutor as? RouteTaskExecutor {
      return await withTaskExecutorPreference(executor, operation: operation)
    }
    return await operation()
  }

  fileprivate func respond(to request: HTTPRequest) async -> RoutedResponse {
    // Try ComfyUI bridge routes first.
    if let bridgeResponse = await comfyBridge.route(request) {
      return bridgeResponse
    }

    switch (request.method, request.path) {
    case ("GET", "/health"):
      // #217: read from the lock-based snapshot instead of `await
      // coordinator.health()`. The coordinator actor is blocked for a whole
      // synchronous render, so awaiting it made /health hang (HTTP 000) for the
      // render's duration. The snapshot is published on every state transition.
      // The SAME builder serves the sync control plane (`serveControlPlaneSync`),
      // so both arms emit identical bytes.
      return .json(healthRouteResponse())

    case ("POST", "/v1/generate"):
      do {
        // #286 (I5): `rawBody` is the EXPANDED request, so a crash-recovery
        // replay repeats the stack this job was accepted with rather than
        // re-resolving the preset against a store that may have changed.
        let (payload, rawBody) = try await decodedGenerateRequest(from: request.body)
        let source = payload.source ?? "api"
        // #1479: absent/false `preempt` (or no video rendering, or a nested
        // attempt) is `.notApplicable` — same call as before this feature.
        switch await attemptPreemption(payload, source: source, rawBody: rawBody) {
        case .notApplicable:
          let result = try await coordinator.enqueueGenerate(payload, source: source, rawBody: rawBody)
          return .json(status: 200, payload: result)
        case .ran(let result):
          return .json(status: 200, payload: result)
        case .ranFailed(let error):
          return .error(response(for: error))
        case .refused(let eta):
          let result = try await coordinator.enqueueGenerate(payload, source: source, rawBody: rawBody)
          let stamped = GenerateResponse(
            success: result.success, outputPath: result.outputPath, durationMs: result.durationMs,
            preemptRefused: true, etaSec: eta, applied: result.applied,
            appliedLoras: result.appliedLoras, presetUnresolved: result.presetUnresolved,
            presetUnresolvedReason: result.presetUnresolvedReason,
            presetStackMismatch: result.presetStackMismatch,
            memoryEstimateBytes: result.memoryEstimateBytes, memoryAvailableBytes: result.memoryAvailableBytes,
            loraStackOrigin: result.loraStackOrigin,
            warmDefaultSkipped: result.warmDefaultSkipped, loraReload: result.loraReload)
          return .json(status: 200, payload: stamped)
        }
      } catch {
        return .error(response(for: error))
      }

    // Queue-submit: returns a job id immediately instead of blocking the HTTP
    // connection for the whole render. Poll GET /v1/generate/status/{id} for
    // completion — same convention as /v1/video/generate + /v1/video/status.
    // Built after a render's Telegram delivery was orphaned by a blocking
    // /v1/generate call outliving the caller's own turn timeout.
    case ("POST", "/v1/generate/async"):
      do {
        let (payload, rawBody) = try await decodedGenerateRequest(from: request.body)
        let source = payload.source ?? "api"
        // #1479: `submitPreempting` runs the SAME `attemptPreemption` check
        // inside the job's own detached Task, so a `preempt`-absent/false
        // submit takes the exact same `coordinator.enqueueGenerate` path as
        // before this feature.
        let status = imageJobTracker.submitPreempting(
          payload, source: source, coordinator: coordinator, rawBody: rawBody,
          preemptor: { [weak self] jobId in
            await self?.attemptPreemption(payload, source: source, rawBody: rawBody, jobId: jobId) ?? .notApplicable
          })
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(status)
        return .json(.rawJSON(status: 202, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("GET", _) where request.path.hasPrefix("/v1/generate/status/"):
      let jobId = String(request.path.dropFirst("/v1/generate/status/".count))
      guard !jobId.isEmpty else {
        return .error(.error(status: 400, message: "Missing job_id in path"))
      }
      guard var status = imageJobTracker.status(jobId: jobId) else {
        return .error(.error(status: 404, message: "Image job not found: \(jobId)"))
      }
      // comfybox#283/#217: additive — the last 5 lifecycle events recorded
      // for this job id, absent (nil) when the ledger has never seen it.
      let tail = lifecycleLedger.tail(jobId: jobId, count: 5)
      if !tail.isEmpty { status.lifecycleTail = tail }
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      let data = try? encoder.encode(status)
      return .json(.rawJSON(status: 200, data: data ?? Data()))

    case ("GET", "/v1/generate/preview"):
      guard let frame = await coordinator.latestPreviewFrame() else {
        return .json(.empty(status: 204))
      }
      return .json(.binary(status: 200, contentType: "image/jpeg", data: frame))

    case ("POST", "/v1/lora/swap"):
      do {
        var payload = try decode(LoRASwapPayload.self, from: request.body)
        payload = stageNearlineLoras(in: payload)
        let result = try await coordinator.enqueueSwap(payload, rawBody: request.body)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    // MARK: - Nearline storage

    case ("GET", "/v1/nearline"):
      return nearlineListResponse()

    case ("POST", "/v1/nearline/scan"):
      let count = nearlineLibrary.scan()
      auditLog.append(kind: "nearline.scan", message: "Nearline scan found \(count) items")
      return nearlineListResponse()

    case ("POST", "/v1/nearline/stage"):
      struct NameBody: Decodable { let name: String }
      do {
        let body = try decode(NameBody.self, from: request.body)
        let staged = try nearlineLibrary.stage(name: body.name)
        auditLog.append(kind: "nearline.stage", message: "Staged \(body.name)", metadata: ["path": staged])
        return nearlineListResponse()
      } catch let error as NearlineError {
        return .error(.error(status: Self.httpStatus(for: error), message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: "Stage failed: \(error.localizedDescription)"))
      }

    case ("POST", "/v1/nearline/anchor"):
      do {
        let body = try decode(NearlineAnchorBody.self, from: request.body)
        guard let existing = nearlineLibrary.item(named: body.id) else {
          return .error(.error(status: 404, message: "Nearline item not in catalog: \(body.id) (rescan?)"))
        }
        guard existing.kind == body.kind else {
          return .error(.error(
            status: 400,
            message: "Kind mismatch for \(body.id): catalog has \(existing.kind), request said \(body.kind)"))
        }
        // #273 fix round 1 (C1): a "lora" kind also rewrites the matching
        // LoRALibraryEntry's relativePath — the issue's actual problem —
        // through LoRALibrary's own update API. "model" kind has no
        // equivalent per-file registry today, so only the nearline flag
        // applies.
        if body.kind == "lora" {
          try NearlineLoRAAnchoring.setAnchored(
            name: body.id, anchored: body.anchored, loraLibrary: loraLibrary, nearlineLibrary: nearlineLibrary)
        } else {
          try nearlineLibrary.setAnchored(name: body.id, anchored: body.anchored)
        }
        auditLog.append(
          kind: "nearline.anchor",
          message: "\(body.anchored ? "Anchored" : "Un-anchored") \(body.id)")
        return nearlineListResponse()
      } catch let error as NearlineError {
        return .error(.error(status: Self.httpStatus(for: error), message: error.localizedDescription))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: "Anchor failed: \(error.localizedDescription)"))
      }

    case ("POST", "/v1/nearline/evict"):
      struct NameBody: Decodable { let name: String }
      do {
        let body = try decode(NameBody.self, from: request.body)
        let evicted = try nearlineLibrary.evict(name: body.name)
        if evicted {
          auditLog.append(kind: "nearline.evict", message: "Evicted \(body.name)")
        }
        return evicted
          ? nearlineListResponse()
          : .error(.error(status: 404, message: "Not staged: \(body.name)"))
      } catch let error as NearlineError {
        return .error(.error(status: Self.httpStatus(for: error), message: error.localizedDescription))
      } catch {
        return .error(.error(status: 400, message: "Invalid evict payload"))
      }

    case ("POST", "/v1/shutdown"):
      do {
        let result = try await coordinator.enqueueShutdown()
        return .shutdown(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("GET", "/v1/models"):
      // 0.B-2: shared with the sync control-plane path so both emit identical bytes.
      if let data = Self.modelsPayloadData() {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize models"))

    case ("GET", "/v1/styles"):
      let styles = ComfyBoxStylePresets.toJSON()
      if let data = try? JSONSerialization.data(
        withJSONObject: ["styles": styles, "count": styles.count]
      ) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize styles"))

    // MARK: - Config
    // The config document is served/accepted in its canonical camelCase shape (matching
    // ~/.comfybox/config.json and the desktop's plain Codable) — not the snake_case DTO
    // convention used by the render/status routes.

    case ("GET", "/v1/config"):
      // 0.B-2: shared with the sync control-plane path so both emit identical bytes.
      return Self.configGetResponse()

    case ("PUT", "/v1/config"):
      // Full-document replace, routed through ServerConfigStore (FDD §3.3, D3).
      // Port/host changes take effect on next server start; the running listener
      // is unchanged. `If-Match` is advisory: honoured when present-and-stale
      // (409), otherwise proceeds with a deprecation `Warning` — no current
      // caller sends it, so requiring it would break every one of them on day one.
      return Self.configPutResponse(request: request)

    case ("PATCH", "/v1/config"):
      // RFC 7386 JSON Merge Patch — the primary write path going forward
      // (FDD §3.3). Merged INSIDE ServerConfigStore's lock against the CURRENT
      // document, so two agents patching different pointers cannot conflict.
      return Self.configPatchResponse(request: request)

    case ("GET", "/v1/controls"):
      // Phase 4 discovery (FDD §3.4, D4): every descriptor in the compile-time
      // ControlRegistry plus its per-request resolved value — the registry
      // never caches a copy. Shared with the sync control plane (0.B-2) so
      // both paths emit identical bytes.
      return .json(controlsResponse())

    case ("GET", "/v1/providers/status"):
      // F4 (comfybox#324): read through ServerConfigStore, not a direct
      // `ComfyBoxServerConfig`-level disk load — the store is the one
      // authoritative in-memory document PUT/PATCH write through; a
      // parallel direct read did no per-request I/O harm on a healthy
      // install, but its own migrate-on-decode-failure branch can `save()`
      // out of band, so a PATCH racing it would merge into (and overwrite)
      // a document the store never saw. Same fix applied at every other
      // direct config-load read site in this file (see git blame).
      let config = ServerConfigStore.shared.current().config
      func status(_ endpoint: AIProviderEndpoint?) -> [String: Any] {
        guard let endpoint else { return ["configured": false] }
        return [
          "configured": true,
          "model": endpoint.model,
          "base_url": endpoint.baseUrl,
          "has_api_key": !(endpoint.apiKey ?? "").isEmpty,
        ]
      }
      let payload: [String: Any] = [
        "prompt_optimization": status(config.providers.promptOptimization),
        "vision": status(config.providers.vision),
        "captioning": status(config.providers.captioning),
        "replicate": ["configured": !(config.replicate?.apiKey ?? "").isEmpty],
      ]
      if let data = try? JSONSerialization.data(withJSONObject: payload) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize provider status"))

    // MARK: - Model Pool Endpoints

    case ("POST", "/v1/model/load"):
      do {
        let payload = try decode(ModelLoadRequest.self, from: request.body)
        let shouldActivate = payload.activate ?? true
        let shouldWait = payload.wait ?? true

        // Resolve CivitAI model IDs (e.g. 'cyberrealistic-v5') to file paths
        let resolvedSpec = Self.parseModelSpec(from: payload.model)
        let resolvedQuantization = payload.quantization ?? Self.parseQuantization(from: payload.model)

        // K-FIX-1 / Codex C2: every MUTATING pool operation goes through the
        // same FIFO as renders, LoRA swaps and the ComfyBridge model switch.
        // Calling `coordinator.poolLoad` here was not serialized: actor
        // isolation does not hold across an await, so the load's eviction and
        // `GPU.clearCache()` could run under an in-flight render.
        let operation = ModelOperation.load(
          modelSpec: resolvedSpec, quantization: resolvedQuantization, activate: shouldActivate)
        if shouldWait {
          guard case .load(let result) = try await coordinator.enqueueModelOperation(operation) else {
            return .error(.error(status: 500, message: "Model load returned the wrong result kind"))
          }
          return .json(status: 200, payload: result)
        } else {
          // #339: `wait: false` is exactly the "202 + job id nobody is
          // waiting on" pattern — `enqueueModelOperationDetached` never
          // persists (no rawBody, same as local video), so a submission that
          // lands mid-replay and a second restart before it finishes would
          // vanish with no trace. Refuse up front instead.
          let loadRecovery = queueRecoveryState.snapshot()
          if QueueRecoveryGate.shouldReject(kind: .modelLoad, recoveryInProgress: loadRecovery.inProgress) {
            logger.warning("Model load: refused wait:false submission — persisted-queue replay in flight (#339)")
            return .error(.queueRecovering(remainingKinds: loadRecovery.remainingKinds))
          }
          // Fire-and-forget is now a TRACKED queue job, not a detached Task:
          // it is listed in /v1/queue under the id returned here, it can be
          // cancelled, and it still cannot begin under a render.
          let jobId = try await coordinator.enqueueModelOperationDetached(operation)
          let ack = ModelLoadResponse(
            // "loading" is kept verbatim: an out-of-repo client (the daemons)
            // may branch on it, and C2's wire change is the ADDED `job_id`.
            status: "loading",
            model: payload.model,
            family: "pending",
            loadTimeMs: 0,
            vramEstimateMB: 0,
            poolSize: await coordinator.modelPool.count(),
            poolBudgetMB: await coordinator.modelPool.budget(),
            jobId: jobId
          )
          return .json(status: 202, payload: ack)
        }
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/model/activate"):
      do {
        let payload = try decode(ModelActivateRequest.self, from: request.body)
        // C2: activation swaps the resident pipeline out from under whatever
        // is rendering unless it is queued behind it.
        guard case .activate(let result) =
          try await coordinator.enqueueModelOperation(.activate(modelId: payload.model))
        else {
          return .error(.error(status: 500, message: "Model activate returned the wrong result kind"))
        }
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("GET", "/v1/model/pool"):
      let result = await coordinator.poolList()
      return .json(status: 200, payload: result)

    case ("GET", "/v1/model/family"):
      // comfybox#359: file-existence detection only (no weight load, no
      // pool mutation) — safe to call for every row in a "Backfill" batch.
      // The desktop derives `checkpoint_family` from this answer plus its
      // own knowledge of the preset's declared LoRA roles (accel vs not),
      // which the engine has no reason to see here.
      guard let spec = request.queryParameters["model"]?.trimmingCharacters(in: .whitespacesAndNewlines),
            !spec.isEmpty else {
        return .error(.error(status: 400, message: "model query parameter is required"))
      }
      return .json(status: 200, payload: ModelFamilyDetector.detect(spec: spec))

    case ("POST", "/v1/model/unload"):
      do {
        let payload = try decode(ModelUnloadRequest.self, from: request.body)
        // C2: an unload releases the pipeline's weights — the same
        // use-after-release hazard as an eviction.
        guard case .unload(let result) =
          try await coordinator.enqueueModelOperation(.unload(modelId: payload.model))
        else {
          return .error(.error(status: 500, message: "Model unload returned the wrong result kind"))
        }
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    // MARK: - LoRA Library Endpoints

    case ("GET", "/v1/loras"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let allEntries = library.list(includeQuarantined: true)
      // #217 pattern applied here too: `await coordinator.activeLoRAIdentifiers`
      // hopped onto the coordinator ACTOR, which stays occupied for the whole of
      // a synchronous render — so listing the LoRA library hung with HTTP 000
      // until the render finished, and any UI listing from this route rendered
      // an EMPTY list (observed 2026-08-10). The catalog is static data and must
      // never depend on GPU state; only the "currently loaded" decoration did.
      // Read that from the same lock-based snapshot /health uses.
      let activeLoRANames = liveHealth.read().0.loras.map { st in
        ((st.source as NSString).lastPathComponent as NSString).deletingPathExtension
      }
      let quarantinedCount = allEntries.filter { $0.quarantined }.count

      var loraList: [[String: Any]] = []
      for entry in allEntries {
        var dict: [String: Any] = [
          "id": entry.id,
          "filename": entry.filename,
          "model_compatibility": entry.modelCompatibility,
          "compatibility_source": entry.compatibilitySource.rawValue,
          "format": entry.format.rawValue,
          "rank": entry.rank,
          "size_bytes": entry.sizeBytes,
          "quarantined": entry.quarantined,
          "tags": entry.tags,
          "category": entry.category,
          "triggerwords": entry.triggerwords,
          "recommended_scale": entry.recommendedScale,
          "date_added": entry.dateAdded,
        ]
        if let reason = entry.quarantineReason { dict["quarantine_reason"] = reason }
        if !entry.notes.isEmpty { dict["notes"] = entry.notes }
        loraList.append(dict)
      }

      let responseDict: [String: Any] = [
        "loras": loraList,
        "active_loras": activeLoRANames,
        "total": allEntries.count,
        "quarantined": quarantinedCount,
      ]
      if let data = try? JSONSerialization.data(withJSONObject: responseDict) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize LoRA list"))

    case ("GET", _) where request.path.hasPrefix("/v1/loras/"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let id = String(request.path.dropFirst("/v1/loras/".count))
      guard !id.isEmpty, !id.contains("/") else {
        return .error(.error(status: 400, message: "Invalid LoRA ID"))
      }
      guard let entry = library.entry(for: id) else {
        return .error(.error(status: 404, message: "LoRA not found: \(id)"))
      }

      var dict: [String: Any] = [
        "id": entry.id,
        "filename": entry.filename,
        "relative_path": entry.relativePath,
        "size_bytes": entry.sizeBytes,
        "size_formatted": entry.sizeFormatted,
        "model_compatibility": entry.modelCompatibility,
        "compatibility_source": entry.compatibilitySource.rawValue,
        "format": entry.format.rawValue,
        "rank": entry.rank,
        "key_count": entry.keyCount,
        "layer_targets": entry.layerTargets,
        "triggerwords": entry.triggerwords,
        "recommended_scale": entry.recommendedScale,
        "scale_range": entry.scaleRange,
        "tags": entry.tags,
        "category": entry.category,
        "notes": entry.notes,
        "date_added": entry.dateAdded,
        "quarantined": entry.quarantined,
      ]
      if let sha = entry.sha256 { dict["sha256"] = sha }
      if let alpha = entry.alpha { dict["alpha"] = alpha }
      if let reason = entry.quarantineReason { dict["quarantine_reason"] = reason }
      if let url = entry.sourceURL { dict["source_url"] = url }
      if let civitaiId = entry.civitaiModelId { dict["civitai_model_id"] = civitaiId }
      if let meta = entry.safetensorsMetadata { dict["safetensors_metadata"] = meta }

      if let data = try? JSONSerialization.data(withJSONObject: dict) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize LoRA entry"))

    case ("POST", "/v1/loras/scan"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      do {
        let force: Bool
        if !request.body.isEmpty,
           let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
           let f = json["force"] as? Bool {
          force = f
        } else {
          force = false
        }
        let result = try library.scan(force: force)
        let responseDict: [String: Any] = [
          "added": result.added,
          "updated": result.updated,
          "removed": result.removed,
          "unchanged": result.unchanged,
          "total": result.total,
          "errors": result.errors.map { ["file": $0.0, "error": $0.1] },
        ]
        if let data = try? JSONSerialization.data(withJSONObject: responseDict) {
          return .json(.rawJSON(status: 200, data: data))
        }
        return .error(.error(status: 500, message: "Failed to serialize scan result"))
      } catch {
        return .error(.error(status: 500, message: "Scan failed: \(error.localizedDescription)"))
      }

    case ("POST", "/v1/loras/import"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      guard let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
            let sourcePath = json["path"] as? String, !sourcePath.isEmpty
      else {
        return .error(.error(status: 400, message: "Missing 'path'"))
      }
      let category = (json["category"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "vault"
      do {
        let entry = try library.importFile(from: sourcePath, category: category)
        let responseDict: [String: Any] = [
          "success": true,
          "id": entry.id,
          "filename": entry.filename,
          "model_compatibility": entry.modelCompatibility,
          "triggerwords": entry.triggerwords,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: responseDict) {
          return .json(.rawJSON(status: 200, data: data))
        }
        return .error(.error(status: 500, message: "Failed to serialize imported entry"))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: "Import failed: \(error.localizedDescription)"))
      }

    case ("POST", _) where request.path.hasSuffix("/quarantine") && request.path.hasPrefix("/v1/loras/"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let pathBody = String(request.path.dropFirst("/v1/loras/".count).dropLast("/quarantine".count))
      guard !pathBody.isEmpty, !pathBody.contains("/") else {
        return .error(.error(status: 400, message: "Invalid LoRA ID"))
      }
      do {
        let reason: String
        if !request.body.isEmpty,
           let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
           let r = json["reason"] as? String {
          reason = r
        } else {
          reason = "Quarantined via API"
        }
        try library.quarantine(pathBody, reason: reason)
        return .json(.rawJSON(status: 200, data: Data("{\"success\":true,\"id\":\"\(pathBody)\",\"quarantined\":true}".utf8)))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: error.localizedDescription))
      }

    case ("DELETE", _) where request.path.hasSuffix("/quarantine") && request.path.hasPrefix("/v1/loras/"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let pathBody = String(request.path.dropFirst("/v1/loras/".count).dropLast("/quarantine".count))
      guard !pathBody.isEmpty, !pathBody.contains("/") else {
        return .error(.error(status: 400, message: "Invalid LoRA ID"))
      }
      do {
        try library.unquarantine(pathBody)
        return .json(.rawJSON(status: 200, data: Data("{\"success\":true,\"id\":\"\(pathBody)\",\"quarantined\":false}".utf8)))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: error.localizedDescription))
      }

    case ("POST", _) where request.path.hasSuffix("/update") && request.path.hasPrefix("/v1/loras/"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let id = String(request.path.dropFirst("/v1/loras/".count).dropLast("/update".count))
      guard !id.isEmpty, !id.contains("/") else {
        return .error(.error(status: 400, message: "Invalid LoRA ID"))
      }
      guard let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
        return .error(.error(status: 400, message: "Invalid request body"))
      }
      // WP-E6: `krea2_relative` is declared by the user — an unknown value
      // is a 400, never silently dropped.
      var krea2Relative: Krea2Variant?
      if let raw = json["krea2_relative"] {
        guard let str = raw as? String, let parsed = Krea2Variant(rawValue: str) else {
          return .error(.error(
            status: 400,
            message: "Invalid krea2_relative '\(raw)': expected one of \(Krea2Variant.allCases.map(\.rawValue))"))
        }
        krea2Relative = parsed
      }
      // #313 (review round 1): model_compatibility is user-declared here,
      // same as krea2_relative above — validated against the scanner's own
      // vocabulary (400 naming the offending value, never silently accepted
      // or silently dropped), then sets provenance to "manual" so a future
      // scan() never overwrites it back to the auto-detected value.
      var modelCompatibility: [String]?
      if let raw = json["model_compatibility"] as? [String] {
        do {
          modelCompatibility = try WarmServer.validateModelCompatibilityTags(raw)
        } catch {
          return .error(response(for: error))
        }
      }
      let patch = LoRAEntryPatch(
        triggerwords: json["triggerwords"] as? [String],
        recommendedScale: (json["recommended_scale"] as? NSNumber)?.floatValue,
        scaleRange: (json["scale_range"] as? [NSNumber])?.map { $0.floatValue },
        tags: json["tags"] as? [String],
        notes: json["notes"] as? String,
        sourceURL: json["source_url"] as? String,
        civitaiModelId: json["civitai_model_id"] as? Int,
        krea2Relative: krea2Relative,
        modelCompatibility: modelCompatibility
      )
      do {
        try library.update(id, patch: patch)
        guard let entry = library.entry(for: id) else {
          return .error(.error(status: 404, message: "LoRA not found: \(id)"))
        }
        let responseDict: [String: Any] = [
          "success": true,
          "id": entry.id,
          "triggerwords": entry.triggerwords,
          "recommended_scale": entry.recommendedScale,
          "tags": entry.tags,
          "notes": entry.notes,
          "model_compatibility": entry.modelCompatibility,
          "compatibility_source": entry.compatibilitySource.rawValue,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: responseDict) {
          return .json(.rawJSON(status: 200, data: data))
        }
        return .error(.error(status: 500, message: "Failed to serialize updated entry"))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: error.localizedDescription))
      }

    // MARK: - Video Endpoints

    // Montage compositor (#232): assemble images (ken-burns) + clips into one
    // MP4 with transitions. Memory-light editorial motion — no LTX-2, no heavy
    // -model admission gate, runs alongside a resident video model. Sync by
    // design: compositing a <30s montage takes seconds.
    case ("POST", "/v1/montage/compose"):
      do {
        let payload = try decode(MontagePayload.self, from: request.body)
        let result = try await composeMontage(payload)
        return .json(status: 200, payload: MontageResponse(
          outputPath: result.outputPath,
          durationS: result.durationS,
          width: result.width,
          height: result.height,
          segmentCount: result.segmentCount,
          frameCount: result.frameCount))
      } catch {
        return .error(response(for: error))
      }

    // Workflow import/run (#238): ComfyUI API-format workflows as first-class
    // stored objects, executed through the existing ComfyBridge parser/executor.
    case ("POST", "/v1/workflows/import"):
      do {
        return try await handleWorkflowImport(body: request.body)
      } catch {
        return .error(response(for: error))
      }

    case ("GET", "/v1/workflows"):
      let items = workflowStore.list().map { $0.summaryJSON() }
      if let data = try? JSONSerialization.data(withJSONObject: ["workflows": items]) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize workflow list"))

    case ("DELETE", _) where request.path.hasPrefix("/v1/workflows/"):
      let id = String(request.path.dropFirst("/v1/workflows/".count))
      guard workflowStore.delete(id) else {
        return .error(.error(status: 404, message: "Workflow not found: \(id)"))
      }
      return .json(status: 200, payload: ["deleted": id])

    case ("POST", _) where request.path.hasPrefix("/v1/workflows/") && request.path.hasSuffix("/run"):
      let id = String(request.path.dropFirst("/v1/workflows/".count).dropLast("/run".count))
      do {
        return try await handleWorkflowRun(id: id, body: request.body)
      } catch {
        return .error(response(for: error))
      }

    case ("GET", _) where request.path.hasPrefix("/v1/workflows/runs/"):
      let runId = String(request.path.dropFirst("/v1/workflows/runs/".count))
      return handleWorkflowRunStatus(runId: runId)

    case ("GET", _) where request.path.hasPrefix("/v1/workflows/"):
      let id = String(request.path.dropFirst("/v1/workflows/".count))
      guard let workflow = workflowStore.get(id) else {
        return .error(.error(status: 404, message: "Workflow not found: \(id)"))
      }
      var record = workflow.summaryJSON()
      record["graph"] = workflow.graph
      if let data = try? JSONSerialization.data(withJSONObject: record) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize workflow"))

    // Storyboard renderer (#237): ordered shot list → chained i2v renders
    // (each anchored on the previous shot's extracted last frame), optional
    // i2i inserts, final assembly. Long-running → 202 + job id; poll
    // GET /v1/video/status/{id} like any local video job.
    case ("POST", "/v1/storyboard/render"):
      do {
        guard configuration.ltx2WeightsPath != nil, configuration.ltx2GemmaPath != nil else {
          return .error(.error(status: 503, message: "Storyboard rendering needs local LTX-2 (--ltx2-weights/--ltx2-gemma)"))
        }
        // #339: a storyboard is a chain of unpersisted local-video renders
        // (`runStoryboard` -> `enqueueLocalVideo` per shot) tracked under one
        // `videoJobTracker` id, exactly as vulnerable to a second restart as
        // a single local video job — refuse up front rather than accept an
        // orchestration that could vanish mid-shot.
        let storyboardRecovery = queueRecoveryState.snapshot()
        if QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: storyboardRecovery.inProgress) {
          logger.warning("Storyboard: refused render — persisted-queue replay in flight (#339)")
          return .error(.queueRecovering(remainingKinds: storyboardRecovery.remainingKinds))
        }
        let payload = try decode(StoryboardPayload.self, from: request.body)
        let spec = try storyboardSpec(from: payload)
        try spec.validate()
        let source = payload.source ?? "api"
        let status = videoJobTracker.submitOrchestrated(source: source, mode: .storyboard) { [weak self] report in
          guard let self else {
            throw StoryboardError.shotFailed(shot: 0, stage: "server", message: "server shutting down")
          }
          return try await self.runStoryboard(spec: spec, source: source, report: report)
        }
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(status)
        return .json(.rawJSON(status: 202, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/video/config/effective"):
      // Finding #16: a HYPOTHETICAL resolution — request-shaped context in,
      // requested_config + derived render_plan out. GET (below) stays as the
      // no-context readout.
      do {
        let q = (try? decode(EffectiveVideoConfigQuery.self, from: request.body)) ?? EffectiveVideoConfigQuery(
          width: nil, height: nil, frames: nil, duration: nil, fps: nil, tuning: nil, preset: nil, twoPass: nil)
        let videoPreset: ImagePreset? = q.preset.flatMap { presetStore.get($0) }
        let effectiveTuning = Self.effectiveVideoTuning(for: q)
        let resolvedTyped = LTX2ConfigResolver.resolveTyped(
          request: effectiveTuning, preset: videoPreset?.videoTuning)

        // Derived plan, mirroring prepareLocalVideo's math step by step —
        // including the config.videoDefaults layer (FDD §3.3, D3).
        let plan = Self.effectiveVideoRenderPlan(
          width: q.width, height: q.height, frames: q.frames, duration: q.duration, fps: q.fps,
          presetWidth: videoPreset?.width, presetHeight: videoPreset?.height,
          videoConfigDefaults: ServerConfigStore.shared.videoDefaults(),
          resolvedTwoStage: resolvedTyped.twoStage)

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        struct Response: Encodable {
          let requestedConfig: [LTX2ResolvedParam]
          let renderPlan: [[String: String]]
          let presetId: String?
        }
        let data = try encoder.encode(Response(
          requestedConfig: resolvedTyped.params
            .map { p -> LTX2ResolvedParam in
              // Overlay request/preset provenance onto the readout rows.
              guard let src = resolvedTyped.provenance[p.name], src != p.source else { return p }
              return LTX2ResolvedParam(
                name: p.name, envKey: p.envKey, tier: p.tier,
                value: resolvedTyped.valueString(for: p.name) ?? p.value,
                source: src, valid: p.valid, note: p.note)
            },
          renderPlan: plan,
          presetId: q.preset))
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("GET", "/v1/video/traces"):
      // Task #19: the Prompt Lab feed — newest-first render traces.
      do {
        let limit = Int(request.queryParameters["limit"] ?? "") ?? 50
        let summaries = renderTraceStore.recentSummaries(limit: min(200, max(1, limit)))
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(["traces": summaries])
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("POST", _) where request.path.hasPrefix("/v1/video/traces/") && request.path.hasSuffix("/promote"):
      // Task #19 finding #5: promote a rated render's optimization pair into
      // the exemplar set. Intent comes from the bound attempt record; falls
      // back to the render prompt when the render skipped optimization.
      let id = String(request.path.dropFirst("/v1/video/traces/".count).dropLast("/promote".count))
      guard !id.isEmpty else { return .error(.error(status: 400, message: "Missing render_id")) }
      let events = renderTraceStore.events(renderId: id)
      guard let submitted = events.first(where: { $0.event == .submitted }) else {
        return .error(.error(status: 404, message: "No trace for render \(id)"))
      }
      let finalPrompt = submitted.payload["prompt"] ?? ""
      var intent = finalPrompt
      var contentMode = ContentModeManager.Mode.neutral.rawValue
      if let attemptId = submitted.payload["optimization_attempt_id"],
         let attempt = renderTraceStore.events(renderId: attemptId).last {
        intent = attempt.payload["intent"] ?? intent
        contentMode = attempt.payload["content_mode"] ?? contentMode
      }
      guard !finalPrompt.isEmpty else {
        return .error(.error(status: 422, message: "Trace has no prompt to promote"))
      }
      ExemplarStore.shared.add(PromptExemplar(
        intent: intent, final: finalPrompt, mediaKind: "video",
        contentMode: contentMode, sourceRenderId: id))
      return .json(status: 200, payload: ["success": true])

    case ("POST", _) where request.path.hasPrefix("/v1/video/traces/") && request.path.hasSuffix("/rating"):
      // Task #19: post-hoc human verdict, appended as a `rated` event.
      let id = String(request.path.dropFirst("/v1/video/traces/".count).dropLast("/rating".count))
      guard !id.isEmpty else { return .error(.error(status: 400, message: "Missing render_id")) }
      struct RatingBody: Decodable { let vote: String; let axis: String?; let note: String? }
      guard let body = try? decode(RatingBody.self, from: request.body) else {
        return .error(.error(status: 400, message: "'vote' is required (up/down or 1-5)"))
      }
      var payload = ["vote": body.vote, "axis": body.axis ?? "overall"]
      if let note = body.note { payload["note"] = note }
      renderTraceStore.append(RenderTraceEvent(
        renderId: id, event: .rated, taskKind: .videoRender, payload: payload))
      renderTraceStore.flush()
      return .json(status: 200, payload: ["success": true])

    case ("GET", "/v1/video/config/effective"):
      // Task #9 Phase 1: the effective Tier A/B video config with provenance
      // per parameter (configFile > env > builtin). The missing-rescale
      // detector: anything the caller expects to be set shows `builtin`.
      do {
        let params = LTX2ConfigResolver.resolveEffective()
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(["params": params])
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/video/generate"):
      // Backward-compatible route. LOCAL renders here still block the HTTP
      // connection for the whole (synchronous) render — kept working for
      // existing callers. New / long renders should POST /v1/video/generate/async
      // and poll GET /v1/video/status/{id}. The Replicate cloud path was always
      // submit-and-poll (202), and stays so.
      let videoIntent = (try? decode(VideoGenerateRequest.self, from: request.body))?.backendIntent ?? .unspecified
      if videoIntent != .cloud {
        if let localResponse = await localVideoResponseIfConfigured(body: request.body) {
          // #339 review r1, item 6: don't log "routing to local" for a
          // request the gate just refused (503) — `localVideoResponseIfConfigured`
          // already logged the refusal with its own reason.
          if case .error = localResponse {} else {
            logger.info("video: routing to local LTX-2 (synchronous)")
          }
          return localResponse
        }
        if videoIntent == .local {
          return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights). Pass backend: \"replicate\" to explicitly use paid cloud."))
        }
        logger.warning("video: local LTX-2 not configured; falling back to PAID Replicate cloud (\(ReplicateVideoProxy.i2vModel)). Pass backend:\"local\" to forbid, backend:\"replicate\" to silence this warning.")
      }
      return await submitReplicateVideo(body: request.body)

    // Async LOCAL video: submit → 202 + job id immediately; poll
    // GET /v1/video/status/{id} for completion. This is the path a multi-minute /
    // multi-chunk render must take — the HTTP connection is never held open for
    // the whole denoise, and /health stays live (#217) so progress can be polled.
    // Cloud requests are already async via the Replicate proxy; this route
    // delegates to it for the cloud/fallback case, so one endpoint covers both.
    case ("POST", "/v1/video/generate/async"):
      let videoIntent = (try? decode(VideoGenerateRequest.self, from: request.body))?.backendIntent ?? .unspecified
      if videoIntent != .cloud {
        if let localResponse = await localVideoAsyncResponseIfConfigured(body: request.body) {
          // #339 review r1, item 6: don't log "async-submitting" for a
          // request the gate just refused (503) — the helper already logged
          // the refusal with its own reason.
          if case .error = localResponse {} else {
            logger.info("video: async-submitting local LTX-2 job")
          }
          return localResponse
        }
        if videoIntent == .local {
          return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights). Pass backend: \"replicate\" to explicitly use paid cloud."))
        }
        logger.warning("video: local LTX-2 not configured; async-submitting to PAID Replicate cloud (\(ReplicateVideoProxy.i2vModel)). Pass backend:\"local\" to forbid, backend:\"replicate\" to silence this warning.")
      }
      return await submitReplicateVideo(body: request.body)

    case ("POST", "/v1/video/rerender"):
      // Winner action: replay a rendered clip's exact request at a higher
      // resolution budget (default 720p). Same seed + stored effective prompt
      // = the same clip, larger. Async job like /v1/video/generate/async.
      return await videoRerenderResponse(body: request.body)

    case ("POST", "/v1/video/extend"):
      // Winner action: chain a fresh continuation from a clip's last frame at
      // the 4s/480p standard (storyboard-style anchoring, new seed).
      return await videoExtendResponse(body: request.body)

    case ("GET", _) where request.path.hasPrefix("/v1/video/status/"):
      let jobId = String(request.path.dropFirst("/v1/video/status/".count))
      guard !jobId.isEmpty else {
        return .error(.error(status: 400, message: "Missing job_id in path"))
      }
      // Local jobs first (this box owns them), then the Replicate proxy. Both
      // report the same `VideoJobStatus` shape, so a single poll loop covers
      // whichever backend produced the job.
      let jobStatus: VideoJobStatus
      if let local = videoJobTracker.status(jobId: jobId) {
        jobStatus = local
      } else if let proxy = replicateVideoProxy, let cloud = proxy.status(jobId: jobId) {
        jobStatus = cloud
      } else {
        return .error(.error(status: 404, message: "Video job not found: \(jobId)"))
      }
      do {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(jobStatus)
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(.error(status: 500, message: "Failed to encode job status"))
      }

    case ("GET", "/v1/video/output"):
      // Download a rendered video's bytes so remote clients don't need SCP.
      // ?path=<server output path>, validated to be within the allowed dir.
      guard let raw = request.queryParameters["path"], !raw.isEmpty,
            let path = raw.removingPercentEncoding else {
        return .error(.error(status: 400, message: "Missing ?path= for video output"))
      }
      do {
        let resolved = try WarmServerOutputPathValidator.resolveOutputPath(
          path, allowedOutputDirectory: configuration.allowedOutputDirectory).path
        guard FileManager.default.fileExists(atPath: resolved),
              let data = FileManager.default.contents(atPath: resolved) else {
          return .error(.error(status: 404, message: "Video output not found (still rendering?): \(path)"))
        }
        return .json(.binary(status: 200, contentType: "video/mp4", data: data))
      } catch {
        return .error(response(for: error))
      }

    // MARK: - Remote gallery (browse the server's output folder)

    case ("GET", "/v1/gallery/list"):
      // List media in the gallery output folder for remote desktop browsing.
      let limit = request.queryParameters["limit"].flatMap { Int($0) } ?? 500
      let dir = (configuration.allowedOutputDirectory as NSString).expandingTildeInPath
      let fm = FileManager.default
      let exts: Set<String> = ["png", "jpg", "jpeg", "webp", "tiff", "heic", "mp4", "mov", "m4v"]
      var items: [[String: Any]] = []
      if let en = fm.enumerator(at: URL(fileURLWithPath: dir), includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey], options: [.skipsHiddenFiles]) {
        for case let url as URL in en {
          let ext = url.pathExtension.lowercased()
          guard exts.contains(ext) else { continue }
          let vals = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
          let isVideo = ["mp4", "mov", "m4v"].contains(ext)
          items.append([
            "path": url.path,
            "filename": url.lastPathComponent,
            "kind": isVideo ? "video" : "image",
            "size": vals?.fileSize ?? 0,
            "modified": (vals?.contentModificationDate.map { ISO8601DateFormatter().string(from: $0) }) ?? "",
          ])
        }
      }
      items.sort { (($0["modified"] as? String) ?? "") > (($1["modified"] as? String) ?? "") }
      if items.count > limit { items = Array(items.prefix(limit)) }
      guard let data = try? JSONSerialization.data(withJSONObject: ["items": items]) else {
        return .error(.error(status: 500, message: "Failed to serialize gallery list"))
      }
      return .json(.rawJSON(status: 200, data: data))

    case ("GET", "/v1/gallery/file"):
      // Serve a gallery file's bytes (validated within the allowed dir).
      guard let raw = request.queryParameters["path"], !raw.isEmpty,
            let path = raw.removingPercentEncoding else {
        return .error(.error(status: 400, message: "Missing ?path="))
      }
      do {
        let resolved = try WarmServerOutputPathValidator.resolveOutputPath(
          path, allowedOutputDirectory: configuration.allowedOutputDirectory).path
        guard FileManager.default.fileExists(atPath: resolved),
              let data = FileManager.default.contents(atPath: resolved) else {
          return .error(.error(status: 404, message: "File not found: \(path)"))
        }
        let ct: String
        switch (resolved as NSString).pathExtension.lowercased() {
        case "png": ct = "image/png"
        case "jpg", "jpeg": ct = "image/jpeg"
        case "webp": ct = "image/webp"
        case "tiff": ct = "image/tiff"
        case "heic": ct = "image/heic"
        case "mp4", "m4v": ct = "video/mp4"
        case "mov": ct = "video/quicktime"
        default: ct = "application/octet-stream"
        }
        return .json(.binary(status: 200, contentType: ct, data: data))
      } catch {
        return .error(response(for: error))
      }

    // MARK: - Upscale Endpoint

    case ("POST", "/v1/upscale"):
      do {
        let payload = try decode(UpscalePayload.self, from: request.body)
        let result = try await handleUpscale(payload)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    // MARK: - Creative Layer: Characters
    // Character registry parity with the image service. Path-parameter routes follow the
    // /v1/loras/ hasPrefix pattern.

    case ("POST", "/v1/enhance"):
      // 0.B-1 (v2.3): async-internals route, lifted off the cooperative pool.
      return await respondOnRouteExecutor { await self.enhancePromptResponse(body: request.body) }

    // MARK: - Queue management

    case ("GET", "/v1/queue"):
      return await queueListResponse()

    // comfybox#283/#217: additive — see QueueLifecycleLedger.swift. Sync-
    // servable (PR #370 review I5) exactly like `/v1/queue`, so this arm is
    // only reached when the sync control plane's classifier is disabled;
    // `queueLifecyclePayloadData` is shared with `syncQueueLifecycleResponse`
    // so both emit identical bytes.
    case ("GET", "/v1/queue/lifecycle"):
      guard let data = queueLifecyclePayloadData(request: request) else {
        return .error(.error(status: 500, message: "Failed to serialize queue lifecycle"))
      }
      return .json(.rawJSON(status: 200, data: data))

    case ("POST", "/v1/queue/interrupt"):
      // comfybox#362: `target` is additive — an absent/empty body (the
      // pre-#362 shape) still means "whatever health shows as active". See
      // `InterruptTarget` for the full vocabulary. Same `InterruptRoute`
      // decode/encode as the sync arm above (review r1, finding 5).
      let target = InterruptRoute.decodeTarget(from: request.body)
      let outcome = await coordinator.cancelActiveRender(target: target)
      let (_, body) = InterruptRouteResponse.build(from: outcome)
      auditLog.append(
        kind: "queue.interrupt",
        message: InterruptRoute.auditMessage(for: body),
        metadata: target.map { ["target": $0] } ?? [:])
      return InterruptRoute.routedResponse(for: outcome)

    case ("POST", "/v1/queue/clear"):
      struct ClearResult: Encodable { let success: Bool; let cleared: Int }
      let cleared = await coordinator.clearPending()
      auditLog.append(kind: "queue.clear", message: "Cleared \(cleared) pending job(s)")
      return .json(status: 200, payload: ClearResult(success: true, cleared: cleared))

    case ("POST", "/v1/queue/pause"), ("POST", "/v1/queue/resume"):
      struct PauseResult: Encodable { let success: Bool; let paused: Bool }
      let paused = request.path.hasSuffix("/pause")
      await coordinator.setPaused(paused)
      auditLog.append(kind: "queue.pause", message: paused ? "Queue paused" : "Queue resumed")
      return .json(status: 200, payload: PauseResult(success: true, paused: paused))

    case ("POST", _) where request.path.hasPrefix("/v1/queue/") && request.path.hasSuffix("/move"):
      let mid = request.path.dropFirst("/v1/queue/".count).dropLast("/move".count)
      guard let id = Self.pathIdComponent(String(mid)) else {
        return .error(.error(status: 400, message: "Invalid job id"))
      }
      struct MoveBody: Decodable { let direction: String }
      let direction = (try? JSONDecoder().decode(MoveBody.self, from: request.body))?.direction ?? "up"
      struct MoveResult: Encodable { let success: Bool; let moved: Bool }
      let moved = await coordinator.movePending(id: id, direction: direction)
      if moved { auditLog.append(kind: "queue.move", message: "Moved job \(id) \(direction)", metadata: ["id": id, "direction": direction]) }
      return .json(status: 200, payload: MoveResult(success: true, moved: moved))

    case ("DELETE", _) where request.path.hasPrefix("/v1/queue/"):
      guard let id = Self.pathIdComponent(String(request.path.dropFirst("/v1/queue/".count))) else {
        return .error(.error(status: 400, message: "Invalid job id"))
      }
      let removed = await coordinator.cancelPending(id: id)
      if removed {
        auditLog.append(kind: "queue.cancel", message: "Cancelled pending job \(id)", metadata: ["id": id])
      }
      return removed
        ? .json(status: 200, payload: DeleteResult(success: true, id: id, deleted: true))
        : .error(.error(status: 404, message: "Job not pending: \(id)"))

    case ("GET", "/v1/characters"):
      return await listCharactersResponse()

    case ("POST", "/v1/characters"), ("PUT", "/v1/characters"):
      return await upsertCharacterResponse(body: request.body)

    case ("GET", _) where request.path.hasPrefix("/v1/characters/"):
      return await getCharacterResponse(rawId: String(request.path.dropFirst("/v1/characters/".count)))

    case ("DELETE", _) where request.path.hasPrefix("/v1/characters/"):
      return await deleteCharacterResponse(rawId: String(request.path.dropFirst("/v1/characters/".count)))

    // MARK: - Creative Layer: Presets

    case ("GET", "/v1/presets"):
      return presetsListResponse()

    case ("POST", "/v1/presets/resolve"):
      // Match before the generic /v1/presets/ prefix routes below.
      return resolvePresetResponse(body: request.body)

    case ("POST", "/v1/presets/import-legacy"):
      struct ImportResult: Encodable { let success: Bool; let imported: Int }
      let count = presetStore.importLegacyImageService()
      if count > 0 {
        auditLog.append(kind: "preset.import", message: "Imported \(count) legacy image-service preset(s)")
      }
      return .json(status: 200, payload: ImportResult(success: true, imported: count))

    case ("POST", "/v1/presets"), ("PUT", "/v1/presets"):
      return upsertPresetResponse(body: request.body)

    case ("GET", _) where request.path.hasPrefix("/v1/presets/"):
      return getPresetResponse(rawId: String(request.path.dropFirst("/v1/presets/".count)))

    case ("DELETE", _) where request.path.hasPrefix("/v1/presets/"):
      return deletePresetResponse(rawId: String(request.path.dropFirst("/v1/presets/".count)))

    // MARK: - Creative Layer: Content modes

    case ("GET", "/v1/content-modes"):
      return contentModesResponse()

    // FDD §3.3, D3 (Class E): writable content modes. PUT sets any of
    // guidanceBoost/promptHint/negativePromptAdditions/styleVariant (fields
    // omitted from the body keep their current value); DELETE reverts a mode
    // to its built-in definition rather than removing it (there is always
    // exactly one definition per ``ContentMode`` case).
    case ("PUT", _) where request.path.hasPrefix("/v1/content-modes/"):
      return putContentModeResponse(
        rawMode: String(request.path.dropFirst("/v1/content-modes/".count)), body: request.body)

    case ("DELETE", _) where request.path.hasPrefix("/v1/content-modes/"):
      return deleteContentModeResponse(
        rawMode: String(request.path.dropFirst("/v1/content-modes/".count)))

    // MARK: - Creative Layer: Stats / memory

    case ("GET", "/v1/stats"):
      return await statsResponse()

    case ("GET", "/v1/memory"):
      return memoryResponse()

    // MARK: - Creative Layer: Audit log

    case ("GET", "/v1/audit-log"):
      return auditLogResponse(query: request.queryParameters)

    // MARK: - CivitAI conduit + prompt repository (#234)

    case ("GET", "/v1/civitai/search"):
      // 0.B-1 (v2.3): async-internals route, lifted off the cooperative pool.
      return await respondOnRouteExecutor { await self.civitaiSearchRoute(request: request) }

    case ("POST", "/v1/civitai/harvest"):
      // 0.B-1 (v2.3): async-internals route, lifted off the cooperative pool.
      return await respondOnRouteExecutor { await self.civitaiHarvestRoute(request: request) }

    case ("GET", "/v1/civitai/repo"):
      return civitaiRepoRoute(request: request)

    default:
      if ["/v1/generate", "/v1/lora/swap", "/v1/shutdown", "/health",
          "/v1/model/load", "/v1/model/activate", "/v1/model/pool", "/v1/model/unload",
          "/v1/model/family",
          "/v1/loras", "/v1/loras/scan", "/v1/video/generate", "/v1/video/generate/async", "/v1/upscale",
          "/v1/characters", "/v1/presets", "/v1/presets/resolve",
          "/v1/content-modes", "/v1/stats", "/v1/memory", "/v1/audit-log", "/v1/config",
          "/v1/controls"
      ].contains(request.path) || request.path.hasPrefix("/v1/loras/")
         || request.path.hasPrefix("/v1/video/status/")
         || request.path.hasPrefix("/v1/characters/")
         || request.path.hasPrefix("/v1/presets/")
         || request.path.hasPrefix("/v1/content-modes/") {
        return .error(.error(status: 405, message: "Method not allowed"))
      }
      return .error(.error(status: 404, message: "Not found"))
    }
  }

  // MARK: - Creative-layer route handlers
  //
  // These back the /v1/characters, /v1/presets, /v1/content-modes, /v1/stats, /v1/memory,
  // and /v1/audit-log routes above. Kept as small private methods so the main route switch
  // stays readable. Responses use the same helpers as the rest of the server:
  // `RoutedResponse.json(status:payload:)` (snake_case JSON) and `.error(.error(...))`.

  /// Small `{ success, id, deleted }` payload for DELETE responses.
  private struct DeleteResult: Encodable {
    let success: Bool
    let id: String
    let deleted: Bool
  }

  /// Validate + percent-decode a single path-parameter id (rejects empty / nested paths),
  /// matching the guard the /v1/loras/{id} routes use.
  private static func pathIdComponent(_ raw: String) -> String? {
    let decoded = raw.removingPercentEncoding ?? raw
    guard !decoded.isEmpty, !decoded.contains("/") else { return nil }
    return decoded
  }

  // Nearline -------------------------------------------------------------------

  /// #273: request body for `POST /v1/nearline/anchor`. A file-scope (not
  /// switch-arm-local) type so its wire shape is directly unit-testable.
  struct NearlineAnchorBody: Decodable {
    let kind: String
    let id: String
    let anchored: Bool
  }

  /// #273 fix round 1 (C2): the HTTP status a `NearlineError` maps to
  /// across the nearline routes — a pure function so the mapping (notably
  /// `.insufficientCapacity` -> 507) is directly unit-tested without a
  /// live server.
  static func httpStatus(for error: NearlineError) -> Int {
    switch error {
    case .insufficientCapacity: return 507
    case .unknownItem, .sourceMissing: return 404
    // #273 fix round 2 (N2): evicting an anchored item is a conflict with
    // its own pinned state, not a missing/malformed request.
    case .anchored: return 409
    }
  }

  /// The per-item JSON shape for `GET /v1/nearline`'s `items` array. A pure
  /// function (no server, no lock) so the wire shape — notably the additive
  /// `anchored` key (#273) — is directly unit-testable.
  static func nearlineItemJSON(_ item: NearlineItem, iso: ISO8601DateFormatter) -> [String: Any] {
    var dict: [String: Any] = [
      "name": item.name,
      "path": item.path,
      "size_mb": item.sizeMB,
      "kind": item.kind,
      "staged": item.staged,
      "anchored": item.anchored,
    ]
    if let stagedPath = item.stagedPath { dict["staged_path"] = stagedPath }
    if let lastUsed = item.lastUsedAt { dict["last_used_at"] = iso.string(from: lastUsed) }
    return dict
  }

  /// GET /v1/nearline payload: config + full catalog with staging state.
  private func nearlineListResponse() -> RoutedResponse {
    let iso = ISO8601DateFormatter()
    let config = nearlineLibrary.configuration
    let payload: [String: Any] = [
      "roots": config.roots,
      "cache_limit_gb": config.cacheLimitGB,
      "staged_mb": nearlineLibrary.stagedMB,
      "items": nearlineLibrary.list().map { Self.nearlineItemJSON($0, iso: iso) },
    ]
    guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
      return .error(.error(status: 500, message: "Failed to serialize nearline catalog"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }

  /// Auto-stage: rewrite bare LoRA filenames that only exist on nearline
  /// storage to their freshly staged local paths, so a preset (or any swap
  /// request) can reference archived LoRAs and they appear on demand.
  private func stageNearlineLoras(in payload: LoRASwapPayload) -> LoRASwapPayload {
    let entries = payload.loras.map { entry -> LoRAEntry in
      // Only bare safetensors filenames are candidates — absolute/relative
      // paths and HF ids resolve through the normal machinery.
      guard !entry.path.hasPrefix("/"), !entry.path.hasPrefix("~"), !entry.path.hasPrefix("."),
            entry.path.hasSuffix(".safetensors"),
            !FileManager.default.fileExists(atPath: (entry.path as NSString).expandingTildeInPath),
            nearlineLibrary.item(named: entry.path) != nil
      else { return entry }
      guard let staged = try? nearlineLibrary.stage(name: entry.path) else { return entry }
      logger.info("Nearline: auto-staged \(entry.path) for LoRA swap")
      // Staging changes only the storage path. The semantic slot is part of
      // the requested stack and must survive (notably `role: "accel"` for
      // Krea-2 distill adapters whose names do not contain `turbo_lora`).
      return LoRAEntry(path: staged, scale: entry.scale, role: entry.role)
    }
    return LoRASwapPayload(loras: entries)
  }

  // Local video (LTX-2) ---------------------------------------------------------

  /// Body for the local LTX-2 video route (snake_case over the wire).
  /// Decode a base64 image (image_base64) to a temp PNG and return its path, so
  /// remote clients can send an init image without a pre-existing server file.
  /// Returns nil when the string is absent/undecodable.
  private static func writeTempImage(base64: String?) -> String? {
    guard let base64, let data = Data(base64Encoded: base64) else { return nil }
    let path = NSTemporaryDirectory() + "zimage-vidinit-\(UUID().uuidString).png"
    return (try? data.write(to: URL(fileURLWithPath: path))) != nil ? path : nil
  }

  /// Body for `POST /v1/video/config/effective` — a HYPOTHETICAL resolution
  /// (Finding #16): request-shaped context in, `requested_config` +
  /// `render_plan` out. Promoted out of the route case (was a locally-scoped
  /// struct, untestable — comfybox#307 review r1, item 2) so a test can
  /// decode a real wire body against it, the same way `LocalVideoRequest`
  /// already is.
  struct EffectiveVideoConfigQuery: Decodable {
    let width: Int?
    let height: Int?
    let frames: Int?
    let duration: Float?
    let fps: Int?
    let tuning: LTX2VideoTuning?
    let preset: String?
    /// comfybox#307 (review r1): this preflight must mirror the real
    /// generate routes — a caller checking "will two_pass do what I expect"
    /// via `two_pass` alone (no `tuning` block) previously got back the
    /// env/preset/builtin answer, silently ignoring the very field it was
    /// probing.
    let twoPass: Bool?
  }

  struct LocalVideoRequest: Decodable {
    let prompt: String
    let negativePrompt: String?
    let imagePath: String?
    /// I2V init image sent as base64 (image_base64) for remote clients.
    let imageBase64: String?
    let width: Int?
    let height: Int?
    let frames: Int?
    let steps: Int?
    let seed: UInt64?
    let strength: Float?
    /// Conditioning compression (libx264 CRF) override for THIS render — the
    /// daemon sends a higher value (more motion) for partnered-action prompts
    /// and a low value (fidelity) for solo/portrait. nil = env default.
    let imgCompression: Int?
    /// CFG guidance override (>1 amplifies action; motion recipe sends 2.0 for
    /// partnered-action, omits for solo=fidelity). nil = env/config default.
    let guidance: Float?
    let extendToSeconds: Float?
    /// Target duration in seconds — the daemon/MCP vocabulary. For local
    /// renders this maps onto `extendToSeconds` (chunked continuation, each
    /// chunk re-anchored on the previous chunk\u{27}s last frame) when it
    /// exceeds one chunk. `extend_to_seconds` still wins when both are set.
    let duration: Float?
    /// Identity re-anchor strength for continuation chunks (0 disables).
    /// Default 0.5 for extended renders \u{2014} counters per-chunk subject drift.
    let identityAnchorStrength: Float?
    let fps: Int?
    let loraPath: String?
    let loraStrength: Float?
    /// Multiple LoRAs, applied in order — same {path, scale} shape as image
    /// LoRA requests. `loraPath`/`loraStrength` still work for a single LoRA.
    let loras: [LoRAEntry]?
    let outputPath: String?
    /// Which client/app submitted this job (desktop, bree, api…) — surfaced in
    /// the async job status and /health, same as image `GeneratePayload.source`.
    let source: String?
    /// Tier A tuning overrides (snake_case JSON via decoder strategy).
    let tuning: LTX2VideoTuning?
    /// comfybox#307: top-level convenience alias for `tuning.two_stage` — the
    /// HQ two-pass quality tier, controllable per request/scheduler cycle
    /// without a caller needing to know about the nested `tuning` object.
    /// `true`/`false` sets the tier explicitly for this render; `null` (or
    /// absent) defers to `tuning.two_stage`, then preset/configFile/env/
    /// builtin — the existing resolution order (`LTX2ConfigResolver`) is
    /// unchanged. When BOTH this and `tuning.two_stage` are set, the more
    /// specific `tuning.two_stage` wins (see `LTX2VideoTuning.merging`).
    let twoPass: Bool?
    /// Server-minted id from /v1/enhance binding this render to its
    /// optimization lineage (task #19, finding #6).
    let optimizationAttemptId: String?
    /// Optional preset id resolved from the shared PresetStore (mediaKind
    /// "video"): LoRAs, prompt prefix/suffix, negative prompt, dims budget,
    /// steps, seed. Explicit request fields always override preset values.
    let preset: String?
    /// Character whose canonical description is prepended to the prompt so the
    /// subject renders on-model. For T2V (no init image) this is the ONLY
    /// identity source; defaults to "kira" when unset. For I2V the init image
    /// already carries identity, so it's injected only when explicitly named.
    let character: String?
    /// Content mode (neutral/apple/banana/avocado) gating the character's
    /// mode-specific description addendum. Defaults to the server's current mode.
    let contentMode: String?
    /// Auto-enhance the prompt through the configured prompt-optimization
    /// provider (Dan's-PE via LM Studio) before encoding. Default on when a
    /// provider is configured; set false to send the raw prompt.
    let enhance: Bool?
    /// Named resolution budget: "480p" | "720p" | "1080p". Maps to a
    /// width x height pixel budget when explicit width/height are absent
    /// (previously this key was silently DROPPED on the local path).
    let resolution: String?
    /// Aspect ratio for the resolution budget: "16:9" (default) or "9:16".
    /// For I2V the source image's aspect still wins (budget only).
    let aspectRatio: String?
    /// Generate synchronized audio (task #21). T2V single-chunk only in v1;
    /// first audio render reloads the transformer with the audio branch.
    let audio: Bool?
    /// Suppress the manual character prepend when the CALLER has already woven
    /// the description into the prompt (Todd 2026-08-07). Mirrors the image
    /// path's `skip_character_injection`, which the video path never had.
    ///
    /// Without this the description is injected twice — once by the caller,
    /// once here — and at ~110 tokens each that alone overruns the 128-token
    /// tokenizer cap, truncating the scene and the camera direction off the
    /// end of the prompt. The idempotency check below cannot be relied on:
    /// it compares the first four words of THIS host's description against a
    /// prompt composed from a DIFFERENT character record on the daemon host,
    /// at a different framing, so it silently misses.
    ///
    /// `character` still applies — it drives preset resolution, the output
    /// directory and gallery attribution. Only the prompt prepend is skipped.
    let skipCharacterInjection: Bool?
    /// Suppress the preset promptPrefix/promptSuffix wrap when the caller's
    /// prompt is a stored EFFECTIVE prompt that already carries them (winner
    /// re-render/extend replay a trace's composed prompt — re-wrapping would
    /// condition on "prefix, prefix, …"). The preset's LoRAs, negatives, dims
    /// and steps still apply; only the prompt wrap is skipped.
    let skipPresetPrompt: Bool?
    /// Temporal beat scheduling (comfybox#310): structured multi-beat
    /// content (snake_case `start_frac`/`end_frac` via the decoder
    /// strategy). Each beat's `text` must be a verbatim substring of
    /// `prompt` — the engine locates it there and drops (fail-open) any
    /// beat it can't find. nil/empty is byte-identical to today's flat
    /// (joined) behavior; unknown-field-ignored convention means older
    /// engines simply drop this key.
    ///
    /// Decode strictness is INTENTIONALLY not fail-open (adversarial review
    /// F11): a structurally malformed `beat_schedule` (wrong types, missing
    /// keys) fails the whole request decode → clean 4xx, like every other
    /// field. Per-beat fail-open applies only to WELL-FORMED beats the
    /// engine can't act on (unlocatable text, degenerate fracs) — a caller
    /// bug should be loud, a tokenizer merge should not.
    ///
    /// T2V ONLY (comfybox#328, see LTX2BeatSchedule.swift's header for why):
    /// on an I2V request (`image_path` set) this is stripped before it
    /// reaches the generator and the response/trace records
    /// `beat_schedule_ignored: "i2v_unsupported"` instead. A non-empty
    /// schedule also makes the server SKIP prompt enhancement for this
    /// request — see `enhance`.
    let beatSchedule: [BeatSegment]?
  }

  /// Map a named resolution + aspect to a width x height budget. Dims are
  /// budgets, not finals — the existing /64 snapping and I2V source-aspect
  /// derivation still apply downstream.
  private static func videoDims(resolution: String?, aspectRatio: String?) -> (width: Int, height: Int)? {
    guard let res = resolution?.lowercased() else { return nil }
    let landscape: (Int, Int)
    switch res {
    case "480p": landscape = (832, 480)
    case "720p": landscape = (1280, 720)
    case "1080p": landscape = (1920, 1080)
    default: return nil
    }
    let portrait = (aspectRatio ?? "16:9") == "9:16"
    return portrait ? (landscape.1, landscape.0) : landscape
  }

  private struct LocalVideoResponse: Encodable {
    let success: Bool
    let outputPath: String
    let frameCount: Int
    let durationSeconds: Float
    let elapsedSeconds: Double
    let backend: String
    /// comfybox#328: non-nil (`"beat_schedule"`) when prompt enhancement
    /// was skipped so the request's beats would locate. Absent field on
    /// older engines/clients is the byte-identical no-op case.
    let enhancementSkipped: String?
    /// comfybox#328: non-nil (`"i2v_unsupported"`) when a `beat_schedule`
    /// on this I2V request was dropped before reaching the generator.
    let beatScheduleIgnored: String?
    /// comfybox#307: non-nil only when `two_stage` was requested and the
    /// refine pass could not run — see `LTX2RefineGate`.
    let refineSkipped: String?
  }

  /// Submit a video render to the paid Replicate cloud proxy and return its 202
  /// job status (already submit-and-poll). Shared by the sync and async video
  /// routes for the cloud / unspecified-fallback case.
  private func submitReplicateVideo(body: Data) async -> RoutedResponse {
    guard let proxy = replicateVideoProxy else {
      return .error(.error(status: 503, message: "Video generation not available: configure LTX-2 (--ltx2-weights) for local video, or a Replicate API key for cloud"))
    }
    logger.info("video: routing to Replicate cloud (\(ReplicateVideoProxy.i2vModel))")
    do {
      var videoRequest = try decode(VideoGenerateRequest.self, from: body)
      // Accept a bytes-uploaded init image (image_base64) when no path is given.
      if videoRequest.imagePath == nil, let tempPath = Self.writeTempImage(base64: videoRequest.imageBase64) {
        videoRequest.imagePath = tempPath
      }
      if let validationError = videoRequest.validate() {
        return .error(.error(status: 400, message: validationError))
      }
      // Enforce output path containment within the allowed output directory
      // (throws WarmServerError.invalidOutputPath -> 400 via response(for:)).
      if let outputPath = videoRequest.outputPath, !outputPath.isEmpty {
        _ = try WarmServerOutputPathValidator.resolveOutputPath(
          outputPath,
          allowedOutputDirectory: configuration.allowedOutputDirectory
        )
      }
      // I2V: verify image_path exists, is a regular file, and has PNG/JPEG
      // magic bytes before it gets base64-uploaded to Replicate.
      if let imagePath = videoRequest.imagePath {
        if let imageError = ReplicateVideoProxy.validateSourceImage(atPath: imagePath) {
          return .error(.error(status: 400, message: imageError))
        }
      }
      let jobStatus = await proxy.submit(videoRequest)
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      let data = try encoder.encode(jobStatus)
      return .json(.rawJSON(status: 202, data: data))
    } catch {
      return .error(response(for: error))
    }
  }

  /// A LTX-2 generator + validated request, ready to enqueue. Shared by the
  /// synchronous (`/v1/video/generate`) and async (`/v1/video/generate/async`)
  /// local video paths so they build the render identically.
  private struct PreparedLocalVideo {
    let generator: LTX2VideoGenerator
    let request: LTX2VideoRequest
    /// t2v when there's no init image, i2v otherwise.
    let mode: VideoMode
    let source: String
    /// Lineage reference from /v1/enhance, if the caller optimized first.
    let optimizationAttemptId: String?
    /// comfybox#328: non-nil when prompt enhancement was skipped so a
    /// `beat_schedule` would survive verbatim in the composed prompt —
    /// stamped onto the response/trace as `enhancement_skipped`.
    let enhancementSkippedReason: String?
    /// comfybox#328 (Codex round 1, finding 5): non-nil (`"i2v_unsupported"`)
    /// when a non-empty `beat_schedule` arrived on an I2V request and was
    /// dropped before reaching the generator — stamped onto the
    /// response/trace as `beat_schedule_ignored`.
    let beatScheduleIgnoredReason: String?
  }

  /// Map the daemon/MCP `duration` field onto chunked continuation: 0 when
  /// the request fits one chunk (single-chunk render, no continuation cost),
  /// else the requested seconds. Pure for unit testing.
  static func extendSecondsFromDuration(_ duration: Float?, framesPerChunk: Int, fps: Int) -> Float {
    guard let seconds = duration, seconds > 0, fps > 0 else { return 0 }
    let singleChunkSeconds = Float(framesPerChunk) / Float(fps)
    return seconds > singleChunkSeconds ? seconds : 0
  }

  /// Whether a video request renders more than one chunk (continuation path).
  /// Mirrors the extendToSeconds resolution above — the identity anchor
  /// defaults on only for these (#231).
  static func isExtendedRender(
    extendToSeconds: Float?, duration: Float?, framesPerChunk: Int, fps: Int
  ) -> Bool {
    if let explicit = extendToSeconds { return explicit > 0 }
    return extendSecondsFromDuration(
      duration, framesPerChunk: framesPerChunk, fps: fps) > 0
  }

  /// Snap a render dimension to the nearest multiple of 64 (floor 256).
  /// LTX-2 renders at dims that are 32-multiples but NOT 64-multiples (e.g.
  /// 480) exhibit progressive haze (#219) — every clean render in the 07-13
  /// bisect used /64 dims, every hazy one used 480.
  /// Resolve the stage-1 (painted) dims for a two-stage render whose request
  /// dims mean the FINAL output size. Pure, so it can be tested at the sizes
  /// callers actually send rather than only the ones convenient to validate.
  ///
  /// The refine SHARPENS what stage 1 painted; it cannot invent detail that was
  /// never generated. Halving a request sized for the old single-pass
  /// convention therefore degrades output silently — Kira's 704x448 became a
  /// 384x256 base pass (a third of her previous pixels) and went visibly
  /// diffuse, while every render validated that day asked for 960x576 and
  /// halved comfortably to 512x320 (2026-08-02).
  ///
  /// Below the floor the request is treated as STAGE-1 dims (pre-halving
  /// behaviour): the clip finishes at 2x the requested size. A sharp surprise
  /// beats a soft silent degradation.
  ///
  /// - Returns: the dims to paint at, and whether halving was applied.
  static func stageOneDims(
    finalWidth: Int, finalHeight: Int, floorPixels: Int = 512 * 320
  ) -> (width: Int, height: Int, halved: Bool) {
    let w = snapDim64(finalWidth / 2)
    let h = snapDim64(finalHeight / 2)
    guard w * h >= floorPixels else {
      return (finalWidth, finalHeight, false)
    }
    return (w, h, true)
  }

  static func snapDim64(_ value: Int) -> Int {
    max(256, Int((Double(value) / 64.0).rounded()) * 64)
  }

  /// Derive I2V render dims matching the source image aspect within the
  /// requested pixel-area budget, both dims /64. Pure for unit testing.
  ///
  /// Rounding each axis to /64 independently compounds error in opposite
  /// directions: a 1664x896 source (aspect 1.857) at a 448x704 budget produced
  /// 768x384 (aspect 2.000) — the height's ideal 412.1 sat almost exactly on a
  /// 64-boundary midpoint and rounded DOWN while the width rounded up, a 7.7%
  /// distortion that visibly squashes the subject (2026-08-01). Search the /64
  /// neighbourhood instead and keep the pair whose aspect is closest to the
  /// source, breaking ties toward the pixel budget.
  static func deriveVideoDims(
    sourceWidth: Int, sourceHeight: Int, budgetWidth: Int, budgetHeight: Int
  ) -> (width: Int, height: Int) {
    guard sourceWidth > 0, sourceHeight > 0 else {
      return (snapDim64(budgetWidth), snapDim64(budgetHeight))
    }
    let aspect = Double(sourceWidth) / Double(sourceHeight)
    let budget = Double(max(budgetWidth, 64) * max(budgetHeight, 64))
    let idealW = (budget * aspect).squareRoot()
    let idealH = idealW / aspect

    let baseW = Int((idealW / 64.0).rounded())
    let baseH = Int((idealH / 64.0).rounded())

    func search(areaCap: Double) -> (w: Int, h: Int, aspectErr: Double, areaErr: Double)? {
      var best: (w: Int, h: Int, aspectErr: Double, areaErr: Double)?
      for dw in -1...1 {
        for dh in -1...1 {
          let w = max(256, (baseW + dw) * 64)
          let h = max(256, (baseH + dh) * 64)
          let area = Double(w * h)
          guard area <= budget * areaCap else { continue }
          let aspectErr = abs(Double(w) / Double(h) - aspect) / aspect
          let areaErr = abs(area - budget) / budget
          if let b = best {
            let better = aspectErr < b.aspectErr - 1e-9
              || (abs(aspectErr - b.aspectErr) <= 1e-9 && areaErr < b.areaErr)
            if better { best = (w, h, aspectErr, areaErr) }
          } else {
            best = (w, h, aspectErr, areaErr)
          }
        }
      }
      return best
    }

    // Prefer staying near the budget; but at small budgets the 256 floor pins one
    // axis and the tight cap can force a badly stretched pair (a halved two-stage
    // budget hit 19% that way), so allow a larger clip rather than distort.
    var pick = search(areaCap: 1.25)
    if pick == nil || pick!.aspectErr > 0.03, let relaxed = search(areaCap: 1.6),
       relaxed.aspectErr < (pick?.aspectErr ?? .infinity) - 1e-9 {
      pick = relaxed
    }
    guard let chosen = pick else {
      return (snapDim64(Int(idealW.rounded())), snapDim64(Int(idealH.rounded())))
    }
    return (chosen.w, chosen.h)
  }

  /// Pixel dimensions of an image file without decoding the bitmap.
  static func imagePixelSize(atPath path: String) -> (width: Int, height: Int)? {
    let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = props[kCGImagePropertyPixelWidth] as? Int,
          let height = props[kCGImagePropertyPixelHeight] as? Int
    else { return nil }
    return (width, height)
  }

  /// Map an LTX-2 (chunk, step) progress tick to an overall 0-100 percent across
  /// all chunks. Pure so it can be unit-tested and reused by both local paths.
  static func localVideoProgressPercent(chunk: Int, totalChunks: Int, step: Int, totalSteps: Int) -> Int {
    let chunks = max(1, totalChunks)
    let steps = max(1, totalSteps)
    let done = max(0, chunk) * steps + max(0, step)
    let total = chunks * steps
    return min(100, max(0, Int((Double(done) / Double(total)) * 100.0)))
  }

  /// Outcome of `resolveVideoEnhancement`: the prompt to render with, and
  /// whether/why enhancement did or didn't run.
  struct VideoEnhancementOutcome: Equatable {
    let effectivePrompt: String
    let enhancedApplied: Bool
    /// comfybox#328: non-nil (`"beat_schedule"`) ONLY when a non-empty,
    /// ACTIVE beat_schedule caused enhancement to be skipped — i.e.
    /// enhancement would otherwise actually have run. Never set when
    /// enhancement was already not going to run for an unrelated reason
    /// (`enhance:false`, no provider configured, kill switch off) — Codex
    /// round 1, finding 3: a false "beat_schedule" marker is worse than none.
    let enhancementSkippedReason: String?
  }

  /// comfybox#328 (Codex round 1, finding 1): the beat-schedule-vs-
  /// enhancement decision AND the enhancement call itself, extracted so a
  /// unit test can inject a spy `optimize` closure and prove — without any
  /// network call, model weights, or running server — that the optimizer is
  /// never invoked when beats are present, and that the composed prompt
  /// retains every beat's exact text.
  ///
  /// `beat_schedule` beats are located as VERBATIM substrings of the
  /// composed prompt (`LTX2BeatScheduleLocator`, matched against the exact
  /// text the tokenizer receives — see `LTX2VideoGenerator`). LLM
  /// enhancement rewrites the prompt wholesale, so none of the caller's beat
  /// phrasing survives and every beat silently fails to locate (fail-open,
  /// one warning each) whenever both would otherwise run together. Beats are
  /// the more specific ask: a non-empty, ACTIVE `beat_schedule` skips
  /// enhancement outright — but ONLY when enhancement would otherwise
  /// actually run (`enhance != false`, a provider configured, and the
  /// `beat_schedule_enabled` kill switch not set) — see finding 3 above.
  static func resolveVideoEnhancement(
    prompt: String,
    enhance: Bool?,
    beatSchedule: [BeatSegment]?,
    beatScheduleEnabled: Bool,
    characterName: String?,
    characterDesc: String?,
    contentMode: String,
    mediaKind: String,
    optimizerEndpoint: AIProviderEndpoint?,
    logger: Logger,
    optimize: (
      _ endpoint: AIProviderEndpoint, _ prompt: String, _ character: String?,
      _ characterDescription: String?, _ contentMode: String, _ mediaKind: String
    ) async -> OptimizeResult
  ) async -> VideoEnhancementOutcome {
    let hasBeats = !(beatSchedule?.isEmpty ?? true)
    let beatsActive = hasBeats && beatScheduleEnabled
    let wouldEnhanceOtherwise = enhance != false && optimizerEndpoint != nil

    if beatsActive && wouldEnhanceOtherwise {
      logger.warning(
        "Video: skipping prompt enhancement — beat_schedule present (\(beatSchedule?.count ?? 0) beat(s)); enhancement rewrites the prompt and would strand every beat (comfybox#328).")
      return VideoEnhancementOutcome(effectivePrompt: prompt, enhancedApplied: false, enhancementSkippedReason: "beat_schedule")
    }
    guard enhance != false, let endpoint = optimizerEndpoint else {
      return VideoEnhancementOutcome(effectivePrompt: prompt, enhancedApplied: false, enhancementSkippedReason: nil)
    }
    let result = await optimize(endpoint, prompt, characterName, characterDesc, contentMode, mediaKind)
    if result.enhanced {
      logger.info("Video: enhanced prompt via \(endpoint.model)\(characterName.map { " (character \($0))" } ?? "").")
      return VideoEnhancementOutcome(effectivePrompt: result.prompt, enhancedApplied: true, enhancementSkippedReason: nil)
    }
    return VideoEnhancementOutcome(effectivePrompt: prompt, enhancedApplied: false, enhancementSkippedReason: nil)
  }

  /// Production `optimize` closure for `resolveVideoEnhancement` — builds the
  /// PromptOptimizer and calls it. The ONLY place a network call happens;
  /// unit tests inject a spy instead.
  private func callPromptOptimizer(
    endpoint: AIProviderEndpoint, prompt: String, character: String?,
    characterDescription: String?, contentMode: String, mediaKind: String
  ) async -> OptimizeResult {
    var base = endpoint.baseUrl
    while base.hasSuffix("/") { base.removeLast() }
    if base.hasSuffix("/v1") { base = String(base.dropLast(3)) }
    while base.hasSuffix("/") { base.removeLast() }
    let optimizer = PromptOptimizer(
      configuration: PromptOptimizer.Configuration(
        ollamaBaseURL: base, lmStudioBaseURL: nil, model: endpoint.model,
        timeoutSeconds: 90, enabled: true),
      logger: logger)
    // i2v: motion-only enhancement (the init image fixes subject/scene); t2v: full scene.
    return await optimizer.optimize(
      prompt: prompt, character: character, characterDescription: characterDescription,
      contentMode: contentMode, mediaKind: mediaKind)
  }

  /// comfybox#307 (review r1, item 3a): the ACTUAL wiring `prepareLocalVideo`
  /// runs to fold the top-level `two_pass` convenience into `tuning` — pulled
  /// out to a static func (mirrors `localVideoProgressPercent` above) so a
  /// test can decode a real wire-format `LocalVideoRequest` and assert on
  /// this exact call, not a re-implementation of it.
  static func effectiveVideoTuning(for req: LocalVideoRequest) -> LTX2VideoTuning? {
    LTX2VideoTuning.merging(req.tuning, twoPass: req.twoPass)
  }

  /// Same merge, for the `/v1/video/config/effective` preflight's query
  /// shape (comfybox#307 review r1, item 2) — one merge rule, two wire
  /// shapes that both carry it.
  static func effectiveVideoTuning(for query: EffectiveVideoConfigQuery) -> LTX2VideoTuning? {
    LTX2VideoTuning.merging(query.tuning, twoPass: query.twoPass)
  }

  /// F3 (comfybox#324, adversarial review of Phase 3 config): the ACTUAL
  /// width/height `??` chain `prepareLocalVideo` (the real LTX-2 generate
  /// prep path — NOT the `/v1/video/config/effective` preview above) applies
  /// before aspect-matching/64-snapping — pulled out so a unit test can call
  /// this exact merge with an injected `VideoDefaultValues` instead of
  /// re-implementing the chain inline (the gap `ServerConfigStoreTests.
  /// testMigrationIsVideoEngineNeutralDespiteDesktopValues` left: it asserted
  /// against local `let ... = nil` shadows of this chain, never this
  /// function). Priority: explicit width/height > named resolution > preset
  /// dims > config.videoDefaults > 704x448 engine default.
  static func resolvedLTX2RequestDims(
    requestWidth: Int?, requestHeight: Int?,
    namedWidth: Int?, namedHeight: Int?,
    presetWidth: Int?, presetHeight: Int?,
    videoConfigDefaults: VideoDefaultValues
  ) -> (width: Int, height: Int) {
    (
      width: requestWidth ?? namedWidth ?? presetWidth ?? videoConfigDefaults.width ?? 704,
      height: requestHeight ?? namedHeight ?? presetHeight ?? videoConfigDefaults.height ?? 448
    )
  }

  /// F3: the frames half of the same real `prepareLocalVideo` chain —
  /// `videoDefaults.frames` (migrated from the desktop's `videoFrames`) slots
  /// between the request and the 97f engine default.
  static func resolvedLTX2Frames(requestFrames: Int?, videoConfigDefaults: VideoDefaultValues) -> Int {
    requestFrames ?? videoConfigDefaults.frames ?? 97
  }

  /// comfybox#307 (review r1, item 2): the derived render plan for
  /// `POST /v1/video/config/effective` — pulled out of the route case,
  /// mirroring `prepareLocalVideo`'s dims/frames math step by step, so
  /// `resolvedTwoStage` (which the route now feeds from the `two_pass`-merged
  /// tuning, not just `tuning.two_stage`) driving the `two_stage_halving`
  /// step is directly testable without a decoder or a live server.
  static func effectiveVideoRenderPlan(
    width: Int?, height: Int?, frames: Int?, duration: Float?, fps: Int?,
    presetWidth: Int?, presetHeight: Int?,
    videoConfigDefaults: VideoDefaultValues,
    resolvedTwoStage: Bool
  ) -> [[String: String]] {
    var plan: [[String: String]] = []
    var w = width ?? presetWidth ?? videoConfigDefaults.width ?? 704
    var h = height ?? presetHeight ?? videoConfigDefaults.height ?? 448
    let snappedW = Self.snapDim64(w), snappedH = Self.snapDim64(h)
    if snappedW != w || snappedH != h {
      plan.append(["step": "snap_64", "note": "\(w)x\(h) -> \(snappedW)x\(snappedH)"])
      w = snappedW; h = snappedH
    }
    if resolvedTwoStage {
      let s1 = Self.stageOneDims(finalWidth: w, finalHeight: h)
      if s1.halved {
        plan.append(["step": "two_stage_halving",
                     "note": "request dims are FINAL; stage 1 paints \(s1.width)x\(s1.height), refine doubles back"])
      } else {
        plan.append(["step": "stage1_floor",
                     "note": "final \(w)x\(h) too small to halve (floor 512x320) — single-scale render"])
      }
    }
    let effectiveFps = fps ?? 24
    var framesPerChunk = frames ?? videoConfigDefaults.frames ?? 97
    var extendSeconds = Self.extendSecondsFromDuration(duration, framesPerChunk: framesPerChunk, fps: effectiveFps)
    if extendSeconds > 0 {
      let targetFrames = Int((extendSeconds * Float(effectiveFps)).rounded())
      if targetFrames <= 289 {
        let singleFrames = min(289, ((max(targetFrames, 9) - 2) / 8) * 8 + 9)
        framesPerChunk = max(framesPerChunk, singleFrames)
        extendSeconds = 0
        plan.append(["step": "single_pass_fold",
                     "note": "\(duration ?? 0)s folds into one \(framesPerChunk)f chunk (continuation chunks degenerate)"])
      } else {
        plan.append(["step": "chunked_continuation",
                     "note": "\(duration ?? 0)s exceeds the 289f window — continuation chunks with identity anchor"])
      }
    }
    plan.append(["step": "final", "note": "\(w)x\(h) @ \(framesPerChunk)f, fps \(effectiveFps)"])
    if width == nil && height == nil {
      plan.append(["step": "caveat",
                   "note": "i2v aspect-matching to a source image is not simulated here (no image supplied)"])
    }
    return plan
  }

  /// comfybox#307 (review r2, item 2a; tightened review r3, minor 3): the
  /// ACTUAL `LTX2VideoRequest` construction `prepareLocalVideo` runs —
  /// pulled out verbatim (every argument expression unchanged) so a test
  /// can call the SAME site production code calls and inspect the
  /// resulting request's `.tuning`.
  ///
  /// `tuning:` is derived HERE, from `req`, via `effectiveVideoTuning(for:)`
  /// — not accepted as a separate parameter — so there is no longer a
  /// caller-supplied value that could silently diverge from the merge (a
  /// one-line revert at the call site used to be able to pass `req.tuning`
  /// instead of the merged value and no test would catch it; now there is
  /// no such parameter to revert). `prepareLocalVideo` still computes its
  /// OWN `effectiveTuning` local separately, for the dims-calc two-stage
  /// check earlier in that function — a harmless duplicate call of the same
  /// pure merge, not a second source of truth for what reaches the request.
  ///
  /// All other parameters are already-resolved values `prepareLocalVideo`
  /// computes before this call — nothing here re-derives anything else.
  static func buildLocalVideoRequest(
    req: LocalVideoRequest, videoPreset: ImagePreset?,
    effectivePrompt: String, effectiveInitImage: String?,
    renderWidth: Int, renderHeight: Int,
    foldedFramesPerChunk: Int, foldedExtendSeconds: Float,
    resolvedLoRAs: [LTX2LoRAReference], effectiveBeatSchedule: [BeatSegment]?,
    resolvedOutput: String
  ) -> LTX2VideoRequest {
    LTX2VideoRequest(
      prompt: effectivePrompt,
      negativePrompt: req.negativePrompt ?? videoPreset?.negativePrompt,
      initImagePath: effectiveInitImage,
      width: renderWidth,
      height: renderHeight,
      framesPerChunk: foldedFramesPerChunk,
      steps: req.steps ?? videoPreset?.steps ?? 8,
      seed: req.seed ?? videoPreset?.seed.map(UInt64.init) ?? 42,
      strength: req.strength ?? 1.0,
      imgCompression: req.imgCompression,
      guidance: req.guidance,
      // Re-enabled by default for EXTENDED renders (#231, 2026-07-16): the
      // 2026-07-13 MLX mutex crash on this path was memory pressure — with
      // the int8 stack (#230) a 12s/3-chunk anchored render completed clean
      // (289f, no crash). Single-chunk renders don't anchor (nothing to
      // drift); callers can still pass 0 to disable.
      // Mid-pass identity re-anchor is OPT-IN and default OFF — it was superseded
      // by the face-region anchor (LTX2_FACE_ANCHOR_STRENGTH), which holds partner
      // faces without the multi-keyframe gap-collapse. Enable explicitly via
      // LTX2_REANCHOR_INTERVAL>0 (+ _STRENGTH); a standard 97f/4s render NEVER takes
      // it unless the interval is set below the frame count. Only the pre-existing
      // extended/chunked anchor stays on by default (unchanged behavior).
      identityAnchorStrength: req.identityAnchorStrength
        ?? (Self.isExtendedRender(
              extendToSeconds: req.extendToSeconds, duration: req.duration,
              framesPerChunk: req.frames ?? 97, fps: req.fps ?? 24)
            ? 0.5
            : ((effectiveInitImage != nil
                && (Int(ProcessInfo.processInfo.environment["LTX2_REANCHOR_INTERVAL"] ?? "") ?? 0) > 0
                && (req.frames ?? 97) > (Int(ProcessInfo.processInfo.environment["LTX2_REANCHOR_INTERVAL"] ?? "") ?? 0))
               ? (Float(ProcessInfo.processInfo.environment["LTX2_REANCHOR_STRENGTH"] ?? "") ?? 0.4) : 0)),
      identityReAnchorInterval: (Int(ProcessInfo.processInfo.environment["LTX2_REANCHOR_INTERVAL"] ?? "") ?? 0),
      extendToSeconds: foldedExtendSeconds,
      fps: req.fps ?? 24,
      loraPath: req.loraPath,
      loraStrength: req.loraStrength ?? 1.0,
      loras: resolvedLoRAs,
      outputPath: resolvedOutput,
      tuning: Self.effectiveVideoTuning(for: req),
      presetTuning: videoPreset?.videoTuning,
      audio: req.audio ?? false,
      beatSchedule: effectiveBeatSchedule
    )
  }

  /// Resolve LTX-2 weights, build + validate the render request. Returns nil when
  /// local LTX-2 isn't configured (caller falls through to Replicate); throws for
  /// a malformed request or invalid output path.
  private func prepareLocalVideo(body: Data) async throws -> PreparedLocalVideo? {
    guard let weights = configuration.ltx2WeightsPath, let gemma = configuration.ltx2GemmaPath else {
      return nil
    }
    let req = try decode(LocalVideoRequest.self, from: body)
    // comfybox#307: fold the top-level `two_pass` convenience into `tuning`
    // before anything downstream reads it — every existing consumer
    // (dims math, `LTX2ConfigResolver.resolveTyped`, the trace snapshot) then
    // sees one authoritative tuning block, unchanged otherwise.
    let effectiveTuning = Self.effectiveVideoTuning(for: req)

    // Video presets — same PresetStore as images (mediaKind "video"). A
    // preset is a named bundle: LoRAs (bare filenames resolve through the
    // LoRA library search roots), prompt shaping, negative prompt, dims
    // budget, steps, seed. Explicit request fields win over the preset.
    var videoPreset: ImagePreset? = nil
    if let presetId = req.preset, !presetId.isEmpty {
      guard let found = presetStore.get(presetId) else {
        throw WarmServerError.invalidRequest(message: "Unknown preset \u{27}\(presetId)\u{27} — see /v1/presets")
      }
      videoPreset = found
      logger.info("LTX-2: applying video preset \u{27}\(presetId)\u{27} (\(found.loras.count) LoRA(s))")
    }

    // Contain the output within the allowed directory. A relative (or absent)
    // output path is resolved against the ALLOWED directory, not the process
    // CWD — under launchd the CWD is the repo checkout, so the old bare-name
    // default failed its own containment check on every submit (#219).
    let requestedOutputRaw = req.outputPath ?? "ltx2-\(UUID().uuidString).mp4"
    let requestedOutput: String
    if requestedOutputRaw.hasPrefix("/") || requestedOutputRaw.hasPrefix("~") {
      requestedOutput = requestedOutputRaw
    } else {
      requestedOutput = (configuration.allowedOutputDirectory as NSString)
        .appendingPathComponent(requestedOutputRaw)
    }
    let resolvedOutput = try WarmServerOutputPathValidator.resolveOutputPath(
      requestedOutput, allowedOutputDirectory: configuration.allowedOutputDirectory).path

    let generator: LTX2VideoGenerator
    if let existing = videoHolder.get() {
      generator = existing
    } else {
      logger.info("LTX-2: resolving weights/text-encoder (downloads on first use if not cached)…")
      let weightsURL = try await ModelResolution.resolve(
        modelSpec: weights,
        filePatterns: ["transformer-distilled.safetensors", "connector.safetensors",
                        "vae_decoder.safetensors", "vae_encoder.safetensors", "config.json"]
      )
      let gemmaURL = try await ModelResolution.resolve(
        modelSpec: gemma,
        filePatterns: ["*.safetensors", "*.json", "tokenizer/*", "*.model"]
      )
      generator = LTX2VideoGenerator(
        config: .init(weightsDir: weightsURL.path, gemmaPath: gemmaURL.path), logger: logger)
    }
    // Publish into the shared holder so the coordinator can evict it before an
    // image load, and so the render queue evicts image models before this one
    // actually loads its ~65GB of weights inside generate() (#218).
    videoHolder.set(generator)
    // #1479: wire telemetry + the (normally never-raised) preemption signal
    // on EVERY generator instance — fresh or reused — so the video path is
    // always preemptible-capable while staying byte-identical unless a job
    // actually raises the signal. Also covers the fresh instance a preemption
    // eviction produces (`VideoGeneratorHolder.release()` deallocates the old
    // one), since this is the same function a post-eviction cold reload runs.
    generator.setTelemetry(ltx2Telemetry)
    generator.setPreemptionSignal(ltx2PreemptionSignal)

    // Accept an init image as bytes (image_base64) when no server path is given.
    let effectiveInitImage = req.imagePath ?? Self.writeTempImage(base64: req.imageBase64)

    var loraEntries: [LoRAEntry] = req.loras ?? []
    if loraEntries.isEmpty, req.loraPath == nil, let preset = videoPreset, !preset.loras.isEmpty {
      loraEntries = preset.loras.map {
        LoRAEntry(path: $0.filename, scale: Float($0.scale), role: $0.role)
      }
    }
    if loraEntries.isEmpty, req.loraPath == nil,
       let defaultLoRA = configuration.ltx2DefaultLoRA, !defaultLoRA.isEmpty {
      // "path" or "path@scale"
      let parts = defaultLoRA.split(separator: "@", maxSplits: 1).map(String.init)
      let scale = parts.count == 2 ? Float(parts[1]) ?? 1.0 : 1.0
      loraEntries = [LoRAEntry(path: parts[0], scale: scale)]
      logger.info("LTX-2: applying default video LoRA \(parts[0]) @ \(scale) (--ltx2-lora)")
    }
    let resolvedLoRAs: [LTX2LoRAReference] = try loraEntries.map { entry in
      let config = try entry.makeConfiguration()
      guard case .local(let url) = config.source else {
        throw WarmServerError.invalidRequest(
          message: "LTX-2 video LoRAs must be local files (got a HuggingFace reference for '\(entry.path)')")
      }
      return LTX2LoRAReference(path: url.path, scale: config.scale)
    }

    // LTX-2 render dims must be divisible by 64: 32-multiples that are not
    // 64-multiples (e.g. 480) produce progressive haze/ghosting (#219). I2V
    // output must additionally match the SOURCE image aspect ratio — a fixed
    // preset like 704x448 applied to a portrait source distorts the
    // conditioning frame and the render drifts off the image. The requested
    // width x height is kept only as a pixel-area budget for I2V.
    // Priority: explicit width/height > named resolution ("720p" etc., FIXED:
    // previously silently dropped) > preset dims > config.videoDefaults >
    // 704x448 engine default (FDD §3.3, D3 — only width/height/frames migrate
    // from the desktop's local settings; steps stays untouched below).
    let namedDims = Self.videoDims(resolution: req.resolution, aspectRatio: req.aspectRatio)
    let videoConfigDefaults = ServerConfigStore.shared.videoDefaults()
    let requestedDims = Self.resolvedLTX2RequestDims(
      requestWidth: req.width, requestHeight: req.height,
      namedWidth: namedDims?.width, namedHeight: namedDims?.height,
      presetWidth: videoPreset?.width, presetHeight: videoPreset?.height,
      videoConfigDefaults: videoConfigDefaults)
    var renderWidth = requestedDims.width
    var renderHeight = requestedDims.height
    if req.width == nil, let nd = namedDims {
      logger.info("LTX-2: resolution '\(req.resolution ?? "")' -> \(nd.width)x\(nd.height) budget")
    }
    if let initPath = effectiveInitImage,
       let sourceSize = Self.imagePixelSize(atPath: initPath) {
      let derived = Self.deriveVideoDims(
        sourceWidth: sourceSize.width, sourceHeight: sourceSize.height,
        budgetWidth: renderWidth, budgetHeight: renderHeight)
      if derived.width != renderWidth || derived.height != renderHeight {
        logger.info(
          "LTX-2 I2V: adjusted \(renderWidth)x\(renderHeight) -> \(derived.width)x\(derived.height) (source \(sourceSize.width)x\(sourceSize.height), aspect-matched, /64)")
        renderWidth = derived.width
        renderHeight = derived.height
      }
    } else {
      let snappedW = Self.snapDim64(renderWidth)
      let snappedH = Self.snapDim64(renderHeight)
      if snappedW != renderWidth || snappedH != renderHeight {
        logger.info("LTX-2: snapped \(renderWidth)x\(renderHeight) -> \(snappedW)x\(snappedH) (dims must be /64, #219)")
        renderWidth = snappedW
        renderHeight = snappedH
      }
    }
    // Two-stage dims convention (2026-08-02): with LTX2_TWO_STAGE=1 the request
    // dims are the FINAL output size (matching ComfyUI and every caller's
    // intuition). ComfyBox's pipeline doubles stage-1 dims through the refine,
    // so hand it the /64-snapped HALF. Without this, enabling two-stage doubled
    // every clip's output size and the refine-volume gate silently skipped the
    // refine — the worst of both worlds. Stage-1 /64 keeps the final /128-ish
    // and matches all validated two-stage renders (stage 1 at 448x256 etc.).
    // Typed resolution honors request/preset tuning overrides (finding #18):
    // a request can enable two-stage without the plist knowing.
    if LTX2ConfigResolver.resolveTyped(request: effectiveTuning, preset: videoPreset?.videoTuning).twoStage {
      let s1 = Self.stageOneDims(finalWidth: renderWidth, finalHeight: renderHeight)
      if s1.halved {
        logger.info(
          "LTX-2 two-stage: request dims \(renderWidth)x\(renderHeight) = FINAL; stage 1 paints \(s1.width)x\(s1.height), refine doubles to \(s1.width * 2)x\(s1.height * 2)")
        renderWidth = s1.width
        renderHeight = s1.height
      } else {
        logger.warning("""
          LTX-2 two-stage: request \(renderWidth)x\(renderHeight) would paint stage 1 at \
          \(Self.snapDim64(renderWidth / 2))x\(Self.snapDim64(renderHeight / 2)) — below the \
          stage-1 floor, which renders SOFT (the refine sharpens, it cannot invent detail). \
          Treating the request as stage-1 dims instead; output will be \
          \(renderWidth * 2)x\(renderHeight * 2). Send ~2x larger dims for the intended size.
          """)
      }
    }
    if let requestedSteps = req.steps, requestedSteps != 8 {
      logger.warning(
        "LTX-2: steps=\(requestedSteps) requested, but the distilled pipeline uses a fixed 8-step sigma schedule — the value is currently ignored (#219)")
    }

    // Character identity + optional prompt enhancement. For T2V (no init image)
    // there is no other identity source, so default to "kira" when the caller
    // names no character. For I2V the init image already carries identity.
    let isT2V = (effectiveInitImage == nil)
    let characterName = req.character ?? (isT2V ? "kira" : nil)
    let charMode = req.contentMode.flatMap { ContentModeManager.Mode(rawValue: $0) } ?? .neutral
    var characterDesc: String? = nil
    if let name = characterName,
       let entry = await characterStore.get(CharacterEntry.slug(name)) {
      characterDesc = entry.resolvedDescription(for: charMode)
    }

    // comfybox#328 (Codex round 1, finding 5): temporal beat scheduling is
    // T2V-only — `LTX2VideoGenerator` never forwards resolved beats into any
    // I2V render path (its I2V branches don't take a beat parameter), so a
    // `beat_schedule` on an I2V request was a SILENT no-op — locate could
    // even succeed and the bias would still be thrown away downstream.
    // Rather than wire beats into I2V in this PR (I2V's frame axis and
    // keyframe chaining differ enough from T2V's single continuous timeline
    // to need its own design), make the no-op loud: strip the schedule
    // before it reaches the generator and record why.
    var beatScheduleIgnoredReason: String? = nil
    let effectiveBeatSchedule: [BeatSegment]?
    if !isT2V, let beats = req.beatSchedule, !beats.isEmpty {
      beatScheduleIgnoredReason = "i2v_unsupported"
      effectiveBeatSchedule = nil
      logger.warning(
        "Video: beat_schedule ignored — I2V rendering doesn't support temporal beat scheduling yet (\(beats.count) beat(s) dropped, comfybox#328: T2V-only). Send a T2V request (no image_path) for beat_schedule.")
    } else {
      effectiveBeatSchedule = req.beatSchedule
    }

    // Auto-enhance the video prompt through the configured prompt-optimization
    // provider (Dan's-PE via LM Studio). The optimizer weaves in the character
    // description AND enriches the scene, so it replaces the manual character
    // prepend. Opt out per request with enhance:false; falls back to the manual
    // prepend when no provider is configured or enhancement fails.
    //
    // comfybox#328: `beat_schedule` beats are located as VERBATIM substrings
    // of this same composed prompt (LTX2BeatScheduleLocator, matched against
    // the exact text the tokenizer receives — see LTX2VideoGenerator). The
    // dolphin enhancement rewrites the prompt wholesale, so none of the
    // caller's beat phrasing survives and every beat silently fails to
    // locate (fail-open, one warning each) whenever both would otherwise run
    // together. Beats are the more specific ask: a non-empty, ACTIVE
    // `beat_schedule` skips enhancement outright — but ONLY when enhancement
    // would otherwise actually run (`enhance != false`, a provider
    // configured, kill switch on), so the trace marker below is never a
    // false claim (Codex round 1, finding 3). `effectiveBeatSchedule` (not
    // `req.beatSchedule`) feeds this — an I2V-ignored schedule shouldn't
    // also cost enhancement quality for beats that will never apply.
    let aiProviderConfig = ServerConfigStore.shared.current().config
    let beatScheduleEnabled = LTX2ConfigResolver.resolveTyped(
      request: req.tuning, preset: videoPreset?.videoTuning
    ).beatScheduleEnabled
    let enhancement = await Self.resolveVideoEnhancement(
      prompt: req.prompt,
      enhance: req.enhance,
      beatSchedule: effectiveBeatSchedule,
      beatScheduleEnabled: beatScheduleEnabled,
      characterName: characterName,
      characterDesc: characterDesc,
      contentMode: charMode.rawValue,
      mediaKind: isT2V ? "video" : "video-i2v",
      optimizerEndpoint: aiProviderConfig.providers.promptOptimization,
      logger: logger,
      optimize: callPromptOptimizer)
    var effectivePrompt = enhancement.effectivePrompt
    let enhancedApplied = enhancement.enhancedApplied
    let enhancementSkippedReason = enhancement.enhancementSkippedReason

    // Fallback: manual character prepend when enhancement didn't run/apply.
    // Skipped outright when the caller says it already wove the description in
    // — see `skipCharacterInjection` on the request for why the idempotency
    // check below is not sufficient on its own.
    if req.skipCharacterInjection == true, characterName != nil {
      logger.info("Video: character injection skipped — caller composed identity.")
    }
    if req.skipCharacterInjection != true,
       !enhancedApplied, let name = characterName, let desc = characterDesc, !desc.isEmpty {
      // Idempotency: skip if the caller already wrote the description in.
      let alreadyPresent = desc.split(separator: " ").prefix(4).allSatisfy {
        effectivePrompt.localizedCaseInsensitiveContains($0)
      }
      if !alreadyPresent {
        effectivePrompt = desc + " " + effectivePrompt
        logger.info("Video: prepended character '\(name)' (mode \(charMode.rawValue)) to prompt.")
      }
    }

    if let preset = videoPreset, req.skipPresetPrompt != true {
      if let prefix = preset.promptPrefix, !prefix.isEmpty { effectivePrompt = prefix + ", " + effectivePrompt }
      if let suffix = preset.promptSuffix, !suffix.isEmpty { effectivePrompt = effectivePrompt + ", " + suffix }
    }

    // Single-pass fold (2026-08-02): continuation chunks DEGENERATE — chunk 2
    // collapses into fragments (long-known for i2v, which is why the daemon
    // sends explicit single-pass frames there; observed for T2V today the
    // moment two-stage went live: chunk 1 clean, chunk 2 psychedelic). The
    // daemon's rule only covers i2v, so fold ANY duration that fits the
    // trained window (289f = 12s) into ONE chunk here, t2v included.
    // FDD §3.3, D3: `videoDefaults.frames` (migrated from the desktop's
    // `videoFrames`) slots between the request and the 97f engine default.
    var foldedFramesPerChunk = Self.resolvedLTX2Frames(requestFrames: req.frames, videoConfigDefaults: videoConfigDefaults)
    var foldedExtendSeconds = req.extendToSeconds
      ?? Self.extendSecondsFromDuration(req.duration, framesPerChunk: foldedFramesPerChunk, fps: req.fps ?? 24)
    if foldedExtendSeconds > 0 {
      let fps = req.fps ?? 24
      let targetFrames = Int((foldedExtendSeconds * Float(fps)).rounded())
      if targetFrames <= 289 {
        let singleFrames = min(289, ((max(targetFrames, 9) - 2) / 8) * 8 + 9)  // 1+8k covering target
        foldedFramesPerChunk = max(foldedFramesPerChunk, singleFrames)
        logger.info(
          "LTX-2: folded \(foldedExtendSeconds)s request into a single \(foldedFramesPerChunk)f chunk (continuation chunks degenerate; ≤289f renders single-pass)")
        foldedExtendSeconds = 0  // 0 = no continuation chunks
      }
    }

    let videoRequest = Self.buildLocalVideoRequest(
      req: req, videoPreset: videoPreset,
      effectivePrompt: effectivePrompt, effectiveInitImage: effectiveInitImage,
      renderWidth: renderWidth, renderHeight: renderHeight,
      foldedFramesPerChunk: foldedFramesPerChunk, foldedExtendSeconds: foldedExtendSeconds,
      resolvedLoRAs: resolvedLoRAs, effectiveBeatSchedule: effectiveBeatSchedule,
      resolvedOutput: resolvedOutput)
    // Validate before enqueuing so bad frames/dims fail fast.
    try generator.validate(videoRequest)

    return PreparedLocalVideo(
      generator: generator,
      request: videoRequest,
      mode: (effectiveInitImage?.isEmpty == false) ? .i2v : .t2v,
      source: req.source ?? "api",
      optimizationAttemptId: req.optimizationAttemptId,
      enhancementSkippedReason: enhancementSkippedReason,
      beatScheduleIgnoredReason: beatScheduleIgnoredReason)
  }

  /// If LTX-2 is configured, ASYNC-submit the local render and return 202 + a
  /// job status immediately; otherwise nil so the caller falls through to the
  /// Replicate proxy. Poll GET /v1/video/status/{id} for completion. This is the
  /// path a long (multi-minute / multi-chunk) render must take — it never holds
  /// the HTTP connection open for the whole denoise.
  private func localVideoAsyncResponseIfConfigured(body: Data) async -> RoutedResponse? {
    // #339: local video is never persisted (QueueRecoveryGate.swift) — refuse
    // it explicitly while a post-restart queue replay is in flight rather
    // than hand out a 202 that a second restart could silently lose. Checked
    // only when local LTX-2 IS configured, so an unconfigured engine still
    // falls through to the Replicate cloud path exactly as before (nil).
    if configuration.ltx2WeightsPath != nil, configuration.ltx2GemmaPath != nil {
      let recovery = queueRecoveryState.snapshot()
      if QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: recovery.inProgress) {
        logger.warning("LTX-2: refused async local video submission — persisted-queue replay in flight (#339)")
        return .error(.queueRecovering(remainingKinds: recovery.remainingKinds))
      }
    }
    do {
      guard let prep = try await prepareLocalVideo(body: body) else { return nil }
      logger.info("LTX-2: local video job submitted (\(prep.request.width)x\(prep.request.height), \(prep.request.framesPerChunk)f)")
      var tracePayload: [String: String] = ["prompt": prep.request.prompt]
      if prep.request.audio {
        tracePayload["has_audio"] = "true"
        tracePayload["audio_seconds"] = String(format: "%.2f",
          Float(prep.request.framesPerChunk) / Float(prep.request.fps))
      }
      if let attemptId = prep.optimizationAttemptId {
        tracePayload["optimization_attempt_id"] = attemptId
      }
      // comfybox#328: visible on GET /v1/video/traces (RenderTraceStore.
      // TraceSummary carries both fields explicitly — see finding 2) —
      // confirms a beat_schedule request actually skipped enhancement
      // instead of silently losing its beats to the rewrite.
      if let reason = prep.enhancementSkippedReason {
        tracePayload["enhancement_skipped"] = reason
      }
      if let reason = prep.beatScheduleIgnoredReason {
        tracePayload["beat_schedule_ignored"] = reason
      }
      // Winner actions (2026-08-10): store the sanitized request + the
      // resolved seed/dims so this render_id is replayable — /v1/video/rerender
      // replays it at 720p, /v1/video/extend chains a continuation.
      if let requestJSON = VideoWinnerActions.sanitizedRequestJSON(fromBody: body) {
        tracePayload["request_json"] = requestJSON
      }
      tracePayload["seed"] = String(prep.request.seed)
      tracePayload["width"] = String(prep.request.width)
      tracePayload["height"] = String(prep.request.height)
      tracePayload["frames"] = String(prep.request.framesPerChunk)
      tracePayload["fps"] = String(prep.request.fps)
      if let initImage = prep.request.initImagePath {
        tracePayload["image_path"] = initImage
      }
      let status = videoJobTracker.submit(
        source: prep.source, mode: prep.mode, coordinator: coordinator,
        // Snapshot at SUBMIT time (finding #15): the authoritative resolution
        // this render will use, durable on the job status.
        resolvedConfig: LTX2ConfigResolver.resolveTyped(
          request: prep.request.tuning, preset: prep.request.presetTuning).params,
        tracePayload: tracePayload,
        wantsAudio: prep.request.audio
      ) { report in
        // #1479: the preemptible entry — a no-op unless a `preempt: true`
        // image job raises `ltx2PreemptionSignal` while this render is
        // in-flight; otherwise behaves exactly like `.generate`.
        try prep.generator.generatePreemptible(prep.request) { chunk, totalChunks, step, totalSteps in
          report(Self.localVideoProgressPercent(
            chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps))
          self.ltx2StepPosition.update(chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps)
        }
      }
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      let data = try encoder.encode(status)
      return .json(.rawJSON(status: 202, data: data))
    } catch let error as LTX2VideoError {
      return .error(.error(status: 400, message: error.localizedDescription))
    } catch {
      return .error(response(for: error))
    }
  }

  // MARK: - Winner actions (2026-08-10: 480p/4s standard, improve the keepers)

  private struct VideoRerenderBody: Decodable {
    let renderId: String?
    let path: String?
    let resolution: String?
    /// Tier-A overrides applied ON TOP of the replayed request — the point is
    /// "same seed, same prompt, but rendered differently". The daemon's hq
    /// quality tier sends { two_stage, audio_refine } here so a cheap 480p
    /// single-pass explore can be promoted to a two-pass keeper without
    /// re-rolling the seed.
    let tuning: LTX2VideoTuning?
    /// comfybox#305 (review r1, item 3): same top-level convenience alias
    /// the generate routes accept (comfybox#307/#355) — `LTX2VideoTuning.
    /// merging` folds this into `tuning.two_stage` before it reaches
    /// `VideoWinnerActions.rerenderBody`, so callers can promote a winner to
    /// two-pass without knowing about the nested `tuning` shape.
    let twoPass: Bool?
  }

  private struct VideoExtendBody: Decodable {
    let renderId: String?
    let path: String?
    let seconds: Int?
    let prompt: String?
  }

  /// "1786475197556_ltx2-ABC.mp4" → "ltx2-ABC.mp4" (daemon temp prefix).
  static func stripTimestampPrefix(_ filename: String) -> String {
    guard let underscore = filename.firstIndex(of: "_"),
      filename[filename.startIndex..<underscore].allSatisfy({ $0.isNumber }),
      filename.distance(from: filename.startIndex, to: underscore) >= 10
    else { return filename }
    return String(filename[filename.index(after: underscore)...])
  }

  /// Locate the trace behind a winner action: by render_id directly, or by
  /// matching a gallery clip's path/filename against recent terminal outputs.
  private func findVideoTrace(
    renderId: String?, path: String?
  ) -> (renderId: String, submitted: [String: String], outputPath: String?)? {
    if let id = renderId, !id.isEmpty {
      let events = renderTraceStore.events(renderId: id)
      guard let submitted = events.first(where: { $0.event == .submitted }) else { return nil }
      let terminal = events.last { $0.event == .terminal }
      return (id, submitted.payload, terminal?.payload["output_path"])
    }
    if let path, !path.isEmpty {
      let summaries = renderTraceStore.recentSummaries(limit: 500)
      let filename = (path as NSString).lastPathComponent
      // Daemon-side copies carry a `<epoch-ms>_` temp prefix on the engine's
      // original basename (fetch-before-save path, 2026-08-11) — the sidecar
      // join key recorded that prefixed name for existing clips, so strip a
      // leading digit run before comparing.
      let normalized = Self.stripTimestampPrefix(filename)
      // Exact full-path match first — a newer clip that merely shares a
      // basename must not shadow the clip the caller actually named.
      let match =
        summaries.first { $0.outputPath == path }
        ?? summaries.first {
          guard let base = $0.outputPath.map({ ($0 as NSString).lastPathComponent }) else { return false }
          return base == filename || base == normalized
        }
      if let match, let out = match.outputPath {
        let events = renderTraceStore.events(renderId: match.renderId)
        let submitted = events.first { $0.event == .submitted }
        return (match.renderId, submitted?.payload ?? [:], out)
      }
    }
    return nil
  }

  private func videoRerenderResponse(body: Data) async -> RoutedResponse {
    // comfybox#305 (review r1, item 2): `try?` here swallowed a malformed
    // `tuning` block behind the misleading "Body must include render_id or
    // path" 400 — decode for real and name the actual parse failure.
    let req: VideoRerenderBody
    do {
      req = try decode(VideoRerenderBody.self, from: body)
    } catch {
      return .error(response(for: error))
    }
    guard req.renderId != nil || req.path != nil else {
      return .error(.error(status: 400, message: "Body must include 'render_id' or 'path'"))
    }
    let resolution = req.resolution ?? "720p"
    if let validationError = VideoGenerateRequest.validateResolution(resolution) {
      return .error(.error(status: 400, message: validationError))
    }
    guard let trace = findVideoTrace(renderId: req.renderId, path: req.path) else {
      return .error(.error(status: 404, message: "No render trace matches that render_id/path"))
    }
    guard let requestJSON = trace.submitted["request_json"] else {
      return .error(.error(
        status: 422,
        message:
          "Trace \(trace.renderId) predates replay support (no stored request) — re-render works for clips rendered after this deploy"))
    }
    do {
      let newBody = try VideoWinnerActions.rerenderBody(
        requestJSON: requestJSON,
        resolvedSeed: trace.submitted["seed"],
        effectivePrompt: trace.submitted["prompt"],
        resolution: resolution,
        initImagePath: trace.submitted["image_path"],
        // comfybox#305 (review r1, item 3): fold the top-level `two_pass`
        // convenience alias into `tuning.two_stage` — the same merge the
        // generate routes apply (comfybox#307/#355) — so a rerender caller
        // doesn't need to know the nested `tuning` shape either.
        tuningOverride: LTX2VideoTuning.merging(req.tuning, twoPass: req.twoPass))
      if let routed = await localVideoAsyncResponseIfConfigured(body: newBody) {
        // #339 review r1, item 6: `routed` can now be a 503 (queue recovery
        // gate) — don't log a "submitted" message for a request that was
        // actually refused; `localVideoAsyncResponseIfConfigured` already
        // logged the refusal with its own reason.
        if case .error = routed {} else {
          logger.info("video: winner re-render of \(trace.renderId) at \(resolution)")
        }
        return routed
      }
      return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights)"))
    } catch {
      return .error(.error(status: 422, message: "\(error)"))
    }
  }

  private func videoExtendResponse(body: Data) async -> RoutedResponse {
    guard let req = try? decode(VideoExtendBody.self, from: body),
      req.renderId != nil || req.path != nil
    else {
      return .error(.error(status: 400, message: "Body must include 'render_id' or 'path'"))
    }
    let trace = findVideoTrace(renderId: req.renderId, path: req.path)
    // Source clip: the caller's path when it exists, else the trace's output.
    let clipPath = [req.path, trace?.outputPath].compactMap { $0 }
      .first { FileManager.default.fileExists(atPath: $0) }
    guard let clipPath else {
      return .error(.error(
        status: 404,
        message: "Source clip not found on disk — pass 'path' or a 'render_id' whose output still exists"))
    }
    var framePath: String?
    do {
      // Same containment rule as /v1/video/output: only gallery clips.
      _ = try WarmServerOutputPathValidator.resolveOutputPath(
        clipPath, allowedOutputDirectory: configuration.allowedOutputDirectory)
      let extracted = NSTemporaryDirectory() + "winner-extend-\(UUID().uuidString).png"
      try LastFrameExtractor.extractLastFrame(from: clipPath, to: extracted)
      framePath = extracted
      let newBody = try VideoWinnerActions.extendBody(
        requestJSON: trace?.submitted["request_json"],
        framePath: extracted,
        seconds: req.seconds ?? 4,
        prompt: req.prompt,
        effectivePrompt: trace?.submitted["prompt"])
      if let routed = await localVideoAsyncResponseIfConfigured(body: newBody) {
        // #339 review r1, item 6: `routed` can now be a 503 (queue recovery
        // gate), not only a successful submission. Only the SUCCESS path
        // keeps the frame alive (the queued render reads it when its GPU
        // turn comes; tmp is system-cleaned between boots) and logs
        // "submitted" — a refusal must clean the frame up immediately (no
        // render was queued to ever read it) and must not log a false
        // success (`localVideoAsyncResponseIfConfigured` already logged the
        // refusal with its own reason).
        if case .error = routed {
          if let framePath { try? FileManager.default.removeItem(atPath: framePath) }
        } else {
          logger.info("video: winner extend of \((clipPath as NSString).lastPathComponent) (+\(req.seconds ?? 4)s)")
        }
        return routed
      }
      if let framePath { try? FileManager.default.removeItem(atPath: framePath) }
      return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights)"))
    } catch {
      // No render was queued — the extracted frame would be orphaned.
      if let framePath { try? FileManager.default.removeItem(atPath: framePath) }
      return .error(response(for: error))
    }
  }

  /// If LTX-2 is configured, generate the video locally and return the result
  /// SYNCHRONOUSLY (blocks the HTTP connection for the whole render); otherwise
  /// nil so the caller falls through to the Replicate proxy. Kept for backward
  /// compatibility — new/long renders should use the async path above.
  private func localVideoResponseIfConfigured(body: Data) async -> RoutedResponse? {
    guard configuration.ltx2WeightsPath != nil, configuration.ltx2GemmaPath != nil else {
      return nil
    }
    // #339: same refusal as the async route — see its comment. The sync
    // route holds the HTTP connection open for the whole render, but the
    // job it starts is exactly as unpersisted, so it is exactly as unsafe
    // to accept mid-replay.
    let syncRecovery = queueRecoveryState.snapshot()
    if QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: syncRecovery.inProgress) {
      logger.warning("LTX-2: refused sync local video submission — persisted-queue replay in flight (#339)")
      return .error(.queueRecovering(remainingKinds: syncRecovery.remainingKinds))
    }
    do {
      guard let prep = try await prepareLocalVideo(body: body) else { return nil }
      let generator = prep.generator
      let videoRequest = prep.request

      logger.info("LTX-2: local video request queued (\(videoRequest.width)x\(videoRequest.height), \(videoRequest.framesPerChunk)f)")
      let result = try await coordinator.enqueueLocalVideo(wantsAudio: videoRequest.audio) { report in
        // #1479: preemptible entry — see the async path's doc comment above.
        try generator.generatePreemptible(videoRequest) { chunk, totalChunks, step, totalSteps in
          report(Self.localVideoProgressPercent(
            chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps))
          self.ltx2StepPosition.update(chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps)
        }
      }
      auditLog.append(kind: "video.local", message: "LTX-2 video \(result.frameCount)f -> \(result.outputPath)")
      return .json(status: 200, payload: LocalVideoResponse(
        success: true,
        outputPath: result.outputPath,
        frameCount: result.frameCount,
        durationSeconds: result.durationSeconds,
        elapsedSeconds: result.elapsedSeconds,
        backend: "ltx2-local",
        enhancementSkipped: prep.enhancementSkippedReason,
        beatScheduleIgnored: prep.beatScheduleIgnoredReason,
        refineSkipped: result.refineSkippedReason
      ))
    } catch let error as LTX2VideoError {
      return .error(.error(status: 400, message: error.localizedDescription))
    } catch {
      // comfybox#322: an operator interrupt is not a server failure, but the
      // STATUS CODE stays 500 (review r1 ruling): images and video must match,
      // and introducing a 499 would be an unversioned HTTP-level change to a
      // production contract (intent.md) — a separate decision from this fix.
      // The message names the cause instead of the bare "CancellationError()",
      // and the async status JSON carries `interrupted: true`.
      if isRenderInterruption(error) {
        logger.info("LTX-2: synchronous video render interrupted by /v1/queue/interrupt.")
        return .error(.error(status: 500, message: "LTX-2 video interrupted by /v1/queue/interrupt"))
      }
      return .error(.error(status: 500, message: "LTX-2 video failed: \(error.localizedDescription)"))
    }
  }

  // MARK: - #1479 preemption entry point
  //
  // Called from BOTH `/v1/generate` and `/v1/generate/async` route handlers,
  // BEFORE either calls into `coordinator` — see the block comment above
  // `RollingMeanSec` for why this cannot itself be an actor method. Returns
  // `.notApplicable` when the flag is absent/false, no video is rendering, or
  // a preemption is already in flight (nested preemption refused, spec) — the
  // caller then falls through to its normal enqueue, unmodified, which is
  // exactly pre-#1479 behaviour.

  /// Result of an `attemptPreemption` call. `fileprivate` (not `private`)
  /// because `ImageJobTracker.submitPreempting`, a sibling top-level type in
  /// this same file, needs to name it in its own parameter type.
  enum PreemptionOutcome {
    /// Nothing to do — caller enqueues normally, unmodified.
    case notApplicable
    /// The refusal guard declined (finishing beats preempting). `eta` is the
    /// projected remaining seconds — callers stamp it onto the normal queued
    /// response/status as `preempt_refused`/`eta_sec`.
    case refused(eta: Double)
    /// The preemption ran end-to-end and the image job completed.
    case ran(GenerateResponse)
    /// The preemption ran end-to-end and the image job itself failed (the
    /// video still resumed — that failure is independent of this one).
    case ranFailed(Error)
  }

  /// `evictReloadRoundTripSec` for the refusal guard: sum of the two observed
  /// rolling means, nil until BOTH have at least one sample (spec: never
  /// refuse on a guess).
  private func combinedEvictReloadRoundTripSec() -> Double? {
    guard let evictMean = ltx2EvictMean.mean(), let reloadMean = ltx2ReloadMean.mean() else { return nil }
    return evictMean + reloadMean
  }

  private func attemptPreemption(
    _ payload: GeneratePayload, source: String, rawBody: Data?, jobId: String? = nil
  ) async -> PreemptionOutcome {
    guard payload.preempt == true, videoHolder.isRendering() else { return .notApplicable }
    guard preemptionInFlight.trySet() else {
      // Nested preemption refused (spec) — a preemptor cannot itself be
      // preempted. Falls through as a normal enqueue.
      return .notApplicable
    }

    let tv = ltx2Telemetry.view()
    let stepsRemaining = ltx2StepPosition.read() ?? 0
    var remainingPhaseMeans: [Double] = []
    if let currentPhase = tv.currentPhase.flatMap(LTX2Phase.init(rawValue:)) {
      for phase in LTX2Phase.allCases where LTX2UnwindGuard.rank(phase) > LTX2UnwindGuard.rank(currentPhase) {
        if let m = tv.phases[phase.rawValue]?.meanSec { remainingPhaseMeans.append(m) }
      }
    }
    if let eta = preemptionRefusalETA(
      stepsRemaining: stepsRemaining, meanStepSec: tv.meanStepSec,
      remainingPhaseMeansSec: remainingPhaseMeans, evictReloadRoundTripSec: combinedEvictReloadRoundTripSec()
    ) {
      preemptionInFlight.clear()
      return .refused(eta: eta)
    }

    // #1479 (review C1, second half): re-check right before raising — the
    // render may have finished on its own in the window between the
    // isRendering()/guard checks above and here (telemetry/guard evaluation
    // takes real, if small, time). Raising into nothing would strand the
    // signal for a FUTURE, unrelated render to observe at its very first
    // unwind point (`resume == nil && isRaised` — instant, near-zero-cost
    // false checkpoint; see the `.localVideo` defer fix for the other half
    // of this bug).
    guard videoHolder.isRendering() else {
      preemptionInFlight.clear()
      return .notApplicable
    }

    logger.info("#1479: preempting in-flight video render for image job (source=\(source))")

    // Checkpoint-failure fallback window (spec, Error handling: "Checkpoint
    // fails -> refuse the preemption, keep rendering"): if the render doesn't
    // observe the signal and yield within this window — it may be deep in an
    // uninterruptible phase, or may simply finish on its own first — clear
    // the signal and run the image job WITHOUT preemption rather than lose it
    // or hang forever waiting for a yield that may never come.
    let windowSec = tv.meanStepSec.map { $0 * 2 + 30 } ?? 120

    do {
      let response: GenerateResponse = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<GenerateResponse, Error>) in
        // #1479 (review I1): the token minted here is what makes this
        // episode's watchdog harmless once the episode is over. Without it a
        // watchdog that wakes late — after its own episode completed and a
        // LATER image job armed the box and raised a fresh signal — would
        // claim the NEXT preemptor, clear its raise and its
        // `preemptionInFlight` flag, and quietly run it unpreempted while its
        // continuation was still outstanding. `claim(matching:)` below only
        // succeeds against the entry this very call parked.
        let episodeToken = pendingPreemptorBox.set(
          .init(payload: payload, source: source, rawBody: rawBody, jobId: jobId, continuation: ContinuationBox(cont)))
        ltx2PreemptionSignal.raise()
        Task { [weak self] in
          try? await Task.sleep(nanoseconds: UInt64(max(0, windowSec) * 1_000_000_000))
          guard let self, let claimed = self.pendingPreemptorBox.claim(matching: episodeToken) else { return }
          self.logger.error("#1479: checkpoint fallback — video render did not yield within \(windowSec)s; running image job without preemption")
          self.ltx2PreemptionSignal.clear()
          self.preemptionInFlight.clear()
          do {
            let result = try await self.coordinator.enqueueGenerate(claimed.payload, source: claimed.source, rawBody: claimed.rawBody)
            claimed.continuation.resume(returning: result)
          } catch {
            claimed.continuation.resume(throwing: error)
          }
        }
      }
      return .ran(response)
    } catch {
      return .ranFailed(error)
    }
  }

  // Queue ----------------------------------------------------------------------

  /// GET /v1/queue: the active operation + every pending job (cancellable by id).
  /// GET /v1/queue — served from the lock-based ``LiveHealthState`` snapshot
  /// instead of `await coordinator.queueSnapshot()` so the Queue tab stays
  /// responsive during a render, matching the /health fix (#217). Hopping
  /// onto the actor here queued this request behind the whole render and
  /// also read `isRendering` off a stale field, so the tab showed an empty,
  /// not-rendering queue while a job was actually active.
  private func queueListResponse() async -> RoutedResponse {
    guard let data = buildQueuePayloadData() else {
      return .error(.error(status: 500, message: "Failed to serialize queue snapshot"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }

  /// GET /v1/queue payload, shared by the async arm and the sync control plane.
  /// Composes the actor-authored snapshot with the lock store's UNDRAINED deltas
  /// (§3.1.4a point 5), so a just-issued cancel/move is reflected IMMEDIATELY —
  /// before the actor next drains — rather than after the in-flight render.
  /// comfybox#283/#217: additive — a `QueueLifecycleEvent` as a snake_case
  /// JSON dictionary, for embedding into the raw `[String: Any]` payloads
  /// this route (and `/health`) build via `JSONSerialization` rather than
  /// `Codable`. `nil` fields are simply absent, matching every other
  /// dictionary this route already builds by hand. No `dateEncodingStrategy`
  /// override needed (PR #370 review I4): `QueueLifecycleEvent` formats its
  /// own `wallTime` field as ISO8601 regardless of the ambient encoder.
  private static func lifecycleEventDict(_ event: QueueLifecycleEvent) -> [String: Any]? {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    guard let data = try? encoder.encode(event),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    return obj
  }

  /// comfybox#283/#217: `GET /v1/queue/lifecycle` payload, shared by the
  /// async arm and the sync control plane (review I5) so both emit
  /// identical bytes — same pattern as `buildQueuePayloadData` above.
  private struct QueueLifecycleListResponse: Encodable {
    let bootId: String
    let count: Int
    let events: [QueueLifecycleEvent]
  }

  private func queueLifecyclePayloadData(request: HTTPRequest) -> Data? {
    let jobId = request.queryParameters["job_id"]
    let requestedLimit = request.queryParameters["limit"].flatMap { Int($0) }
    let limit = max(1, min(requestedLimit ?? 200, 2000))
    let events = lifecycleLedger.events(jobId: jobId, limit: limit)
    let payload = QueueLifecycleListResponse(bootId: lifecycleLedger.bootId, count: events.count, events: events)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    return try? encoder.encode(payload)
  }

  private func buildQueuePayloadData() -> Data? {
    let (snap, progress) = liveHealth.read()
    let pending = QueueDeltaApplier.apply(liveHealth.undrainedDeltas(), to: snap.pending, id: { $0.id })
    let iso = ISO8601DateFormatter()
    var payload: [String: Any] = [
      "is_rendering": snap.isRendering,
      "is_paused": snap.isPaused,
      "max_pending": snap.maxPending,
      "render_count": snap.renderCount,
      "failed_count": snap.failedRenderCount,
      "pending": pending.map { job in
        var entry: [String: Any] = [
          "id": job.id,
          "kind": job.kind,
          "summary": job.summary,
          "source": job.source,
          "enqueued_at": iso.string(from: job.enqueuedAt),
        ]
        // comfybox#283/#217: additive — the last recorded lifecycle event for
        // this pending job (usually `.enqueued`), absent for anything the
        // ledger has never seen (e.g. a job enqueued before this instrument
        // existed and still sitting in a recovered snapshot).
        if let last = lifecycleLedger.lastEvent(jobId: job.id), let dict = Self.lifecycleEventDict(last) {
          entry["last_event"] = dict
        }
        return entry
      },
    ]
    if let id = snap.activeJobId { payload["active_job_id"] = id }
    if let summary = snap.activeSummary { payload["active_summary"] = summary }
    if let source = snap.activeSource { payload["active_source"] = source }
    if let started = snap.activeRenderStartedAt { payload["active_started_at"] = iso.string(from: started) }
    if let pct = progress { payload["progress_percent"] = pct }
    // comfybox#283/#217: additive — the last recorded lifecycle event for the
    // active job, giving `/v1/queue` the one operator-visible signal #283
    // finding 1 found missing (e.g. `kind: "replayed_after_restart"` right
    // after a bounce, distinguishing a recovered job from a brand-new one).
    if let activeId = snap.activeJobId, let last = lifecycleLedger.lastEvent(jobId: activeId),
       let dict = Self.lifecycleEventDict(last) {
      payload["active_last_event"] = dict
    }
    // #1479: LTX-2 phase telemetry — additive, lock-based (no actor hop).
    let tv = ltx2Telemetry.view()
    if let phase = tv.currentPhase { payload["phase"] = phase }
    if let m = tv.maxUninterruptibleSec { payload["max_uninterruptible_sec"] = m }
    payload["phase_timings"] = tv.phases.mapValues { ["mean_sec": $0.meanSec, "samples": $0.samples] }
    return try? JSONSerialization.data(withJSONObject: payload)
  }

  // MARK: - 0.B-2 sync control plane (FDD §3.1.4)

  /// Serve the SYNC-SERVABLE control set on the caller's OWN connection queue,
  /// synchronously — zero cooperative threads, no actor hop — so these routes
  /// answer even when the pool is exhausted by a blocking render. Returns nil for
  /// anything not classified (→ the async `respond` path). Consulted only when
  /// `ControlPlaneSyncFlag.isEnabled`; with the flag off this is never called and
  /// every route falls through to today's async dispatch byte-for-byte.
  fileprivate func serveControlPlaneSync(_ request: HTTPRequest) -> HTTPResponse? {
    guard ControlPlaneClassifier.isSyncServable(method: request.method, path: request.path) else {
      return nil
    }
    switch (request.method, request.path) {
    case ("GET", "/health"):       return healthRouteResponse()
    case ("GET", "/v1/queue"):     return syncQueueResponse()
    case ("GET", "/v1/queue/lifecycle"): return syncQueueLifecycleResponse(request: request)
    case ("GET", "/v1/models"):    return syncModelsResponse()
    case ("GET", "/v1/model/family"): return syncModelFamilyResponse(request: request)
    case ("GET", "/v1/stats"):     return syncStatsResponse()
    case ("GET", "/v1/config"):    return syncConfigResponse()
    case ("GET", "/v1/controls"):  return controlsResponse()
    case ("POST", "/v1/queue/pause"):     return syncPauseResponse(paused: true)
    case ("POST", "/v1/queue/resume"):    return syncPauseResponse(paused: false)
    case ("POST", "/v1/queue/clear"):     return syncClearResponse()
    case ("POST", "/v1/queue/interrupt"): return syncInterruptResponse(request: request)
    case ("POST", _) where request.path.hasPrefix("/v1/queue/") && request.path.hasSuffix("/move"):
      return syncMoveResponse(request: request)
    case ("DELETE", _) where request.path.hasPrefix("/v1/queue/"):
      return syncCancelResponse(request: request)
    default:
      return nil
    }
  }

  static func modelsPayloadData() -> Data? {
    let models = ComfyBoxModelRegistry.allModels.map { model -> [String: Any] in
      [
        "id": model.id,
        "family": model.family.rawValue,
        "variant": model.variant.rawValue,
        "quantization": model.quantization.rawValue,
        "display_name": model.displayName,
        "description": model.description,
        "parameters_b": model.parametersBillions,
        "default_steps": model.defaultSteps,
        "default_guidance": model.defaultGuidance,
        "supports_guidance": model.supportsGuidance,
        "supports_lora": model.supportsLoRA,
        "supports_controlnet": model.supportsControlNet,
        "supports_img2img": model.supportsImg2Img,
        "default_resolution": "\(model.defaultWidth)x\(model.defaultHeight)",
        "estimated_vram_gb": model.estimatedVRAM_GB,
        "huggingface_id": model.huggingFaceId,
      ] as [String: Any]
    }
    return try? JSONSerialization.data(withJSONObject: ["models": models, "count": models.count])
  }

  /// GET /v1/config payload, shared by the async arm and the sync control plane
  /// so both emit identical bytes. FDD §3.3, D3: reads the lock-serialized
  /// ``ServerConfigStore`` (an in-memory snapshot) rather than the disk —
  /// Phase-0-compatible (no I/O, no actor hop on the request path).
  static func configPayloadData() -> (data: Data, etag: String)? {
    let snapshot = ServerConfigStore.shared.current()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    guard let data = try? encoder.encode(snapshot.config) else { return nil }
    return (data, snapshot.etag)
  }

  /// GET /v1/config route handler (async path).
  static func configGetResponse() -> RoutedResponse {
    guard let (data, etag) = Self.configPayloadData() else {
      return .error(.error(status: 500, message: "Failed to serialize config"))
    }
    var response = HTTPResponse.rawJSON(status: 200, data: data)
    response.extraHeaders["ETag"] = etag
    return .json(response)
  }

  /// PUT /v1/config route handler (async path): full-document replace through
  /// ``ServerConfigStore``. `If-Match` is advisory — present-and-stale is `409`;
  /// absent proceeds with a deprecation `Warning` (FDD §3.3).
  static func configPutResponse(request: HTTPRequest) -> RoutedResponse {
    do {
      let updated = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: request.body)
      let ifMatch = request.headers["if-match"]
      let snapshot = try ServerConfigStore.shared.replace(with: updated, ifMatch: ifMatch)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let data = try encoder.encode(snapshot.config)
      var response = HTTPResponse.rawJSON(status: 200, data: data)
      response.extraHeaders["ETag"] = snapshot.etag
      if ifMatch == nil {
        response.extraHeaders["Warning"] =
          "299 - \"PUT /v1/config without If-Match is deprecated; migrate to PATCH /v1/config\""
      }
      return .json(response)
    } catch let error as ServerConfigStoreError {
      if case .etagMismatch = error {
        return .error(.error(status: 409, message: error.description))
      }
      return .error(.error(status: 400, message: error.description))
    } catch {
      return .error(.error(status: 400, message: "Invalid config: \(error.localizedDescription)"))
    }
  }

  /// PATCH /v1/config route handler (async path): RFC 7386 JSON Merge Patch,
  /// merged inside ``ServerConfigStore``'s lock against the current document —
  /// the primary write path going forward (FDD §3.3).
  static func configPatchResponse(request: HTTPRequest) -> RoutedResponse {
    do {
      guard let patchObject = try JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
        return .error(.error(status: 400, message: "Invalid merge-patch body: expected a JSON object"))
      }
      let ifMatch = request.headers["if-match"]
      let snapshot = try ServerConfigStore.shared.applyMergePatch(patchObject, ifMatch: ifMatch)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let data = try encoder.encode(snapshot.config)
      var response = HTTPResponse.rawJSON(status: 200, data: data)
      response.extraHeaders["ETag"] = snapshot.etag
      return .json(response)
    } catch let error as ServerConfigStoreError {
      if case .etagMismatch = error {
        return .error(.error(status: 409, message: error.description))
      }
      return .error(.error(status: 400, message: error.description))
    } catch {
      return .error(.error(status: 400, message: "Invalid merge-patch: \(error.localizedDescription)"))
    }
  }

  private func syncQueueResponse() -> HTTPResponse {
    guard let data = buildQueuePayloadData() else {
      return .error(status: 500, message: "Failed to serialize queue snapshot")
    }
    return .rawJSON(status: 200, data: data)
  }

  /// comfybox#283/#217 review I5.
  private func syncQueueLifecycleResponse(request: HTTPRequest) -> HTTPResponse {
    guard let data = queueLifecyclePayloadData(request: request) else {
      return .error(status: 500, message: "Failed to serialize queue lifecycle")
    }
    return .rawJSON(status: 200, data: data)
  }

  private func syncModelsResponse() -> HTTPResponse {
    guard let data = Self.modelsPayloadData() else {
      return .error(status: 500, message: "Failed to serialize models")
    }
    return .rawJSON(status: 200, data: data)
  }

  /// comfybox#359: shared with the async arm (`GET /v1/model/family`) so
  /// both emit identical bytes — pure file-existence detection, no actor hop.
  private func syncModelFamilyResponse(request: HTTPRequest) -> HTTPResponse {
    Self.modelFamilyRouteResponse(request: request)
  }

  /// The whole sync `GET /v1/model/family` route as a pure function of the
  /// request — no `WarmServer` state needed (pure file-existence detection),
  /// so `WarmServerQueueProbe.syncModelFamilyRoute` can drive it directly in
  /// a unit test without constructing a `WarmServer` (`init` starts the
  /// local-video readiness monitor and scans the LIVE `~/Models/loras`
  /// library, which agents must not touch — same reason `interruptRouteResponse`
  /// is static).
  ///
  /// comfybox#380: this is what exercises the real `request.queryParameters`
  /// percent-decoding for a query value with a space/`#`/`%25`/non-ASCII text,
  /// rather than calling `ModelFamilyDetector.detect` directly and skipping
  /// the query-parsing layer the bug actually lived in.
  fileprivate static func modelFamilyRouteResponse(request: HTTPRequest) -> HTTPResponse {
    guard let spec = request.queryParameters["model"]?.trimmingCharacters(in: .whitespacesAndNewlines),
          !spec.isEmpty else {
      return .error(status: 400, message: "model query parameter is required")
    }
    return .json(status: 200, payload: ModelFamilyDetector.detect(spec: spec))
  }

  private func syncConfigResponse() -> HTTPResponse {
    guard let (data, etag) = Self.configPayloadData() else {
      return .error(status: 500, message: "Failed to serialize config")
    }
    var response = HTTPResponse.rawJSON(status: 200, data: data)
    response.extraHeaders["ETag"] = etag
    return response
  }

  /// GET /v1/controls — Phase 4 discovery (FDD §3.4, D4), served identically by
  /// the async arm and the sync control plane (0.B-2 classified: a lock read of
  /// ServerConfigStore, a small-file ContentModeStore read — the same cost the
  /// config/content-mode routes already pay — and the lock-based queue
  /// snapshot; no actor hop, no cooperative threads). Values are resolved
  /// per-request by dereferencing each descriptor's `read.pointer`; the
  /// registry never caches a copy (§3.4's one rule).
  private func controlsResponse() -> HTTPResponse {
    let queueDocument = buildQueuePayloadData()
      .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
    guard let data = ControlRegistry.controlsPayload(
      config: ServerConfigStore.shared.current().config,
      contentModes: ContentModeStore.loadOrCreate(),
      queueDocument: queueDocument)
    else {
      return .error(status: 500, message: "Failed to serialize controls")
    }
    return .rawJSON(status: 200, data: data)
  }

  /// Stats without an actor hop: the render counters come from the lock snapshot
  /// (the same numbers `coordinator.queueStatus()` returns, published on every
  /// transition), so this answers during a render.
  private func syncStatsResponse() -> HTTPResponse {
    let (snap, _) = liveHealth.read()
    let config = ServerConfigStore.shared.current().config
    let snapshot = statsProvider.snapshot(
      memory: statsProvider.sampleMemoryStatus(),
      uptimeSeconds: StatsProvider.uptimeSeconds(startTime: serverStartTime),
      renderCount: snap.renderCount,
      failedRenderCount: snap.failedRenderCount,
      pendingCount: snap.pending.count,
      config: config)
    return .json(status: 200, payload: snapshot)
  }

  private func syncPauseResponse(paused: Bool) -> HTTPResponse {
    struct PauseResult: Encodable { let success: Bool; let paused: Bool }
    // Authoritative + persisted, immediately visible via LiveHealthState.read().
    liveHealth.setPaused(paused)
    if !paused {
      // The wake (§3.1.4a point 1) — fire-and-forget, NEVER a mailbox command.
      // resume's only job through the actor is to (re)start the parked loop;
      // decoupling the ACK from that effect is what avoids the v1 wedge.
      Task { await coordinator.setPaused(false) }
    }
    auditLog.append(kind: "queue.pause", message: paused ? "Queue paused" : "Queue resumed")
    // F-1 (adversarial review): BOTH arms return 200. The authoritative
    // lock-store write completes before this response is built (resume’s wake
    // is fire-and-forget, but `isPaused` itself is already false), and clients
    // guard pause/resume on 200 — a 202 here made every UI resume throw while
    // the engine actually resumed.
    return .json(status: 200, payload: PauseResult(success: true, paused: paused))
  }

  private func syncClearResponse() -> HTTPResponse {
    struct ClearResult: Encodable { let success: Bool; let cleared: Int }
    // Record a cancel delta for each currently-pending job (composed view), applied
    // at the next drain; the jobs disappear from the composed GET /v1/queue at once.
    let (snap, _) = liveHealth.read()
    let pending = QueueDeltaApplier.apply(liveHealth.undrainedDeltas(), to: snap.pending, id: { $0.id })
    for job in pending { liveHealth.recordDelta(.cancel(job.id)) }
    if !pending.isEmpty { Task { await coordinator.drainControlDeltas() } }
    auditLog.append(kind: "queue.clear", message: "Cleared \(pending.count) pending job(s)")
    return .json(status: 200, payload: ClearResult(success: true, cleared: pending.count))
  }

  private func syncInterruptResponse(request: HTTPRequest) -> HTTPResponse {
    Self.interruptRouteResponse(request: request, liveHealth: liveHealth, auditLog: auditLog)
  }

  /// The whole sync `/v1/queue/interrupt` route, as a function of the only two
  /// things it needs — the lock store it cancels through and the audit log it
  /// records to.
  ///
  /// comfybox#362 review r2, item 3: `WarmServerQueueProbe.syncInterruptRoute`
  /// calls THIS, so the route test drives the same decode → cancel → audit →
  /// encode chain the server runs, rather than a copy of it that could drift.
  /// It is static because a unit test cannot construct a `WarmServer`:
  /// `init` starts the local-video readiness monitor and scans the LIVE
  /// `~/Models/loras` library, which agents must not touch.
  ///
  /// `target` is additive — see `InterruptTarget` for the vocabulary.
  fileprivate static func interruptRouteResponse(
    request: HTTPRequest, liveHealth: LiveHealthState, auditLog: AuditLog
  ) -> HTTPResponse {
    let target = InterruptRoute.decodeTarget(from: request.body)
    let outcome = liveHealth.cancelActiveRender(target: target)
    let (_, body) = InterruptRouteResponse.build(from: outcome)
    auditLog.append(
      kind: "queue.interrupt",
      message: InterruptRoute.auditMessage(for: body),
      metadata: target.map { ["target": $0] } ?? [:])
    return InterruptRoute.response(for: outcome)
  }

  private func syncMoveResponse(request: HTTPRequest) -> HTTPResponse {
    struct MoveResult: Encodable { let success: Bool; let moved: Bool }
    let mid = request.path.dropFirst("/v1/queue/".count).dropLast("/move".count)
    guard let id = Self.pathIdComponent(String(mid)) else {
      return .error(status: 400, message: "Invalid job id")
    }
    struct MoveBody: Decodable { let direction: String }
    let direction = (try? JSONDecoder().decode(MoveBody.self, from: request.body))?.direction ?? "up"
    let (snap, _) = liveHealth.read()
    let pending = QueueDeltaApplier.apply(liveHealth.undrainedDeltas(), to: snap.pending, id: { $0.id })
    let present = pending.contains { $0.id == id }
    if present {
      liveHealth.recordDelta(.move(id, direction: direction))
      Task { await coordinator.drainControlDeltas() }
      auditLog.append(kind: "queue.move", message: "Moved job \(id) \(direction)", metadata: ["id": id, "direction": direction])
    }
    return .json(status: 200, payload: MoveResult(success: true, moved: present))
  }

  private func syncCancelResponse(request: HTTPRequest) -> HTTPResponse {
    guard let id = Self.pathIdComponent(String(request.path.dropFirst("/v1/queue/".count))) else {
      return .error(status: 400, message: "Invalid job id")
    }
    let (snap, _) = liveHealth.read()
    let pending = QueueDeltaApplier.apply(liveHealth.undrainedDeltas(), to: snap.pending, id: { $0.id })
    guard pending.contains(where: { $0.id == id }) else {
      return .error(status: 404, message: "Job not pending: \(id)")
    }
    liveHealth.recordDelta(.cancel(id))
    Task { await coordinator.drainControlDeltas() }
    auditLog.append(kind: "queue.cancel", message: "Recorded cancel for pending job \(id)", metadata: ["id": id])
    // F-3 (adversarial review): between the presence read above and the drain,
    // the loop may dequeue-and-start this job — so the sync path must never
    // claim `deleted: true`. Record the delta, ACK the recording, and let the
    // composed GET /v1/queue (where the job is already absent) and job status
    // tell the truth. The flag-off async arm still reports deleted:true
    // because it actually removes the job before responding.
    return .json(status: 202, payload: SyncCancelAccepted.ack(id: id))
  }

  // Prompt enhancement --------------------------------------------------------

  /// POST /v1/enhance body (snake_case over the wire).
  private struct EnhanceRequest: Decodable {
    let prompt: String
    let character: String?
    let characterDescription: String?
    let contentMode: String?
    /// Target model family: "image" (Z-Image, default) or "video" (LTX). Selects
    /// the prompt FORMAT — image uses YOUR CONTEXT/YOUR PHOTO, video uses LTX-2.3
    /// cinematic prose.
    let mediaKind: String?
  }

  /// Enhance a prompt through the configured prompt-optimization provider
  /// (Settings → AI Providers; e.g. Dan's heresy model on LM Studio). Falls
  /// back to the raw prompt when the provider is unreachable — the optimizer
  /// never blocks a render.
  private func enhancePromptResponse(body: Data) async -> RoutedResponse {
    guard let req = try? decode(EnhanceRequest.self, from: body),
          !req.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else {
      return .error(.error(status: 400, message: "'prompt' is required"))
    }

    let config = ServerConfigStore.shared.current().config
    guard let endpoint = config.providers.promptOptimization else {
      return .error(.error(
        status: 503,
        message: "No prompt-optimization provider configured (Settings → AI Providers)"))
    }

    // PromptOptimizer appends /v1/chat/completions itself; the configured
    // baseUrl is an OpenAI-style root that usually already ends in /v1.
    var base = endpoint.baseUrl
    while base.hasSuffix("/") { base.removeLast() }
    if base.hasSuffix("/v1") { base = String(base.dropLast(3)) }
    while base.hasSuffix("/") { base.removeLast() }

    let optimizer = PromptOptimizer(
      configuration: PromptOptimizer.Configuration(
        ollamaBaseURL: base,
        lmStudioBaseURL: nil,
        model: endpoint.model,
        timeoutSeconds: 90,
        enabled: true
      ),
      logger: logger
    )

    // Resolve a named character to its mode-gated description when the
    // caller didn't supply one.
    let mode = req.contentMode ?? ContentModeManager.Mode.neutral.rawValue
    var characterDescription = req.characterDescription
    if characterDescription == nil, let name = req.character,
       let entry = await characterStore.get(CharacterEntry.slug(name)) {
      characterDescription = entry.resolvedDescription(
        for: ContentModeManager.Mode(rawValue: mode) ?? .neutral)
    }

    let result = await optimizer.optimize(
      prompt: req.prompt,
      character: req.character,
      characterDescription: characterDescription,
      contentMode: mode,
      mediaKind: req.mediaKind ?? "image"
    )

    // Task #19 (Codex finding #6): a server-minted attempt id bound to
    // input, result, template and outcome — render submissions reference
    // this instead of shipping client-echoed strings. Persisted as a trace
    // event so the lineage survives the 1h job prune.
    let attemptId = "opt-" + UUID().uuidString
    renderTraceStore.append(RenderTraceEvent(
      renderId: attemptId, event: .terminal, taskKind: .videoRender,
      payload: [
        "kind": "optimization_attempt",
        "intent": req.prompt,
        "optimized": result.prompt,
        "outcome": result.outcome,
        "template_id": result.templateId ?? "",
        "template_hash": result.templateHash ?? "",
        "template_source": result.templateSource ?? "",
        "media_kind": req.mediaKind ?? "image",
        "content_mode": mode,
      ]))

    var payload: [String: Any] = [
      "success": true,
      "prompt": result.prompt,
      "enhanced": result.enhanced,
      "optimization_attempt_id": attemptId,
      "optimizer_outcome": result.outcome,
    ]
    if let tid = result.templateId { payload["template_id"] = tid }
    if let th = result.templateHash { payload["template_hash"] = th }
    if let note = result.note { payload["note"] = note }
    guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
      return .error(.error(status: 500, message: "Failed to serialize enhance response"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }

  // Characters ---------------------------------------------------------------

  private func listCharactersResponse() async -> RoutedResponse {
    .json(status: 200, payload: await characterStore.list())
  }

  private func getCharacterResponse(rawId: String) async -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid character id"))
    }
    guard let character = await characterStore.get(id) else {
      return .error(.error(status: 404, message: "Character not found: \(id)"))
    }
    return .json(status: 200, payload: character)
  }

  private func upsertCharacterResponse(body: Data) async -> RoutedResponse {
    do {
      let character = try decode(CharacterEntry.self, from: body)
      guard !character.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        return .error(.error(status: 400, message: "Character 'name' is required"))
      }
      let saved = await characterStore.upsert(character)
      auditLog.append(
        kind: "character.upsert",
        message: "Upserted character \(saved.id)",
        metadata: ["id": saved.id, "name": saved.name]
      )
      return .json(status: 200, payload: saved)
    } catch {
      return .error(.error(status: 400, message: "Invalid character payload: \(error.localizedDescription)"))
    }
  }

  private func deleteCharacterResponse(rawId: String) async -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid character id"))
    }
    let deleted = await characterStore.delete(id)
    if deleted {
      auditLog.append(kind: "character.delete", message: "Deleted character \(id)", metadata: ["id": id])
    }
    return .json(status: deleted ? 200 : 404, payload: DeleteResult(success: deleted, id: id, deleted: deleted))
  }

  // Presets ------------------------------------------------------------------

  private func presetsListResponse() -> RoutedResponse {
    Self.presetsList(store: presetStore)
  }

  /// `GET /v1/presets` (WP-E20, AC-44c): every preset, flat, with
  /// `invalid` / `invalid_reason` so a flagged preset is visible and
  /// un-selectable by the desktop app, the bridge and MCP alike.
  static func presetsList(store: PresetStore) -> RoutedResponse {
    .json(status: 200, payload: store.listing())
  }

  private func getPresetResponse(rawId: String) -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid preset id"))
    }
    // WP-E20: the single-preset read carries the same validity flag as the list.
    guard let entry = presetStore.listing().first(where: { $0.preset.id == id }) else {
      return .error(.error(status: 404, message: "Preset not found: \(id)"))
    }
    return .json(status: 200, payload: entry)
  }

  private func upsertPresetResponse(body: Data) -> RoutedResponse {
    let (response, saved) = Self.upsertPreset(store: presetStore, body: body)
    if let saved {
      auditLog.append(kind: "preset.upsert", message: "Upserted preset \(saved.id)", metadata: ["id": saved.id])
    }
    return response
  }

  /// `POST`/`PUT /v1/presets` (WP-E20, AC-44b): decode, validate through
  /// `PresetStore.upsert` (recipe-name resolution, range checks, the
  /// deprecated-kroma migration) and persist. A refused preset is a 400
  /// naming the preset and the field; nothing is stored. Returns the saved
  /// preset for the audit log.
  static func upsertPreset(store: PresetStore, body: Data) -> (RoutedResponse, saved: ImagePreset?) {
    do {
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      let preset = try decoder.decode(ImagePreset.self, from: body)
      let saved = try store.upsert(preset)
      return (.json(status: 200, payload: saved), saved)
    } catch let error as PresetStoreError {
      return (presetErrorResponse(error), nil)
    } catch {
      return (.error(.error(status: 400, message: "Invalid preset payload: \(error.localizedDescription)")), nil)
    }
  }

  private func deletePresetResponse(rawId: String) -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid preset id"))
    }
    do {
      let deleted = try presetStore.delete(id)
      if deleted {
        auditLog.append(kind: "preset.delete", message: "Deleted preset \(id)", metadata: ["id": id])
      }
      return .json(status: deleted ? 200 : 404, payload: DeleteResult(success: deleted, id: id, deleted: deleted))
    } catch {
      return .error(.error(status: 500, message: "Failed to delete preset: \(error.localizedDescription)"))
    }
  }

  private func resolvePresetResponse(body: Data) -> RoutedResponse {
    Self.resolvePreset(store: presetStore, body: body)
  }

  /// `POST /v1/presets/resolve`. A preset flagged invalid at load (WP-E20,
  /// AC-44c) is a 400 naming it and the reason — it can never be selected.
  static func resolvePreset(store: PresetStore, body: Data) -> RoutedResponse {
    struct ResolveRequest: Decodable { let id: String }
    do {
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      let request = try decoder.decode(ResolveRequest.self, from: body)
      let resolved = try store.resolve(request.id)
      return .json(status: 200, payload: resolved)
    } catch let error as PresetStoreError {
      return presetErrorResponse(error)
    } catch {
      return .error(.error(status: 400, message: #"Invalid resolve request (expected {"id": ...}): \#(error.localizedDescription)"#))
    }
  }

  /// Map a ``PresetStoreError`` to the right HTTP status: validation -> 400,
  /// invalid (flagged on disk) -> 400, notFound -> 404.
  private static func presetErrorResponse(_ error: PresetStoreError) -> RoutedResponse {
    switch error {
    case .validation(let message):
      return .error(.error(status: 400, message: message))
    case .invalid:
      return .error(.error(status: 400, message: error.description))
    case .notFound(let id):
      return .error(.error(status: 404, message: "Preset not found: \(id)"))
    }
  }

  // Content modes ------------------------------------------------------------

  private func contentModesResponse() -> RoutedResponse {
    // Re-read fresh rather than the `let contentModeStore` snapshot captured at
    // server start: PUT/DELETE below mutate the on-disk store directly (FDD
    // §3.3), so a stale in-memory copy here would hide a write made moments
    // earlier in the same process. `ContentModeStore.loadOrCreate` is cheap
    // (small JSON, same cost the config route already pays per request).
    .json(status: 200, payload: ContentModeStore.loadOrCreate().listModes())
  }

  /// PUT /v1/content-modes/{mode} — FDD §3.3, D3 (Class E). Sets any of
  /// guidanceBoost/promptHint/negativePromptAdditions/styleVariant; fields the
  /// body omits keep their current value (tolerant partial update, matching
  /// `ContentModeDefinition`'s own tolerant decode). `400` on an unknown mode,
  /// unknown `styleVariant`, or an out-of-range `guidanceBoost`.
  private func putContentModeResponse(rawMode: String, body: Data) -> RoutedResponse {
    guard let mode = ContentMode(rawValue: rawMode) else {
      return .error(.error(status: 404, message: "Unknown content mode '\(rawMode)' (expected one of: neutral, banana, avocado)"))
    }
    struct ContentModePatchBody: Decodable {
      let guidanceBoost: Double?
      let promptHint: String?
      let negativePromptAdditions: [String]?
      let styleVariant: String?
    }
    let patch: ContentModePatchBody
    do {
      patch = try JSONDecoder().decode(ContentModePatchBody.self, from: body)
    } catch {
      return .error(.error(status: 400, message: "Invalid content-mode body: \(error.localizedDescription)"))
    }
    var styleVariant: ContentStyleVariant?
    if let rawVariant = patch.styleVariant {
      guard let parsed = ContentStyleVariant(rawValue: rawVariant) else {
        return .error(.error(status: 400, message: "Unknown styleVariant '\(rawVariant)' (expected one of: neutral, sensual, nsfw)"))
      }
      styleVariant = parsed
    }
    if let boost = patch.guidanceBoost, !ContentModeStore.guidanceBoostRange.contains(boost) {
      return .error(.error(
        status: 400,
        message: "guidanceBoost must be between \(ContentModeStore.guidanceBoostRange.lowerBound) and \(ContentModeStore.guidanceBoostRange.upperBound) (got \(boost))"))
    }
    do {
      let updated = try ContentModeStore.update(
        mode: mode,
        guidanceBoost: patch.guidanceBoost,
        promptHint: patch.promptHint,
        negativePromptAdditions: patch.negativePromptAdditions,
        styleVariant: styleVariant)
      auditLog.append(kind: "content_mode.change", message: "Updated content mode '\(mode.rawValue)'")
      return .json(status: 200, payload: updated)
    } catch {
      return .error(.error(status: 400, message: "Failed to save content mode: \(error.localizedDescription)"))
    }
  }

  /// DELETE /v1/content-modes/{mode} — reverts a mode to its built-in
  /// definition rather than removing it (there is always exactly one
  /// definition per ``ContentMode`` case).
  private func deleteContentModeResponse(rawMode: String) -> RoutedResponse {
    guard let mode = ContentMode(rawValue: rawMode) else {
      return .error(.error(status: 404, message: "Unknown content mode '\(rawMode)' (expected one of: neutral, banana, avocado)"))
    }
    let reverted = ContentModeStore.reset(mode: mode)
    auditLog.append(kind: "content_mode.change", message: "Reverted content mode '\(mode.rawValue)' to its built-in definition")
    return .json(status: 200, payload: reverted)
  }

  // Stats / memory -----------------------------------------------------------

  private func statsResponse() async -> RoutedResponse {
    let queue = await coordinator.queueStatus()
    let config = ServerConfigStore.shared.current().config
    let snapshot = statsProvider.snapshot(
      memory: statsProvider.sampleMemoryStatus(),
      uptimeSeconds: StatsProvider.uptimeSeconds(startTime: serverStartTime),
      renderCount: queue.renderCount,
      failedRenderCount: queue.failedCount,
      pendingCount: queue.pendingCount,
      config: config
    )
    return .json(status: 200, payload: snapshot)
  }

  private func memoryResponse() -> RoutedResponse {
    .json(status: 200, payload: statsProvider.sampleMemoryStatus())
  }

  // Audit log ----------------------------------------------------------------

  private func auditLogResponse(query: [String: String]) -> RoutedResponse {
    let limit = query["limit"].flatMap { Int($0) } ?? 100
    let entries = auditLog.recent(limit: max(0, limit))
    // Custom encoder: ISO8601 timestamps (matching the on-disk JSONL). No snake_case
    // conversion — AuditEntry keys are already flat single words, and converting would
    // also mangle arbitrary `metadata` dictionary keys.
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = [.sortedKeys]
    guard let data = try? encoder.encode(entries) else {
      return .error(.error(status: 500, message: "Failed to serialize audit log"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }


  /// Bridge a ComfyUI workflow request to the internal generate pipeline.
  /// Called by ComfyBridgeExecutor via the closure set in init.
  /// Read PNG dimensions from IHDR chunk (bytes 16-23 of a valid PNG).
  private func pngDimensions(from data: Data) -> (width: Int, height: Int)? {
    guard data.count >= 24, data.prefix(Self.pngSignature.count).elementsEqual(Self.pngSignature) else { return nil }
    let w = Int(data[16]) << 24 | Int(data[17]) << 16 | Int(data[18]) << 8 | Int(data[19])
    let h = Int(data[20]) << 24 | Int(data[21]) << 16 | Int(data[22]) << 8 | Int(data[23])
    return (w, h)
  }

  /// Round up to nearest multiple of 16 (for latent alignment).
  private func roundTo16(_ n: Int) -> Int {
    return ((n + 15) / 16) * 16
  }

  /// Default LoRA directory path — matches ComfyBridgeObjectInfo discovery path.
  private static let loraDirectoryPath = ("~/bin/zimage/loras" as NSString).expandingTildeInPath

  /// Default ControlNet directory path — matches ComfyBridgeObjectInfo discovery path.
  private static let controlnetDirectoryPath = ("~/bin/zimage/controlnet" as NSString).expandingTildeInPath
  private static let krea2ControlLoRAPath = "/Volumes/Bolt/Models/krea2-controlnet/depth-control-lora.safetensors"

  private func bridgeGenerate(_ request: ComfyBridgeGenerateRequest, progressCallback: ComfyBridgeProgressHandler?, latentPreviewCallback: ComfyBridgeLatentPreviewHandler? = nil) async throws -> ComfyBridgeGenerateResult {
    // --- Phase 4: Dynamic LoRA swap ---
    // If the workflow contains LoraLoader nodes, swap LoRAs before generating.
    // The coordinator serializes operations, so swap completes before generate starts.
    if !request.loras.isEmpty {
      let loraEntries = request.loras.map { lora -> LoRAEntry in
        // Resolve the LoRA name to a path. Prefer an uploaded LoRA in the bridge
        // dir; otherwise pass the BARE filename so the applicator resolves it
        // against the LoRA library — the same resolution /v1/lora/swap uses.
        // (Blindly prepending the upload dir turned resolvable library LoRAs like
        // "Anneliese_Zbase3.safetensors" into non-existent paths → fileNotFound.)
        let resolvedPath: String
        if lora.name.contains("/") || lora.name.hasPrefix("~") {
          resolvedPath = (lora.name as NSString).expandingTildeInPath
        } else {
          let uploadPath = Self.loraDirectoryPath + "/" + lora.name
          resolvedPath = FileManager.default.fileExists(atPath: uploadPath) ? uploadPath : lora.name
        }
        return LoRAEntry(path: resolvedPath, scale: lora.scale)
      }
      let swapPayload = LoRASwapPayload(loras: loraEntries)
      let swapResult = try await coordinator.enqueueSwap(swapPayload)
      logger.info("WarmServer: bridge LoRA swap complete — \(swapResult.loraCount) LoRA(s) active")
    }

    // Derive dimensions from inpaint image if parser returned 0x0
    // (happens when workflow has no ImageCrop or EmptyLatentImage nodes)
    var genWidth = request.width
    var genHeight = request.height
    if genWidth == 0 || genHeight == 0, let imgData = request.inpaintImageData {
      if let dims = pngDimensions(from: imgData) {
        genWidth = roundTo16(dims.width)
        genHeight = roundTo16(dims.height)
        logger.info("WarmServer: derived dimensions from inpaint image: \(dims.width)x\(dims.height) -> \(genWidth)x\(genHeight)")
      } else {
        genWidth = 1024
        genHeight = 1024
        logger.warning("WarmServer: could not read inpaint image dimensions, falling back to 1024x1024")
      }
    }

    // --- Phase 4: ControlNet routing ---
    // If the workflow contains ControlNet nodes, route to ZImageControlPipeline
    // instead of the standard ZImagePipeline.
    // ControlNet is not supported for Flux 2 or Krea-2 models.
    if request.isControlNet, let controlnetModel = request.controlnetModel {
      let family = await coordinator.modelFamily
      if family == .flux2 || family == .krea2 {
        throw WarmServerError.controlNetNotSupported
      }
      // #339: ControlNet generate closes over resolved temp-file paths, so
      // (like local video) it can never be persisted — refuse before writing
      // any of those temp files rather than accept work that could vanish if
      // the engine restarts again before this replay finishes.
      let controlRecovery = queueRecoveryState.snapshot()
      if QueueRecoveryGate.shouldReject(kind: .controlnet, recoveryInProgress: controlRecovery.inProgress) {
        throw WarmServerError.queueRecoveryInProgress(retryAfterSeconds:
          QueueRecoveryGate.retryAfterSeconds(remainingKinds: controlRecovery.remainingKinds))
      }
      logger.info("WarmServer: routing to ControlNet pipeline — model=\(controlnetModel), strength=\(request.controlnetStrength)")

      // Resolve controlnet model name to a path or HuggingFace ID
      let resolvedControlnetWeights: String
      if controlnetModel.contains("/") || controlnetModel.hasPrefix("~") || controlnetModel.hasPrefix(".") {
        // Already a path or HuggingFace ID — use as-is
        resolvedControlnetWeights = controlnetModel
      } else {
        // Bare name — check if it's a local directory/file in the controlnet dir
        let localPath = Self.controlnetDirectoryPath + "/" + controlnetModel
        if FileManager.default.fileExists(atPath: localPath) {
          resolvedControlnetWeights = localPath
        } else {
          // Treat as HuggingFace ID
          resolvedControlnetWeights = controlnetModel
        }
      }

      // Write control image data to a temp file if we have it
      var controlImageURL: URL? = nil
      if let controlData = request.controlImageData {
        let tempPath = NSTemporaryDirectory() + "zimage-control-\(UUID().uuidString).png"
        try controlData.write(to: URL(fileURLWithPath: tempPath))
        controlImageURL = URL(fileURLWithPath: tempPath)
        logger.info("WarmServer: wrote control image to \(tempPath) (\(controlData.count) bytes)")
      }

      // Write inpaint image to temp file if present
      var inpaintImageURL: URL? = nil
      if let inpaintData = request.inpaintImageData {
        let tempPath = NSTemporaryDirectory() + "zimage-inpaint-\(UUID().uuidString).png"
        try inpaintData.write(to: URL(fileURLWithPath: tempPath))
        inpaintImageURL = URL(fileURLWithPath: tempPath)
      }

      // Write mask to temp file if present
      var maskImageURL: URL? = nil
      if let maskData = request.maskImageData {
        let tempPath = NSTemporaryDirectory() + "zimage-mask-\(UUID().uuidString).png"
        try maskData.write(to: URL(fileURLWithPath: tempPath))
        maskImageURL = URL(fileURLWithPath: tempPath)
      }

      let outputURL = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("zimage-\(UUID().uuidString).png")

      // Build LoRA configurations for the control pipeline
      let controlLoRAs: [LoRAConfiguration] = request.loras.map { lora in
        let resolvedPath: String
        if lora.name.contains("/") || lora.name.hasPrefix("~") {
          resolvedPath = lora.name
        } else {
          resolvedPath = Self.loraDirectoryPath + "/" + lora.name
        }
        return .local(resolvedPath, scale: lora.scale)
      }

      let controlRequest = ZImageControlGenerationRequest(
        prompt: request.prompt,
        negativePrompt: nil,
        controlImage: controlImageURL,
        inpaintImage: inpaintImageURL,
        maskImage: maskImageURL,
        controlContextScale: request.controlnetStrength,
        width: genWidth,
        height: genHeight,
        steps: request.steps,
        guidanceScale: 0.0,
        seed: request.seed,
        outputPath: outputURL,
        model: nil,
        textEncoderPath: configuration.textEncoderPath,
        controlnetWeights: resolvedControlnetWeights,
        // For HuggingFace repos with multiple safetensors, specify the 8-step variant
        controlnetWeightsFile: resolvedControlnetWeights.contains("alibaba-pai")
          ? "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors" : nil,
        maxSequenceLength: configuration.maxSequenceLength,
        loras: controlLoRAs,
        progressCallback: progressCallback.map { callback in
          return { progress in
            if progress.stage == "Denoising" {
              callback(progress.stepIndex, progress.totalSteps)
            }
          }
        },
        enhancePrompt: false,
        enhanceMaxTokens: 512,
        // #154: the workflow's ModelSamplingAuraFlow / SD3 shift reaches the
        // ControlNet arm too.
        //
        // This arm returns before `bridgeGenerate`'s family switch and its
        // `auraFlowNodeGate`, and does not need it, because `runControlGenerate`
        // REFUSES the request outright on krea2 / flux2 / fibo / chroma
        // (`WarmServerError.controlNetNotSupported`) before any render — so the
        // only family that can reach a ControlNet render is `.flux1`, where the
        // node's LINEAR shift is exactly what `shift` means. The refusal a
        // client sees on the ControlNet path is therefore `controlNetNotSupported`,
        // NOT the node-named 400 the txt2img path returns; that is a
        // pre-existing property of this arm, not something #154 introduced.
        //
        // It also leaves `sigmaSchedule` at its `.flow` default, which reads the
        // shift, so `applied_shift` here always reports a value that reached the
        // grid.
        shift: request.shift
      )

      let start = Date()
      let result = try await coordinator.enqueueControlGenerate(controlRequest)
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)

      // Clean up temp files
      if let url = controlImageURL { try? FileManager.default.removeItem(at: url) }
      if let url = inpaintImageURL { try? FileManager.default.removeItem(at: url) }
      if let url = maskImageURL { try? FileManager.default.removeItem(at: url) }

      return ComfyBridgeGenerateResult(
        outputPath: result.outputPath,
        durationMs: result.durationMs
      )
    }

    // Family-aware defaults for step clamping, guidance, and negative prompts.
    let family = await coordinator.modelFamily
    let resolvedSteps: Int
    let resolvedGuidance: Float
    let resolvedNegativePrompt: String?
    let resolvedSampler: String?

    switch family {
    case .krea2:
      // WP-E19 (FDD §3.5, D13, AC-5a): split by the PHYSICAL variant the
      // engine loaded. .turbo is byte-identical to today (clamp 12, guidance
      // 0, negative dropped); .raw takes what Krita sent — no clamp, CFG and
      // the negative prompt live, variant defaults only when absent. A krea2
      // family with no known variant is a fault, never "turbo".
      let resolution = try BridgeKrea2Arm.resolve(request, variant: await coordinator.currentKrea2Variant)
      resolvedSteps = resolution.steps
      resolvedGuidance = resolution.guidance
      resolvedNegativePrompt = resolution.negativePrompt
      resolvedSampler = resolution.sampler
      let clampNote: String = resolution.stepsClamped ? " (clamped from \(request.steps))" : ""
      let negativeNote: String = resolution.negativePrompt.map { "\($0.count) chars" } ?? "none"
      let samplerNote: String = resolution.sampler ?? "nil"
      let scheduleNote: String = request.sigmaSchedule ?? "nil"
      let armLine = "WarmServer: bridge krea2 arm (\(resolution.variant.rawValue)): steps=\(resolution.steps)\(clampNote) "
        + "guidance=\(resolution.guidance) (requested \(request.guidance)) negative=\(negativeNote) "
        + "sampler=\(samplerNote) sigma_schedule=\(scheduleNote)"
      logger.info("\(armLine)")
    case .fibo:
      // FIBO: use model defaults, no step clamping
      resolvedSteps = request.steps
      resolvedGuidance = request.guidance > 0 ? request.guidance : 4.0
      resolvedNegativePrompt = request.negativePrompt
      resolvedSampler = request.sampler
    case .chroma:
      // Chroma: 28 steps default, guidance 0.0 (unconditioned)
      resolvedSteps = request.steps > 0 ? request.steps : 28
      resolvedGuidance = request.guidance
      resolvedNegativePrompt = nil
      resolvedSampler = request.sampler
    case .flux1:
      let zimageVariant = await coordinator.currentZImageVariant
      if zimageVariant == .base {
        // Z-Image Base / undistilled checkpoints (Moody, etc.): the ComfyUI/Krita
        // KSampler defaults are tuned for Turbo (9 steps, euler) and produce noise on
        // undistilled models. When the request still carries those turbo defaults, apply
        // the undistilled recommendations (40 steps, dpmpp_2m). If the user changed a
        // value, respect it. (Model-aware defaults from PR #164, @bree.)
        resolvedSteps = request.steps <= 9 ? 40 : request.steps
        resolvedGuidance = request.guidance > 0 ? request.guidance : ZImageModelMetadata.Base.recommendedGuidanceScale
        resolvedNegativePrompt = request.negativePrompt
        let sampler = request.sampler ?? "euler"
        resolvedSampler = sampler == "euler" ? "dpmpp_2m" : sampler
        if resolvedSteps != request.steps || resolvedSampler != request.sampler {
          logger.info("[WarmServer] Z-Image Base override: steps=\(resolvedSteps) (was \(request.steps)), sampler=\(resolvedSampler ?? "nil") (was \(request.sampler ?? "nil"))")
        }
      } else {
        // Z-Image Turbo: distilled, optimal at 9 steps. Honor the requested
        // guidance rather than hardcoding 0 — merged/finetuned "turbo"
        // checkpoints do respond to CFG, so forcing 0 removed real user control
        // (0 is the recommended default, passed through when the client sends it).
        resolvedSteps = min(request.steps, 9)
        resolvedGuidance = request.guidance
        resolvedNegativePrompt = nil
        resolvedSampler = request.sampler
      }
    case .flux2:
      // Base (non-distilled) models support guidance > 1.0 and default to 50 steps;
      // distilled models default to 4 steps and guidance 1.0.
      let isBaseModel = await coordinator.isFlux2BaseModel
      resolvedSteps = request.steps                 // Klein: no step clamp
      resolvedGuidance = isBaseModel ? request.guidance : 1.0
      resolvedNegativePrompt = nil                  // Klein: CFG only when guidance > 1.0
      resolvedSampler = request.sampler
    }

    // One constructor for every family arm (BridgeKrea2Arm.swift) so the
    // field set is asserted once, field-for-field (AC-5a).
    let payload = request.makeGeneratePayload(
      width: genWidth, height: genHeight,
      steps: resolvedSteps, guidance: resolvedGuidance,
      negativePrompt: resolvedNegativePrompt, sampler: resolvedSampler)
    // WP-E4 (§3.4): the bridge builds its payload directly, so it runs the
    // same fail-loud name resolution the /v1/generate decoder does — a Krita
    // style whose sampler we do not implement is refused by name, never
    // rendered as euler. The krea2 tier gates run here too (D18).
    let recipeNames = try payload.validateRecipeNames()
    // I5: the bridge builds its payload directly and forwards the request's
    // sampler on every family arm, so it runs the same family capability gate
    // the REST dispatch does — a Krita style naming a sampler this family
    // cannot drive is refused, never rendered as euler under that name.
    if let error = GeneratePayload.validateFamilyRecipe(recipeNames, family: family) {
      throw error
    }
    // #154: the bridge can now carry a `ModelSamplingAuraFlow` shift. The node
    // is honoured ONLY on the Z-Image family, whose `shift` is the same linear
    // warp the node applies; on any other family (Krea 2's `shift` is a
    // log-shift `mu`, the rest read none) the workflow is REFUSED by name,
    // never silently rendered on the model's own schedule.
    if let error = GeneratePayload.auraFlowNodeGate(payload.shift, family: family) {
      throw error
    }
    if let message = GeneratePayload.validateShift(payload.shift, family: family) {
      throw WarmServerError.invalidRequest(message: message)
    }
    if let message = GeneratePayload.validateShiftSchedule(
      payload.shift, sigmaSchedule: recipeNames.sigmaSchedule, family: family) {
      throw WarmServerError.invalidRequest(message: message)
    }
    // WP-E17: the bridge builds its payload directly too, so it runs the same
    // stage-2 gate. It never SETS `stage2` today; the gate is here so that
    // adding it later cannot skip the family check.
    if let error = GeneratePayload.stage2Gate(payload, family: family) {
      throw error
    }
    if family == .krea2 {
      try payload.validateKrea2TierGates(recipeNames)
    }

    // Convert bridge progress callback to pipeline progress handler.
    let pipelineProgress: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = progressCallback.map { callback in
      return { progress in
        if progress.stage == .denoising {
          callback(progress.stepIndex, progress.totalSteps)
        }
      }
    }

    // Forward the latent preview callback directly — it uses the same
    // (MLXArray, Int, Int, Int, Int) signature as the pipeline handler.
    let pipelineLatentPreview: ZImagePipeline.LatentPreviewHandler? = latentPreviewCallback

    // Batch generation: if batchSize > 1 (from RepeatLatentBatch), loop and return last result.
    if request.batchSize > 1 {
      logger.info("WarmServer: batch generation — \(request.batchSize) images")
      var lastResult: ComfyBridgeGenerateResult?
      var totalDurationMs = 0
      for i in 0..<request.batchSize {
        // Vary seed per batch item for unique outputs.
        var batchPayload = payload
        if let baseSeed = request.seed {
          batchPayload = GeneratePayload(
            prompt: payload.prompt,
            negativePrompt: payload.negativePrompt,
            width: payload.width,
            height: payload.height,
            steps: payload.steps,
            guidance: payload.guidance,
            seed: baseSeed + UInt64(i),
            outputPath: payload.outputPath,
            levelsMin: payload.levelsMin,
            levelsMax: payload.levelsMax,
            scheduler: payload.scheduler,
            sigmaSchedule: payload.sigmaSchedule,
            shift: payload.shift,
            inpaintImageData: payload.inpaintImageData,
            maskData: payload.maskData,
            denoise: payload.denoise,
            maskGrow: payload.maskGrow,
            maskFeather: payload.maskFeather,
            maskCropX: payload.maskCropX,
            maskCropY: payload.maskCropY
          )
        }
        let result = try await coordinator.enqueueGenerate(batchPayload, progressHandler: pipelineProgress, latentPreviewHandler: pipelineLatentPreview, source: "comfyui")
        totalDurationMs += result.durationMs
        lastResult = ComfyBridgeGenerateResult(outputPath: result.outputPath, durationMs: totalDurationMs)
      }
      return lastResult!
    }

    let result = try await coordinator.enqueueGenerate(payload, progressHandler: pipelineProgress, latentPreviewHandler: pipelineLatentPreview, source: "comfyui")
    return ComfyBridgeGenerateResult(
      outputPath: result.outputPath,
      durationMs: result.durationMs
    )
  }

  /// Known ESRGAN model name patterns.
  /// If the upscale model name matches any of these, route to ESRGANPipeline.
  private static let esrganModelPatterns: [String] = [
    "RealESRGAN_x4",
    "4x-UltraSharp",
    "4xNomos8k",
    "4x_NMKD-Superscale",
    "OmniSR_",
  ]

  /// Whether the given upscale model name should be routed to ESRGAN.
  private static func isESRGANModel(_ modelName: String) -> Bool {
    esrganModelPatterns.contains { modelName.hasPrefix($0) || modelName.contains($0) }
  }

  /// Bridge a ComfyUI upscale workflow request to the appropriate upscale pipeline.
  /// Routes to ESRGANPipeline for ESRGAN-family models, SeedVR2Pipeline for SeedVR2.
  /// Both pipelines are lazy-loaded on first use to avoid startup memory costs.
  private func bridgeUpscale(
    imageData: Data,
    modelName: String,
    progressCallback: ComfyBridgeProgressHandler?
  ) async throws -> ComfyBridgeGenerateResult {
    if Self.isESRGANModel(modelName) {
      return try await bridgeUpscaleESRGAN(imageData: imageData, modelName: modelName)
    } else {
      return try await bridgeUpscaleSeedVR2(imageData: imageData, modelName: modelName, progressCallback: progressCallback)
    }
  }

  /// ESRGAN upscale path. Lazy-loads the ESRGANPipeline on first use.
  /// Weights are resolved from ~/bin/zimage/upscale_models/<modelName>/.
  private func bridgeUpscaleESRGAN(
    imageData: Data,
    modelName: String
  ) async throws -> ComfyBridgeGenerateResult {
    // Resolve weights directory: ~/bin/zimage/upscale_models/<modelName>/
    // Strip file extension if present (e.g. "4x-UltraSharp.pth" -> "4x-UltraSharp")
    let baseName: String
    if let dotIndex = modelName.lastIndex(of: ".") {
      baseName = String(modelName[modelName.startIndex..<dotIndex])
    } else {
      baseName = modelName
    }
    let weightsDir = URL(fileURLWithPath: Self.upscaleModelsDirectoryPath)
      .appendingPathComponent(baseName)

    // Lazy-load ESRGAN pipeline (re-create if model changed)
    let pipeline = try loadESRGANPipelineIfNeeded(weightsDirectory: weightsDir)

    // Write input image data to a temp file.
    let inputTempPath = NSTemporaryDirectory() + "zimage-esrgan-input-\(UUID().uuidString).png"
    try imageData.write(to: URL(fileURLWithPath: inputTempPath))
    logger.info("WarmServer: wrote ESRGAN input to \(inputTempPath) (\(imageData.count) bytes)")

    let outputTempPath = NSTemporaryDirectory() + "zimage-esrgan-output-\(UUID().uuidString).png"

    let start = Date()
    do {
      let outputPath = try pipeline.upscaleAndSave(
        imagePath: inputTempPath,
        outputPath: outputTempPath
      )
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      try? FileManager.default.removeItem(atPath: inputTempPath)
      logger.info("WarmServer: ESRGAN upscale complete — \(durationMs)ms, output=\(outputPath)")
      return ComfyBridgeGenerateResult(outputPath: outputPath, durationMs: durationMs)
    } catch {
      try? FileManager.default.removeItem(atPath: inputTempPath)
      try? FileManager.default.removeItem(atPath: outputTempPath)
      throw error
    }
  }

  /// SeedVR2 upscale path. Lazy-loads on first use.
  private func bridgeUpscaleSeedVR2(
    imageData: Data,
    modelName: String,
    progressCallback: ComfyBridgeProgressHandler?
  ) async throws -> ComfyBridgeGenerateResult {
    guard let weightsPath = seedvr2WeightsPath else {
      throw SeedVR2Pipeline.PipelineError.weightsDirectoryNotFound("No SeedVR2 weights path configured")
    }

    // Lazy-load the SeedVR2 pipeline on first upscale request.
    let pipeline = try loadSeedVR2PipelineIfNeeded(weightsPath: weightsPath)

    // Write input image data to a temp file.
    let inputTempPath = NSTemporaryDirectory() + "zimage-upscale-input-\(UUID().uuidString).png"
    try imageData.write(to: URL(fileURLWithPath: inputTempPath))
    logger.info("WarmServer: wrote upscale input to \(inputTempPath) (\(imageData.count) bytes)")

    let outputTempPath = NSTemporaryDirectory() + "zimage-upscale-output-\(UUID().uuidString).png"

    let start = Date()
    do {
      let outputPath = try pipeline.upscaleAndSave(
        imagePath: inputTempPath,
        outputPath: outputTempPath,
        progressHandler: progressCallback
      )
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      try? FileManager.default.removeItem(atPath: inputTempPath)
      logger.info("WarmServer: upscale complete — \(durationMs)ms, output=\(outputPath)")
      return ComfyBridgeGenerateResult(outputPath: outputPath, durationMs: durationMs)
    } catch {
      try? FileManager.default.removeItem(atPath: inputTempPath)
      try? FileManager.default.removeItem(atPath: outputTempPath)
      throw error
    }
  }

  /// Get or lazily create the SeedVR2 pipeline. Double-checked under
  /// `upscalePipelineLock` so concurrent first-use requests cannot
  /// double-load the ~6GB weights.
  private func loadSeedVR2PipelineIfNeeded(weightsPath: String) throws -> SeedVR2Pipeline {
    upscalePipelineLock.lock()
    defer { upscalePipelineLock.unlock() }

    if let pipeline = seedvr2Pipeline {
      return pipeline
    }

    logger.info("WarmServer: lazy-loading SeedVR2 pipeline from \(weightsPath)...")
    let pipeline = try SeedVR2Pipeline(weightsPath: weightsPath, logger: logger)
    seedvr2Pipeline = pipeline
    logger.info("WarmServer: SeedVR2 pipeline ready (\(pipeline.modelConfig == .preset7B ? "7B" : "3B"))")
    return pipeline
  }

  /// Get or lazily create the ESRGAN pipeline for the given weights directory,
  /// re-creating it when the requested model changes. Serialized under
  /// `upscalePipelineLock` like SeedVR2 to prevent concurrent double-loads.
  private func loadESRGANPipelineIfNeeded(weightsDirectory weightsDir: URL) throws -> ESRGANPipeline {
    upscalePipelineLock.lock()
    defer { upscalePipelineLock.unlock() }

    if let pipeline = esrganPipeline, pipeline.weightsDirectory.path == weightsDir.path {
      return pipeline
    }

    logger.info("WarmServer: lazy-loading ESRGAN pipeline from \(weightsDir.path)...")
    let pipeline = try ESRGANPipeline(weightsDirectory: weightsDir, logger: logger)
    esrganPipeline = pipeline
    logger.info("WarmServer: ESRGAN pipeline ready (scale=\(pipeline.config.scale)x, blocks=\(pipeline.config.numBlock))")
    return pipeline
  }

  // MARK: - Upscale Handler

  /// Handle a direct upscale request via the REST API.
  /// Lazy-loads the SeedVR2 pipeline on first call.
  private func handleUpscale(_ payload: UpscalePayload) async throws -> UpscaleResponse {
    guard let weightsPath = seedvr2WeightsPath else {
      throw WarmServerError.invalidRequest(
        message: "SeedVR2 upscale not available: no weights path configured"
      )
    }

    // Validate input file exists
    guard FileManager.default.fileExists(atPath: payload.imagePath) else {
      throw WarmServerError.invalidRequest(
        message: "Input image not found: \(payload.imagePath)"
      )
    }

    let targetResolution = payload.targetResolution ?? 1024
    let softness = payload.softness ?? 0.0

    // Resolution guard
    if let error = UpscalePayload.validateResolution(targetResolution) {
      throw WarmServerError.invalidRequest(message: error)
    }

    // Validate softness range
    if let error = UpscalePayload.validateSoftness(softness) {
      throw WarmServerError.invalidRequest(message: error)
    }

    // Model variant validation
    if let error = UpscalePayload.validateModel(payload.model) {
      throw WarmServerError.invalidRequest(message: error)
    }

    // Lazy-load pipeline
    let pipeline = try loadSeedVR2PipelineIfNeeded(weightsPath: weightsPath)

    // Check model variant matches request
    if let requestedModel = payload.model {
      let is7B = pipeline.modelConfig == .preset7B
      let requested7B = requestedModel == "seedvr2-7b"
      if is7B != requested7B {
        let loaded = is7B ? "seedvr2-7b" : "seedvr2-3b"
        throw WarmServerError.invalidRequest(
          message: "Requested \(requestedModel) but loaded weights are \(loaded)"
        )
      }
    }

    // Build warning for experimental resolutions
    let warning = UpscalePayload.resolutionWarning(for: targetResolution)

    let start = Date()

    // Resolve output path
    let resolvedOutputPath: String?
    if let op = payload.outputPath {
      resolvedOutputPath = try WarmServerOutputPathValidator
        .resolveOutputPath(op, allowedOutputDirectory: configuration.allowedOutputDirectory)
        .path
    } else {
      resolvedOutputPath = nil
    }

    let outputPath = try pipeline.upscaleAndSave(
      imagePath: payload.imagePath,
      outputPath: resolvedOutputPath,
      targetResolution: targetResolution,
      seed: payload.seed,
      softness: softness
    )

    let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
    let modelName = pipeline.modelConfig == .preset7B ? "seedvr2-7b" : "seedvr2-3b"

    // Read output image dimensions for the response
    let outputResolution = readImageDimensions(at: outputPath)
    let inputResolution = readImageDimensions(at: payload.imagePath)

    return UpscaleResponse(
      success: true,
      outputPath: outputPath,
      durationMs: durationMs,
      inputResolution: inputResolution,
      outputResolution: outputResolution,
      model: modelName,
      warning: warning
    )
  }

  /// Read image dimensions as "WxH" string. Returns "unknown" on failure.
  private func readImageDimensions(at path: String) -> String {
    guard let source = CGImageSourceCreateWithURL(
      URL(fileURLWithPath: path) as CFURL, nil
    ),
    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
    let width = properties[kCGImagePropertyPixelWidth] as? Int,
    let height = properties[kCGImagePropertyPixelHeight] as? Int else {
      return "unknown"
    }
    return "\(width)x\(height)"
  }

  fileprivate func requestShutdownAfterResponse() {
    initiateShutdown()
  }

  /// Initiate a clean shutdown. Cancels the listener and exits.
  /// Safe to call from any thread — idempotent via shutdownSignalled flag.
  private func initiateShutdown(exitCode: Int32 = 0) {
    lifecycleLock.lock()
    defer { lifecycleLock.unlock() }

    guard !shutdownSignalled else { return }
    shutdownSignalled = true

    logger.info("Server shutting down (exit code \(exitCode))...")
    listener?.cancel()

    // Give in-flight connections 1 second to drain, then exit.
    DispatchQueue.global().asyncAfter(deadline: .now() + 1.0) {
      exit(exitCode)
    }
  }

  private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
    try Self.decode(type, from: data)
  }

  /// The wire decoder every route uses: snake_case in, camelCase properties.
  static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: data)
  }

  // MARK: - Workflows (#238)

  private struct WorkflowImportPayload: Decodable {
    let name: String?
    /// The ComfyUI workflow as a JSON string. Objects also accepted via the
    /// raw-body fallback in the handler.
    let workflowJson: String?
  }

  private func handleWorkflowImport(body: Data) async throws -> RoutedResponse {
    // Accept {name, workflow_json: "<string>"} or {name, workflow: {...}} or a
    // bare graph as the whole body.
    var name = "imported-workflow"
    var graphData: Data = body
    if let envelope = try? JSONSerialization.jsonObject(with: body) as? [String: Any] {
      if let n = envelope["name"] as? String, !n.isEmpty { name = n }
      if let s = envelope["workflow_json"] as? String, let d = s.data(using: .utf8) {
        graphData = d
      } else if let obj = envelope["workflow"] as? [String: Any] {
        graphData = try JSONSerialization.data(withJSONObject: obj)
      }
    }

    let graph = try WorkflowStore.apiGraph(fromJSON: graphData)
    // Dry-run normalization + parse so the compat report reflects
    // runnability without touching input files.
    var parses = true
    var parseError: String? = nil
    do {
      let normalized = try WorkflowStore.normalizeGenericNodes(graph, stageImage: nil)
      _ = try ComfyBridgeWorkflowParser.parseWorkflow(
        ["prompt": normalized, "prompt_id": "import-validate", "client_id": "workflow-api"])
    } catch {
      parses = false
      parseError = "\(error)"
    }

    let workflow = StoredWorkflow(
      id: UUID().uuidString.lowercased(),
      name: name,
      importedAt: Date(),
      graph: graph,
      compat: WorkflowStore.compatReport(for: graph, parses: parses, parseError: parseError))
    try workflowStore.save(workflow)
    logger.info("Workflows: imported '\(name)' (\(workflow.compat.nodeCount) nodes, parses=\(parses)) as \(workflow.id)")
    auditLog.append(kind: "workflow.import", message: "\(name) -> \(workflow.id)", metadata: [:])
    if let data = try? JSONSerialization.data(withJSONObject: workflow.summaryJSON()) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize import result"))
  }

  private struct WorkflowRunPayload: Decodable {
    let prompt: String?
    let negativePrompt: String?
    let seed: UInt64?
    let outputPath: String?
    let timeoutS: Double?
  }

  /// Pending workflow runs: run id → (workflow id, requested output path).
  /// The status route consumes this to place the finished image on disk.
  private let workflowRunLock = NSLock()
  private var workflowRuns: [String: (workflowId: String, outputPath: String?, resolved: String?)] = [:]

  /// Submit a workflow run. Long renders (base models run 40 steps — several
  /// minutes) outlive the HTTP connection, so this is async by design: 202 +
  /// run_id, polled via GET /v1/workflows/runs/{run_id} — the same convention
  /// as /v1/generate/async and /v1/video/generate/async.
  private func handleWorkflowRun(id: String, body: Data) async throws -> RoutedResponse {
    guard let workflow = workflowStore.get(id) else {
      return .error(.error(status: 404, message: "Workflow not found: \(id)"))
    }
    let payload: WorkflowRunPayload = body.isEmpty
      ? WorkflowRunPayload(prompt: nil, negativePrompt: nil, seed: nil, outputPath: nil, timeoutS: nil)
      : try decode(WorkflowRunPayload.self, from: body)

    // Normalize with REAL staging: LoadImage files go into the bridge cache.
    let normalized = try WorkflowStore.normalizeGenericNodes(workflow.graph) { data in
      let cacheId = "wf-\(UUID().uuidString.lowercased())"
      guard self.comfyBridge.imageCache.store(id: cacheId, data: data) else {
        throw WorkflowError.storeFailed("image cache store failed")
      }
      return cacheId
    }

    let promptId = "wfrun-\(UUID().uuidString.lowercased())"
    try comfyBridge.submitWorkflowGraph(
      normalized,
      promptId: promptId,
      promptOverride: payload.prompt,
      negativePromptOverride: payload.negativePrompt,
      seedOverride: payload.seed)
    workflowRunLock.lock()
    workflowRuns[promptId] = (workflow.id, payload.outputPath, nil)
    workflowRunLock.unlock()
    logger.info("Workflows: run \(promptId) of '\(workflow.name)' submitted")

    let response: [String: Any] = [
      "run_id": promptId,
      "workflow_id": workflow.id,
      "status": "queued",
    ]
    if let data = try? JSONSerialization.data(withJSONObject: response) {
      return .json(.rawJSON(status: 202, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize run submission"))
  }

  private func handleWorkflowRunStatus(runId: String) -> RoutedResponse {
    workflowRunLock.lock()
    let run = workflowRuns[runId]
    workflowRunLock.unlock()
    guard let run else {
      return .error(.error(status: 404, message: "Workflow run not found: \(runId) (runs are tracked in memory — a server restart loses them)"))
    }

    func respond(_ dict: [String: Any]) -> RoutedResponse {
      if let data = try? JSONSerialization.data(withJSONObject: dict) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize run status"))
    }

    // Already finalized on a previous poll.
    if let resolved = run.resolved {
      return respond(["run_id": runId, "workflow_id": run.workflowId,
                      "status": "succeeded", "output_path": resolved])
    }
    guard let entry = comfyBridge.history.entry(for: runId) else {
      return respond(["run_id": runId, "workflow_id": run.workflowId, "status": "running"])
    }
    // History filenames carry a .png suffix for ComfyUI /view compatibility;
    // the cache is keyed on the bare id — try both.
    let imageData: Data? = Self.firstImageId(inHistoryEntry: entry).flatMap { imageId in
      comfyBridge.imageCache.retrieve(id: imageId)
        ?? comfyBridge.imageCache.retrieve(id: (imageId as NSString).deletingPathExtension)
    }
    guard let imageData else {
      var detail = "no output image recorded"
      if let status = entry["status"],
         let statusData = try? JSONSerialization.data(withJSONObject: status),
         let statusText = String(data: statusData, encoding: .utf8) {
        detail = String(statusText.prefix(400))
      }
      return respond(["run_id": runId, "workflow_id": run.workflowId,
                      "status": "failed", "error": detail])
    }

    // Success: place the image at the requested (contained) path exactly once.
    do {
      let requestedRaw = run.outputPath ?? "workflow-\(run.workflowId.prefix(8))-\(runId.suffix(8)).png"
      let requested = requestedRaw.hasPrefix("/") || requestedRaw.hasPrefix("~")
        ? requestedRaw
        : (configuration.allowedOutputDirectory as NSString).appendingPathComponent(requestedRaw)
      let resolved = try WarmServerOutputPathValidator.resolveOutputPath(
        requested, allowedOutputDirectory: configuration.allowedOutputDirectory)
      try imageData.write(to: resolved)
      workflowRunLock.lock()
      workflowRuns[runId] = (run.workflowId, run.outputPath, resolved.path)
      workflowRunLock.unlock()
      logger.info("Workflows: run \(runId) -> \(resolved.path)")
      auditLog.append(kind: "workflow.run", message: "\(run.workflowId) -> \(resolved.path)", metadata: [:])
      return respond(["run_id": runId, "workflow_id": run.workflowId,
                      "status": "succeeded", "output_path": resolved.path])
    } catch {
      return respond(["run_id": runId, "workflow_id": run.workflowId,
                      "status": "failed", "error": "output write failed: \(error.localizedDescription)"])
    }
  }

  /// Dig the first output image id out of a bridge history entry
  /// ({outputs: {<nodeId>: {images: [{filename,...}]}}} — the bridge stores
  /// cache ids in `filename`).
  static func firstImageId(inHistoryEntry entry: [String: Any]) -> String? {
    guard let outputs = entry["outputs"] as? [String: Any] else { return nil }
    for value in outputs.values {
      guard let node = value as? [String: Any],
            let images = node["images"] as? [[String: Any]] else { continue }
      for image in images {
        if let filename = image["filename"] as? String, !filename.isEmpty {
          return filename
        }
      }
    }
    return nil
  }

  // MARK: - Storyboard (#237)

  /// Wire payload for POST /v1/storyboard/render (snake_case).
  struct StoryboardPayload: Decodable {
    struct InsertSpec: Decodable {
      let prompt: String
      let creativity: Double?
      let negativePrompt: String?
      let maskPath: String?
      let maskRegion: String?
      let maskInvert: Bool?
      let maskGrow: Int?
      let maskFeather: Int?
      let seed: UInt64?
    }
    struct ShotSpec: Decodable {
      let prompt: String
      let durationS: Double?
      let anchorImage: String?
      let insert: InsertSpec?
      let negativePrompt: String?
      let seed: UInt64?
    }
    struct TransitionSpec: Decodable {
      let type: String
      let durationS: Double?
    }
    struct OutputSpec: Decodable {
      let width: Int?
      let height: Int?
      let fps: Int?
      let path: String?
    }
    let shots: [ShotSpec]
    let transitions: [TransitionSpec]?
    let output: OutputSpec?
    let loras: [LoRAEntry]?
    let source: String?
  }

  private func storyboardSpec(from payload: StoryboardPayload) throws -> StoryboardSpec {
    var transitions: [MontageTransition] = []
    for t in payload.transitions ?? [] {
      guard let kind = MontageTransition.Kind(rawValue: t.type) else {
        throw WarmServerError.invalidRequest(
          message: "transitions[].type must be cut|fade|dissolve (got '\(t.type)')")
      }
      transitions.append(MontageTransition(kind: kind, durationS: t.durationS ?? 0.5))
    }
    let shots = payload.shots.map { s in
      StoryboardSpec.Shot(
        prompt: s.prompt,
        durationS: s.durationS,
        anchorImage: s.anchorImage,
        insert: s.insert.map { ins in
          StoryboardSpec.Insert(
            prompt: ins.prompt,
            creativity: ins.creativity ?? 0.35,
            negativePrompt: ins.negativePrompt,
            maskPath: ins.maskPath,
            maskRegion: ins.maskRegion,
            maskInvert: ins.maskInvert ?? false,
            maskGrow: ins.maskGrow ?? 0,
            maskFeather: ins.maskFeather ?? 0,
            seed: ins.seed)
        },
        negativePrompt: s.negativePrompt,
        seed: s.seed)
    }
    return StoryboardSpec(
      shots: shots,
      transitions: transitions,
      output: StoryboardSpec.Output(
        width: payload.output?.width ?? 640,
        height: payload.output?.height ?? 640,
        fps: payload.output?.fps ?? 24,
        path: payload.output?.path),
      loras: (payload.loras ?? []).map { ($0.path, $0.scale ?? 1.0) })
  }

  /// Execute a storyboard: per-shot [i2i insert →] i2v render, chained on the
  /// previous shot's extracted last frame, then montage assembly. Runs OUTSIDE
  /// the GPU queue (via submitOrchestrated) and takes a normal queue turn for
  /// each render, so other jobs interleave between shots. Intermediates are
  /// named storyboard-* (NOT ltx2-*) so the daemon's orphan reconciler ignores
  /// them.
  private func runStoryboard(
    spec: StoryboardSpec,
    source: String,
    report: @escaping @Sendable (Int) -> Void
  ) async throws -> LTX2VideoResult {
    let started = Date()
    let session = UUID().uuidString.prefix(8)
    let dir = configuration.allowedOutputDirectory
    var anchor: String? = nil
    var clips: [String] = []
    // Assembly gets the final ~4%; shots split the rest evenly.
    let shotWeight = 96.0 / Double(spec.shots.count)

    for (i, shot) in spec.shots.enumerated() {
      var shotAnchor: String
      if let explicit = shot.anchorImage, !explicit.isEmpty {
        guard FileManager.default.fileExists(atPath: explicit) else {
          throw StoryboardError.anchorNotFound(shot: i, path: explicit)
        }
        shotAnchor = explicit
      } else if let chained = anchor {
        shotAnchor = chained
      } else {
        throw StoryboardError.shotFailed(shot: i, stage: "anchor", message: "no previous last frame to chain from")
      }

      // Optional i2i insert on the anchor (e.g. add an element i2v can't
      // invent) — uses the #239 selective-inpainting path.
      if let insert = shot.insert {
        logger.info("Storyboard[\(session)] shot \(i): i2i insert (creativity \(insert.creativity))")
        let insertOut = (dir as NSString)
          .appendingPathComponent("storyboard-\(session)-shot\(i)-insert.png")
        let payload = GeneratePayload(
          prompt: insert.prompt,
          negativePrompt: insert.negativePrompt,
          width: spec.output.width,
          height: spec.output.height,
          seed: insert.seed,
          outputPath: insertOut,
          maskGrow: insert.maskGrow,
          maskFeather: insert.maskFeather,
          imagePath: shotAnchor,
          creativity: Float(insert.creativity),
          maskPath: insert.maskPath,
          maskRegion: insert.maskRegion,
          maskInvert: insert.maskInvert,
          source: source)
        do {
          let generated = try await coordinator.enqueueGenerate(payload, source: source)
          guard FileManager.default.fileExists(atPath: generated.outputPath) else {
            throw StoryboardError.shotFailed(shot: i, stage: "insert", message: "no output produced")
          }
          shotAnchor = generated.outputPath
        } catch let error as StoryboardError {
          throw error
        } catch {
          // comfybox#322: never launder an interrupt into a shot failure.
          // `shotFailed` keeps only a message STRING, so wrapping destroys the
          // `CancellationError` the trackers classify on — the storyboard job
          // would report "failed: CancellationError()" instead of interrupted.
          if ltx2IsCancellation(error) { throw error }
          throw StoryboardError.shotFailed(shot: i, stage: "insert", message: error.localizedDescription)
        }
      }

      // i2v render for this shot, through the same preparation the video
      // routes use (dims snapping, presets, LoRA resolution, validation).
      logger.info("Storyboard[\(session)] shot \(i)/\(spec.shots.count): i2v from \((shotAnchor as NSString).lastPathComponent)")
      var body: [String: Any] = [
        "prompt": shot.prompt,
        "image_path": shotAnchor,
        "width": spec.output.width,
        "height": spec.output.height,
        "fps": spec.output.fps,
        "output_path": "storyboard-\(session)-shot\(i).mp4",
        "source": source,
      ]
      if let d = shot.durationS { body["duration"] = d }
      if let n = shot.negativePrompt { body["negative_prompt"] = n }
      if let s = shot.seed { body["seed"] = s }
      if !spec.loras.isEmpty {
        body["loras"] = spec.loras.map { ["path": $0.path, "scale": $0.scale] as [String: Any] }
      }
      let shotResult: LTX2VideoResult
      do {
        let bodyData = try JSONSerialization.data(withJSONObject: body)
        guard let prep = try await prepareLocalVideo(body: bodyData) else {
          throw StoryboardError.shotFailed(shot: i, stage: "i2v", message: "local LTX-2 not configured")
        }
        let base = Double(i) * shotWeight
        shotResult = try await coordinator.enqueueLocalVideo { coordReport in
          // #1479: storyboard shots stay on the legacy, non-preemptible
          // entry — the checkpoint's generator-level context travels with a
          // single request, and a multi-shot orchestration issues its own
          // per-shot coordinator enqueues (see `submitOrchestrated`'s doc
          // comment), which the preemption episode does not model. `.generate`
          // never yields, so this is always `.completed`.
          .completed(try prep.generator.generate(prep.request) { chunk, totalChunks, step, totalSteps in
            let pct = Self.localVideoProgressPercent(
              chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps)
            coordReport(pct)
            report(Int(base + Double(pct) * shotWeight / 100.0))
          })
        }
      } catch let error as StoryboardError {
        throw error
      } catch {
        // comfybox#322: same as the insert stage above — an interrupted shot is
        // an interrupted storyboard, not a failed one. This is also the path a
        // cancelled `generate()` (non-preemptible) shot takes out of
        // `LTX2Pipeline.nonPreemptible`.
        if ltx2IsCancellation(error) { throw error }
        throw StoryboardError.shotFailed(shot: i, stage: "i2v", message: error.localizedDescription)
      }
      clips.append(shotResult.outputPath)

      // Chain: the next shot's default anchor is this shot's last frame.
      if i < spec.shots.count - 1 {
        let framePath = (dir as NSString)
          .appendingPathComponent("storyboard-\(session)-shot\(i)-lastframe.png")
        do {
          anchor = try LastFrameExtractor.extractLastFrame(from: shotResult.outputPath, to: framePath)
        } catch {
          throw StoryboardError.shotFailed(shot: i, stage: "last-frame", message: error.localizedDescription)
        }
      }
    }

    // Assembly (hard cuts unless the spec provided transitions).
    let requestedOutput = spec.output.path ?? "storyboard-\(session).mp4"
    let requested = requestedOutput.hasPrefix("/") || requestedOutput.hasPrefix("~")
      ? requestedOutput
      : (dir as NSString).appendingPathComponent(requestedOutput)
    let resolvedOutput = try WarmServerOutputPathValidator.resolveOutputPath(
      requested, allowedOutputDirectory: dir).path
    logger.info("Storyboard[\(session)]: assembling \(clips.count) shot(s) -> \(resolvedOutput)")
    let montage = try MontageComposer.compose(
      segments: clips.map { MontageSegment(kind: .clip, path: $0) },
      transitions: spec.transitions,
      width: spec.output.width,
      height: spec.output.height,
      fps: spec.output.fps,
      outputPath: resolvedOutput)
    report(100)
    auditLog.append(
      kind: "video.storyboard",
      message: "\(spec.shots.count) shots -> \(resolvedOutput)", metadata: [:])

    // comfybox#401: each per-shot clip already got its own sidecar+atom via
    // `LTX2VideoGenerator.render` (the storyboard's i2v shots go through the
    // same `prepareLocalVideo` -> `generate` path every other video route
    // does). The FINAL assembled clip does not — `MontageComposer` is a
    // different writer (composition/export, not `LTX2PostProcess.writeMP4`)
    // — so it gets its own aggregate record here. Sidecar only (ruling 2's
    // "mandatory" half); no single request's seed/steps/model apply to an
    // assembly of N independently-seeded shots, so those fields are absent
    // rather than misleadingly picking one shot's.
    let storyboardRecord = VideoGenerationRecord(
      prompt: spec.shots.map(\.prompt).joined(separator: " / "),
      model: "ltx2-storyboard",
      width: spec.output.width, height: spec.output.height,
      frames: montage.frameCount, fps: spec.output.fps,
      resolvedWidth: spec.output.width, resolvedHeight: spec.output.height,
      twoPass: false, refine: false, audio: false,
      kind: "storyboard",
      loras: spec.loras.map {
        VideoGenerationRecord.LoRAEntry(name: VideoGenerationRecord.basename($0.path), scale: $0.scale)
      })
    VideoSidecar.write(storyboardRecord, forMediaAt: montage.outputPath)

    return LTX2VideoResult(
      outputPath: montage.outputPath,
      frameCount: montage.frameCount,
      durationSeconds: Float(montage.durationS),
      elapsedSeconds: Date().timeIntervalSince(started),
      generationRecord: storyboardRecord)
  }

  // MARK: - Montage (#232)

  /// Wire response for POST /v1/montage/compose.
  struct MontageResponse: Encodable {
    let outputPath: String
    let durationS: Double
    let width: Int
    let height: Int
    let segmentCount: Int
    let frameCount: Int

    enum CodingKeys: String, CodingKey {
      case outputPath = "output_path"
      case durationS = "duration_s"
      case width, height
      case segmentCount = "segment_count"
      case frameCount = "frame_count"
    }
  }

  /// Wire payload for POST /v1/montage/compose (snake_case; see the FDD).
  struct MontagePayload: Decodable {
    struct Segment: Decodable {
      struct KenBurnsSpec: Decodable {
        /// [startScale, endScale]
        let zoom: [Double]?
        /// [[x0,y0],[x1,y1]] normalized output-unit offsets
        let pan: [[Double]]?
      }
      let type: String
      let path: String
      let durationS: Double?
      let kenburns: KenBurnsSpec?
    }
    struct Transition: Decodable {
      let type: String
      let durationS: Double?
    }
    struct Output: Decodable {
      let width: Int?
      let height: Int?
      let fps: Int?
      let path: String?
    }
    let segments: [Segment]
    let transitions: [Transition]?
    let output: Output?
    let aspectPolicy: String?
  }

  private func composeMontage(_ payload: MontagePayload) async throws -> MontageResult {
    var segments: [MontageSegment] = []
    for (i, s) in payload.segments.enumerated() {
      guard let kind = MontageSegment.Kind(rawValue: s.type) else {
        throw WarmServerError.invalidRequest(
          message: "segments[\(i)].type must be image|clip (got '\(s.type)')")
      }
      var kenBurns: MontageSegment.KenBurns? = nil
      if let spec = s.kenburns {
        var kb = MontageSegment.KenBurns()
        if let zoom = spec.zoom {
          guard zoom.count == 2 else {
            throw WarmServerError.invalidRequest(
              message: "segments[\(i)].kenburns.zoom must be [start, end]")
          }
          kb.zoomStart = zoom[0]
          kb.zoomEnd = zoom[1]
        }
        if let pan = spec.pan {
          guard pan.count == 2, pan[0].count == 2, pan[1].count == 2 else {
            throw WarmServerError.invalidRequest(
              message: "segments[\(i)].kenburns.pan must be [[x0,y0],[x1,y1]]")
          }
          kb.panStart = (pan[0][0], pan[0][1])
          kb.panEnd = (pan[1][0], pan[1][1])
        }
        kenBurns = kb
      }
      segments.append(MontageSegment(kind: kind, path: s.path, durationS: s.durationS, kenBurns: kenBurns))
    }

    var transitions: [MontageTransition] = []
    for t in payload.transitions ?? [] {
      guard let kind = MontageTransition.Kind(rawValue: t.type) else {
        throw WarmServerError.invalidRequest(
          message: "transitions[].type must be cut|fade|dissolve (got '\(t.type)')")
      }
      transitions.append(MontageTransition(kind: kind, durationS: t.durationS ?? 0.5))
    }

    let aspectPolicy: MontageAspectPolicy
    if let raw = payload.aspectPolicy {
      guard let parsed = MontageAspectPolicy(rawValue: raw) else {
        throw WarmServerError.invalidRequest(
          message: "aspect_policy must be fill_crop|fit_pad (got '\(raw)')")
      }
      aspectPolicy = parsed
    } else {
      aspectPolicy = .fillCrop
    }

    // Output containment — same convention as the video routes (#219).
    let requestedRaw = payload.output?.path ?? "montage-\(UUID().uuidString).mp4"
    let requested: String
    if requestedRaw.hasPrefix("/") || requestedRaw.hasPrefix("~") {
      requested = requestedRaw
    } else {
      requested = (configuration.allowedOutputDirectory as NSString)
        .appendingPathComponent(requestedRaw)
    }
    let resolvedOutput = try WarmServerOutputPathValidator.resolveOutputPath(
      requested, allowedOutputDirectory: configuration.allowedOutputDirectory).path

    let width = payload.output?.width ?? 448
    let height = payload.output?.height ?? 768
    let fps = payload.output?.fps ?? 30

    logger.info("Montage: \(segments.count) segment(s), \(transitions.count) transition(s) -> \(width)x\(height)@\(fps)")
    let start = Date()
    // Compositing is CPU-bound and synchronous — run it off the request task's
    // executor so a long montage can't starve other routes.
    let result = try await Task.detached(priority: .userInitiated) {
      try MontageComposer.compose(
        segments: segments,
        transitions: transitions,
        width: width, height: height, fps: fps,
        aspectPolicy: aspectPolicy,
        outputPath: resolvedOutput)
    }.value
    let elapsed = Int(Date().timeIntervalSince(start) * 1000)
    logger.info("Montage: wrote \(result.outputPath) (\(String(format: "%.2f", result.durationS))s, \(result.frameCount) frames) in \(elapsed)ms")
    auditLog.append(kind: "montage.compose", message: "\(segments.count) segments -> \(result.outputPath)", metadata: [:])
    return result
  }

  /// Shared decode + validation for both the synchronous and queue-submit
  /// generate routes, so output-path containment can't drift between them.
  ///
  /// Round 2 (I6): this is a one-line forward to the static below. The static
  /// is the WHOLE decode — parse, init-image, output-path containment, recipe
  /// names AND #286's preset expansion — so there is no separate "expansion
  /// call at the decode site" that could be removed while the decode survived,
  /// and the tests that drive it are testing the route.
  ///
  /// #22 (PR #363 review, C2): `gateSubmission` defaults `true` — a live
  /// `/v1/generate`/`/v1/generate/async` call. `recoverPersistedQueue`'s
  /// crash-recovery replay is the ONE caller that passes `false`: a job
  /// already accepted before a restart must never be re-refused by a gate
  /// that did not exist (or had different config) when it was submitted —
  /// `async` only to `await coordinator.modelFamily` for I4's warm-family
  /// resolution, skipped entirely when not gating.
  func decodedGeneratePayload(from body: Data, gateSubmission: Bool = true) async throws -> GeneratePayload {
    let warmFamily: WarmModelFamily? = gateSubmission ? await coordinator.modelFamily : nil
    return try Self.decodedGeneratePayload(
      from: body, store: presetStore, configuration: configuration,
      stageNearline: { entries in self.stageNearlineLoras(in: LoRASwapPayload(loras: entries)).loras },
      log: { line in self.logger.info("\(line)") },
      gateSubmission: gateSubmission, warmFamily: warmFamily)
  }

  /// The generate routes' decode, over an EXPLICIT store and configuration so
  /// it is testable without a warm server (same shape as
  /// ``WarmServer/upsertPreset(store:body:)``).
  ///
  /// The preset is read through ``PresetStore/lookup(_:)`` — one lock over the
  /// preset AND its validity flag, the same read `POST /v1/presets/resolve`
  /// makes — so the two routes cannot disagree about a given revision.
  static func decodedGeneratePayload(
    from body: Data,
    store: PresetStore,
    configuration: WarmServerConfiguration,
    stageNearline: ([LoRAEntry]) -> [LoRAEntry] = { $0 },
    loraExists: (LoRAEntry) -> Bool = WarmServer.loRASourceExists,
    log: (String) -> Void = { _ in },
    gateSubmission: Bool = true,
    warmFamily: WarmModelFamily? = nil
  ) throws -> GeneratePayload {
    var payload = try decode(GeneratePayload.self, from: body)
    // Bytes-uploaded img2img init image (init_image_base64) — write it to a
    // temp file so remote clients don't need a pre-existing server path.
    if let initData = payload.initImageData, payload.imagePath == nil {
      let tempPath = NSTemporaryDirectory() + "zimage-init-\(UUID().uuidString).png"
      try initData.write(to: URL(fileURLWithPath: tempPath))
      payload.imagePath = tempPath
    }
    try payload.validateOutputPath(configuration: configuration)
    // WP-E4 (D22): name resolution + structural checks happen ONCE, here, for
    // /v1/generate, /v1/generate/async and persisted-queue replay alike. An
    // unknown sampler/schedule name is a 400 before anything is enqueued —
    // never euler/flow by coercion. Family-specific tier gates (eta on krea2,
    // …) live in runKrea2Generate, NOT here: the family is unknown at this
    // point and Z-Image `eta` is a shipped parameter (D18, AC-28).
    _ = try payload.validateRecipeNames()
    // #286: expand a named `preset` HERE — model, LoRA stack and declared
    // steps/guidance — so /v1/generate, /v1/generate/async and persisted-queue
    // replay all go through it and the existing per-job model/LoRA application
    // at dequeue does the work. Before this, `preset` was a provenance label on
    // the image path and a preset-by-name render used whatever adapters the
    // warm pipeline happened to hold, on whatever base was active.
    var expanded = try expandGeneratePayload(
      payload, store: store, stageNearline: stageNearline, loraExists: loraExists, log: log)
    // #22 (PR #363 review, C2): resolution/memory preflight — refuses an
    // oversized request BEFORE it is enqueued (let alone before any model
    // load), with the estimate/available/cap named in the refusal. Runs
    // AFTER preset expansion so a preset that changes `model` (and therefore
    // which transformer profile the estimate uses) is checked accurately —
    // presets never touch width/height (#286: model/LoRA/steps/guidance
    // only), so this does not change which requests are gated, only which
    // family they are gated as. `gateSubmission: false` (replay) skips this
    // entirely — deleting this call makes the 6000×6000 wiring test fail.
    if gateSubmission {
      try expanded.validateImageMemoryPreflight(
        warmFamily: warmFamily, log: { line in log("ImageMemoryPreflight: \(line)") })
    }
    return expanded
  }

  /// #286 — the preset expansion half of the decode. Split out only so the
  /// two halves read separately; `decodedGeneratePayload` is the entry point.
  static func expandGeneratePayload(
    _ payload: GeneratePayload,
    store: PresetStore,
    stageNearline: ([LoRAEntry]) -> [LoRAEntry] = { $0 },
    loraExists: (LoRAEntry) -> Bool = WarmServer.loRASourceExists,
    log: (String) -> Void = { _ in }
  ) throws -> GeneratePayload {
    var out = try GeneratePayload.expandingPreset(payload) { id in
      let (found, invalidReason) = store.lookup(id)
      guard let preset = found else { return .notFound }
      if let reason = invalidReason { return .invalid(reason: reason) }
      return .resolved(store.resolve(preset: preset), declared: preset)
    } normalizeModelSpec: { spec in
      WarmServer.parseModelSpec(from: spec)
    } log: { line in log(line) }

    // Nothing was expanded (no preset, an unresolvable one, or the request
    // brought its own `loras`) — the two gates below apply only to a stack the
    // ENGINE produced. An explicit `loras` entry that does not resolve keeps
    // its long-standing 400 at dequeue; that contract is not this ticket's.
    guard let expanded = out.loras, out.presetUnresolved == nil,
          payload.loras == nil, let presetId = payload.preset
    else { return out }

    // I3: a preset may name an adapter that lives only on nearline storage.
    // `/v1/lora/swap` stages those; the expanded stack must be staged the same
    // way or a valid preset fails on a file that is merely archived.
    let staged = expanded.isEmpty ? expanded : stageNearline(expanded)

    // Round 2, finding 3: a preset naming a LoRA that is not on disk (and could
    // not be staged) used to become a 400 at DEQUEUE — turning a harmless
    // provenance label into a failed render for every caller of that preset.
    // Resolve the sources here, while the request can still fall back, and
    // treat an unresolvable one as an unexpandable preset like any other.
    if let missing = staged.first(where: { !loraExists($0) }) {
      let name = (missing.path as NSString).lastPathComponent
      return payload.asUnresolvedPreset(
        presetId,
        PresetExpansion.Unresolved(
          code: "missing_lora:\(name)",
          message: "preset '\(presetId)' names LoRA '\(missing.path)', which is not on disk and "
            + "could not be staged from nearline storage"),
        log: log)
    }

    out.loras = staged
    return out
  }

  /// #286 — decode AND expand, returning the body to persist alongside the
  /// payload.
  ///
  /// I5: the queue snapshot must carry the EXPANDED request. A crash-recovery
  /// replay of the original body would re-resolve the preset against whatever
  /// the store says at replay time — so editing or deleting a preset after a
  /// job was accepted could change or invalidate a queued render. The rewritten
  /// body carries the accepted `loras`/`model`/`steps`/`guidance`, and because
  /// it now has explicit `loras` the replay takes the request-wins branch and
  /// resolves nothing again.
  private func decodedGenerateRequest(from body: Data) async throws -> (GeneratePayload, Data) {
    let payload = try await decodedGeneratePayload(from: body)
    return (payload, Self.rawBody(body, expandedWith: payload))
  }

  /// #286 round 3 (minor 2): is this LoRA actually there?
  ///
  /// `LoRAEntry.makeConfiguration()` only *searches* for a bare filename; an
  /// absolute, `~`-prefixed or relative path is taken at its word and returned
  /// unchecked, so the missing-LoRA rule held for one source form and not the
  /// others. Resolve first (which tilde-expands and searches the library
  /// roots), then STAT the resolved local path.
  ///
  /// A HuggingFace reference is not a local file and is fetched at load time;
  /// there is nothing to stat, so it passes here and fails loudly later if the
  /// repo is wrong.
  static func loRASourceExists(_ entry: LoRAEntry) -> Bool {
    guard let configuration = try? entry.makeConfiguration() else { return false }
    switch configuration.source {
    case .local(let url):
      return FileManager.default.fileExists(atPath: url.path)
    case .huggingFace:
      return true
    }
  }

  /// Merge the fields #286's expansion may have filled in back into the raw
  /// JSON, so the persisted job replays the stack that was accepted. Returns
  /// the original bytes unchanged when nothing was expanded or the body is not
  /// a JSON object.
  static func rawBody(_ original: Data, expandedWith payload: GeneratePayload) -> Data {
    guard payload.preset?.isEmpty == false,
          var object = (try? JSONSerialization.jsonObject(with: original)) as? [String: Any]
    else { return original }
    var changed = false
    if let loras = payload.loras, object["loras"] == nil {
      object["loras"] = loras.map { entry -> [String: Any] in
        var row: [String: Any] = ["path": entry.path, "scale": entry.scale ?? 1.0]
        if let role = entry.role { row["role"] = role }
        return row
      }
      changed = true
    }
    for (key, value) in [
      ("model", payload.model as Any?), ("steps", payload.steps as Any?),
      ("guidance", payload.guidance as Any?), ("vae", payload.vae as Any?),
      // #154: a preset-owned schedule shift must survive a crash-recovery
      // replay, exactly like the preset-owned model/steps/guidance/vae — a
      // replayed body that dropped it would silently render on the model's
      // own schedule instead.
      ("shift", payload.shift as Any?),
    ] where object[key] == nil {
      if let value {
        object[key] = value
        changed = true
      }
    }
    guard changed, let data = try? JSONSerialization.data(withJSONObject: object) else {
      return original
    }
    return data
  }

  /// #339 review r3, item 1b (corrected r4): races "the job becomes
  /// observably admitted" — present as either the active job or somewhere
  /// in `pending`, which happens the instant an `enqueueXxx` call's
  /// synchronous append (`pending.append` + the `persistQueueState()` that
  /// immediately follows it) runs — against "the enqueue Task itself
  /// already finished" (an immediate `queueFull`/`shuttingDown` throw never
  /// appends at all, so polling for admission alone would burn the whole
  /// timeout every time recovery hits the capacity gate).
  ///
  /// r3's version narrowed the tail unconditionally after this call
  /// returned, INCLUDING on a timeout — but the coordinator actor's
  /// cooperative thread pool can legitimately be starved by an in-flight
  /// render well past 5s (a documented, known risk in this codebase — see
  /// the "#300" note on `WarmServerCoordinator`), so a timeout does not
  /// mean the job failed to admit, only that admission was not YET
  /// observed. The caller (`recoverPersistedQueue`) now uses the returned
  /// `AdmissionRaceOutcome` via `AdmissionNarrowingPolicy` (pure, tested
  /// directly) to decide whether narrowing now is safe — only `.admitted`
  /// is. `.timedOut`/`.renderFinishedFirst` leave the tail as-is; the
  /// caller's OWN next loop iteration narrows past this job safely once its
  /// `renderTask.value` has been awaited (proving it is truly done, success
  /// or failure, either way) — so the job is NEVER at risk of being dropped
  /// from the tail before it is durably represented elsewhere.
  private func waitForAdmissionOrCompletion<T>(
    jobId: String, renderTask: Task<T, Error>, timeout: TimeInterval = 5.0
  ) async -> AdmissionRaceOutcome {
    let health = liveHealth
    return await withTaskGroup(of: AdmissionRaceOutcome.self) { group in
      group.addTask {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
          let (snapshot, _) = health.read()
          if snapshot.activeJobId == jobId || snapshot.pending.contains(where: { $0.id == jobId }) {
            return .admitted
          }
          try? await Task.sleep(nanoseconds: 2_000_000)  // 2ms — cheap; keeps the duplicate window tiny
        }
        return .timedOut
      }
      group.addTask {
        _ = try? await renderTask.value
        return .renderFinishedFirst
      }
      let outcome = await group.next() ?? .timedOut
      group.cancelAll()
      return outcome
    }
  }

  /// Replay any queue jobs left over from before a crash (see
  /// QueuePersistence.swift) — the "active" slot (if any) first, since it
  /// was originally at the front, then everything still pending, in order.
  /// Each job is decoded through the exact same path its live route handler
  /// uses, then re-enqueued with no caller to respond to (fire-and-forget —
  /// the original HTTP connection is long gone by the time a crashed process
  /// restarts). Runs as a detached background task so a large recovered
  /// queue never delays the listener from coming up.
  private func recoverPersistedQueue() {
    // 0.B-2 (FDD §3.1.4a point 4): fold any undrained deltas from the sidecar
    // into the recovered set BEFORE re-enqueue, resolving them against the
    // persisted snapshot (not the live actor `pending` mid-replay). This is what
    // makes "cancel → bounce → stays cancelled" deterministic: a persisted cancel
    // keeps its job from resurrecting, a persisted move survives the bounce, and
    // the sidecar is cleared exactly once.
    let deltas = QueueDeltaStore.load()
    let state = QueueStateStore.load()
    // #339 review r3, item 1a: dedupe by id (first occurrence wins, active
    // before pending) — see `RecoverySnapshotMerger.deduplicated`'s doc
    // comment for why a snapshot can (briefly) carry the same id twice, and
    // why that must never turn into replaying it twice.
    var jobs = RecoverySnapshotMerger.deduplicated((state?.active.map { [$0] } ?? []) + (state?.pending ?? []))
    if !deltas.isEmpty {
      let before = jobs.count
      jobs = QueueDeltaApplier.apply(deltas, to: jobs, id: { $0.id })
      logger.info("Queue recovery: folded \(deltas.count) undrained delta(s), \(before) -> \(jobs.count) job(s)")
      // comfybox#386 review round 2, item 3: `clearDeltas()` now persists the
      // empty snapshot itself, through the same generation-guarded path as
      // every other sidecar write — a bare `QueueDeltaStore.clear()` here
      // raced an in-flight sync cancel/move (the listener is already up
      // while this recovery task runs) and could let a write from BEFORE
      // this clear land AFTER it, resurrecting what was just folded in.
      liveHealth.clearDeltas()
    }
    guard !jobs.isEmpty else { return }
    logger.info("Queue recovery: replaying \(jobs.count) job(s) left over from before a restart")

    // #339: while this replay is in flight, a submission for a queue-job
    // kind that is never persisted (local video, ControlNet generate, a
    // Krita model switch, a detached model load) cannot be trusted — a
    // second restart before it finishes would lose it with no trace. Gate
    // those kinds at the route (QueueRecoveryGate) for the span of this
    // Task; "generate"/"lora_swap" are unaffected, they queue durably
    // behind this same backlog either way (and, per review r1 item 1, are
    // now durable even against a SECOND restart mid-replay — see
    // `setRecoveryUnadmittedTail`/`RecoverySnapshotMerger`).
    queueRecoveryState.begin(jobKinds: jobs.map { QueueJobKind(rawValue: $0.kind) ?? .generate })
    Task {
      // #339 review r2, item 5: both cleanup steps run in `defer`s (LIFO —
      // the tail clears BEFORE `finish()` flips `inProgress` false) so a
      // cancelled replay Task (not currently possible in production, but
      // this Task is unstructured and nothing guarantees that stays true)
      // never leaves either a ghost ungated window OR a ghost ungated ALLOW
      // — ordering doesn't matter for correctness here since both only ever
      // narrow what's refused, but clearing the tail first keeps the two
      // signals consistent with each other for the single instant between them.
      defer { queueRecoveryState.finish() }
      defer { Task { await self.coordinator.setRecoveryUnadmittedTail([]) } }
      for index in jobs.indices {
        let job = jobs[index]
        // #339 review r2, item 4: publish `jobs[index...]` — INCLUDING this
        // job, not just what comes after it — as the still-unadmitted tail
        // BEFORE attempting to admit `job`. Review r1's version excluded
        // `job` itself here, on the theory that it was "about to be admitted
        // anyway" — but decoding the payload, staging nearline LoRAs for a
        // swap (a synchronous copy that can take real time), or even just
        // the `enqueueGenerate`/`enqueueSwap` call's own brief window all
        // happen AFTER this line and BEFORE `job` is actually durable in
        // `pending`. A crash in that gap, with the tail already narrowed to
        // exclude `job`, lost it — the exact "one-job admission window" r2
        // found. Publishing `jobs[index...]` keeps `job` visible in the
        // merged snapshot (`RecoverySnapshotMerger`) for that entire span;
        // once `job` actually lands in `pending`/`active`, it may appear in
        // BOTH the tail and the admitted state for a brief instant (a
        // harmless, transient duplicate — never a loss) until the NEXT
        // iteration narrows the tail to `jobs[(index+1)...]`.
        await coordinator.setRecoveryUnadmittedTail(Array(jobs[index...]))
        defer { queueRecoveryState.jobAdmitted() }
        // comfybox#283 finding 1: the operator-visible signal that a restart
        // is about to resurrect-and-replay this SAME job id — nothing in the
        // system reported this before. `classifyReplay` inspects this job
        // id's own prior ledger history for an unresolved `.checkpointed`
        // event (`ReplayClassifier`, pure/tested); today `generate`/
        // `lora_swap` (the only two kinds ever reaching this loop) never
        // checkpoint, so `fromStep1` is always `true` in production — see
        // that type's doc comment for why the classifier stays general
        // rather than hard-coding that answer. Read-only: recorded before
        // the replay attempt below, regardless of whether it goes on to
        // succeed or fail.
        let replayClassification = lifecycleLedger.classifyReplay(jobId: job.id)
        lifecycleLedger.record(
          jobId: job.id, kind: .replayedAfterRestart, jobKind: job.kind, source: job.source,
          step: replayClassification.resumeStep, chunk: replayClassification.resumeChunk,
          originalJobId: job.id, fromStep1: replayClassification.fromStep1)
        do {
          switch job.kind {
          case QueueJobKind.generate.rawValue:
            // #22 (C2, PR #363 review): `gateSubmission: false` — a job the
            // server already accepted before this restart must never be
            // re-refused by the image-memory preflight on replay.
            let payload = try await decodedGeneratePayload(from: job.rawBody, gateSubmission: false)
            // #339 review r3, item 1b (corrected r4): run the render in a
            // detached child Task and race admission against the Task's
            // own completion (`waitForAdmissionOrCompletion`) instead of
            // awaiting `enqueueGenerate` directly — which only returns once
            // the render COMPLETES, so r2's version left the tail (still
            // listing this job) and the admitted state (also listing it, as
            // `active`) both carrying it for the render's ENTIRE duration.
            // Only narrow when admission was actually OBSERVED
            // (`AdmissionNarrowingPolicy`, pure, tested directly) — r3's
            // version narrowed even on a timeout, which can drop the job
            // from the tail before the coordinator actually holds it if a
            // render blocks the actor's cooperative pool past 5s (#300).
            let renderTask = Task { try await self.coordinator.enqueueGenerate(payload, source: job.source, rawBody: job.rawBody, jobId: job.id) }
            let outcome = await waitForAdmissionOrCompletion(jobId: job.id, renderTask: renderTask)
            if AdmissionNarrowingPolicy.shouldNarrowNow(outcome) {
              await coordinator.setRecoveryUnadmittedTail(Array(jobs[jobs.index(after: index)...]))
            }
            // AC-18: replay under the job's OWN id (the client-visible one for
            // an async job), so a second restart persists the same name.
            _ = try await renderTask.value
            logger.info("Queue recovery: completed generate job \(job.id)")
          case QueueJobKind.loraSwap.rawValue:
            let payload = stageNearlineLoras(in: try decode(LoRASwapPayload.self, from: job.rawBody))
            // Same admit-then-narrow fix as generate above. `jobId: job.id`
            // (new — `enqueueSwap` had no way to name a job before r3) is
            // what makes this job observable to `waitForAdmissionOrCompletion` at all.
            let swapTask = Task { try await self.coordinator.enqueueSwap(payload, rawBody: job.rawBody, jobId: job.id) }
            let swapOutcome = await waitForAdmissionOrCompletion(jobId: job.id, renderTask: swapTask)
            if AdmissionNarrowingPolicy.shouldNarrowNow(swapOutcome) {
              await coordinator.setRecoveryUnadmittedTail(Array(jobs[jobs.index(after: index)...]))
            }
            _ = try await swapTask.value
            logger.info("Queue recovery: completed lora_swap job \(job.id)")
          default:
            logger.warning("Queue recovery: unknown job kind '\(job.kind)' for \(job.id), skipping")
          }
        } catch {
          // WP-E4 (D22, AC-18): a persisted job that fails replay is marked
          // FAILED with the reason on its own id (GET /v1/generate/status/{id})
          // and in the audit log — never rendered, never silently dropped.
          logger.error("Queue recovery: job \(job.id) (\(job.kind)) failed — \(error.localizedDescription)")
          // #339 review r4, item 3: a failed lora_swap replay is recorded
          // here too, not just generate — there is no dedicated swap-job
          // tracker, but `imageJobTracker` is keyed by id, not by kind, so
          // `GET /v1/generate/status/{id}` still surfaces the failure
          // reason instead of a bare 404 for a swap job's own id.
          if job.kind == QueueJobKind.generate.rawValue || job.kind == QueueJobKind.loraSwap.rawValue {
            imageJobTracker.recordFailedReplay(jobId: job.id, source: job.source, error: error)
          }
          auditLog.append(
            kind: "queue.recovery_failed",
            message: "job \(job.id) (\(job.kind)) failed replay: \(error.localizedDescription)",
            metadata: ["job_id": job.id, "kind": job.kind, "source": job.source])
        }
      }
      // Every job admitted (successfully or not) — no unadmitted remainder.
      // (Also covered by the `defer` above; explicit here so the common,
      // non-cancelled path clears it immediately rather than waiting on a
      // second Task hop.)
      await coordinator.setRecoveryUnadmittedTail([])
    }
  }

  private func response(for error: Error) -> HTTPResponse {
    Self.errorResponse(for: error)
  }

  // MARK: - CivitAI conduit (#234)

  /// Message returned on every /v1/civitai/* route when no API key resolves
  /// via CivitAISecrets — never crash/trap on a missing key.
  private static let civitaiKeyMissingMessage =
    "CivitAI API key not configured. Set --civitai-key on `serve`, export CIVITAI_API_KEY, " +
    "or save a key in the Desktop app's CivitAI settings (shared Keychain entry, " +
    "service com.barkadabrew.comfybox.desktop / account civitai)."

  private func civitaiSearchRoute(request: HTTPRequest) async -> RoutedResponse {
    guard let apiKey = CivitAISecrets.resolve(explicit: configuration.civitaiApiKey) else {
      return .error(.error(status: 503, message: Self.civitaiKeyMissingMessage))
    }
    let q = CivitAISearchQuery(queryParameters: request.queryParameters)
    // P1-1: the site allowlist (CivitAIHostAllowlist, shared with the
    // harvest route) must pass BEFORE a client carrying the Bearer key is
    // ever constructed — an unlisted host would receive the CivitAI key.
    guard let baseURL = q.validatedBaseURL else {
      return .error(.error(status: 400, message: CivitAIHostAllowlist.rejectionMessage(forSite: q.site)))
    }
    let client = CivitAIClient(baseURL: baseURL, apiKey: apiKey)
    do {
      let page = try await client.searchModels(
        query: q.query, types: q.types, baseModel: q.baseModel,
        sort: q.sort, period: q.period, nsfw: q.nsfw, cursor: q.cursor, limit: q.limit)
      let payload = CivitAISearchResponse(
        models: page.items.map(CivitAISearchResultModel.init),
        count: page.items.count,
        nextCursor: page.nextCursor)
      return .json(status: 200, payload: payload)
    } catch {
      return .error(.error(status: 502, message: "CivitAI search failed: \(error.localizedDescription)"))
    }
  }

  private func civitaiHarvestRoute(request: HTTPRequest) async -> RoutedResponse {
    guard let apiKey = CivitAISecrets.resolve(explicit: configuration.civitaiApiKey) else {
      return .error(.error(status: 503, message: Self.civitaiKeyMissingMessage))
    }
    let body: CivitAIHarvestRequestBody
    do {
      body = try request.body.isEmpty
        ? CivitAIHarvestRequestBody()
        : decode(CivitAIHarvestRequestBody.self, from: request.body)
    } catch {
      return .error(.error(status: 400, message: "Invalid harvest request: \(error.localizedDescription)"))
    }
    // P1-1: same shared allowlist as the search route, same reason.
    guard let baseURL = body.validatedBaseURL else {
      return .error(.error(status: 400, message: CivitAIHostAllowlist.rejectionMessage(forSite: body.site)))
    }
    let client = CivitAIClient(baseURL: baseURL, apiKey: apiKey)
    do {
      // Behavior notes (P1-2): the runner clamps body.limit to
      // CivitAIHarvestRunner.maxModelsPerHarvest (200) models per call,
      // upserts page-by-page, and stops after ~60s with truncated: true in
      // the summary — partial results are already persisted.
      let summary = try await CivitAIHarvestRunner.run(client: client, request: body)
      return .json(status: 200, payload: summary)
    } catch {
      return .error(.error(status: 502, message: "CivitAI harvest failed: \(error.localizedDescription)"))
    }
  }

  private func civitaiRepoRoute(request: HTTPRequest) -> RoutedResponse {
    let q = CivitAIRepoQuery(queryParameters: request.queryParameters)
    // Result cap (P2): default 100 entries, raisable via ?limit= up to 500 —
    // never the whole store per request.
    let entries = PromptRepositoryStore.query(
      baseModel: q.baseModel, act: q.act, tag: q.tag, keyword: q.keyword, limit: q.limit)
    let payload = CivitAIRepoResponse(entries: entries, count: entries.count)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.dateEncodingStrategy = .iso8601
    guard let data = try? encoder.encode(payload) else {
      return .error(.error(status: 500, message: "Failed to serialize prompt repository"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }

  /// Error → HTTP mapping, static so the 400/500 split is unit-testable
  /// without a listening server (WP-E4 `WarmServerRejectionTests`).
  static func errorResponse(for error: Error) -> HTTPResponse {
    switch error {
    case let error as WarmServerCoordinator.ServerError:
      switch error {
      case .queueFull(let maxPending):
        return .error(status: 429, message: "Queue full (\(maxPending) pending max)")
      case .modelOperationQueueFull(let maxPending):
        return .error(
          status: 429,
          message: "Model operation queue full (\(maxPending) pending model operations max)")
      case .shuttingDown:
        return .error(status: 503, message: "Server is shutting down")
      case .cancelled:
        return .error(status: 409, message: "Request cancelled (queue cleared)")
      }

    case let error as ZImagePipeline.PipelineError:
      switch error {
      case .invalidDimensions(let message):
        return .error(status: 400, message: message)
      case .loraError(let loraError):
        return .error(status: 400, message: loraError.localizedDescription)
      default:
        return .error(status: 500, message: error.localizedDescription)
      }

    case let error as LoRAError:
      return .error(status: 400, message: error.localizedDescription)

    // WP-E9 (AC-56): a VAE the caller named that is not on disk, or a file in
    // a key layout the engine cannot name, is the caller's error — named in
    // full, never substituted.
    case let error as Krea2VAESelectionError:
      return .error(status: 400, message: error.localizedDescription)
    case let error as Krea2VAEKeyMapError:
      return .error(status: 400, message: error.localizedDescription)

    case let error as WarmServerError:
      switch error {
      // #22: image-memory preflight refusal — 413 (Payload Too Large), with
      // the estimate/available/cap numbers as additive JSON fields so a
      // client can branch on `error_code` without string-matching `error`.
      case .imageMemoryPreflightRefused(let code, let reason, let estimate, let available, let cap):
        return .json(status: 413, payload: ErrorPayload(
          success: false, error: "[\(code)] \(reason)", errorCode: code,
          estimateBytes: estimate, availableBytes: available, capBytes: cap))
      case .loraSwapNotSupported, .controlNetNotSupported:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      case .invalidOutputPath, .invalidRequest:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      // #286: a preset/model contradiction is a CONFLICT, not a malformed
      // request — the caller sent two valid things that cannot both hold.
      case .presetModelConflict:
        return .error(status: 409, message: error.localizedDescription ?? "Preset/model conflict")
      // WP-E4: a bad recipe name / key conflict / unimplemented tier is the
      // caller's error, named in full (AC-15, AC-28).
      case .unknownSampler, .unknownSigmaSchedule, .mutuallyExclusive, .unsupportedRecipeField,
           .unsupportedSampler, .orphanField, .projectorScaleOutOfRange, .unknownNoiseType,
           .implicitStepsOutOfRange, .c2OutOfRange:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      case .flux2NotLoaded, .flux2DetectionFailed, .fiboNotLoaded, .fiboDetectionFailed,
           .chromaNotLoaded, .chromaDetectionFailed, .krea2NotLoaded, .krea2VariantUnknown:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      case .invalidPort:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      // comfybox#322: an operator interrupt is not a server fault, but it
      // reports 500 like every other terminal error here (review r1 ruling —
      // no unversioned HTTP change; images and video must match). The
      // `interrupted: true` field on the job status is the machine-readable
      // signal.
      case .renderInterrupted:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      case .queueRecoveryInProgress(let retryAfterSeconds):
        return .queueRecovering(retryAfterSeconds: retryAfterSeconds)
      }

    case let error as Flux2Pipeline.Flux2PipelineError:
      switch error {
      case .invalidDimensions:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      default:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      }

    case let error as FiboPipeline.FiboPipelineError:
      switch error {
      case .invalidDimensions:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      default:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      }

    case let error as ModelPoolError:
      switch error {
      case .modelNotInPool, .cannotUnloadActive:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      case .budgetExceeded:
        return .error(status: 507, message: error.localizedDescription ?? error.localizedDescription)
      case .alreadyLoaded:
        return .error(status: 409, message: error.localizedDescription ?? error.localizedDescription)
      case .loadFailed, .modelDetectionFailed:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      }

    // WP-E4 / WP-E17: the pipeline's own fail-loud refusals are the CALLER's
    // error, not the server's. Every one of these is pre-empted by a wire gate
    // on the paths that have one; a non-server caller — and any path a gate
    // does not cover — must still get a 400 naming the field rather than a 500
    // naming nothing.
    case let error as Krea2ScheduleError:
      return .error(status: 400, message: error.description)
    case let error as Krea2StageError:
      return .error(status: 400, message: error.description)
    case let error as SchedulerFactoryError:
      switch error {
      // "this schedule cannot be built at that step count" is the request's
      // problem. `missingMu` is the engine failing to hand the factory a shift
      // it owns, which is ours.
      case .stepCountBelowMinimum:
        return .error(status: 400, message: error.description)
      case .missingMu:
        return .error(status: 500, message: error.description)
      }

    case let error as DecodingError:
      return .error(status: 400, message: "Invalid JSON body: \(describe(decodingError: error))")

    default:
      return .error(status: 500, message: error.localizedDescription)
    }
  }

  /// #313 (review round 1): validate a caller-supplied `model_compatibility`
  /// patch against `LoRAScanner.knownCompatibilityTags` — the same treatment
  /// `krea2_relative` already gets on this route (an unrecognized value is a
  /// 400 naming the offending value and the valid set, never silently
  /// accepted). Also rejects an empty array: `model_compatibility: []` would
  /// otherwise silently strip a LoRA's compatibility down to nothing.
  /// Static and pure so the 400 is unit-testable without a listening server,
  /// same as `errorResponse(for:)` above.
  static func validateModelCompatibilityTags(_ tags: [String]) throws -> [String] {
    guard !tags.isEmpty else {
      throw WarmServerError.invalidRequest(message: "model_compatibility must not be empty")
    }
    for tag in tags {
      guard LoRAScanner.knownCompatibilityTags.contains(tag.lowercased()) else {
        throw WarmServerError.invalidRequest(
          message: "Invalid model_compatibility tag '\(tag)': expected one of "
            + "\(LoRAScanner.knownCompatibilityTags.sorted())")
      }
    }
    return tags
  }

  private static func describe(decodingError: DecodingError) -> String {
    switch decodingError {
    case .dataCorrupted(let context):
      return context.debugDescription
    case .keyNotFound(let key, let context):
      return "Missing key '\(key.stringValue)' (\(context.debugDescription))"
    case .typeMismatch(_, let context):
      return context.debugDescription
    case .valueNotFound(_, let context):
      return context.debugDescription
    @unknown default:
      return decodingError.localizedDescription
    }
  }

  private static func currentMemoryFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
      }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return info.phys_footprint
  }

  /// The /health body: the snake_case encoding of `health` plus the video
  /// section, with the telemetry-contract keys ALWAYS present (JSON null when
  /// idle) so clients decode them unconditionally — `current_job_id`,
  /// `progress_percent`, and (WP-E10) `last_recipe`, `model_alias`,
  /// `model_variant`. Static so the contract is unit-testable (`HealthSinkTests`).
  ///
  /// `video.available`/`video.backend` keep their pre-existing, Replicate-only
  /// meaning (#298 review finding 4) — nothing routes on local readiness yet.
  /// `localVideoReadiness` is an ADDITIVE, already-computed snapshot (from
  /// `LocalVideoReadinessMonitor`, never touched synchronously on this path)
  /// surfaced as `video.local_ready` / `local_reason` / `local_checked_at` /
  /// `local_backend`, plus the detailed asset breakdown under `local_assets`.
  static func healthJSON(
    _ health: HealthResponse,
    videoAvailable: Bool,
    activeVideoJobs: Int,
    localVideoReadiness: LocalVideoReadiness = .unchecked
  ) -> Data? {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    guard var healthJSON = try? JSONSerialization.jsonObject(with: encoder.encode(health)) as? [String: Any] else {
      return nil
    }
    healthJSON["current_job_id"] = (health.currentJobId as Any?) ?? NSNull()
    healthJSON["progress_percent"] = (health.progressPercent as Any?) ?? NSNull()
    healthJSON["model_variant"] = (health.modelVariant as Any?) ?? NSNull()
    healthJSON["model_alias"] = (health.modelAlias as Any?) ?? NSNull()
    if healthJSON["last_recipe"] == nil { healthJSON["last_recipe"] = NSNull() }
    healthJSON["video"] = [
      "available": videoAvailable,
      "backend": videoAvailable ? "replicate" : "none",
      "active_jobs": activeVideoJobs,
      "local_ready": localVideoReadiness.ready,
      "local_reason": (localVideoReadiness.reason as Any?) ?? NSNull(),
      "local_checked_at": localVideoReadiness.checkedAt.map { ISO8601DateFormatter().string(from: $0) } as Any? ?? NSNull(),
      "local_backend": localVideoReadiness.ready ? "local_ltx2" : NSNull(),
      "local_assets": [
        "required": localVideoReadiness.requiredAssets.map { $0.json },
        "optional": localVideoReadiness.optionalAssets.map { $0.json },
      ],
    ] as [String: Any]
    return try? JSONSerialization.data(withJSONObject: healthJSON, options: [.sortedKeys])
  }

  /// Max render age (ms) before /health flags the render as likely deadlocked.
  private static let healthRenderStaleThresholdMs = 300_000  // 5 minutes (#141)

  /// The WHOLE `GET /health` response, assembled SYNCHRONOUSLY (#217).
  ///
  /// Every input is lock-based or an immutable value — `LiveHealthState.read()`,
  /// `VideoJobTracker.activeJobCount`, `ReplicateVideoProxy.activeJobCount`,
  /// `LocalVideoReadinessMonitor.current()`, `currentMemoryFootprintBytes()` and
  /// the configuration — so there is nothing to await and no subset of the
  /// payload has to be split onto a separate route. Shared by the async dispatch
  /// arm and `serveControlPlaneSync`, so the two cannot emit different bytes.
  fileprivate func healthRouteResponse() -> HTTPResponse {
    let memoryBytes = Self.currentMemoryFootprintBytes()
    let health = liveHealthResponse(memoryBytes: memoryBytes)
    if let data = Self.healthJSON(
      health, videoAvailable: replicateVideoProxy != nil,
      activeVideoJobs: videoJobTracker.activeJobCount + (replicateVideoProxy?.activeJobCount ?? 0),
      localVideoReadiness: localVideoReadinessMonitor.current()) {
      return .rawJSON(status: 200, data: data)
    }
    return .json(status: 200, payload: health)
  }

  /// Assemble the /health payload from the lock-based ``LiveHealthState``
  /// snapshot — NO actor hop — so /health stays responsive during a render (#217).
  private func liveHealthResponse(memoryBytes: UInt64) -> HealthResponse {
    Self.liveHealthPayload(
      liveHealth: liveHealth, configuration: configuration,
      serverStartTime: serverStartTime, memoryBytes: memoryBytes)
  }

  /// The pure payload assembly, split out of `liveHealthResponse` so the
  /// no-actor-hop claim can be driven from a unit test (`WarmServerQueueProbe`)
  /// against a real coordinator that is BUSY — the exact condition #217 is
  /// about. Depends on nothing but the lock store, the immutable configuration
  /// and two clocks.
  fileprivate static func liveHealthPayload(
    liveHealth: LiveHealthState, configuration: WarmServerConfiguration,
    serverStartTime: Date, memoryBytes: UInt64, now: Date = Date()
  ) -> HealthResponse {
    let (snap, progress) = liveHealth.read()
    let uptimeSeconds = Int(now.timeIntervalSince(serverStartTime))
    let activeAgeMs = snap.activeRenderStartedAt.map { Int(now.timeIntervalSince($0) * 1000.0) }
    let status = WarmServerHealthStatus.derive(
      shuttingDown: snap.shuttingDown,
      activeRenderAgeMs: activeAgeMs,
      staleThresholdMs: healthRenderStaleThresholdMs)
    // comfybox#386 review round 3, item 1c: additive — surfaces the
    // liveness-over-durability tradeoff `drainQueueDeltas` makes when the
    // undrained-delta sidecar is unwritable.
    let deltaStatus = liveHealth.deltaDurabilityStatus()
    return HealthResponse(
      status: status,
      model: snap.model.isEmpty ? (configuration.modelSpec ?? ZImageRepository.id) : snap.model,
      modelFamily: snap.modelFamily,
      modelVariant: snap.modelVariant,
      modelAlias: snap.modelAlias,
      buildSha: BuildInfo.gitSHA,
      textEncoderPath: configuration.textEncoderPath,
      loaded: snap.loaded,
      loras: snap.loras,
      uptimeSeconds: uptimeSeconds,
      renderCount: snap.renderCount,
      failedRenderCount: snap.failedRenderCount,
      pendingCount: snap.pendingCount,
      maxPending: configuration.maxPendingRequests,
      isRendering: snap.isRendering,
      isPaused: snap.isPaused,
      activeRequestAgeMs: activeAgeMs,
      currentJobId: snap.activeJobId,
      progressPercent: progress,
      memoryUsageBytes: memoryBytes,
      memoryUsageMB: memoryBytes / (1024 * 1024),
      lastRenderDurationMs: snap.lastRenderDurationMs,
      lastError: snap.lastError,
      lastRecipe: snap.lastRecipe,
      queueDeltaSidecarDegraded: deltaStatus.isDegraded,
      queueDeltaNonDurableCount: deltaStatus.nonDurableCount
    )
  }

  // MARK: - Krita Model Detection Helpers

  /// Parse quantization suffix from a model ID string.
  /// e.g. "z-image-turbo-q8" -> "q8", "klein-4b-q8" -> "q8", "briaai/FIBO" -> nil
  static func parseQuantization(from modelId: String) -> String? {
    let lowered = modelId.lowercased()
    if lowered.hasSuffix("-q4") { return "q4" }
    if lowered.hasSuffix("-q8") { return "q8" }
    if lowered.hasSuffix("-bf16") { return "bf16" }
    return nil
  }

  /// Parse the model spec from a pool-style model ID.
  /// Strips quantization suffixes since poolLoad takes them separately.
  /// e.g. "z-image-turbo-q8" -> "z-image-turbo", "briaai/FIBO" -> "briaai/FIBO"
  /// Specs this function passes through untouched — engine-known model ids.
  /// Exposed (comfybox#359) so `ModelFamilyDetector` can answer "would
  /// `/v1/generate` accept this as `model`?" from THIS list rather than
  /// growing a second copy of it.
  static let knownModelSpecs: Set<String> = [
    "briaai/FIBO",
    "chroma-8.9b",
    "z-image-turbo",
    "z-image-turbo-bf16",
    "klein-4b",
    "klein-9b",
  ]

  /// CivitAI checkpoint path mappings (Moody family). Same reason as
  /// `knownModelSpecs` for being a stored table rather than a local.
  static let civitaiCheckpointPaths: [String: String] = [
    "moody-wild-v4": "~/Models-working/moody-wild-mix/moody-wild-v4-fp16-full.safetensors",
    "moody-wild-v4-distilled": "~/Models-working/moody-wild-mix/moody-wild-v4-distilled-10step-fp16.safetensors",
    "moody-wild-v4-fp8": "~/Models-working/moody-wild-mix/moody-wild-v4-fp8.safetensors",
    "moody-real-v6": "~/Models-working/moody-real-v6/moody-real-v6.safetensors",
    "cyberrealistic-v5": "~/Models-working/cyberrealistic-z-image/cyberrealisticZImage_v50.safetensors",
  ]

  static func parseModelSpec(from modelId: String) -> String {
    if knownModelSpecs.contains(modelId) { return modelId }

    if let path = civitaiCheckpointPaths[modelId] {
      return NSString(string: path).expandingTildeInPath
    }
    // Krea-2 family installs (kroma-v0.2-turbo, krea2-raw, …) live in ONE
    // declared spec→directory table (WP-E5) — seeded from config.json
    // `krea2Models` over the built-in defaults — so this function never grows
    // a second one. The directory is then detected fail-closed.
    if let dir = Krea2ModelDetection.specDirectory(modelId) {
      return dir.path
    }

    let suffixes = ["-q4", "-q8", "-bf16"]
    for suffix in suffixes {
      if modelId.lowercased().hasSuffix(suffix) {
        return parseModelSpec(from: String(modelId.dropLast(suffix.count)))
      }
    }
    return modelId
  }
}

/// Snapshot of the coordinator's health-relevant state, published to the
/// lock-based ``LiveHealthState`` so GET /health can be served WITHOUT hopping
/// onto the ``WarmServerCoordinator`` actor.
///
/// The actor is blocked for the full duration of a synchronous GPU render
/// (seconds to minutes). Routing /health through `await coordinator.health()`
/// made the endpoint queue behind the render and return nothing (HTTP 000) for
/// the render's whole duration, then respond instantly once it finished — the
/// Desktop queue/progress UI and external monitors went stale mid-render (#217).
/// The coordinator publishes this snapshot at each state transition instead.
private struct HealthSnapshot: Sendable {
  var shuttingDown: Bool
  var model: String
  var modelFamily: String
  var modelVariant: String?
  /// WP-E10 "E9b" (AC-34b): the declared alias `model` resolved from
  /// (`krea2-raw`), when the active krea2 spec is one — `model` itself carries
  /// the resolved directory path once `parseModelSpec` has expanded it.
  var modelAlias: String? = nil
  /// WP-E10 sink 3: the record of the last successful Krea 2 render.
  var lastRecipe: AppliedRecordSlot? = nil
  var loaded: Bool
  var loras: [LoRAState]
  var renderCount: Int
  var failedRenderCount: Int
  var pendingCount: Int
  var isRendering: Bool
  var activeRenderStartedAt: Date?
  var activeJobId: String?
  var lastRenderDurationMs: Int?
  var lastError: String?
  /// Queue-specific fields (#217 follow-up: GET /v1/queue also used to hop
  /// onto the actor via `coordinator.queueSnapshot()`, so the Queue tab went
  /// stale during a render exactly like /health used to before this snapshot
  /// existed). Populated alongside everything else in `publishHealth()`.
  var isPaused: Bool = false
  var activeSummary: String?
  var activeSource: String?
  var pending: [WarmServerCoordinator.QueueJobInfo] = []
  var maxPending: Int = 0

  static let initial = HealthSnapshot(
    shuttingDown: false, model: "", modelFamily: WarmModelFamily.flux1.rawValue,
    modelVariant: nil, loaded: false, loras: [], renderCount: 0, failedRenderCount: 0,
    pendingCount: 0, isRendering: false, activeRenderStartedAt: nil, activeJobId: nil,
    lastRenderDurationMs: nil, lastError: nil)
}

/// Lock-based publisher for ``HealthSnapshot`` + live progress. Written on the
/// actor at each state transition and from the off-actor progress callback;
/// read by the /health route with no actor hop, so /health stays responsive
/// throughout a render (#217).
private final class LiveHealthState: @unchecked Sendable {
  private let lock = NSLock()
  private var snapshot = HealthSnapshot.initial
  private var progressPercent: Int?

  // 0.B-2 (FDD §3.1.5): `isPaused` and the undrained-delta list are
  // AUTHORITATIVE here, not on the actor. Off-actor/sync control writes land
  // here and are visible to the read path IMMEDIATELY (not after the actor
  // catches up at the end of a render). The actor is a READER of `paused` (its
  // between-items gate) and DRAINS `deltas` at its scheduling points.
  private var paused: Bool
  private var deltas: [StampedDelta] = []
  /// The in-flight render's task AND the identity that names it, published
  /// here (both `Sendable`) so the SYNC `/v1/queue/interrupt` can resolve and
  /// cancel with no actor hop. Set/cleared by the process loop around each
  /// render, as ONE write.
  ///
  /// comfybox#362: during a preemption episode this holds the PREEMPTING IMAGE
  /// job's own task and id, not the checkpointed video underneath it — the
  /// coordinator republishes it for the episode's duration
  /// (`runAsPublishedActiveRender`) so this slot and `/health`'s active job
  /// agree, and a plain `/v1/queue/interrupt` (no `target`) cancels what an
  /// operator actually sees as active instead of silently abandoning the video
  /// that is no longer the visible render.
  ///
  /// Review r1, finding 1: the task and the identity are one value on purpose
  /// — see `PublishedRender`. The interrupt route must never be able to read a
  /// half-swapped pair.
  private var activeRender = PublishedRender.none
  /// comfybox#362: the checkpointed video's OWN task + ids, published
  /// separately from `activeRender` for the span of a preemption episode so
  /// `target: "video"` can still reach it even while `activeRender` holds the
  /// preempting image job. `PublishedRender.none` outside an episode.
  private var checkpointedVideo = PublishedRender.none

  /// Cap on undrained deltas. Deltas drain promptly (every processLoop
  /// iteration + every startProcessingIfNeeded), so this is only reached under a
  /// pathological flood while the loop is wedged; oldest is evicted so a flood
  /// cannot grow memory unbounded. Set generously: a dropped cancel would
  /// resurrect a job, so eviction must stay a pathological-only event.
  static let maxDeltas = 512

  init() {
    // Seed the authoritative pause flag from the same sentinel the actor used to
    // read directly, so a paused queue survives a restart exactly as before.
    // Undrained deltas are NOT preloaded here: `recoverPersistedQueue` is the
    // single startup consumer of the sidecar (§3.1.4a point 4); preloading would
    // double-count.
    self.paused = FileManager.default.fileExists(atPath: WarmServerCoordinator.pauseSentinelPath)
  }

  func publish(_ s: HealthSnapshot) {
    lock.lock(); snapshot = s; lock.unlock()
    notifyPublication()
  }

  /// comfybox#362 review r2, item 1: publish the health snapshot AND the
  /// interrupt triple in ONE lock acquisition.
  ///
  /// The preemption episode swaps which job is "active" in two observable
  /// places — `/health`'s `active_job_id` and the task `/v1/queue/interrupt`
  /// cancels. As two separate writes there is always a moment, however short,
  /// where a reader on another thread sees one swapped and the other not: with
  /// health first, an operator's interrupt kills the VIDEO while health names
  /// the image job (the original #362 bug, in miniature); with the triple
  /// first, it kills the IMAGE job while health still names the video. One
  /// lock, one write, no window either way.
  func publish(_ s: HealthSnapshot, activeRender publication: PublishedRender) {
    lock.lock(); snapshot = s; activeRender = publication; lock.unlock()
    notifyPublication()
  }

  func setProgress(_ p: Int?) { lock.lock(); progressPercent = p; lock.unlock() }

  /// The published interrupt triple — a NON-destructive read of what
  /// `/v1/queue/interrupt`'s default target would reach, for tests that need
  /// to wait for a publication without cancelling it.
  func activeRenderPublication() -> PublishedRender {
    lock.lock(); defer { lock.unlock() }; return activeRender
  }

  #if DEBUG
  /// comfybox#362 review r2, item 1 test seam. Fired after EVERY write that
  /// changes either observable state — the health snapshot or the interrupt
  /// triple — with the pair as a reader would then see it.
  ///
  /// The invariant it exists to pin: **whenever the triple names a job, health
  /// must name the SAME job.** If someone ever splits the atomic publish above
  /// back into two writes, this fires twice per transition and the middle
  /// sample violates that, so the sequencing test fails instead of the bug
  /// shipping.
  nonisolated(unsafe) static var publicationObserver:
    (@Sendable (_ healthJobId: String?, _ targetJobId: String?, _ targetKind: String?) -> Void)?

  private func notifyPublication() {
    guard let observer = Self.publicationObserver else { return }
    lock.lock()
    let healthJobId = snapshot.activeJobId
    let target = activeRender
    lock.unlock()
    observer(healthJobId, target.jobId, target.kind)
  }
  #else
  private func notifyPublication() {}
  #endif

  /// Read the published snapshot with the AUTHORITATIVE pause overlaid (§3.1.5:
  /// "the read path composes lockStore.isPaused with the actor-authored
  /// remainder"). `publishHealth()` no longer writes `isPaused`, so this overlay
  /// is the sole source of `is_paused` for /health and /v1/queue.
  func read() -> (HealthSnapshot, Int?) {
    lock.lock(); defer { lock.unlock() }
    var s = snapshot
    s.isPaused = paused
    return (s, progressPercent)
  }

  // MARK: pause authority

  func isPausedAuthoritative() -> Bool { lock.lock(); defer { lock.unlock() }; return paused }

  /// Authoritative pause write + its persistence (the sentinel IS the on-disk
  /// form of this flag). Idempotent, so the sync route and the actor's
  /// `setPaused` can both call it without racing to an inconsistent sentinel.
  func setPaused(_ value: Bool) {
    lock.lock(); paused = value; lock.unlock()
    if value {
      FileManager.default.createFile(
        atPath: WarmServerCoordinator.pauseSentinelPath,
        contents: Data("paused \(Date())\n".utf8))
    } else {
      try? FileManager.default.removeItem(atPath: WarmServerCoordinator.pauseSentinelPath)
    }
  }

  // MARK: delta mailbox

  /// One undrained delta plus the generation it was recorded at (review round
  /// 2, item 1). The generation is what lets the drain distinguish a delta
  /// that is merely visible in memory from one that is actually durable on
  /// disk — see `peekDeltas` below.
  private struct StampedDelta {
    let generation: Int
    let command: QueueControlCommand
  }

  /// comfybox#386: the sidecar write below must never run while holding
  /// `lock` — `lock` is also what `read()` (the sync-servable `/health` and
  /// `/v1/queue` routes' only dependency) needs, and `QueueDeltaStore.save` is
  /// disk I/O that can stall on a slow or nearly-full volume. Every mutator
  /// here follows the same shape: mutate + snapshot + stamp a generation
  /// number while holding `lock`, release it, THEN persist.
  ///
  /// `recordDelta`, `commitDrainedDeltas`, and `clearDeltas` all write this
  /// same sidecar file and can genuinely race each other. Two invariants,
  /// both restored in review round 2 after round 1's lock-scoping fix traded
  /// them away:
  ///
  /// 1. **No lost/double-applied delta on a crash (WAL).** Round 1 let
  ///    `peekDeltas` (the drain's read) return a delta the instant it was
  ///    appended in memory — BEFORE its sidecar write reached disk. If the
  ///    drain then applied that delta and `persistQueueState()` ran, and the
  ///    process died before the sidecar write landed, the delta was durable
  ///    in NEITHER file: gone from memory (the crash) and never written to
  ///    the sidecar — a cancelled job could resurrect. Fixed by stamping each
  ///    delta with its own generation and having `peekDeltas` (drain-only)
  ///    return only the PREFIX whose generation is already `<=
  ///    lastPersistedDeltaGeneration` — only deltas the drain could also
  ///    recover from disk if the process died right after applying them.
  ///    `undrainedDeltas` (the sync `/health`/`/v1/queue` read) is
  ///    deliberately NOT gated this way: it is a display of what has been
  ///    accepted, not a "safe for the drain to act on" signal, so it still
  ///    shows every recorded delta the instant it is recorded.
  /// 2. **No stale write ever looks durable.** `sidecarLock` serializes the
  ///    ENTIRE check-then-write-then-publish span for one write (not just the
  ///    write itself), so two concurrent writers can never both decide
  ///    they're allowed to write and then race each other to the rename —
  ///    the loser's write is a no-op instead of clobbering fresher content.
  ///    And `lastPersistedDeltaGeneration` (read/written under `lock`, in
  ///    short non-blocking hops from inside that span) never claims a
  ///    generation is durable unless `QueueDeltaStore.save` actually reported
  ///    success for it — a failed write must never advance the marker (item
  ///    2: `save`'s old silent `try?` swallow is now a `Bool` result).
  ///
  /// A background/async writer queue was rejected: it would let these
  /// methods return before the sidecar reflects their effect, which the
  /// existing WAL (`testSidecarSurvivesUntilCanonicalStatePersists`) and
  /// recovery-replay tests assert synchronously.
  ///
  /// comfybox#386 review round 4, item 1: **every call site that can still
  /// BLOCK on `sidecarLock`, and the thread/context each runs on.** Only
  /// `persistDeltaSidecar` (the blocking wrapper) can ever wait for
  /// `sidecarLock`; everything reachable from the coordinator actor uses the
  /// non-blocking `tryPersistDeltaSidecar` instead (see its own doc comment
  /// for why). By construction — grep `liveHealth.recordDelta(` and
  /// `liveHealth.clearDeltas(` in this file — the only two callers of the
  /// blocking path are:
  ///   - `recordDelta`, from `WarmServer.syncCancelResponse`,
  ///     `syncMoveResponse`, `syncClearResponse` — the sync `DELETE
  ///     /v1/queue/{id}`, `POST /v1/queue/{id}/move`, `POST /v1/queue/clear`
  ///     routes (0.B-2), each served SYNCHRONOUSLY on ITS OWN connection's
  ///     `DispatchQueue` before any `Task {}` — so several of these can run
  ///     truly concurrently (one per connection), but NONE of them is the
  ///     coordinator actor.
  ///   - `clearDeltas`, from `WarmServer.recoverPersistedQueue()`, called
  ///     synchronously (not inside a `Task`) from `WarmServer.run()` during
  ///     process boot — strictly BEFORE the listener binds its port and
  ///     BEFORE the coordinator actor's `processLoop` ever starts. A single
  ///     one-time call on the boot thread, never concurrent with anything.
  /// `commitDrainedDeltas` — the one mutator the actor's OWN
  /// `drainQueueDeltas` calls — used the blocking path through round 3; round
  /// 4 moved it to `tryPersistDeltaSidecar` (see its doc comment) precisely
  /// because it is the one call site that WAS reachable from the actor.
  private let sidecarLock = NSLock()
  private var deltaGeneration = 0
  /// The highest delta generation `QueueDeltaStore.save` has actually
  /// confirmed on disk. Read/written under `lock` — never under `sidecarLock`
  /// while `save` itself runs — so `peekDeltas`/`read()`/`undrainedDeltas`
  /// never wait on disk; only `persistDeltaSidecarLocked` (called with
  /// `sidecarLock` already held, by either wrapper) advances it, and only
  /// after a successful write.
  private var lastPersistedDeltaGeneration = 0
  private let sidecarLogger = Logger(label: "z-image.live-health.sidecar")
  /// Edge-triggered so a stuck-full-disk episode logs once, not once per
  /// mutation — but a NEW failure streak (after an intervening success) logs
  /// again.
  private var hasLoggedSidecarWriteFailure = false
  /// When the CURRENT unresolved failure streak began — `nil` while writes
  /// are succeeding, and reset (review round 4, item 2) once the streak ages
  /// out — see `resetFailureStreakIfStaleLocked`. Read/written under `lock`
  /// alongside `lastFailureAt`/`consecutiveFailureCount`: together they're
  /// the input to `deltaDurabilityStatus().isDegraded`.
  private var firstUnresolvedFailureAt: Date?
  /// The most recent failed attempt, regardless of streak — comfybox#386
  /// review round 4, item 2: this is what "ages out" a streak. A single
  /// hiccup that nobody ever retried (a PARKED loop: nothing enqueuing,
  /// cancelling, or moving, so nothing calls `persistDeltaSidecar` again) has
  /// no business staying "degraded" forever just because wall-clock time
  /// passed with no further evidence the disk is still broken.
  private var lastFailureAt: Date?
  private var consecutiveFailureCount = 0

  /// comfybox#386 review round 3, item 1b: once the sidecar has been failing
  /// continuously for this long, OR for this many consecutive attempts —
  /// whichever comes first — the drain applies non-durable deltas anyway
  /// rather than starve a cancel forever behind a broken disk (see
  /// `WarmServerCoordinator.drainQueueDeltas`). Liveness wins past this
  /// point, matching the pre-comfybox#386 behavior (which had no durability
  /// gate at all) — but the tradeoff is now OBSERVABLE via `/health`'s
  /// `queue_delta_sidecar_degraded` instead of a cancel silently vanishing
  /// from `/v1/queue` while the job keeps rendering. Both bounds are
  /// deliberately small: a real outage (nearly-full disk) fails fast and
  /// repeatedly, so genuine liveness needs are met within a couple of
  /// scheduling points, and a merely-slow (not broken) disk keeps making
  /// progress well under either bound.
  static let degradedModeWindowSeconds: TimeInterval = 5.0
  static let degradedModeFailureCountThreshold = 3

  /// comfybox#386 review round 4, item 3: bounded self-heal so a PARKED loop
  /// (nothing enqueuing/cancelling/moving — no natural trigger for a retry)
  /// still recovers once the disk comes back, rather than depending on the
  /// next unrelated mutation or drain pass to notice. Small delay, small cap:
  /// a real outage is either transient (heals within a couple of seconds) or
  /// prolonged, in which case degraded mode (item 1b) already gives liveness
  /// and repeated background attempts add nothing but log noise.
  static let backgroundHealMaxAttempts = 5
  /// `var`, not `let` (review round 5, item 2): a test driving the
  /// multi-failure heal chain (`testBackgroundHealRetriesThroughMultipleFailuresBeforeSucceeding`)
  /// needs to shrink this to avoid burning several real seconds per run —
  /// tests lower it and MUST restore it afterward (nothing resets this
  /// automatically; there is no per-instance override).
  static var backgroundHealRetryDelaySeconds: TimeInterval = 1.0
  /// Guards against stacking more than one heal CHAIN at once — a burst of
  /// several failed writes in quick succession must not spawn several
  /// independent retry chains. Touched only under `lock`; cleared when a
  /// chain ends (success, gives up at the cap, or finds itself already
  /// durable by the time it runs).
  private var backgroundHealScheduled = false

  /// Total undrained deltas, how many are durable, and whether the drain
  /// should apply the rest anyway (review round 3, item 1). One read under
  /// `lock` so `/health`'s telemetry and the drain's own decision never
  /// disagree with each other. `now` is injectable (review round 4, item 2
  /// test seam) — defaults to the real clock everywhere in production.
  struct DeltaDurabilityStatus {
    let totalCount: Int
    let durableCount: Int
    let isDegraded: Bool
    var nonDurableCount: Int { totalCount - durableCount }
  }

  func deltaDurabilityStatus(now: Date = Date()) -> DeltaDurabilityStatus {
    lock.lock(); defer { lock.unlock() }
    resetFailureStreakIfStaleLocked(now: now)
    let durableCount = deltas.filter { $0.generation <= lastPersistedDeltaGeneration }.count
    // review round 4, item 2: the TIME half requires at least 2 consecutive
    // failures, not just 1 — a single transient failure sitting unresolved
    // must never trip degraded mode purely because 5 seconds happened to
    // pass (e.g. nothing else was going on). The COUNT half already implies
    // this (its own threshold is 3), so this only tightens the time half.
    let degraded = consecutiveFailureCount >= Self.degradedModeFailureCountThreshold
      || (consecutiveFailureCount >= 2
        && firstUnresolvedFailureAt.map { now.timeIntervalSince($0) >= Self.degradedModeWindowSeconds } ?? false)
    return DeltaDurabilityStatus(totalCount: deltas.count, durableCount: durableCount, isDegraded: degraded)
  }

  /// comfybox#386 review round 4, item 2: a failure streak "ages out" once
  /// nothing has actually failed (or been retried) in
  /// `degradedModeWindowSeconds` — the CALLER must already hold `lock`.
  /// Invoked both here (on every status read, so a parked loop's stale
  /// streak clears itself the next time anyone asks) and from
  /// `persistDeltaSidecarLocked`'s own failure branch (so a brand-new failure
  /// arriving long after the last one starts a FRESH streak of 1, not a
  /// continuation of ancient history).
  private func resetFailureStreakIfStaleLocked(now: Date) {
    guard let last = lastFailureAt, now.timeIntervalSince(last) >= Self.degradedModeWindowSeconds else { return }
    firstUnresolvedFailureAt = nil
    lastFailureAt = nil
    consecutiveFailureCount = 0
  }

  #if DEBUG
  /// comfybox#386 review round 5, item 3 test seam: directly stamp the
  /// failure-streak fields. `persistDeltaSidecarLocked` always uses the real
  /// wall clock to record actual failures, so this is the only way to
  /// construct "first failure old, last failure recent" (the TIME half's
  /// positive case: `firstUnresolvedFailureAt` past the window while
  /// `lastFailureAt` is still within it, i.e. NOT aged out) on demand,
  /// without a real multi-second sleep. No legitimate production use.
  func testSeamStampFailureStreak(first: Date, last: Date, count: Int) {
    lock.lock()
    firstUnresolvedFailureAt = first
    lastFailureAt = last
    consecutiveFailureCount = count
    lock.unlock()
  }
  #endif

  func recordDelta(_ delta: QueueControlCommand) {
    lock.lock()
    deltaGeneration += 1
    let generation = deltaGeneration
    deltas.append(StampedDelta(generation: generation, command: delta))
    if deltas.count > Self.maxDeltas { deltas.removeFirst(deltas.count - Self.maxDeltas) }
    let snapshot = deltas.map { $0.command }
    lock.unlock()
    persistDeltaSidecar(generation: generation, snapshot: snapshot)
  }

  /// Every recorded delta, durable or not — what `/health` and `/v1/queue`
  /// (and the sync cancel/move routes' own present-check) compose against.
  /// Deliberately NOT gated on durability — see the class-level comment above
  /// `sidecarLock`.
  func undrainedDeltas() -> [QueueControlCommand] {
    lock.lock(); defer { lock.unlock() }
    return deltas.map { $0.command }
  }

  /// Snapshot the DURABLE undrained deltas for the actor to apply — WITHOUT
  /// clearing (F-2, WAL ordering): the sidecar must outlive the canonical
  /// `persistQueueState()` write, so a kill mid-drain replays the deltas on
  /// the next boot instead of resurrecting a cancelled job. The actor calls
  /// `commitDrainedDeltas` only AFTER canonical state is on disk.
  ///
  /// comfybox#386 review round 2: filtered to `generation <=
  /// lastPersistedDeltaGeneration` — a delta the sidecar hasn't confirmed yet
  /// is invisible to the drain, so the drain can never act on something that
  /// wouldn't also survive a crash right now. Deltas are appended (and
  /// generations assigned) in order, so "durable" is always a PREFIX of
  /// `deltas` — exactly what `commitDrainedDeltas`'s prefix-drop assumes.
  func peekDeltas() -> [QueueControlCommand] {
    lock.lock(); defer { lock.unlock() }
    return deltas.filter { $0.generation <= lastPersistedDeltaGeneration }.map { $0.command }
  }

  /// WAL commit point (F-2): canonical queue state is persisted; drop the first
  /// `count` deltas — the ones the drain applied — and rewrite the sidecar to
  /// the remainder, so deltas recorded DURING the drain survive to the next one.
  ///
  /// comfybox#386 review round 4, item 1: this is the ONE mutator the
  /// coordinator actor calls directly (`WarmServerCoordinator.drainQueueDeltas`),
  /// so it MUST use the non-blocking `tryPersistDeltaSidecar` — the blocking
  /// path here was the last wedge of this class: a stuck/slow disk write
  /// (from a totally unrelated `recordDelta` call, or this same call on a
  /// previous drain pass) would park the render loop behind `sidecarLock`.
  ///
  /// Review round 5 correction: memory (`deltas`) is trimmed ABOVE
  /// regardless of this write's outcome. That is safe — but NOT because
  /// "nothing is left on disk to double-apply": a failed/skipped rewrite
  /// here leaves the SIDECAR FILE still listing the just-drained commands,
  /// exactly like any other failed write would. The actual guarantee is the
  /// WAL replay ordering `drainQueueDeltas`'s own doc comment already
  /// establishes (F-2): canonical `persistQueueState()` always runs BEFORE
  /// this call, so by the time anything ever replays that stale sidecar (a
  /// crash here, then `recoverPersistedQueue` on the next boot), the
  /// canonical state it folds against ALREADY reflects the drained
  /// command's effect — replaying an already-applied `.cancel` there is the
  /// documented no-op (the id is already absent from the recovered jobs).
  /// (`.move`'s replay parity against an already-moved canonical order isn't
  /// separately proven by this argument — a pre-existing characteristic of
  /// the F-2 design this round didn't touch, not something introduced here.)
  /// A failed/skipped attempt here just leaves `lastPersistedDeltaGeneration`
  /// behind; the drain's own item-1a retry (next pass, `peekDeltas` empty
  /// because the marker lags) or the background self-heal (item 3) catches
  /// the sidecar back up to the now-shorter truth.
  func commitDrainedDeltas(_ count: Int) {
    lock.lock()
    deltas.removeFirst(min(count, deltas.count))
    deltaGeneration += 1
    let generation = deltaGeneration
    let snapshot = deltas.map { $0.command }
    lock.unlock()
    _ = tryPersistDeltaSidecar(generation: generation, snapshot: snapshot)
  }

  /// Persist `snapshot` to the sidecar. `sidecarLock` wraps the WHOLE
  /// check-then-write-then-publish span (not just the write) so two
  /// concurrent callers can never both pass the staleness check and then race
  /// each other to the actual write — the loser would otherwise clobber
  /// fresher content with stale, or (worse) leave
  /// `lastPersistedDeltaGeneration` claiming durability for content that
  /// isn't actually the latest thing on disk. The brief `lock` hops inside
  /// are pure in-memory reads/writes of a single `Int` — never wrapping
  /// `QueueDeltaStore.save` itself — so nothing blocked on `lock` (`read()`,
  /// `undrainedDeltas`, `peekDeltas`) ever waits on this disk write, however
  /// slow or stuck.
  ///
  /// comfybox#386 review round 2, item 2: the marker advances ONLY on a
  /// successful write. The delta is never lost on a failure: it stays in
  /// `deltas` and rides along, unchanged, in the FULL snapshot the next
  /// successful write sends (every snapshot is the whole undrained list,
  /// never an increment) — so a later write recovers everything a failed one
  /// missed.
  @discardableResult
  private func persistDeltaSidecar(generation: Int, snapshot: [QueueControlCommand]) -> Bool {
    sidecarLock.lock(); defer { sidecarLock.unlock() }
    return persistDeltaSidecarLocked(generation: generation, snapshot: snapshot)
  }

  /// comfybox#386 review round 5 (critical): what a non-blocking attempt
  /// actually accomplished — distinct from a plain `Bool`, which conflated
  /// "we acquired `sidecarLock`" with "the write succeeded." `runBackgroundHeal`
  /// used to treat `sidecarLock.try()` SUCCEEDING as "done" regardless of
  /// whether `QueueDeltaStore.save` then failed: a sustained outage (lock
  /// always free, write always failing) got exactly ONE real attempt before
  /// the chain incorrectly declared victory and stopped — the ≤5 cap only
  /// ever applied to LOST TRY-LOCK RACES, never to actual write failures.
  private enum SidecarWriteOutcome {
    /// `sidecarLock` was already held elsewhere; nothing was attempted.
    case lockBusy
    /// The write landed, or this generation was already covered by an
    /// even-newer persisted one — either way the sidecar now reflects
    /// something at least as fresh as `snapshot` as of when this call ran.
    case wroteOK
    /// The lock was acquired, an attempt was made, and `QueueDeltaStore.save`
    /// reported failure.
    case writeFailed
  }

  /// Non-blocking sibling of `persistDeltaSidecar`. Every caller that can run
  /// ON (or be triggered synchronously by) the coordinator actor, or that
  /// must never contend with a route handler's in-flight write, goes through
  /// this: `retryPendingSidecarWrite` (`drainQueueDeltas`'s item-1a liveness
  /// retry), `commitDrainedDeltas` (the drain's own WAL commit — review round
  /// 4, item 1: this was the last blocking call site reachable from the
  /// actor), and `runBackgroundHeal` (review round 4, item 3's off-actor
  /// self-heal — which MUST branch on `.wroteOK` specifically, per the
  /// round-5 critical fix above). Actor code in particular must never block
  /// waiting for `sidecarLock` — a single slow/stuck disk write from a
  /// totally unrelated caller would otherwise stall the ENTIRE render loop
  /// behind it, trading a `/health`-only liveness bug for a much worse one.
  /// `NSLock.try()` gives up immediately if another write already holds
  /// `sidecarLock`; each of these callers has its own way of trying again
  /// (the drain's next scheduling point, the heal chain's next delayed
  /// attempt), so nothing is lost by not blocking.
  private func tryPersistDeltaSidecar(generation: Int, snapshot: [QueueControlCommand]) -> SidecarWriteOutcome {
    guard sidecarLock.`try`() else { return .lockBusy }
    defer { sidecarLock.unlock() }
    return persistDeltaSidecarLocked(generation: generation, snapshot: snapshot) ? .wroteOK : .writeFailed
  }

  /// The check-then-write-then-publish body shared by `persistDeltaSidecar`
  /// and `tryPersistDeltaSidecar` — the CALLER must already hold
  /// `sidecarLock` for its whole span (not just the write) so two concurrent
  /// callers can never both pass the staleness check and then race each
  /// other to the actual write — the loser would otherwise clobber fresher
  /// content with stale, or (worse) leave `lastPersistedDeltaGeneration`
  /// claiming durability for content that isn't actually the latest thing on
  /// disk. The brief `lock` hops inside are pure in-memory reads/writes of a
  /// few small values — never wrapping `QueueDeltaStore.save` itself — so
  /// nothing blocked on `lock` (`read()`, `undrainedDeltas`, `peekDeltas`,
  /// `deltaDurabilityStatus`) ever waits on this disk write, however slow or
  /// stuck.
  ///
  /// comfybox#386 review round 2, item 2: the durability marker advances
  /// ONLY on a successful write. The delta is never lost on a failure: it
  /// stays in `deltas` and rides along, unchanged, in the FULL snapshot the
  /// next successful write sends (every snapshot is the whole undrained
  /// list, never an increment) — so a later write recovers everything a
  /// failed one missed. Review round 3, item 1b: failures also update
  /// `firstUnresolvedFailureAt`/`consecutiveFailureCount`, the input to
  /// `deltaDurabilityStatus().isDegraded`. Review round 4, item 2: a failure
  /// long after the last one starts a FRESH streak, not a continuation of
  /// ancient history (`resetFailureStreakIfStaleLocked`). Review round 4,
  /// item 3: a failure also schedules the bounded background self-heal.
  /// Returns whether the sidecar now reflects `snapshot` (or something even
  /// fresher) — `true` for a successful write OR a stale no-op (already
  /// covered by a newer persisted generation), `false` only when
  /// `QueueDeltaStore.save` itself reported failure. This is what
  /// `tryPersistDeltaSidecar` maps onto `.wroteOK`/`.writeFailed`.
  @discardableResult
  private func persistDeltaSidecarLocked(generation: Int, snapshot: [QueueControlCommand]) -> Bool {
    lock.lock()
    let isStale = generation <= lastPersistedDeltaGeneration
    lock.unlock()
    guard !isStale else { return true }

    let succeeded = QueueDeltaStore.save(snapshot)

    lock.lock()
    if succeeded {
      lastPersistedDeltaGeneration = generation
      firstUnresolvedFailureAt = nil
      lastFailureAt = nil
      consecutiveFailureCount = 0
    } else {
      let now = Date()
      resetFailureStreakIfStaleLocked(now: now)
      if firstUnresolvedFailureAt == nil { firstUnresolvedFailureAt = now }
      lastFailureAt = now
      consecutiveFailureCount += 1
      scheduleBackgroundHealLocked()
    }
    lock.unlock()

    if succeeded {
      hasLoggedSidecarWriteFailure = false
    } else if !hasLoggedSidecarWriteFailure {
      hasLoggedSidecarWriteFailure = true
      sidecarLogger.error(
        "comfybox#386: failed to persist queue-deltas.json sidecar (generation \(generation)) — undrained deltas will not survive a restart until a write succeeds")
    }

    return succeeded
  }

  /// comfybox#386 review round 4, item 3: schedule (at most one concurrent
  /// chain of) a bounded background retry — the CALLER must already hold
  /// `lock`. Entirely off the actor and off whatever thread is failing right
  /// now: a detached `Task` sleeping `backgroundHealRetryDelaySeconds` before
  /// trying again via the non-blocking `tryPersistDeltaSidecar`, so it can
  /// never contend with (or be blocked by) anything a route handler or the
  /// coordinator is doing. This is what heals a PARKED loop — one with no
  /// enqueue/cancel/move ever happening again to naturally retry — instead of
  /// requiring an unrelated mutation or drain pass to notice the disk is back.
  private func scheduleBackgroundHealLocked() {
    guard !backgroundHealScheduled else { return }
    backgroundHealScheduled = true
    Task.detached { [weak self] in
      await self?.runBackgroundHeal(attempt: 1)
    }
  }

  /// `attempt` counts real attempts MADE so far — the call this invocation is
  /// about to make is attempt number `attempt`, and `backgroundHealMaxAttempts`
  /// bounds how many of those actually happen (attempts `1...maxAttempts`
  /// each get to try; there is no `maxAttempts + 1`th call).
  private func runBackgroundHeal(attempt: Int) async {
    try? await Task.sleep(nanoseconds: UInt64(Self.backgroundHealRetryDelaySeconds * 1_000_000_000))

    lock.lock()
    let generation = deltaGeneration
    let alreadyDurable = generation <= lastPersistedDeltaGeneration
    let snapshot = deltas.map { $0.command }
    lock.unlock()

    // Durable already (something else — a route handler's write, the
    // drain's own retry — beat us to it): end this chain without spending an
    // attempt.
    guard !alreadyDurable else {
      lock.lock(); backgroundHealScheduled = false; lock.unlock()
      return
    }

    // comfybox#386 review round 5 (critical): MUST branch on `.wroteOK`
    // specifically, not merely "the call returned something" — the prior
    // `if tryPersistDeltaSidecar(...)` (a plain `Bool`) treated ACQUIRING
    // `sidecarLock` as success regardless of whether `QueueDeltaStore.save`
    // then failed, so a sustained outage (lock always free, write always
    // failing) got exactly ONE real attempt before this incorrectly declared
    // victory and stopped — the cap below never actually applied to a real
    // write failure, only to a lost try-lock race.
    switch tryPersistDeltaSidecar(generation: generation, snapshot: snapshot) {
    case .wroteOK:
      lock.lock(); backgroundHealScheduled = false; lock.unlock()
      return
    case .writeFailed, .lockBusy:
      // Still failing, or lost the try-lock race to a concurrent write —
      // either way this was a real attempt at healing and counts against
      // the cap. Under the cap: keep the chain's `backgroundHealScheduled`
      // claim and reschedule rather than clearing and re-setting it (which
      // would let a second chain slip in during the gap).
      guard attempt < Self.backgroundHealMaxAttempts else {
        lock.lock(); backgroundHealScheduled = false; lock.unlock()
        return
      }
      await runBackgroundHeal(attempt: attempt + 1)
    }
  }

  /// comfybox#386 review round 3, item 1a: re-attempt persisting whatever is
  /// CURRENTLY undrained, at the CURRENT generation — no new mutation, no new
  /// delta, just another try at the same content a previous attempt failed
  /// to land. Used by the drain (`WarmServerCoordinator.drainQueueDeltas`)
  /// when `peekDeltas` comes back empty but deltas still exist: the
  /// durability marker is stuck behind a prior failure, and nothing will
  /// move it forward unless someone tries the write again.
  ///
  /// NON-BLOCKING (`tryPersistDeltaSidecar`): this can run on the actor
  /// itself, which must never stall behind `sidecarLock` contention — if
  /// another write is already in flight, this attempt is simply skipped
  /// rather than parked, and the drain's own guard falls through to the
  /// degraded-mode check or gives up for this pass, trying again next time.
  /// comfybox#386 review round 5: branches on `SidecarWriteOutcome.wroteOK`
  /// specifically (not merely "the call returned"), so this can never
  /// mistake "we acquired the lock" for "the write actually landed" — the
  /// exact conflation that made `runBackgroundHeal` declare victory after a
  /// single failed write (see its own doc comment). Returns whether the
  /// current generation is durable once this call returns (`true` also when
  /// there was nothing to persist or it was already durable).
  @discardableResult
  func retryPendingSidecarWrite() -> Bool {
    lock.lock()
    let generation = deltaGeneration
    let alreadyDurable = generation <= lastPersistedDeltaGeneration
    let snapshot = deltas.map { $0.command }
    lock.unlock()
    guard !alreadyDurable else { return true }
    switch tryPersistDeltaSidecar(generation: generation, snapshot: snapshot) {
    case .wroteOK: return true
    case .writeFailed, .lockBusy: return false
    }
  }

  /// Clear the in-memory deltas (recovery already folded them into the
  /// recovered queue) — but ONLY once the empty snapshot is CONFIRMED
  /// durable. comfybox#386 review round 2, item 3: a clear persists the empty
  /// snapshot at a new generation through the same `sidecarLock`/generation
  /// machinery as every other mutation, instead of a bare, unprotected
  /// `QueueDeltaStore.clear()` call. The one production caller,
  /// `WarmServer.recoverPersistedQueue()`, actually runs SYNCHRONOUSLY from
  /// `WarmServer.run()` during process boot — strictly before the listener
  /// binds its port, so no real HTTP-driven cancel/move can race it in
  /// production today. The generation guard earns its keep anyway: it is
  /// what makes this call SAFE to reuse from a context that CAN race (this
  /// file's own tests drive it concurrently with an in-flight older write —
  /// `testClearDeltasCannotBeResurrectedByAnOlderInFlightWrite` — and a
  /// future caller should not have to relearn this the hard way), and unlike
  /// the bare `QueueDeltaStore.clear()` it replaced, it is provably correct
  /// rather than "safe because nothing races it right now."
  ///
  /// Review round 3, item 2: memory is stripped only AFTER
  /// `persistDeltaSidecar` confirms the write (`generation <=
  /// lastPersistedDeltaGeneration` afterward — true whether THIS call's own
  /// write succeeded or a later, superseding write already covers it). Round
  /// 2's version cleared `deltas` unconditionally up front: on a failed
  /// write, disk kept the STALE pre-clear content while memory had already
  /// forgotten it, so a crash before any later write happened to overwrite
  /// the sidecar left the next boot's `recoverPersistedQueue` re-folding
  /// deltas it had already applied this session — `.move` is not idempotent,
  /// so a repeat could silently reorder the queue wrong. On failure here,
  /// nothing is lost: `deltas` (and disk) are untouched, and the very next
  /// successful write ANYWHERE (`recordDelta`, `commitDrainedDeltas`, or a
  /// later `clearDeltas`) resends the full current snapshot and corrects the
  /// stale disk content — there is no separate retry loop to maintain.
  /// `removeAll(where:)` (not a blind `removeAll()`) so a delta that was
  /// recorded concurrently, strictly AFTER this clear's generation was
  /// stamped, survives — it was never part of what this clear persisted.
  func clearDeltas() {
    lock.lock()
    deltaGeneration += 1
    let generation = deltaGeneration
    lock.unlock()

    persistDeltaSidecar(generation: generation, snapshot: [])

    lock.lock(); defer { lock.unlock() }
    guard generation <= lastPersistedDeltaGeneration else { return }
    deltas.removeAll { $0.generation <= generation }
  }

  // MARK: active-render interrupt (sync /v1/queue/interrupt)

  /// Publish (or clear, with `.none`) the active render — task and identity in
  /// ONE write under the lock (review r1, finding 1).
  func setActiveRender(_ publication: PublishedRender) {
    lock.lock(); activeRender = publication; lock.unlock()
    notifyPublication()
  }

  /// comfybox#362: publish (or clear, with `.none`) the checkpointed video's
  /// own task + ids for the span of a preemption episode. See
  /// `checkpointedVideo`'s doc comment.
  func setCheckpointedVideo(_ publication: PublishedRender) {
    lock.lock(); checkpointedVideo = publication; lock.unlock()
  }

  /// Cancel per `target` (comfybox#362) — see `InterruptTarget` for the target
  /// vocabulary. `Task.cancel` is Sendable, so the sync interrupt route calls
  /// this with no actor hop.
  ///
  /// Both slots are read under ONE lock acquisition, so the pair handed to
  /// `InterruptExecutor` is a consistent snapshot; and `InterruptExecutor` is
  /// literally the same function the async fallback runs (review r1, finding
  /// 2), not a parallel implementation of the same rules.
  func cancelActiveRender(target: String? = nil) -> InterruptCancelOutcome {
    lock.lock()
    let active = activeRender
    let video = checkpointedVideo
    lock.unlock()
    return InterruptExecutor.cancel(target: target, active: active, checkpointedVideo: video)
  }
}

/// Thread-safe holder for the active render's progress percent. Written from
/// the (off-actor, `@Sendable`) pipeline progress callback and read by the
/// actor's `queueStatus()` — lock-protected so it can cross the actor boundary
/// safely without an actor hop on every denoising step.
private final class RenderProgressTracker: @unchecked Sendable {
  private let lock = NSLock()
  private var percent: Int?
  func set(_ value: Int?) { lock.lock(); percent = value; lock.unlock() }
  func get() -> Int? { lock.lock(); defer { lock.unlock() }; return percent }
}

// MARK: - #1479 preemption support (lock-based, no actor hop)
//
// The coordinator actor is synchronously blocked for the whole duration of an
// in-flight LTX-2 render (`WarmServerCoordinator.processLoop`'s `.localVideo`
// case calls `body(report)` with no internal `await`), so a preempt decision
// that needs to run WHILE that render is in flight cannot itself be an actor
// call — it would simply queue behind the render and only run once the render
// already finished on its own, defeating the whole feature (see the #217
// PreemptionSignal precedent this mirrors). Everything below is therefore
// lock-protected, exactly like `PreemptionSignal`/`LTX2PhaseTelemetry`, and
// lives on `WarmServer` (not the actor) so the `/v1/generate` route handler
// can read/raise it before ever calling into the coordinator.

/// Rolling mean of an observed duration (evict, reload). Recorded by the
/// coordinator when it actually performs the operation during a preemption
/// episode; read with no actor hop by the pre-raise refusal guard.
final class RollingMeanSec: @unchecked Sendable {
  private let lock = NSLock()
  private var sumSec: Double = 0
  private var samples: Int = 0
  func record(_ seconds: Double) { lock.lock(); sumSec += seconds; samples += 1; lock.unlock() }
  func mean() -> Double? { lock.lock(); defer { lock.unlock() }; return samples > 0 ? sumSec / Double(samples) : nil }
}

/// Live "steps remaining" of the in-flight video render, fed from the same
/// per-chunk/per-step progress callback the video path already reports
/// through. Read (no actor hop) by the pre-raise refusal guard; cleared when
/// the render ends (completed or failed).
final class LTX2StepPosition: @unchecked Sendable {
  private let lock = NSLock()
  private var remaining: Int?
  /// #1479 (review I3): a multi-chunk render's true remaining step count is
  /// (steps left in THIS chunk) + (steps per chunk) x (whole chunks left) —
  /// counting only the current chunk's remainder under-projects a multi-
  /// chunk render's remaining time, causing spurious refusals.
  func update(chunk: Int, totalChunks: Int, step: Int, totalSteps: Int) {
    lock.lock()
    let inCurrentChunk = max(0, totalSteps - step)
    let fullChunksLeft = max(0, totalChunks - chunk - 1)
    remaining = inCurrentChunk + fullChunksLeft * max(0, totalSteps)
    lock.unlock()
  }
  func clear() { lock.lock(); remaining = nil; lock.unlock() }
  func read() -> Int? { lock.lock(); defer { lock.unlock() }; return remaining }
}

/// Single-owner lock flag: `trySet()` succeeds for exactly one caller until
/// `clear()`. Used to refuse nested preemption (spec: "a preemptor cannot
/// itself be preempted") without an actor hop.
final class LockedFlag: @unchecked Sendable {
  private let lock = NSLock()
  private var value = false
  @discardableResult
  func trySet() -> Bool {
    lock.lock(); defer { lock.unlock() }
    if value { return false }
    value = true
    return true
  }
  func clear() { lock.lock(); value = false; lock.unlock() }
  func get() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
}

/// A lock-protected, exactly-once single-slot handoff whose occupant carries
/// an *episode token* — a monotonically increasing stamp minted by `set`.
///
/// The token exists because a slot can be re-armed by a LATER episode while an
/// EARLIER episode's timeout watchdog is still asleep (#1479 review I1): the
/// stale watchdog would otherwise wake up, take an unqualified `claim()`, and
/// hijack the *next* episode's occupant — silently degrading a preemption that
/// was about to be honoured and releasing its in-flight flag out from under a
/// live continuation. `claim(matching:)` gives a holder a way to claim only the
/// occupant it armed itself; `claim()` (unqualified) is for the authoritative
/// consumer that always wants whatever is currently parked.
///
/// Generic purely so the claim semantics are unit-testable without building a
/// live `GeneratePayload`/continuation pair (see `PendingPreemptorBoxTests`).
final class TokenedSlot<Value>: @unchecked Sendable {
  private let lock = NSLock()
  private var value: Value?
  private var token: UInt64 = 0

  /// Park `v`, evicting anything already there, and return the episode token
  /// that identifies THIS occupant. Tokens never repeat within a process.
  @discardableResult
  func set(_ v: Value) -> UInt64 {
    lock.lock(); defer { lock.unlock() }
    token &+= 1
    value = v
    return token
  }

  /// Unconditional, exactly-once claim: whoever calls first gets the occupant,
  /// everyone else gets nil.
  func claim() -> Value? {
    lock.lock(); defer { lock.unlock() }
    let v = value
    value = nil
    return v
  }

  /// Token-qualified claim: succeeds only if the slot still holds the exact
  /// occupant `t` was minted for. Returns nil if it was already claimed, or if
  /// the slot has since been re-armed by a later episode.
  func claim(matching t: UInt64) -> Value? {
    lock.lock(); defer { lock.unlock() }
    guard token == t, let v = value else { return nil }
    value = nil
    return v
  }
}

/// The preempting image job, parked between the HTTP route handler raising
/// `PreemptionSignal` and whichever side observes the render yield first — the
/// coordinator's `.localVideo` case (the normal path, which takes the
/// unqualified `claim()`) or the checkpoint-failure watchdog Task (the render
/// never yielded in time, or finished on its own before the signal was
/// observed; it takes `claim(matching:)` with its own episode's token, so a
/// stale watchdog can never hijack a later preemptor — #1479 review I1).
struct PendingPreemptor {
  fileprivate let payload: GeneratePayload
  fileprivate let source: String
  fileprivate let rawBody: Data?
  /// The client-visible async job id, when the preemptor came through
  /// `/v1/generate/async` (AC-18) — the preempting render takes it as its
  /// active-job identity so the persisted snapshot names the same job.
  fileprivate let jobId: String?
  fileprivate let continuation: ContinuationBox<GenerateResponse>

  fileprivate init(payload: GeneratePayload, source: String, rawBody: Data?, jobId: String?, continuation: ContinuationBox<GenerateResponse>) {
    self.payload = payload
    self.source = source
    self.rawBody = rawBody
    self.jobId = jobId
    self.continuation = continuation
  }
}

typealias PendingPreemptorBox = TokenedSlot<PendingPreemptor>

/// Holds the latest live-denoising preview JPEG for polling clients (the
/// Desktop app, which already polls /health for progress_percent — see
/// GH #216). Krita/ComfyUI get previews pushed over their own WebSocket;
/// this is the same frame made available to REST/polling clients instead.
private final class RenderPreviewTracker: @unchecked Sendable {
  private let lock = NSLock()
  private var frame: Data?
  func set(_ value: Data?) { lock.lock(); frame = value; lock.unlock() }
  func get() -> Data? { lock.lock(); defer { lock.unlock() }; return frame }
}

/// State of an async-submitted image generation job — mirrors `VideoJobState`
/// so image and video generation share one submit/poll convention.
public enum ImageJobState: String, Codable, Sendable {
  case queued
  case processing
  case succeeded
  case failed
}

/// Wire status for `POST /v1/generate/async` / `GET /v1/generate/status/{id}`.
public struct ImageJobStatus: Codable, Sendable {
  public let jobId: String
  public let status: ImageJobState
  public let source: String
  public let outputPath: String?
  public let durationMs: Int?
  public let error: String?
  public let elapsedMs: Int
  /// #1479: set when this job asked to preempt an in-flight video render but
  /// the refusal guard declined — the job still ran normally (just queued),
  /// `etaSec` is the guard's projected remaining seconds. Both Optional (not
  /// merely defaulted) so `Codable`'s synthesized decode tolerates JSON that
  /// predates this field.
  public let preemptRefused: Bool?
  public let etaSec: Double?
  /// WP-E10 sink 4 (FDD §3.10): the provenance record of a succeeded Krea 2
  /// job — the same `applied` the sync response carries. Optional so
  /// persisted pre-upgrade JSON still decodes (AC-64); null for other
  /// families (D12) and until the job succeeds.
  public let applied: AppliedRecordSlot?
  /// #286: the flat `applied_loras` stack the sync response carries, so an
  /// async caller can verify its render the same way. Optional so persisted
  /// pre-#286 JSON still decodes.
  public let appliedLoras: [LoRAState]?
  /// #286 (C2/I1): the same additive flags the sync response carries.
  public let presetUnresolved: String?
  public let presetUnresolvedReason: String?
  public let presetStackMismatch: Bool?
  /// #22 (PR #363 review, C1b): the same memory-advisory numbers the sync
  /// response carries, set at accept time (before the job runs).
  public let memoryEstimateBytes: UInt64?
  public let memoryAvailableBytes: UInt64?
  /// #282: `lora_stack_origin` — `request` | `preset` | `warm_default`. Known
  /// only once the job DEQUEUES (the warm default is read then, not at
  /// submit), so it is absent on the 202 and on a job that never ran.
  public let loraStackOrigin: String?
  /// #282 review r1: the same additive markers the sync response carries.
  public let warmDefaultSkipped: String?
  public let loraReload: Bool?
  /// #154: `applied_shift` — the same flat echo the sync response carries, so
  /// an async caller can verify its schedule the same way. Absent until the
  /// job succeeds, and absent on a render that used the model's own schedule.
  public let appliedShift: Float?
  /// comfybox#283/#217: the last 5 recorded lifecycle events for this job id
  /// (see QueueLifecycleLedger.swift), newest last. `var` with an in-line
  /// default (rather than an `init` parameter) so every existing call site
  /// that constructs `ImageJobStatus` — none of which know about the
  /// ledger — is untouched; the one route that DOES know about it
  /// (`GET /v1/generate/status/{id}`) sets this field on the value AFTER
  /// construction. Additive: absent on any JSON predating this field.
  public var lifecycleTail: [QueueLifecycleEvent]? = nil

  /// The record itself; see ``AppliedRecordSlot`` for absent-vs-null.
  public var appliedRecord: RenderRecipe? { applied?.record }

  public init(
    jobId: String, status: ImageJobState, source: String, outputPath: String?, durationMs: Int?,
    error: String?, elapsedMs: Int, preemptRefused: Bool?, etaSec: Double?,
    applied: AppliedRecordSlot? = nil, appliedLoras: [LoRAState]? = nil,
    presetUnresolved: String? = nil, presetUnresolvedReason: String? = nil,
    presetStackMismatch: Bool? = nil,
    memoryEstimateBytes: UInt64? = nil, memoryAvailableBytes: UInt64? = nil,
    loraStackOrigin: String? = nil,
    warmDefaultSkipped: String? = nil, loraReload: Bool? = nil,
    appliedShift: Float? = nil
  ) {
    self.jobId = jobId
    self.status = status
    self.source = source
    self.outputPath = outputPath
    self.durationMs = durationMs
    self.error = error
    self.elapsedMs = elapsedMs
    self.preemptRefused = preemptRefused
    self.etaSec = etaSec
    self.applied = applied
    self.appliedLoras = appliedLoras
    self.presetUnresolved = presetUnresolved
    self.presetUnresolvedReason = presetUnresolvedReason
    self.presetStackMismatch = presetStackMismatch
    self.memoryEstimateBytes = memoryEstimateBytes
    self.memoryAvailableBytes = memoryAvailableBytes
    self.loraStackOrigin = loraStackOrigin
    self.warmDefaultSkipped = warmDefaultSkipped
    self.loraReload = loraReload
    self.appliedShift = appliedShift
  }
}

/// Internal mutable state for a tracked async image generation job.
private final class ImageJob: @unchecked Sendable {
  let id: String
  let source: String
  let startTime = Date()
  var state: ImageJobState = .queued
  var outputPath: String?
  var durationMs: Int?
  var error: String?
  var completedAt: Date?
  /// #1479: set once, before the job is enqueued, if `attemptPreemption`
  /// refused it.
  var preemptRefused: Bool?
  var etaSec: Double?
  /// WP-E10: set with the result on success (tri-state, see AppliedRecordSlot).
  var applied: AppliedRecordSlot?
  /// #286: the flat applied stack, set with the result on success.
  var appliedLoras: [LoRAState]?
  /// #286: the preset flags, set with the result on success.
  var presetUnresolved: String?
  var presetUnresolvedReason: String?
  var presetStackMismatch: Bool?
  /// #22: set at accept time from the payload (before the job runs) — see
  /// `job.memoryEstimateBytes = payload.memoryEstimateBytes` at submit.
  var memoryEstimateBytes: UInt64?
  var memoryAvailableBytes: UInt64?
  /// #282: set from the result, since the origin is only decided at dequeue.
  var loraStackOrigin: String?
  var warmDefaultSkipped: String?
  var loraReload: Bool?
  /// #154: set from the result on success — the schedule shift that applied.
  var appliedShift: Float?

  init(id: String, source: String) {
    self.id = id
    self.source = source
  }

  var elapsedMs: Int {
    let end = completedAt ?? Date()
    return Int(end.timeIntervalSince(startTime) * 1000)
  }

  func toStatus() -> ImageJobStatus {
    ImageJobStatus(
      jobId: id, status: state, source: source, outputPath: outputPath,
      durationMs: durationMs, error: error, elapsedMs: elapsedMs,
      preemptRefused: preemptRefused, etaSec: etaSec, applied: applied,
      appliedLoras: appliedLoras, presetUnresolved: presetUnresolved,
      presetUnresolvedReason: presetUnresolvedReason,
      presetStackMismatch: presetStackMismatch,
      memoryEstimateBytes: memoryEstimateBytes, memoryAvailableBytes: memoryAvailableBytes,
      loraStackOrigin: loraStackOrigin,
      warmDefaultSkipped: warmDefaultSkipped, loraReload: loraReload,
      appliedShift: appliedShift
    )
  }
}

/// Submit-and-poll wrapper around `WarmServerCoordinator.enqueueGenerate` so
/// callers (Bree's async envelope, the Telegram bot, MCP tools) can fire a
/// render without holding a connection open for the whole denoising run.
/// A blocking `POST /v1/generate` that takes minutes is what orphaned a
/// Telegram delivery in production once: the caller's own turn timeout
/// (180s) expired before the render finished, and there was no live turn
/// left to deliver through. Queue-submit decouples render time from the
/// caller's timeout — submit returns a job id immediately, and the caller
/// polls `GET /v1/generate/status/{id}` (or `/v1/video/status/{id}`'s twin)
/// until it sees `succeeded`/`failed`, exactly like the video path already
/// does via `ReplicateVideoProxy`.
final class ImageJobTracker: @unchecked Sendable {
  private let lock = NSLock()
  private var jobs: [String: ImageJob] = [:]

  /// Submit a job. Returns immediately with `queued` status; the render
  /// itself runs in a detached Task against the existing FIFO render queue,
  /// so submitting async doesn't skip the line ahead of synchronous callers.
  fileprivate func submit(_ payload: GeneratePayload, source: String, coordinator: WarmServerCoordinator, rawBody: Data? = nil) -> ImageJobStatus {
    submit(payload, source: source, rawBody: rawBody) { jobId in
      try await coordinator.enqueueGenerate(payload, source: source, rawBody: rawBody, jobId: jobId)
    }
  }

  /// The enqueue seam (WP-E10 "E9b", AC-18): the tracker's OWN id — the one
  /// the `/v1/generate/async` caller receives — is handed to the queue, so
  /// the persisted job, a failed replay after a restart, and the status
  /// route all name the same job. `enqueue` receives that id.
  func submit(
    _ payload: GeneratePayload, source: String, rawBody: Data?,
    enqueue: @escaping @Sendable (String) async throws -> GenerateResponse
  ) -> ImageJobStatus {
    let jobId = UUID().uuidString
    let job = ImageJob(id: jobId, source: source)
    lock.lock(); jobs[jobId] = job; lock.unlock()

    Task { [weak self] in
      guard let self else { return }
      self.markProcessing(jobId)
      do {
        let result = try await enqueue(jobId)
        self.markSucceeded(jobId, result: result)
      } catch {
        self.markFailed(jobId, error: error)
      }
    }
    return job.toStatus()
  }

  /// #1479: submit-and-poll with a preemption attempt tried FIRST, inside
  /// this job's own detached Task. `preemptor` is `WarmServer.attemptPreemption`
  /// bound to this payload — kept as a closure (rather than a direct
  /// dependency on `WarmServer`) so `ImageJobTracker`'s state machine stays
  /// unit-testable in isolation, matching the file's existing convention.
  /// `.notApplicable` (flag absent/false, no video rendering, nested
  /// preemption) takes the EXACT SAME `coordinator.enqueueGenerate` path
  /// `submit` above does.
  fileprivate func submitPreempting(
    _ payload: GeneratePayload, source: String, coordinator: WarmServerCoordinator, rawBody: Data? = nil,
    preemptor: @escaping @Sendable (String) async -> WarmServer.PreemptionOutcome
  ) -> ImageJobStatus {
    submitPreempting(payload, source: source, rawBody: rawBody, preemptor: preemptor) { jobId in
      try await coordinator.enqueueGenerate(payload, source: source, rawBody: rawBody, jobId: jobId)
    }
  }

  /// Closure-seam twin of `submit(_:source:rawBody:enqueue:)` for the
  /// preempting path; both the preemptor and the enqueue receive the
  /// client-visible id (AC-18).
  func submitPreempting(
    _ payload: GeneratePayload, source: String, rawBody: Data?,
    preemptor: @escaping @Sendable (String) async -> WarmServer.PreemptionOutcome,
    enqueue: @escaping @Sendable (String) async throws -> GenerateResponse
  ) -> ImageJobStatus {
    let jobId = UUID().uuidString
    let job = ImageJob(id: jobId, source: source)
    // #286: the preset flags are known at SUBMIT, not only on success — an
    // async caller must see `preset_unresolved` on the 202 rather than having
    // to poll for it, and a job that later fails must still report it.
    job.presetUnresolved = payload.presetUnresolved
    job.presetUnresolvedReason = payload.presetUnresolvedReason
    job.presetStackMismatch = payload.presetStackMismatch
    // #22: same "known at submit" posture as the preset flags above.
    job.memoryEstimateBytes = payload.memoryEstimateBytes
    job.memoryAvailableBytes = payload.memoryAvailableBytes
    lock.lock(); jobs[jobId] = job; lock.unlock()

    Task { [weak self] in
      guard let self else { return }
      switch await preemptor(jobId) {
      case .ran(let result):
        self.markSucceeded(jobId, result: result)
        return
      case .ranFailed(let error):
        self.markFailed(jobId, error: error)
        return
      case .refused(let eta):
        self.markPreemptRefused(jobId, eta: eta)
      case .notApplicable:
        break
      }
      self.markProcessing(jobId)
      do {
        let result = try await enqueue(jobId)
        self.markSucceeded(jobId, result: result)
      } catch {
        self.markFailed(jobId, error: error)
      }
    }
    return job.toStatus()
  }

  func status(jobId: String) -> ImageJobStatus? {
    lock.lock(); defer { lock.unlock() }
    return jobs[jobId]?.toStatus()
  }

  /// WP-E4 (D22, AC-18): register a persisted-queue job that failed replay
  /// validation as FAILED under its original id, with the reason, so a client
  /// polling `GET /v1/generate/status/{id}` across a restart sees why it never
  /// rendered instead of a 404.
  func recordFailedReplay(jobId: String, source: String, error: Error) {
    let job = ImageJob(id: jobId, source: source)
    job.state = .failed
    job.error = error.localizedDescription
    job.completedAt = Date()
    lock.lock(); jobs[jobId] = job; lock.unlock()
  }

  private func markProcessing(_ jobId: String) {
    lock.lock(); jobs[jobId]?.state = .processing; lock.unlock()
  }

  /// #1479: the refusal guard declined — this job still queues normally
  /// right after (see `submitPreempting`), just annotated.
  private func markPreemptRefused(_ jobId: String, eta: Double) {
    lock.lock()
    jobs[jobId]?.preemptRefused = true
    jobs[jobId]?.etaSec = eta
    lock.unlock()
  }

  private func markSucceeded(_ jobId: String, result: GenerateResponse) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .succeeded
      job.outputPath = result.outputPath
      job.durationMs = result.durationMs
      job.applied = result.applied
      job.appliedLoras = result.appliedLoras
      job.presetUnresolved = result.presetUnresolved ?? job.presetUnresolved
      job.presetUnresolvedReason = result.presetUnresolvedReason ?? job.presetUnresolvedReason
      job.presetStackMismatch = result.presetStackMismatch ?? job.presetStackMismatch
      job.loraStackOrigin = result.loraStackOrigin ?? job.loraStackOrigin
      job.warmDefaultSkipped = result.warmDefaultSkipped ?? job.warmDefaultSkipped
      job.loraReload = result.loraReload ?? job.loraReload
      job.appliedShift = result.appliedShift ?? job.appliedShift
      job.completedAt = Date()
    }
    lock.unlock()
  }

  private func markFailed(_ jobId: String, error: Error) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .failed
      job.error = error.localizedDescription
      job.completedAt = Date()
    }
    lock.unlock()
  }

  /// Drop completed/failed jobs older than `ttl` so this doesn't grow
  /// unboundedly on a long-running server. Mirrors `ReplicateVideoProxy`'s
  /// prune convention.
  func pruneCompleted(olderThan ttl: TimeInterval = 3600) {
    lock.lock(); defer { lock.unlock() }
    let cutoff = Date().addingTimeInterval(-ttl)
    jobs = jobs.filter { _, job in
      guard let completedAt = job.completedAt else { return true }
      return completedAt > cutoff
    }
  }
}

/// Internal mutable state for a tracked async LOCAL LTX-2 video job. Mirrors
/// `ImageJob`, with the extra video fields (`mode`, `frameCount`,
/// `videoDurationSeconds`, live `progressPercent`) the wire `VideoJobStatus`
/// carries.
private final class LocalVideoJob: @unchecked Sendable {
  let id: String
  let source: String
  let mode: VideoMode
  let startTime = Date()
  var state: VideoJobState = .queued
  var outputPath: String?
  var frameCount: Int?
  var videoDurationSeconds: Int?
  var durationMs: Int?
  var error: String?
  var progressPercent: Int?
  var completedAt: Date?
  /// comfybox#322: this render was stopped by `/v1/queue/interrupt`, not by a
  /// failure. Surfaced as the additive `interrupted` field on the status JSON
  /// (see `VideoJobStatus.interrupted` for why `status` itself stays `failed`).
  var interrupted = false
  /// Authoritative config snapshot, set at submit (finding #15).
  var resolvedConfig: [LTX2ResolvedParam]?
  /// comfybox#307: set on success when `two_stage` was requested and the
  /// refine could not run — see `LTX2RefineGate`.
  var refineSkippedReason: String?
  /// comfybox#401: set on success — see `VideoJobStatus.generationRecord`.
  var generationRecord: VideoGenerationRecord?

  init(id: String, source: String, mode: VideoMode) {
    self.id = id
    self.source = source
    self.mode = mode
  }

  var elapsedMs: Int {
    let end = completedAt ?? Date()
    return Int(end.timeIntervalSince(startTime) * 1000)
  }

  func toStatus() -> VideoJobStatus {
    VideoJobStatus(
      jobId: id,
      status: state,
      mode: mode,
      backend: "ltx2-local",
      outputPath: outputPath,
      durationMs: durationMs,
      videoDurationSeconds: videoDurationSeconds,
      error: error,
      elapsedMs: elapsedMs,
      progressPercent: progressPercent,
      resolvedConfig: resolvedConfig,
      frameCount: frameCount,
      interrupted: interrupted ? true : nil,
      refineSkipped: refineSkippedReason,
      generationRecord: generationRecord
    )
  }
}

/// Submit-and-poll tracker for LOCAL LTX-2 video renders — the video twin of
/// ``ImageJobTracker``. A local render can run for minutes across multiple
/// chunks; holding the HTTP connection open for the whole thing is what the
/// async `POST /v1/video/generate/async` + `GET /v1/video/status/{id}` pair
/// avoids. Submit returns a job id immediately; the render itself runs on the
/// coordinator's serial GPU queue (so it never shares the GPU with an image
/// render), streaming progress into the job as it goes.
///
/// The state-transition surface (`register`/`markProcessing`/`markSucceeded`/
/// `markFailed`/`setProgress`/`status`/`pruneCompleted`) is deliberately kept
/// free of any coordinator dependency so the state machine is unit-testable in
/// isolation; `submit` is the thin production wrapper that drives it against the
/// real render queue.
final class VideoJobTracker: @unchecked Sendable {
  /// Task #19: append-only lifecycle trace (submitted/started/terminal).
  /// nil in unit tests that only exercise the state machine.
  var traceStore: RenderTraceStore?
  private let lock = NSLock()
  private var jobs: [String: LocalVideoJob] = [:]

  /// Queued/processing/paused jobs currently owned by this local tracker.
  /// Terminal jobs carry `completedAt` and remain queryable for the TTL, but
  /// must not inflate `/health.video.active_jobs` during that hour.
  var activeJobCount: Int {
    lock.lock(); defer { lock.unlock() }
    return jobs.values.filter { $0.completedAt == nil }.count
  }

  /// Create a tracked job in `.queued` and return (jobId, its status). Testable
  /// without a coordinator.
  @discardableResult
  func register(
    source: String, mode: VideoMode,
    resolvedConfig: [LTX2ResolvedParam]? = nil,
    tracePayload: [String: String] = [:]
  ) -> (jobId: String, status: VideoJobStatus) {
    let jobId = UUID().uuidString
    let job = LocalVideoJob(id: jobId, source: source, mode: mode)
    job.resolvedConfig = resolvedConfig
    lock.lock(); jobs[jobId] = job; lock.unlock()
    var payload = tracePayload
    payload["source"] = source
    payload["mode"] = mode.rawValue
    if let rc = resolvedConfig {
      payload["config"] = rc.map { "\($0.name)=\($0.value)(\($0.source.rawValue))" }.joined(separator: " ")
    }
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .submitted, taskKind: .videoRender, payload: payload))
    return (jobId, job.toStatus())
  }

  /// Submit a local render. Returns immediately with a `.queued` status; the
  /// render runs in a detached Task against the coordinator's FIFO GPU queue.
  /// `render` receives a `report(percent)` callback to stream progress; the
  /// tracker fans that out to both this job's status and (via the coordinator's
  /// own report wired in `enqueueLocalVideo`) the /health + /queue trackers.
  fileprivate func submit(
    source: String,
    mode: VideoMode,
    coordinator: WarmServerCoordinator,
    resolvedConfig: [LTX2ResolvedParam]? = nil,
    tracePayload: [String: String] = [:],
    wantsAudio: Bool = false,
    render: @escaping @Sendable (@escaping @Sendable (Int) -> Void) throws -> LTX2RenderOutcome
  ) -> VideoJobStatus {
    let (jobId, queued) = register(
      source: source, mode: mode, resolvedConfig: resolvedConfig,
      tracePayload: tracePayload)
    Task { [weak self] in
      guard let self else { return }
      self.markProcessing(jobId)
      do {
        // #1479: pass this job's tracker id so the coordinator can mark it
        // paused-for-preemption / resumed by id (see `enqueueLocalVideo`).
        let result = try await coordinator.enqueueLocalVideo(wantsAudio: wantsAudio, videoJobId: jobId) { coordReport in
          try render { pct in
            // Fan progress to both the coordinator's health/queue trackers and
            // this job's own status.
            coordReport(pct)
            self.setProgress(jobId, pct)
          }
        }
        self.markSucceeded(jobId, result: result)
      } catch {
        self.markFailed(jobId, error: error)
      }
    }
    return queued
  }

  /// Submit a multi-step orchestration (storyboard, #237). Unlike `submit`,
  /// the work closure is NOT wrapped in a single `enqueueLocalVideo` — it
  /// issues its OWN coordinator enqueues (one per shot render / i2i insert),
  /// so each step takes a normal turn on the FIFO GPU queue and other jobs
  /// can interleave between shots. Wrapping the whole storyboard in one queue
  /// entry would deadlock: the closure would enqueue from inside the queue.
  fileprivate func submitOrchestrated(
    source: String,
    mode: VideoMode,
    work: @escaping @Sendable (@escaping @Sendable (Int) -> Void) async throws -> LTX2VideoResult
  ) -> VideoJobStatus {
    let (jobId, queued) = register(source: source, mode: mode)
    Task { [weak self] in
      guard let self else { return }
      self.markProcessing(jobId)
      do {
        let result = try await work { pct in
          self.setProgress(jobId, pct)
        }
        self.markSucceeded(jobId, result: result)
      } catch {
        self.markFailed(jobId, error: error)
      }
    }
    return queued
  }

  func status(jobId: String) -> VideoJobStatus? {
    lock.lock(); defer { lock.unlock() }
    return jobs[jobId]?.toStatus()
  }

  func markProcessing(_ jobId: String) {
    lock.lock(); jobs[jobId]?.state = .processing; lock.unlock()
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .started, taskKind: .videoRender, payload: [:]))
  }

  /// #1479: checkpointed for an in-flight preemption — not terminal, no trace
  /// event (the trace store's `.terminal` kind means "done"; this isn't).
  func markPausedForPreemption(_ jobId: String) {
    lock.lock(); jobs[jobId]?.state = .pausedForPreemption; lock.unlock()
  }

  /// #1479: the preempting image job finished (success or failure) and the
  /// video render resumed. Reuses `.processing` — this job was never
  /// "finished" from the tracker's point of view, so there is no distinct
  /// "resumed" state to model.
  func markResumedFromPreemption(_ jobId: String) {
    lock.lock(); jobs[jobId]?.state = .processing; lock.unlock()
  }

  func setProgress(_ jobId: String, _ percent: Int) {
    lock.lock(); jobs[jobId]?.progressPercent = min(100, max(0, percent)); lock.unlock()
  }

  func markSucceeded(_ jobId: String, result: LTX2VideoResult) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .succeeded
      job.outputPath = result.outputPath
      job.frameCount = result.frameCount
      job.videoDurationSeconds = Int(result.durationSeconds.rounded())
      job.durationMs = Int(result.elapsedSeconds * 1000)
      job.progressPercent = 100
      job.completedAt = Date()
      job.refineSkippedReason = result.refineSkippedReason
      job.generationRecord = result.generationRecord
    }
    lock.unlock()
    var payload = [
      "status": "succeeded",
      "output_path": result.outputPath,
      "frames": String(result.frameCount),
      "elapsed_ms": String(Int(result.elapsedSeconds * 1000)),
    ]
    // comfybox#307: durable on the trace even after the job itself prunes —
    // `RenderTraceStore` outlives `pruneCompleted`'s 1h job TTL.
    if let reason = result.refineSkippedReason {
      payload["refine_skipped"] = reason
    }
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .terminal, taskKind: .videoRender, payload: payload))
  }

  func markFailed(_ jobId: String, error: Error) {
    // comfybox#322: an operator interrupt is not a failure. Both spellings of
    // it (a raw `CancellationError` out of a pipeline loop, and the named
    // `WarmServerError.renderInterrupted` the video queue case substitutes)
    // land here, because `submit`'s only terminal-error path is this method.
    if isRenderInterruption(error) {
      markInterrupted(jobId)
      return
    }
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .failed
      job.error = error.localizedDescription
      job.completedAt = Date()
    }
    lock.unlock()
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .terminal, taskKind: .videoRender,
      payload: ["status": "failed", "error": error.localizedDescription]))
  }

  /// comfybox#322: terminal, but not a failure — `/v1/queue/interrupt` stopped
  /// this render mid-flight.
  ///
  /// `state` stays `.failed` so every existing polling client still sees a
  /// terminal status it knows (see `VideoJobStatus.interrupted`); the additive
  /// `interrupted` flag and the trace's `status: interrupted` carry the real
  /// outcome. The trace has no wire-compat constraint, so it says the truth.
  func markInterrupted(_ jobId: String) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .failed
      job.interrupted = true
      job.error = "Render interrupted by /v1/queue/interrupt"
      job.completedAt = Date()
    }
    lock.unlock()
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .terminal, taskKind: .videoRender,
      payload: ["status": "interrupted", "interrupted": "true"]))
  }

  /// Drop completed/failed jobs older than `ttl`. Mirrors `ImageJobTracker`.
  func pruneCompleted(olderThan ttl: TimeInterval = 3600) {
    lock.lock(); defer { lock.unlock() }
    let cutoff = Date().addingTimeInterval(-ttl)
    jobs = jobs.filter { _, job in
      guard let completedAt = job.completedAt else { return true }
      return completedAt > cutoff
    }
  }
}

private actor WarmServerCoordinator {
  enum ServerError: Error {
    case queueFull(maxPending: Int)
    /// The model-operation cap (`maxPendingModelOps`), which is counted and
    /// reported separately from the render queue so the message names the
    /// limit the caller actually hit (WP-E8 review, finding 1).
    case modelOperationQueueFull(maxPending: Int)
    case shuttingDown
    /// The pending request was removed by a queue clear (not a server shutdown).
    case cancelled
  }

  /// #300: this actor's isolated work (including the synchronous render call)
  /// otherwise runs on the Swift cooperative thread pool
  /// (`com.apple.root.utility-qos.cooperative`), which is width-capped at
  /// ~core count and does NOT grow when a worker blocks. A `sample` of the
  /// live process during a render showed 2964/2972 samples parked in
  /// `__psynch_cvwait` on that pool — starving every other actor hop,
  /// including `Task { await respond(...) }` for async HTTP routes (HTTP 000
  /// at 120s while sync routes stayed fine). Giving the coordinator its own
  /// serial executor (SE-0392) moves its work off the shared cooperative pool
  /// entirely; actor serialization semantics are unchanged.
  private let executorQueue = DispatchSerialQueue(label: "z-image.warm-server.coordinator", qos: .userInitiated)
  nonisolated var unownedExecutor: UnownedSerialExecutor {
    executorQueue.asUnownedSerialExecutor()
  }

  private let configuration: WarmServerConfiguration
  private let logger: Logger
  private var pipeline: ZImagePipeline
  /// Flux 2 pipeline — created when the model is detected as Flux 2 Klein.
  private var flux2Pipeline: Flux2Pipeline?
  /// FIBO pipeline — created when the model is detected as FIBO.
  private var fiboPipeline: FiboPipeline?
  /// Chroma pipeline — created when the model is detected as Chroma.
  private var chromaPipeline: ChromaPipeline?

  /// Krea-2 pipeline (native port), loaded when the model spec is Krea-2.
  private var krea2Pipeline: Krea2Pipeline?
  /// The physical Krea-2 variant the resident pipeline loaded (WP-E5, D7) —
  /// beside `zimageVariant`. nil when no Krea-2 model is resident.
  private var krea2Variant: Krea2Variant?
  /// Trigger lookups for the rewriter-proof guard (set by WarmServer.run()).
  var loraLibrary: LoRALibrary?
  func setLoraLibrary(_ library: LoRALibrary) { loraLibrary = library }
  /// Chroma tokenizer — loaded alongside the Chroma pipeline.
  private var chromaTokenizer: ChromaTokenizer?
  /// Which model family is loaded — determines generation routing.
  private var currentModelFamily: WarmModelFamily = .flux1
  /// Detected Flux 2 model info (variant, configs) — nil when running Flux 1.
  private var detectedFlux2Model: Flux2DetectedModel?
  /// Detected FIBO model info — nil when running Flux 1/2.
  private var detectedFiboModel: FiboDetectedModel?
  /// Detected Z-Image variant (Base vs Turbo) — only set when running Flux 1 (Z-Image).
  private var zimageVariant: ZImageVariant = .turbo
  /// Lazy-initialized ControlNet pipeline — only created when first ControlNet request arrives.
  private var controlPipeline: ZImageControlPipeline?
  private let startTime = Date()
  /// What the ACTIVE pipeline currently holds — the engine's belief about
  /// residency, reconciled from the pipeline on pool activation and published
  /// as `/health.loras`. Since #282 this is a CONSEQUENCE of the last job's
  /// resolved stack, never an input to the next one's.
  private var activeLoRAs: [LoRAConfiguration]
  /// #282 — the WARM DEFAULT stack: the stack a request that named neither
  /// `preset` nor `loras` renders with.
  ///
  /// This is the whole of what `POST /v1/lora/swap` now does. The route, its
  /// payload and its response JSON are unchanged (the daemon contract is
  /// production), but a swap no longer publishes a stack that later jobs
  /// silently inherit — it publishes a DEFAULT, and a job picks it up only by
  /// asking for nothing. Seeded from the engine's launch-time `--lora`
  /// arguments so a bare render behaves identically to before on a boot that
  /// declared them.
  ///
  /// Review r1 (I3): the stack stored here is the one that was actually
  /// APPLIED — `applyActiveLoRAs` folds Krea-2 relativity into it (request >
  /// library > seed) and the pre-fold configs would make `warm_default_stack`
  /// disagree with `/health.loras` about the same adapters.
  private var warmDefaultStack: [LoRAConfiguration]
  /// #282 review r1 (C1) — the base the warm default was published under. A
  /// bare request that dequeues onto a different checkpoint does NOT take it;
  /// see ``RequestStackResolver/admitWarmDefault(isEmpty:tag:requestFamily:requestModelSpec:)``.
  private var warmDefaultTag: RequestStackResolver.WarmDefaultTag = .untagged
  /// A queued operation tagged with identity + arrival time so the queue can
  /// be listed and individual pending jobs cancelled.
  private struct PendingJob {
    /// The job's id everywhere it is named: /v1/queue, the persisted
    /// snapshot, a failed replay. Supplied by the caller when a client already
    /// holds one (AC-18); defaults to a fresh UUID.
    var id: String = UUID().uuidString
    let enqueuedAt = Date()
    /// Which client/app submitted this job (desktop, comfyui/krita, bree, api…).
    var source: String = "api"
    let operation: QueuedOperation
    /// The original raw JSON request body, kept only for kinds that can be
    /// replayed after a crash (see QueuePersistence.swift). nil for kinds
    /// that can't be recovered (modelSwitch/localVideo close over live
    /// in-memory state; controlGenerate closes over resolved temp files;
    /// shutdown never needs recovery).
    var rawBody: Data? = nil
  }

  private var pending: [PendingJob] = []
  /// Human-readable summary of the operation the loop is currently running.
  private var activeJobSummary: String?
  /// Source/app of the currently-running job.
  private var activeJobSource: String?
  private var isProcessing = false
  // 0.B-2 (FDD §3.1.5): `isPaused` is no longer owned here. It is AUTHORITATIVE
  // in the lock store (`LiveHealthState`), persisted by the same sentinel file
  // (2026-08-10 rationale unchanged: a "paused" that un-pauses itself across a
  // watchdog kickstart / crash / deploy is how the mystery-GPU-usage incidents
  // happen). This actor is a READER of it — see the `processLoop` gate and
  // `setPaused` below.

  /// Sentinel marking the queue paused; survives engine restarts. Computed
  /// from `QueueStateStore.stateDirectory` so it follows `COMFYBOX_STATE_DIR`
  /// (K-FIX-1: a test driving a real coordinator must not read — or clear —
  /// the LIVE engine's pause flag).
  static var pauseSentinelPath: String {
    QueueStateStore.stateDirectory.appendingPathComponent("queue-paused").path
  }
  private var shuttingDown = false
  private var successfulRenderCount = 0
  private var failedRenderCount = 0
  private var lastRenderDurationMs: Int?
  private var lastError: String?
  /// WP-E10 sink 3: the record of the last successful Krea 2 render,
  /// published into /health as `last_recipe`. Set only from a completed
  /// render (a failed one writes no record), never from a request.
  private var lastRecipe: AppliedRecordSlot?

  /// comfybox#308: apply one render's terminal outcome to the counters
  /// `/health` and `/v1/queue` publish. The six image `run*Generate` methods
  /// still hand-increment inline (unchanged, working, and out of scope here);
  /// this is for paths — currently just `.localVideo` — that had no
  /// equivalent at all.
  private func recordRenderCompletion(_ event: RenderCompletionEvent) {
    var counters = RenderHealthCounters(
      successCount: successfulRenderCount, failedCount: failedRenderCount,
      lastDurationMs: lastRenderDurationMs)
    counters.apply(event)
    successfulRenderCount = counters.successCount
    failedRenderCount = counters.failedCount
    lastRenderDurationMs = counters.lastDurationMs
  }

  /// comfybox#308 (review r2, item 2b): the ONE place all `.localVideo`
  /// completion bookkeeping goes through — consolidates what were three
  /// independent `recordRenderCompletion(...); lastError = ...` pairs (one
  /// per exit point: success, thrown error, memory-admission refusal) so
  /// there is exactly one function a `#if DEBUG` seam
  /// (`WarmServerQueueProbe.finishLocalVideoTestSeam`) can drive to prove
  /// the counters move for all three outcomes.
  private func finishLocalVideo(_ outcome: LocalVideoCompletionOutcome, lastError message: String?) {
    recordRenderCompletion(.forLocalVideoCompletion(outcome))
    lastError = message
  }

  /// comfybox#308/#322 (review r3): the `.localVideo` case's generic
  /// `catch` — shared by the production catch block and the `#if DEBUG`
  /// seam (`testSeamHandleLocalVideoCatch`) so both run the exact same
  /// classify-then-finish decision. `localVideoCatchOutcome(for:)` returns
  /// nil for an operator interrupt (including a WRAPPED cancellation) —
  /// `finishLocalVideo` is not called at all in that case, matching the
  /// sibling `catch is CancellationError` branch, which never touches the
  /// counters either.
  private func handleLocalVideoCatch(_ error: Error) {
    if let outcome = localVideoCatchOutcome(for: error) {
      finishLocalVideo(outcome, lastError: error.localizedDescription)
    }
  }

  /// Re-decide whether `lastRecipe` may still be published, from what is
  /// resident RIGHT NOW (WP-E10 sink 3).
  ///
  /// `/health` prints `model`, `loaded`, `model_variant` and `last_recipe`
  /// side by side. A record that outlived its checkpoint reads as provenance
  /// and is not — most starkly during an LTX-2 render, where the whole image
  /// stack is evicted (#218) and `/health` would otherwise show a full Krea 2
  /// provenance block beside `loaded: false` for tens of minutes.
  ///
  /// EVERY writer of `krea2Pipeline` calls this immediately after: pool
  /// activation, `prepare()`, and `releaseImageModelsForVideo()`. Adding a
  /// fourth writer without this call is the bug this comment exists to catch.
  /// A record that no longer describes what is resident is dropped ENTIRELY
  /// (the key goes absent), not turned into a `null` — `null` means "a Krea 2
  /// render just refused its record", which is a different statement.
  private func revalidateLastRecipe() {
    guard let slot = lastRecipe else { return }
    guard let record = slot.record else {
      // A refused record describes a render, not a checkpoint; once the
      // resident model may have changed there is nothing left to say.
      lastRecipe = currentModelFamily == .krea2 ? slot : nil
      return
    }
    lastRecipe = RenderRecipe.retained(
      record,
      family: currentModelFamily,
      krea2TransformerFile: krea2Pipeline?.paths.transformerFile.path
    ).map(AppliedRecordSlot.init(record:))
  }
  private var activeRenderStartedAt: Date?
  /// Synthetic id for the currently-rendering job — surfaced as `current_job_id`.
  private var activeJobId: String?
  /// Handle for the in-flight render — retained so /interrupt can cancel it.
  /// The pipelines observe cancellation via Task.checkCancellation() in their
  /// denoise loops; the render's continuation then resumes with CancellationError.
  ///
  /// comfybox#362: during a preemption episode this holds the PREEMPTING IMAGE
  /// job's own task and identity — see `runAsPublishedActiveRender` for the
  /// swap, and `LiveHealthState.checkpointedVideo` for the video's own handle
  /// during that span.
  ///
  /// Written ONLY through `publishActiveRender`, which mirrors it into
  /// `liveHealth` in the same breath. This field is the actor's bookkeeping
  /// copy (what a preemption episode captures to restore afterwards); the
  /// AUTHORITY both interrupt paths read is the `liveHealth` publication.
  private var activeRender = PublishedRender.none
  /// Live progress (0-100) of the active render; nil when idle. Updated from the
  /// pipeline denoising callback, read by `queueStatus()`.
  private let progressTracker = RenderProgressTracker()
  private let previewTracker = RenderPreviewTracker()
  private var pipelinePrepared = false
  /// When a pool model is activated, this holds its modelSpec so that
  /// generation requests use the pool model instead of the startup
  /// configuration.modelSpec. Reset to nil when the startup model is
  /// re-activated or the pool model is unloaded.
  private var activePoolModelSpec: String?

  /// Model hot-swap pool — holds loaded pipelines with LRU eviction.
  let modelPool: ModelPool

  /// Shared, lock-based owner of the LTX-2 video generator (#218). The video
  /// stack lives outside the pool; this lets the coordinator evict it before an
  /// image load, keeping a single heavy model resident across image + video.
  private let videoHolder: VideoGeneratorHolder

  /// Pure single-heavy-model residency accounting (#218).
  private let heavyAdmission = HeavyModelAdmission()

  /// Lock-based health snapshot the /health route reads without an actor hop (#217).
  private let liveHealth: LiveHealthState
  /// comfybox#283/#217: shared with `WarmServer` — see its declaration for
  /// what this records and why. Read-only from this actor's point of view:
  /// every call site here OBSERVES a transition, never gates or reorders one.
  private let lifecycleLedger: QueueLifecycleLedger
  /// When the current queue job started processing — the /health start time even
  /// before a render method sets `activeRenderStartedAt` past its first await.
  private var currentJobStartedAt: Date?
  /// Raw request body + kind of the currently-active job, kept only long
  /// enough to persist it as the "active" slot in QueueStateStore — cleared
  /// alongside the other activeJob* fields once the job finishes (see
  /// QueuePersistence.swift for why only these two kinds are recoverable).
  private var activeJobRawBody: Data?
  private var activeJobKindForPersistence: String?
  /// #339 review r1: the not-yet-admitted remainder of a persisted-queue
  /// replay in progress (see `RecoverySnapshotMerger`). `persistQueueState()`
  /// merges this into every snapshot it writes so a SECOND restart mid-replay
  /// does not lose it — the "deeper bug" review r1 found: the old sequential
  /// one-job-at-a-time replay left jobs 2..N invisible to `queue-state.json`
  /// until each was individually re-admitted. Set by
  /// `WarmServer.recoverPersistedQueue` before each job's admission; cleared
  /// to `[]` once the whole batch is admitted (`defer`, so it always clears).
  private var recoveryUnadmittedTail: [PersistedQueueJob] = []

  /// True after the image models were released to make room for LTX-2 video —
  /// the next image render must reload before it can run (#218).
  private var imageModelsEvicted = false
  /// The image model that was active when it was evicted for video, so the next
  /// image render can restore exactly that model.
  private var lastActiveImageSpec: String?

  // MARK: - #1479 preemption

  /// Async-video job tracker, so the coordinator can mark the render
  /// paused-for-preemption / resumed by job id (nil for the synchronous
  /// `/v1/video/generate` path, which has no tracker entry).
  private let videoJobTracker: VideoJobTracker
  /// Installed on every LTX-2 generator instance (`prepareLocalVideo` wires
  /// the initial one; re-wired here on a fresh post-eviction instance).
  private let ltx2Telemetry: LTX2PhaseTelemetry
  private let ltx2PreemptionSignal: PreemptionSignal
  private let ltx2StepPosition: LTX2StepPosition
  private let ltx2EvictMean: RollingMeanSec
  private let ltx2ReloadMean: RollingMeanSec
  /// Cleared unconditionally once a preemption episode is fully resolved
  /// (resumed or handed to the checkpoint-fallback watchdog) — see
  /// `runPreemptionEpisode`'s defer.
  private let preemptionInFlight: LockedFlag
  private let pendingPreemptorBox: PendingPreemptorBox
  /// The one checkpoint this coordinator may hold at a time (spec: hold
  /// exactly one checkpoint; nested preemption refused separately via
  /// `preemptionInFlight`). Non-nil only for the span between the video
  /// render yielding and the video being resumed.
  private var checkpointedVideo: LTX2ResumeState?

  init(
    configuration: WarmServerConfiguration, logger: Logger, videoHolder: VideoGeneratorHolder, liveHealth: LiveHealthState,
    videoJobTracker: VideoJobTracker, ltx2Telemetry: LTX2PhaseTelemetry, ltx2PreemptionSignal: PreemptionSignal,
    ltx2StepPosition: LTX2StepPosition, ltx2EvictMean: RollingMeanSec, ltx2ReloadMean: RollingMeanSec,
    preemptionInFlight: LockedFlag, pendingPreemptorBox: PendingPreemptorBox,
    lifecycleLedger: QueueLifecycleLedger = QueueLifecycleLedger()
  ) {
    self.configuration = configuration
    self.logger = logger
    self.videoHolder = videoHolder
    self.liveHealth = liveHealth
    self.videoJobTracker = videoJobTracker
    self.ltx2Telemetry = ltx2Telemetry
    self.ltx2PreemptionSignal = ltx2PreemptionSignal
    self.ltx2StepPosition = ltx2StepPosition
    self.ltx2EvictMean = ltx2EvictMean
    self.ltx2ReloadMean = ltx2ReloadMean
    self.preemptionInFlight = preemptionInFlight
    self.pendingPreemptorBox = pendingPreemptorBox
    self.lifecycleLedger = lifecycleLedger
    self.pipeline = ZImagePipeline(logger: logger, retentionPolicy: .keepLoaded)
    self.activeLoRAs = configuration.initialLoRAs
    self.warmDefaultStack = configuration.initialLoRAs
    self.modelPool = ModelPool(
      textEncoderPath: configuration.textEncoderPath,
      maxSequenceLength: configuration.maxSequenceLength,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly,
      logger: logger
    )
  }

  // MARK: - Single-heavy-model residency (#218)

  /// Release EVERY resident image model — the pool (including the active model)
  /// and the coordinator's own per-family pipelines — to vacate unified memory
  /// for the ~65GB LTX-2 video stack. Records what was active so the next image
  /// render can restore it. Returns the estimated MB freed.
  @discardableResult
  func releaseImageModelsForVideo() async -> Int {
    lastActiveImageSpec = activePoolModelSpec ?? configuration.modelSpec
    let freedMB = await modelPool.releaseAll()
    pipeline.unloadModel()
    flux2Pipeline = nil
    fiboPipeline = nil
    chromaPipeline = nil
    krea2Pipeline = nil
    revalidateLastRecipe()  // WP-E10: no checkpoint, no record (#218)
    chromaTokenizer = nil
    controlPipeline = nil
    pipelinePrepared = false
    activePoolModelSpec = nil
    imageModelsEvicted = true
    GPU.clearCache()
    logger.info("Released image models for LTX-2 video (~\(freedMB)MB pool est; base pipeline + per-family pipelines unloaded) (#218)")
    publishHealth()
    return freedMB
  }

  /// If image models were evicted for a video render, reload the previously
  /// active image model (or the one this request explicitly asks for) before
  /// rendering. Throws if the reload fails. No-op when nothing was evicted.
  private func reloadImageModelIfEvicted(requestedModel: String?) async throws {
    guard imageModelsEvicted else { return }
    let spec = requestedModel.map { WarmServer.parseModelSpec(from: $0) }
      ?? lastActiveImageSpec
      ?? configuration.modelSpec
    guard let reloadSpec = spec else { imageModelsEvicted = false; return }
    let quant = requestedModel.flatMap { WarmServer.parseQuantization(from: $0) }
    logger.info("Reloading image model '\(reloadSpec)' after video eviction (#218)")
    _ = try await poolLoad(modelSpec: reloadSpec, quantization: quant, activate: true)
    imageModelsEvicted = false
  }

  /// Shed the least-recently-used *inactive* pool model under memory pressure.
  /// Never touches the active model or an in-flight render. Returns MB freed.
  @discardableResult
  func shedInactivePoolModelUnderPressure() async -> Int {
    await modelPool.releaseLRUInactive()
  }

  // MARK: - #1479 preemption orchestration
  //
  // Entered from the `.localVideo` case in `processLoop` when
  // `body(report)` (or a prior `resume(from:)`) returns `.yielded` — i.e.
  // `ltx2PreemptionSignal` was raised by an image job's route handler
  // (`WarmServer.attemptPreemption`, which runs BEFORE any actor call — see
  // the block comment above `RollingMeanSec`) and the render loop observed
  // it at a step boundary. Everything from here on runs on the actor, in the
  // SAME Task that was blocked inside the video's synchronous render, so no
  // further actor hop is needed to reach `pipeline`/`runGenerate`/etc.

  /// #1479/#230/#218/#34: the LTX-2 stack's memory need assuming it stays
  /// WARM (resident) — the warm-stack-discount / audio-mode-mismatch logic
  /// `vacateImageModelsAndAdmitVideo`'s admission gate uses. (Until the final
  /// review this was also consulted by a no-eviction "fast path" decision in
  /// `runPreemptionEpisode`; that path is gone — see `runPreemptionEpisode`'s
  /// doc comment — so this now has exactly one caller.)
  private func ltx2WarmNeedBytes(wantsAudio: Bool) -> UInt64 {
    let gen = videoHolder.get()
    let audioModeMatches = (gen?.isAudioLoaded ?? false) == wantsAudio
    let videoStackWarm = gen?.isLoaded == true && audioModeMatches
    let audioDelta: UInt64 = wantsAudio ? 13 * 1024 * 1024 * 1024 : 0
    return videoStackWarm
      ? 24 * 1024 * 1024 * 1024
      : HeavyModelAdmission.ltx2EstimateBytes(forWeightsPath: configuration.ltx2WeightsPath) + audioDelta
  }

  /// Rebuild the LTX-2 generator after an eviction. After `videoHolder
  /// .release()` the previous instance (and anything it had wired) is gone —
  /// this is a cold start by the exact same resolution `prepareLocalVideo`
  /// uses, publishing the fresh instance into `videoHolder` so the rest of
  /// the codebase (admission, `/health`, later renders) sees it immediately.
  /// Only resolves paths and constructs the object — the actual multi-GB
  /// weight load is lazy, inside `resume(from:)` -> `render` -> `load(...)`
  /// (review I1: do not time THIS call as "the reload").
  private func reloadVideoGeneratorAfterEviction() async throws -> LTX2VideoGenerator {
    guard let weights = configuration.ltx2WeightsPath, let gemma = configuration.ltx2GemmaPath else {
      throw WarmServerError.invalidRequest(
        message: "#1479: LTX-2 not configured — cannot reload after a preemption eviction")
    }
    let weightsURL = try await ModelResolution.resolve(
      modelSpec: weights,
      filePatterns: ["transformer-distilled.safetensors", "connector.safetensors",
                      "vae_decoder.safetensors", "vae_encoder.safetensors", "config.json"]
    )
    let gemmaURL = try await ModelResolution.resolve(
      modelSpec: gemma,
      filePatterns: ["*.safetensors", "*.json", "tokenizer/*", "*.model"]
    )
    let gen = LTX2VideoGenerator(
      config: .init(weightsDir: weightsURL.path, gemmaPath: gemmaURL.path), logger: logger)
    videoHolder.set(gen)
    return gen
  }

  /// #1479 (review C2): the SAME admission gate a cold video start runs
  /// (`.localVideo`'s admission block, now just a thin wrapper around this),
  /// extracted so a preemption resume can run it too. The preempting image
  /// job loaded ITS OWN weights while the video was away; resuming into
  /// whatever memory is left over without re-vacating image models and
  /// re-checking the drain/admission gate is the documented #218/#34 SIGKILL
  /// condition — "a failed tap must never cost a video" includes never
  /// costing it an OOM kill.
  private func vacateImageModelsAndAdmitVideo(
    wantsAudio: Bool
  ) async -> (admitted: Bool, availableMB: Int, neededMB: Int) {
    let freedForVideoMB = await releaseImageModelsForVideo()
    var availableForVideo = MemoryProbe.systemAvailableMemoryBytes()
    // Precision-keyed (#230), warm-stack discount, audio-mode-mismatch cold
    // fallback (task #21, Codex #2) — factored into `ltx2WarmNeedBytes`, which
    // is now this gate's only consumer, so no second, independent estimate can
    // disagree with it (review round 2, finding 3).
    let genForLog = videoHolder.get()
    let videoStackWarm = genForLog?.isLoaded == true && (genForLog?.isAudioLoaded ?? false) == wantsAudio
    let ltx2Need = ltx2WarmNeedBytes(wantsAudio: wantsAudio)
    // Drain-until-settled (#34): back-to-back renders (e.g. Kira's i2v →
    // multi-keyframe in the same second) start while the previous job's
    // MLX buffer pool + lazy macOS reclaim still hold tens of GB. Admission
    // then either refuses spuriously OR passes on memory that isn't really
    // reclaimed yet — and the render dies ~60s in on a Metal allocation
    // abort (SIGKILL, no app error; 3x reproduced 2026-07-25). Actively
    // drain and re-probe until free ≥ need + margin, up to ~18s, before
    // deciding. clearCache() returns pooled buffers; the settle sleep gives
    // the OS time to actually reclaim them.
    let drainMargin: UInt64 = 6 * 1024 * 1024 * 1024
    var drainAttempts = 0
    while availableForVideo < ltx2Need + drainMargin && drainAttempts < 6 {
      GPU.clearCache()
      try? await Task.sleep(nanoseconds: 3_000_000_000)
      availableForVideo = MemoryProbe.systemAvailableMemoryBytes()
      drainAttempts += 1
      logger.info("LTX-2 admission drain #\(drainAttempts): \(availableForVideo >> 20)MB free (target \((ltx2Need + drainMargin) >> 20)MB)")
    }
    let admitVideo = heavyAdmission.admitsAfterEvict(
      needBytes: ltx2Need, freeBytes: availableForVideo)
    logger.info("LTX-2 admission: freed ~\(freedForVideoMB)MB image, \(availableForVideo >> 20)MB free after \(drainAttempts) drain(s), need ~\(ltx2Need >> 20)MB\(videoStackWarm ? " (warm stack)" : "") → admit=\(admitVideo) (#218/#34)")
    return (admitVideo, Int(availableForVideo >> 20), Int(ltx2Need >> 20))
  }

  #if DEBUG
  /// comfybox#322 test seam. The #218 admission gate needs ~65-80GB of real
  /// free RAM, which no unit test can arrange on a machine that is also
  /// serving production — so the coordinator seam tests (which exist to prove
  /// the `.localVideo` case PUBLISHES a cancellable render task, not to
  /// re-test admission) skip the gate. Same precedent as the `.synthetic`
  /// queue kind: `#if DEBUG`, never compiled into the shipped engine, and only
  /// `WarmServerQueueProbe` can set it.
  var bypassVideoAdmissionForTests = false
  func setBypassVideoAdmission(_ value: Bool) { bypassVideoAdmissionForTests = value }

  /// comfybox#362 test seam: stage a fake "checkpointed video" exactly as
  /// `runPreemptionEpisode` publishes it for the span of a real episode —
  /// lets a test drive `target: "video"` resolution (and the
  /// default-target/health-agreement behaviour of `runAsPublishedActiveRender`)
  /// without a real LTX-2 checkpoint or model weights. `statusJobId` is the
  /// `/v1/video/status/{id}` id (comfybox#283: it differs from the queue id).
  func setCheckpointedVideoForTest(
    task: Task<Void, Never>?, jobId: String?, statusJobId: String? = nil
  ) {
    liveHealth.setCheckpointedVideo(
      task.map {
        PublishedRender(
          task: $0, jobId: jobId, statusJobId: statusJobId, kind: QueueJobKind.video.rawValue)
      } ?? .none)
  }

  /// comfybox#362 test seam: publish an active render directly, as the queue
  /// loop does around every job — lets a test stage a REAL published active
  /// task (rather than leaving the slot empty, which made the "the video was
  /// not cancelled" assertion vacuous; review r1, finding 6).
  /// Sets the identity fields too, and publishes both atomically — the state
  /// the queue loop leaves behind for a running job, so a sequencing test
  /// starts from a consistent published pair rather than a triple that names
  /// a job `/health` has never heard of.
  func setActiveRenderForTest(
    task: Task<Void, Never>?, jobId: String?, statusJobId: String? = nil, kind: String?
  ) {
    activeJobId = jobId
    activeJobKindForPersistence = kind
    publishActiveRenderAndHealth(
      task.map { PublishedRender(task: $0, jobId: jobId, statusJobId: statusJobId, kind: kind) }
        ?? .none)
  }

  /// comfybox#362 test seam: the currently published active render, so a test
  /// can capture it and hand it back as `restoringTo` exactly as
  /// `runPreemptionEpisode` does.
  var activeRenderForTest: PublishedRender { activeRender }

  /// comfybox#362 test seam: the PRODUCTION function `runPreemptionEpisode`
  /// wraps its preempting image job in — publishes `work`'s own task as the
  /// active render (`activeRender` + `liveHealth`) for its duration, then
  /// restores `restoringTo`. A test drives this with fake tasks instead of a
  /// real checkpoint/render to prove the publish/restore sequence in
  /// isolation from model weights.
  /// `preemptorIdentity`/`restoredIdentity` stand in for
  /// `runPreemptionEpisode`'s identity swap and restore, so the test drives
  /// the REAL `runAsPublishedActiveRender` — including the ordering review r2
  /// item 1 is about — with no model weights anywhere.
  func runAsPublishedActiveRenderForTest(
    restoringTo restoreTo: PublishedRender,
    preemptorIdentity: (jobId: String, kind: String)? = nil,
    restoredIdentity: (jobId: String?, kind: String?)? = nil,
    _ work: @escaping @Sendable () async -> Void
  ) async {
    await runAsPublishedActiveRender(
      restoringTo: restoreTo,
      swapIdentity: {
        if let preemptorIdentity {
          activeJobId = preemptorIdentity.jobId
          activeJobKindForPersistence = preemptorIdentity.kind
        }
      },
      restoreIdentity: {
        if let restoredIdentity {
          activeJobId = restoredIdentity.jobId
          activeJobKindForPersistence = restoredIdentity.kind
        }
      },
      work)
  }

  /// comfybox#362 test seam: the ASYNC `/v1/queue/interrupt` fallback path
  /// (`ControlPlaneSyncFlag` off) — exposed so a test can prove it agrees
  /// with the sync path (`WarmServerQueueProbe.controlInterrupt(target:)`)
  /// rather than assuming parity.
  func cancelActiveRenderForTest(target: String? = nil) -> InterruptCancelOutcome {
    cancelActiveRender(target: target)
  }

  /// comfybox#362: read back the DEBUG-only admission bypass — lets a test
  /// prove `setBypassVideoAdmission(false)` (the reset `makeQueueProbe`'s
  /// teardown now performs) actually took effect, rather than assuming it.
  var bypassVideoAdmissionForTestsValue: Bool { bypassVideoAdmissionForTests }

  /// comfybox#362 review r1, finding 2 — reproduces the window `runGenerate`'s
  /// `defer` opens, without model weights.
  ///
  /// That `defer` (see `runGenerate`) sets `activeJobId = nil` and does NOT
  /// call `publishHealth()`, so between a render finishing and the queue
  /// loop's own `defer` running, the ACTOR's `activeJobId` is nil while the
  /// PUBLISHED snapshot still names the job. While the two interrupt
  /// implementations sourced the job id from those two different places, a
  /// job-id `target` got two different answers in that window (the sync path
  /// cancelled and named the job; the async fallback 404'd). Both now read the
  /// one published triple, and `testSyncAndAsyncAgreeOnAJobIdTarget…` proves
  /// it by driving exactly this state.
  func clearActiveJobIdWithoutPublishingForTest() { activeJobId = nil }
  #endif

  /// The admission decision for a `.localVideo` job, in one place so the queue
  /// case reads the same gate a resume does.
  private func admitVideoForRender(
    wantsAudio: Bool
  ) async -> (admitted: Bool, availableMB: Int, neededMB: Int) {
    #if DEBUG
    if bypassVideoAdmissionForTests { return (true, 0, 0) }
    #endif
    return await vacateImageModelsAndAdmitVideo(wantsAudio: wantsAudio)
  }

  /// Reload (if needed) and resume the checkpointed video render.
  ///
  /// Runs the full #218/#34 admission gate (`vacateImageModelsAndAdmitVideo`)
  /// before touching the generator (review C2).
  ///
  /// **Residency is the authority** (final review, C1): there is no `evicted`
  /// parameter any more. The caller's belief about whether it evicted the
  /// video is not trustworthy — `#218` makes ANY image load release the video
  /// stack underneath it (`poolLoad`'s first statement is
  /// `videoHolder.release()`), and the memory-pressure guard can release it
  /// too — so this asks `videoHolder` what is actually resident and rebuilds
  /// when nothing is. Absence is recoverable (a cold rebuild costs a reload,
  /// which is exactly what the always-evict episode already budgets for);
  /// throwing on it, as the previous `evicted: false` branch did, destroyed
  /// the checkpoint and failed the video job outright.
  ///
  /// Binding constraint from Task 4's reviews, enforced HERE so no call site
  /// can get it wrong: a fresh post-eviction instance has no signal/telemetry
  /// wired (`VideoGeneratorHolder.release()` deallocates the old one) —
  /// re-wired unconditionally below. The OTHER Task 4 constraint — the
  /// signal must be cleared before `resume(from:)` — is enforced by
  /// `runPreemptionEpisode`'s caller instead (review round 2, finding 2):
  /// clearing it HERE, after this function's own admission gate can `await`
  /// for up to ~18s in the drain loop, left a window where a SECOND
  /// preemptor could raise a NEW signal during that wait, which this clear
  /// would then erase — stranding the second preemptor behind its own 120s
  /// watchdog. `runPreemptionEpisode` clears while `preemptionInFlight` is
  /// still held, before anything here can run, so no such window exists.
  ///
  /// Throws on failure (config drift, missing generator, admission refused)
  /// — the caller's `continuation` then resumes with the error, i.e. the
  /// video job fails loudly. No silent restart from step 0 (spec, Error
  /// handling).
  private func resumeCheckpointedVideo(
    state: LTX2ResumeState, wantsAudio: Bool, report: @escaping @Sendable (Int) -> Void
  ) async throws -> LTX2RenderOutcome {
    let admission = await vacateImageModelsAndAdmitVideo(wantsAudio: wantsAudio)
    guard admission.admitted else {
      throw WarmServerError.invalidRequest(
        message: "#1479: insufficient memory to resume LTX-2 video after preemption: only \(admission.availableMB)MB free (need ~\(admission.neededMB)MB)")
    }
    let gen: LTX2VideoGenerator
    if let existing = videoHolder.get() {
      // Still resident — nothing released it while the preemptor ran. Reuse
      // it; the checkpoint resumes with no weight reload at all.
      logger.info("#1479: resuming LTX-2 video — generator still resident, no reload needed")
      gen = existing
    } else {
      // Nothing resident: the expected state after an episode's eviction, and
      // also the recovery path if something ELSE released it out from under
      // the checkpoint (#218's release-before-image-load, the memory-pressure
      // guard). Either way this is recoverable — rebuild cold and resume from
      // the checkpoint rather than throwing the render away.
      logger.info("#1479: resuming LTX-2 video — no generator resident, rebuilding cold before resume")
      gen = try await reloadVideoGeneratorAfterEviction()
    }
    gen.setPreemptionSignal(ltx2PreemptionSignal)
    gen.setTelemetry(ltx2Telemetry)
    return try gen.resume(from: state) { chunk, totalChunks, step, totalSteps in
      report(WarmServer.localVideoProgressPercent(chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps))
    }
  }

  /// Sum of observed `modelLoad` phase durations (`meanSec x samples`) —
  /// the real instrument for "how long does a weight reload take" (review
  /// round 2, finding 1). The load itself is bracketed by
  /// `telemetry?.begin(.modelLoad)`/`.end(.modelLoad)` inside
  /// `LTX2VideoGenerator`'s `render()`, around the actual `load(...)` call —
  /// snapshotting this before/after a resume and recording the DELTA (only
  /// when it grew, i.e. a load actually happened) is accurate regardless of
  /// how long the REST of the resumed render takes, unlike wall-clocking the
  /// resume call itself (which the previous fix for I1 got wrong — see the
  /// call site below).
  private func ltx2ModelLoadTotalSec() -> Double {
    guard let p = ltx2Telemetry.view().phases["modelLoad"] else { return 0 }
    return p.meanSec * Double(p.samples)
  }

  /// comfybox#322 (review r1, Critical): run `work` to completion regardless of
  /// whether THIS task is cancelled.
  ///
  /// The one caller is the preemption episode's image job. Structured
  /// concurrency would be wrong there: the episode is awaited from inside the
  /// video's render task, so a child task inherits the video's cancellation and
  /// an interrupt aimed at the video takes an unrelated image render down with
  /// it. An unstructured task inherits no cancellation, and `.value` on a
  /// `Task<Void, Never>` is non-throwing, so this returns only when `work` has
  /// actually finished.
  ///
  /// Isolation note: `work` is a `@Sendable` closure, so it is nonisolated and
  /// hops back onto this actor at its own `await`. The window that opens while
  /// this task is suspended is the same one `runGenerate`'s own first `await`
  /// already opened before this shield existed — no new reentrancy, and the
  /// queue loop stays parked on the video render task throughout.
  ///
  /// Exposed to `WarmServerQueueProbe` (DEBUG) so a test can prove the shielded
  /// work does NOT observe the caller's cancellation.
  func runShieldedFromCancellation(_ work: @escaping @Sendable () async -> Void) async {
    await startShieldedFromCancellation(work).value
  }

  /// Starts `work` in a new unstructured task that inherits no cancellation
  /// (see `runShieldedFromCancellation`'s doc comment) and returns the task
  /// WITHOUT awaiting it.
  ///
  /// comfybox#362: split out of `runShieldedFromCancellation` so
  /// `runAsPublishedActiveRender` can publish the task's own handle as the
  /// active render BEFORE waiting on it — `runShieldedFromCancellation`
  /// itself still exists unchanged (as a thin wrapper) for
  /// `WarmServerQueueProbe`'s existing shield-only test.
  private func startShieldedFromCancellation(_ work: @escaping @Sendable () async -> Void) -> Task<Void, Never> {
    Task { await work() }
  }

  /// comfybox#362: run `work` (shielded from the caller's cancellation, exactly
  /// like `runShieldedFromCancellation`) as the PUBLISHED active render for its
  /// duration, then restore `restoringTo` as the published active render
  /// afterward.
  ///
  /// This is what makes `/health`, `/v1/queue` and `/v1/queue/interrupt`'s
  /// default target agree during a preemption episode: without it,
  /// `activeRenderTask`/`liveHealth` kept pointing at the checkpointed video's
  /// own task for the ENTIRE episode (including while the preempting image job
  /// — the thing health actually shows as active — was running), so a plain
  /// interrupt cancelled the invisible video instead of the visible image
  /// render. `runPreemptionEpisode` is the one caller, publishing the video's
  /// own task as `restoringTo` so it flows straight back into
  /// `resumeCheckpointedVideo`'s time.
  ///
  /// The checkpointed-video slot is left untouched here — that is
  /// `runPreemptionEpisode`'s responsibility (set once for the whole episode,
  /// not once per shielded call), so `target: "video"` can still reach the
  /// video while this function's task is what `target: "active"` reaches.
  ///
  /// Review r1, finding 1: the publish and the restore are each ONE write of a
  /// complete `PublishedRender`. The preemptor's identity (`activeJobId` /
  /// `activeJobKindForPersistence`) has already been swapped in by the caller
  /// when this runs, so the triple published here names the preemptor and
  /// cancels the preemptor — an interrupt can never cancel one job while
  /// reporting another's id.
  /// Review r2, item 1: the identity swap and the triple publish are ONE step.
  ///
  /// Round 1 bundled the task with its identity, which closed the "cancel one
  /// job, report another's id" hole — but the episode still swapped
  /// `activeJobId`/`activeJobSummary`/… and called `publishHealth()` +
  /// `persistQueueState()` BEFORE this function published the triple, and
  /// restored them AFTER it restored the triple. Between those two points a
  /// reader saw health naming the preemptor while the interrupt still pointed
  /// at the video — the #362 failure itself, just narrowed to a window.
  ///
  /// So the caller hands its identity swap/restore in as `swapIdentity` /
  /// `restoreIdentity`, and each runs immediately before the corresponding
  /// publication, with no `await` in between (both closures are synchronous
  /// and actor-isolated, so `work` — which is unstructured and hops onto this
  /// actor at its own first `await` — cannot interleave). The publication
  /// itself is atomic: `publishActiveRenderAndHealth` writes the snapshot and
  /// the triple under one lock.
  private func runAsPublishedActiveRender(
    restoringTo restoreTo: PublishedRender,
    swapIdentity: () -> Void = {},
    restoreIdentity: () -> Void = {},
    _ work: @escaping @Sendable () async -> Void
  ) async {
    // Start the shielded task FIRST so its handle exists for the publication
    // below — it cannot enter this actor until the `await` further down, so
    // everything between here and there is one indivisible step.
    let task = startShieldedFromCancellation(work)
    swapIdentity()
    publishActiveRenderAndHealth(
      PublishedRender(task: task, jobId: activeJobId, kind: activeJobKindForPersistence))
    persistQueueState()
    await task.value
    restoreIdentity()
    publishActiveRenderAndHealth(restoreTo)
    persistQueueState()
  }

  /// The ONE writer of the active-render publication: the actor's bookkeeping
  /// copy and the lock-store publication, always together, always a complete
  /// triple (review r1, finding 1).
  private func publishActiveRender(_ publication: PublishedRender) {
    activeRender = publication
    liveHealth.setActiveRender(publication)
  }

  /// Publish the triple AND the health snapshot it must agree with, atomically
  /// (review r2, item 1) — used wherever the two change together, i.e. the
  /// preemption episode's swap and restore.
  private func publishActiveRenderAndHealth(_ publication: PublishedRender) {
    activeRender = publication
    liveHealth.publish(makeHealthSnapshot(), activeRender: publication)
  }

  /// Publish `task` as the active render, taking the identity from the fields
  /// the queue loop has ALREADY set for this job (`activeJobId`,
  /// `activeJobKindForPersistence`) — so the published id/kind and the
  /// persisted/health identity are the same values by construction, not by
  /// two call sites agreeing to pass the same thing. `nil` clears the slot.
  /// `statusJobId` is the `/v1/video/status/{id}` id for a video job
  /// (comfybox#283: it differs from the queue id).
  private func publishActiveRender(task: Task<Void, Never>?, statusJobId: String? = nil) {
    publishActiveRender(
      task.map {
        PublishedRender(
          task: $0, jobId: activeJobId, statusJobId: statusJobId,
          kind: activeJobKindForPersistence)
      } ?? .none)
  }

  /// The full preemption episode: checkpoint received -> evict the video
  /// weights -> run the preempting image job -> ALWAYS resume the video.
  ///
  /// **v1 always evicts** (final review C1 + controller ruling, 2026-08-15;
  /// the spec's Decision 1 has been amended to match). The original design had
  /// a "fits alongside" fast path that kept the video weights resident and
  /// skipped the evict/reload round trip. That path was not merely an
  /// optimisation that sometimes failed to pay off — it was actively
  /// destructive: `runGenerate` begins with `reloadImageModelIfEvicted` ->
  /// `poolLoad`, whose FIRST statement is `videoHolder.release()` (#218's
  /// invariant that video must vacate before any image load, and
  /// `imageModelsEvicted` is always true while a video is resident, so that
  /// reload always runs). The layer below therefore released the very weights
  /// the fast path was preserving, and the resume then found nothing resident
  /// and threw — losing the checkpoint and failing the video job. On a 128GB
  /// box the fast path was the LIKELY branch. Honouring pause-in-place would
  /// require a preemption-aware carve-out in `poolLoad` that lets both heavy
  /// stacks co-reside — the exact 2x22GB pool-budget hazard the spec's
  /// Decision 1 exists to avoid — so v1 simply always evicts, and
  /// `resumeCheckpointedVideo` treats live residency (not a caller's belief)
  /// as the authority on whether a rebuild is needed.
  ///
  /// Always-resume is enforced two ways: structurally, the image-job call
  /// sits in its own `do`/`catch` (review I4) so that even if `runGenerate`
  /// ever grows a `throws` in the future, control still reaches the resume
  /// below instead of silently skipping it; today `runGenerate` is
  /// non-throwing by construction (every internal failure already resolves
  /// `claimed.continuation` via `continuation.resume(throwing:)`), so that
  /// `catch` is presently unreachable — it exists as a guardrail, not because
  /// it fires. A failed preemptor therefore costs the video nothing but the
  /// wall-clock time it ran (spec, Error handling).
  ///
  /// Checkpoint/paused/in-flight bookkeeping is cleared BEFORE the (possibly
  /// long, up to the rest of the render) resume call, not after (review I2):
  /// clearing after would hold `preemptionInFlight` for the render's entire
  /// remaining duration, making a SECOND preemption of the resumed render
  /// permanently unreachable — exactly the double-preempt path Task 4's
  /// forward-only unwind guard exists to handle correctly. `resolved` guards
  /// the `defer` so the two code paths (claimed vs. not-claimed) each clear
  /// exactly once, and a throw before either path reaches its own clearing
  /// (e.g. `pendingPreemptorBox.claim()` itself can't throw, but future edits
  /// might add one) still cleans up via the `defer`.
  ///
  /// `ltx2PreemptionSignal.clear()` happens FIRST, before anything else
  /// (review round 2, finding 2): this runs while `preemptionInFlight` is
  /// STILL held (it isn't released until `clearEpisodeState()` below), so no
  /// other caller can have raised a competing signal in between — clearing
  /// it any later (previously done inside `resumeCheckpointedVideo`, after
  /// its own admission gate could `await` up to ~18s in the drain loop) left
  /// a window where a second preemptor could raise, get its raise erased by
  /// this render's stale clear, and be stranded behind its own 120s
  /// watchdog. `checkpointedVideo` is read once, right after, as a
  /// single-checkpoint sanity check (spec: hold exactly one at a time) —
  /// this function must never be reentered while a checkpoint is already
  /// held.
  private func runPreemptionEpisode(
    state: LTX2ResumeState, videoJobId: String?, wantsAudio: Bool, report: @escaping @Sendable (Int) -> Void
  ) async throws -> LTX2RenderOutcome {
    ltx2PreemptionSignal.clear()
    assert(checkpointedVideo == nil, "#1479: hold exactly one checkpoint at a time (spec) — runPreemptionEpisode entered with one already held")
    checkpointedVideo = state
    if let videoJobId { videoJobTracker.markPausedForPreemption(videoJobId) }
    // comfybox#362: capture the video's whole published triple BEFORE
    // anything below can touch it (the identity swap further down), and
    // republish it as the checkpointed-video slot for the episode's duration
    // — this is what lets `target: "video"` still reach the video once the
    // active slot is republished to the preemptor.
    //
    // Review r1, finding 3: the captured triple already carries BOTH of the
    // video's ids — the queue id (`jobId`) and the `/v1/video/status/{id}` id
    // (`statusJobId`), which comfybox#283 documents as different — because
    // the `.localVideo` queue case published both. So `target: <id>` matches
    // whichever one the client happens to hold, and it is one write, not two.
    let videoPublication = activeRender
    liveHealth.setCheckpointedVideo(videoPublication)
    publishHealth()

    var resolved = false
    // PR #370 review I3: this used to record `.resumed` unconditionally,
    // including on the `.abandonVideo` disposition below — where the render
    // is NEVER actually resumed, the checkpoint is dropped. `abandoned`
    // (`ReplayClassifier` treats it identically to `.resumed`: either way
    // the checkpoint is closed out) names the OTHER outcome correctly; the
    // caller computes the disposition once and passes it in so this
    // function never has to re-derive it (and can't disagree with the
    // abandon check that follows each call site).
    func clearEpisodeState(abandoned: Bool) {
      resolved = true
      checkpointedVideo = nil
      // comfybox#362: symmetric with the publish at the top of this function
      // — the episode is over (resumed or abandoned) either way, so the
      // separately-published "video" interrupt target goes away with it.
      liveHealth.setCheckpointedVideo(.none)
      if let videoJobId { videoJobTracker.markResumedFromPreemption(videoJobId) }
      // comfybox#283/#217: read-only. `activeJobId` here is always the VIDEO
      // job's own id — either this runs from the early-return branch (before
      // any preemptor identity swap happened below) or the swap has already
      // been reversed by the restore block just above this function's second
      // call site — never the preemptor's id.
      if let jobId = activeJobId {
        lifecycleLedger.record(
          jobId: jobId, kind: abandoned ? .abandoned : .resumed, jobKind: QueueJobKind.video.rawValue,
          step: state.stepIndex, chunk: state.chunkIndex, reason: state.phase.rawValue)
      }
      // Idempotent: the checkpoint-fallback watchdog may have already
      // cleared this if it raced ahead of the yield.
      preemptionInFlight.clear()
    }
    defer {
      if !resolved {
        clearEpisodeState(abandoned: LTX2PreemptionEpisode.disposition(videoInterrupted: Task.isCancelled) == .abandonVideo)
      }
    }

    guard let claimed = pendingPreemptorBox.claim() else {
      // The checkpoint-fallback watchdog already handled this preemptor (it
      // raced ahead of the yield, or the render finished on its own before
      // the signal was observed) — nothing left to run. Resume immediately.
      logger.warning("#1479: video yielded but no preemptor was waiting (checkpoint-fallback watchdog already handled it) — resuming immediately")
      // comfybox#322: …unless the video itself was interrupted. There is no
      // preemptor to protect on this branch, so this is purely "do not bring
      // a killed render back to life". Computed ONCE (PR #370 review I3) so
      // `clearEpisodeState` and this check can never disagree about which
      // outcome just happened.
      let disposition = LTX2PreemptionEpisode.disposition(videoInterrupted: Task.isCancelled)
      clearEpisodeState(abandoned: disposition == .abandonVideo)
      if disposition == .abandonVideo {
        logger.info("#1479/#322: video interrupted with no preemptor pending — checkpoint dropped, no resume.")
        throw CancellationError()
      }
      return try await resumeCheckpointedVideo(state: state, wantsAudio: wantsAudio, report: report)
    }

    logger.info("#1479: video checkpointed at chunk \(state.chunkIndex), phase \(state.phase.rawValue), step \(state.stepIndex) — running preempting image job (source=\(claimed.source))")
    // comfybox#283/#217: read-only — `activeJobId` is still the video job's
    // own id here (the identity swap to the preemptor happens below).
    if let jobId = activeJobId {
      lifecycleLedger.record(
        jobId: jobId, kind: .checkpointed, jobKind: QueueJobKind.video.rawValue,
        step: state.stepIndex, chunk: state.chunkIndex, reason: state.phase.rawValue)
    }

    // Always evict (see doc comment): the image load below would release these
    // weights anyway, one layer down and without timing it. Doing it here
    // explicitly keeps `ltx2EvictMean` — half of the refusal guard's
    // round-trip estimate — measuring the real thing.
    let t0 = Date()
    videoHolder.release()
    ltx2EvictMean.record(Date().timeIntervalSince(t0))
    logger.info("#1479: evicted LTX-2 video weights to admit the preempting image job")

    // #1479 (final review I2/M13): the preemptor is a REAL render — it just
    // didn't arrive through `pending`. Swap the entire active-job identity
    // over to it (id, summary, source, start time) and, critically, the
    // persistence pair (rawBody + kind) so the durable queue snapshot names
    // the image job while the image job is what's actually running. Without
    // this, a crash during the preemptor lost it outright (the persisted
    // "active" slot still described the paused video, whose own rawBody is
    // nil and therefore unrecoverable), and /health + /v1/queue reported the
    // video as active for the whole image render. Persist AFTER each swap so
    // the on-disk snapshot is never behind the in-memory truth.
    let videoIdentity = (
      id: activeJobId, summary: activeJobSummary, source: activeJobSource,
      rawBody: activeJobRawBody, kind: activeJobKindForPersistence,
      startedAt: currentJobStartedAt, renderStartedAt: activeRenderStartedAt
    )
    let preemptorJobId = claimed.jobId ?? UUID().uuidString  // AC-18: the async caller's id when it has one
    //
    // comfybox#362 review r2, item 1: the swap is NOT performed here. It is
    // handed to `runAsPublishedActiveRender` as `swapIdentity`, which runs it
    // immediately before publishing the interrupt triple + health snapshot
    // atomically. Performing it here (with its own `publishHealth()` +
    // `persistQueueState()`) left a real window in which `/health` named the
    // preemptor while `/v1/queue/interrupt` still pointed at the video — an
    // operator interrupting on what they saw would have killed the video,
    // which is #362 itself. The restore below is handed over the same way.

    // Run the image job exactly as a normal render — the same actor method
    // every non-preempting `.generate` job runs, including its own
    // model/LoRA application and (since `imageModelsEvicted` is already true
    // whenever a video is resident, #218) its own image-model reload. See
    // this function's doc comment (review I4) for why this is its own
    // do/catch rather than a bare call.
    // comfybox#322 (review r1, Critical): SHIELDED. This episode runs inside
    // the video's render task, and `runGenerate` has been cancellation-aware
    // since comfybox#304 — so the plain `await` that used to be here inherited
    // the video's cancellation, and `/v1/queue/interrupt` aimed at the video
    // killed the preempting IMAGE job mid-denoise. That is exactly the "died
    // as cancel collateral with an opaque error" failure #322 exists to end.
    // `runShieldedFromCancellation` runs it in an unstructured task, which
    // does not inherit cancellation, and waits for it either way.
    //
    // This also replaces the old do/catch guardrail with a STATIC one: the
    // shield takes a non-throwing closure, so if `runGenerate` ever grows a
    // `throws` this stops compiling — louder than a `catch` that logs and
    // carries on.
    //
    // comfybox#362: `runAsPublishedActiveRender` (not the bare
    // `runShieldedFromCancellation`) — it ALSO republishes this shielded
    // task's own handle as `activeRenderTask`/`liveHealth`'s active render for
    // its duration, restoring `videoTask` afterward. Before this, those
    // handles kept pointing at the video's own task for the whole episode, so
    // a plain `/v1/queue/interrupt` (no explicit target) — the exact request
    // an operator sends after watching `/health` show the image job active —
    // cancelled the invisible video instead of the visible image render.
    await runAsPublishedActiveRender(
      restoringTo: videoPublication,
      swapIdentity: {
        activeJobId = preemptorJobId
        activeJobSummary = "Render (preempting): \(claimed.payload.prompt.prefix(100))"
        activeJobSource = claimed.source
        activeJobRawBody = claimed.rawBody
        activeJobKindForPersistence = "generate"
        currentJobStartedAt = Date()
        activeRenderStartedAt = nil
      },
      // Restore the video's identity — symmetric with the swap, and done
      // BEFORE its (possibly long, synchronous) resume so /health, /v1/queue
      // and the persisted snapshot all go back to describing the video job.
      // The preemptor is finished by now, so leaving it in the persisted
      // active slot would replay a completed image job on the next restart.
      restoreIdentity: {
        activeJobId = videoIdentity.id
        activeJobSummary = videoIdentity.summary
        activeJobSource = videoIdentity.source
        activeJobRawBody = videoIdentity.rawBody
        activeJobKindForPersistence = videoIdentity.kind
        currentJobStartedAt = videoIdentity.startedAt
        activeRenderStartedAt = videoIdentity.renderStartedAt
      }
    ) {
      await self.runGenerate(claimed.payload, continuation: claimed.continuation)
    }
    // PR #370 review M: the preemptor bypasses the normal enqueue/admit path
    // entirely (a mailbox handoff, not the FIFO — see WarmServer.swift's
    // "Known limitations" note in the PR body), so it never reaches a
    // terminal event through `record`, and its progress-throttle entry would
    // otherwise never be cleared — one leaked dictionary entry per
    // preemption for the life of the process. `preemptorJobId`, not
    // `activeJobId` — `runGenerate`'s own `defer` already reset
    // `activeJobId` to nil by the time this line runs.
    lifecycleLedger.clearProgressThrottle(jobId: preemptorJobId)

    // Clear BEFORE the resume, not in the defer (review I2) — see doc comment.
    // Computed ONCE (PR #370 review I3), same reasoning as the other call
    // site: `clearEpisodeState` and the abandon check right after it must
    // never disagree about which outcome just happened.
    let disposition = LTX2PreemptionEpisode.disposition(videoInterrupted: Task.isCancelled)
    clearEpisodeState(abandoned: disposition == .abandonVideo)

    // comfybox#322 (review r1, Critical): the operator may have interrupted the
    // video while the preemptor ran. The preemptor was protected above and has
    // finished normally; the video's checkpoint is dropped here rather than
    // resumed — carrying on for another 20 minutes would defeat the interrupt
    // entirely. `clearEpisodeState()` above already released the checkpoint and
    // the in-flight flag, and the video weights were evicted at the top of this
    // episode and are deliberately NOT reloaded: an operator who interrupted to
    // free the box gets the box. `CancellationError` propagates to the queue
    // case, which reports the job interrupted.
    if disposition == .abandonVideo {
      logger.info("#1479/#322: video interrupted during the preemption episode — preemptor completed normally, video checkpoint dropped (no resume).")
      throw CancellationError()
    }

    // Review round 2, finding 1: snapshot the REAL instrument (modelLoad
    // phase telemetry), not a wall clock around the whole resume call — the
    // resume call's return time includes the rest of the render (15-60min),
    // which would poison a cumulative rolling mean into a de facto refuse-
    // everything kill switch. Record only the delta, and gate it on a real
    // reload having happened (review round 3, finding 1): `load()`'s
    // idempotent early-return still sits INSIDE the telemetry bracket
    // (`LTX2VideoGenerator.swift`), so a resume that reuses a resident
    // generator emits its own microseconds-scale `modelLoad` sample —
    // `modelLoadDelta > 0` alone is true there too, and a near-zero sample
    // decays the rolling mean toward zero, degrading the refusal guard to
    // never-refuse (the opposite failure this metric exists to avoid). The
    // gate is therefore "was the generator absent going in", read from the
    // same authority `resumeCheckpointedVideo` uses and evaluated on the
    // actor with nothing able to change residency in between; the delta is
    // only for accuracy of the reload's DURATION, not for detecting whether
    // it occurred. This also swallows the ~1e-13 float round-trip noise the
    // mean*samples sum reconstruction can produce. Under the always-evict
    // rule this is normally true — it stays a check rather than a constant so
    // a future preemption-aware carve-out cannot silently poison the metric.
    //
    // Single-flight assumption: this snapshot/delta is only correct because
    // exactly one video render is ever in flight at a time (the coordinator
    // serializes video on the same queue as image renders); a future
    // concurrent-video path would let a second render's modelLoad samples
    // land inside this window and silently corrupt the delta.
    let willReload = videoHolder.get() == nil
    let modelLoadBefore = ltx2ModelLoadTotalSec()
    let outcome = try await resumeCheckpointedVideo(state: state, wantsAudio: wantsAudio, report: report)
    let modelLoadDelta = ltx2ModelLoadTotalSec() - modelLoadBefore
    if willReload, modelLoadDelta > 0 {
      ltx2ReloadMean.record(modelLoadDelta)
    }
    return outcome
  }

  func prepare() async throws {
    // Resolve model snapshot path for family detection
    let modelSpec = configuration.modelSpec
    var isFlux2 = false
    var snapshotURL: URL?

    var isFibo = false
    var isChroma = false
    var isKrea2 = false

    if let spec = modelSpec {
      // Check by known model ID first
      if Krea2ModelDetection.isKnownKrea2Model(spec) {
        isKrea2 = true
      } else if ChromaModelDetection.isKnownChromaModel(spec) {
        isChroma = true
      } else if FiboModelDetection.isKnownFiboModel(spec) {
        isFibo = true
      } else if Flux2ModelDetection.isKnownFlux2Model(spec) {
        isFlux2 = true
      }

      if !isKrea2 {
        // Resolve snapshot — needed for both detection and loading
        let resolved = try await ModelResolution.resolveOrDefault(
          modelSpec: spec,
          filePatterns: ["*.safetensors", "*.json", "tokenizer/*"]
        )
        snapshotURL = resolved

        // If not already detected by name, check the snapshot directory
        if !isFibo && !isFlux2 && !isChroma {
          if Krea2ModelDetection.isKrea2ModelDirectory(resolved) {
            isKrea2 = true
          } else if ChromaModelDetection.detect(at: resolved) != nil {
            isChroma = true
          } else if FiboModelDetection.detect(at: resolved) != nil {
            isFibo = true
          } else if Flux2ModelDetection.detectFamily(at: resolved) == .flux2 {
            isFlux2 = true
          }
        }
      }
    }

    if isKrea2, let spec = modelSpec {
      // --- Krea-2 path (native port) — variant read off disk, fail-closed (WP-E5) ---
      currentModelFamily = .krea2
      let paths = try Krea2ModelDetection.resolve(spec: spec)
      logger.info(
        "Detected Krea-2 \(paths.variant.rawValue) (\(paths.transformerFile.path)) — 8-bit transformer, estimated GPU memory: ~22GB")
      krea2Pipeline = try Krea2Pipeline(paths: paths, quantizeTransformer: 8)
      krea2Variant = paths.variant
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (Krea-2 \(paths.variant.rawValue))")
    } else if isChroma, let snapshot = snapshotURL {
      // --- Chroma path ---
      currentModelFamily = .chroma

      guard let detected = ChromaModelDetection.detect(at: snapshot) else {
        throw WarmServerError.chromaDetectionFailed(modelSpec ?? "unknown")
      }

      logger.info("Detected Chroma model — estimated GPU memory: ~17GB")

      let components = try ChromaInitializer.load(
        from: snapshot,
        paths: detected.componentPaths,
        config: detected.config,
        dtype: .bfloat16,
        logger: logger
      )

      // Load tokenizer
      let tokenizer = try ChromaTokenizer.load(from: detected.componentPaths.tokenizerPath)

      chromaPipeline = ChromaPipeline(
        transformer: components.transformer,
        t5: components.t5,
        vae: components.vae,
        config: detected.config
      )
      chromaTokenizer = tokenizer
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (Chroma)")
    } else if isFibo, let snapshot = snapshotURL {
      // --- FIBO path ---
      currentModelFamily = .fibo

      guard let detected = FiboModelDetection.detect(at: snapshot) else {
        throw WarmServerError.fiboDetectionFailed(modelSpec ?? "unknown")
      }
      detectedFiboModel = detected
      logger.info("Detected FIBO model — estimated GPU memory: ~16GB")

      let fp = FiboPipeline(logger: logger)
      try fp.loadModel(
        from: snapshot,
        transformerConfig: detected.transformerConfig,
        vaeConfig: detected.vaeConfig,
        textEncoderConfig: detected.textEncoderConfig
      )
      fiboPipeline = fp
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (FIBO)")
    } else if isFlux2, let snapshot = snapshotURL {
      // --- Flux 2 Klein path ---
      currentModelFamily = .flux2

      guard let detected = Flux2ModelDetection.detect(at: snapshot) else {
        throw WarmServerError.flux2DetectionFailed(modelSpec ?? "unknown")
      }
      detectedFlux2Model = detected

      // Log memory estimate
      let estimatedGB: String
      switch detected.variant {
      case "klein-4b", "klein-base-4b": estimatedGB = "~15GB"
      case "klein-9b", "klein-base-9b": estimatedGB = "~25GB"
      default: estimatedGB = "unknown"
      }
      let modelType = detected.isBaseModel ? "base (non-distilled)" : "distilled"
      logger.info("Detected Flux 2 Klein \(detected.variant) [\(modelType)] — estimated GPU memory: \(estimatedGB)")

      let f2 = Flux2Pipeline(logger: logger)
      try f2.loadModel(
        from: snapshot,
        config: detected.transformerConfig,
        textEncoderConfig: detected.textEncoderConfig,
        isBase: detected.isBaseModel
      )
      flux2Pipeline = f2
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (Flux 2 Klein \(detected.variant))")
    } else {
      // --- Flux 1 / Z-Image path ---
      currentModelFamily = .flux1

      // Detect Z-Image variant (Base vs Turbo)
      if let spec = modelSpec, let variant = ZImageVariant.fromModelSpec(spec) {
        zimageVariant = variant
      } else if let spec = modelSpec, spec.hasSuffix(".safetensors") {
        // Detect from CivitAI checkpoint inspection
        let localURL = URL(fileURLWithPath: spec)
        if FileManager.default.fileExists(atPath: localURL.path) {
          let inspection = CivitAICheckpoint.inspect(fileURL: localURL)
          if let variant = inspection.variant {
            zimageVariant = variant
          }
        }
      } else if let resolvedSnapshot = snapshotURL {
        zimageVariant = ZImageVariant.fromSnapshot(at: resolvedSnapshot)
      } else if let spec = modelSpec {
        // Resolve and detect from snapshot if not already resolved
        if let resolved = try? await ModelResolution.resolveOrDefault(
          modelSpec: spec,
          filePatterns: ["*.safetensors", "*.json", "tokenizer/*"]
        ) {
          zimageVariant = ZImageVariant.fromSnapshot(at: resolved)
        }
      }
      let variantLabel = zimageVariant == .base ? "Base (non-distilled)" : "Turbo (distilled)"
      logger.info("Preloading warm server pipeline (Flux 1 / Z-Image \(variantLabel))")
      try await pipeline.prepare(
        modelSpec: modelSpec,
        textEncoderPath: configuration.textEncoderPath,
        loras: activeLoRAs,
        forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly
      )
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (Flux 1 / Z-Image \(zimageVariant.rawValue))")

      // Pre-load the full VAE encoder for img2img support.
      // Without this, the first img2img request triggers synchronous weight
      // loading inside the actor-isolated render path, which can deadlock
      // the cooperative thread pool (issue #141).
      do {
        try pipeline.prepareFullVAE()
        logger.info("Full VAE encoder pre-loaded for img2img")
      } catch {
        logger.warning("Failed to pre-load full VAE encoder: \(error). First img2img request will attempt lazy load.")
      }
    }
    // WP-E10 sink 3: `prepare()` is the third writer of `krea2Pipeline`, and
    // its non-Krea-2 arms replace the ACTIVE family out from under a record.
    // Placed after the whole chain so every arm is covered by the one rule.
    revalidateLastRecipe()

    // Register the initial model in the pool so it appears in pool listings
    // and can be managed alongside hot-swapped models.
    // We register the already-loaded pipeline to avoid double-loading.
    if let spec = modelSpec {
      let box: PipelineBox
      let detectedInfo: Any?
      let vramMB: Int
      switch currentModelFamily {
      case .chroma:
        box = PipelineBox(pipeline: chromaPipeline! as AnyObject)
        if let tok = chromaTokenizer { box.context["tokenizer"] = tok as AnyObject }
        detectedInfo = nil
        vramMB = 17408
      case .fibo:
        box = PipelineBox(pipeline: fiboPipeline! as AnyObject)
        detectedInfo = detectedFiboModel
        vramMB = 22528
      case .flux2:
        box = PipelineBox(pipeline: flux2Pipeline! as AnyObject)
        detectedInfo = detectedFlux2Model
        vramMB = (detectedFlux2Model?.variant.contains("9b") ?? false) ? 18432 : 8704
      case .flux1:
        box = PipelineBox(pipeline: pipeline as AnyObject)
        detectedInfo = zimageVariant
        vramMB = 12288
      case .krea2:
        box = PipelineBox(pipeline: krea2Pipeline! as AnyObject)
        detectedInfo = krea2Pipeline!.variant
        vramMB = 22528
      }
      let poolKey = ModelPool.poolKey(for: spec)
      await modelPool.registerExisting(
        poolKey: poolKey,
        modelSpec: spec,
        family: currentModelFamily,
        box: box,
        vramEstimateMB: vramMB,
        detectedInfo: detectedInfo
      )
      logger.info("ModelPool: initial model '\(poolKey)' registered and activated")
    }
    // Seed the lock-based health snapshot now that the model is loaded, so
    // GET /health returns real data before the first render (#217).
    publishHealth()
  }

  /// Expose the current model family for routing decisions outside the actor.
  var modelFamily: WarmModelFamily {
    currentModelFamily
  }

  /// Active LoRA identifiers (bare filenames without path or extension) for the library API.
  var activeLoRAIdentifiers: [String] {
    activeLoRAs.map { config in
      switch config.source {
      case .local(let url):
        return (url.lastPathComponent as NSString).deletingPathExtension
      case .huggingFace(let modelId, let filename):
        if let filename {
          return (filename as NSString).deletingPathExtension
        }
        return modelId.components(separatedBy: "/").last ?? modelId
      }
    }
  }

  /// Whether the loaded Flux 2 model is a base (non-distilled) variant.
  var isFlux2BaseModel: Bool {
    detectedFlux2Model?.isBaseModel ?? false
  }

  /// The detected Z-Image variant (Base vs Turbo) for Flux 1 models.
  var currentZImageVariant: ZImageVariant {
    zimageVariant
  }

  /// The physical Krea-2 variant of the resident pipeline (WP-E5). nil when
  /// the active family is not krea2 — callers on the krea2 arm must treat nil
  /// as a fault, never as "turbo".
  var currentKrea2Variant: Krea2Variant? {
    currentModelFamily == .krea2 ? krea2Variant : nil
  }

  // MARK: - Model Pool Operations

  /// Load a model into the pool, optionally activating it.
  /// QUEUE-INTERNAL (K-FIX-1 / Codex C2). Call this ONLY from inside the
  /// process loop — a queued `.modelOperation`, the ComfyBridge switch's
  /// `enqueueModelSwitch` body, or a render's own #218 reload. A route or
  /// handler that reaches it directly reintroduces the race: actor isolation
  /// does not hold across an await, so the pool's eviction and
  /// `GPU.clearCache()` would be free to run under an active render. From
  /// outside the loop use `enqueueModelOperation` / `enqueueModelOperationDetached`.
  func poolLoad(modelSpec: String, quantization: String?, activate: Bool) async throws -> ModelLoadResponse {
    // #218: an image load must vacate a resident LTX-2 video stack first — the
    // two heavy models can't co-reside on a 128GB box. Safe here because
    // poolLoad and the video render are serialized on the same actor/queue, so
    // no video render is ever in flight at this point.
    if videoHolder.release() {
      logger.info("Released resident LTX-2 video stack before image load (#218)")
    }
    // D17 (AC-59a): every ACTUAL base handoff that touches the krea2 family
    // logs outgoing and incoming spec/variant, so a slow A/B is attributable,
    // not mysterious. `Krea2Handoff.logLine` is nil for a no-op re-activation
    // of the resident base and for a cold start with nothing resident.
    let outgoing: Krea2Handoff.Side? = {
      guard pipelinePrepared, let spec = activePoolModelSpec ?? configuration.modelSpec else { return nil }
      return Krea2Handoff.Side(spec: spec, family: currentModelFamily, krea2Variant: currentKrea2Variant)
    }()
    let start = Date()
    let entry = try await modelPool.load(
      modelSpec: modelSpec,
      quantization: quantization,
      initialLoRAs: activeLoRAs,
      // A load that intends to activate is a HANDOFF — the pool may evict the
      // current active model to make room (two ~22GB krea2-family models
      // cannot co-reside; without this a switch 507s, 2026-08-11).
      allowActiveEviction: activate
    )
    let loadTimeMs = Int(Date().timeIntervalSince(start) * 1000.0)

    if activate {
      try await poolActivate(modelId: entry.id)
      let incoming = Krea2Handoff.Side(spec: entry.modelSpec, family: entry.family, krea2Variant: currentKrea2Variant)
      if let line = Krea2Handoff.logLine(outgoing: outgoing, incoming: incoming, loadTimeMs: loadTimeMs) {
        logger.info("\(line)")
      }
    }

    return ModelLoadResponse(
      status: "loaded",
      model: entry.modelSpec,
      family: entry.family.rawValue,
      loadTimeMs: loadTimeMs,
      vramEstimateMB: entry.vramEstimateMB,
      poolSize: await modelPool.count(),
      poolBudgetMB: await modelPool.budget()
    )
  }

  /// Activate a model that is already in the pool.
  @discardableResult
  /// QUEUE-INTERNAL (K-FIX-1 / Codex C2). Call this ONLY from inside the
  /// process loop — a queued `.modelOperation`, the ComfyBridge switch's
  /// `enqueueModelSwitch` body, or a render's own #218 reload. A route or
  /// handler that reaches it directly reintroduces the race: actor isolation
  /// does not hold across an await, so the pool's eviction and
  /// `GPU.clearCache()` would be free to run under an active render. From
  /// outside the loop use `enqueueModelOperation` / `enqueueModelOperationDetached`.
  func poolActivate(modelId: String) async throws -> ModelActivateResponse {
    // Try by pool key first, then by model spec.
    let entry: PoolEntry
    if let e = await modelPool.findEntry(for: modelId) {
      entry = try await modelPool.activate(modelId: e.id)
    } else {
      throw ModelPoolError.modelNotInPool(modelId)
    }

    // Sync coordinator state from pool entry.
    currentModelFamily = entry.family
    // An image model is now resident and active — clear the video-eviction flag
    // so a later render doesn't redundantly reload (#218).
    imageModelsEvicted = false
    // Track the activated pool model's spec so generation requests use
    // the correct model instead of the startup configuration.modelSpec.
    activePoolModelSpec = entry.modelSpec
    switch entry.family {
    case .krea2:
      krea2Pipeline = entry.box.pipeline as? Krea2Pipeline
      // The pipeline is the physical fact; the pool entry carries the same
      // value back from loadPipeline (WP-E5).
      krea2Variant = krea2Pipeline?.variant ?? (entry.detectedInfo as? Krea2Variant)
      if krea2Pipeline == nil {
        logger.warning(
          "ModelPool: activated krea2 entry '\(entry.id)' but its pipeline could not be read back — publishing an EMPTY LoRA stack (I1)")
      }
    case .chroma:
      chromaPipeline = entry.box.pipeline as? ChromaPipeline
      chromaTokenizer = entry.box.context["tokenizer"] as? ChromaTokenizer
    case .fibo:
      fiboPipeline = entry.box.pipeline as? FiboPipeline
      detectedFiboModel = entry.detectedInfo as? FiboDetectedModel
    case .flux2:
      flux2Pipeline = entry.box.pipeline as? Flux2Pipeline
      detectedFlux2Model = entry.detectedInfo as? Flux2DetectedModel
    case .flux1:
      // Reassign the pipeline so that runSwap and runFlux1Generate
      // operate on the pool-loaded pipeline, not the original one (#138).
      if let poolZImage = entry.box.pipeline as? ZImagePipeline {
        pipeline = poolZImage
        // Pre-load full VAE for the pool-activated pipeline to avoid
        // deadlock on first img2img request (same issue as #141).
        do {
          try poolZImage.prepareFullVAE()
        } catch {
          logger.warning("Failed to pre-load full VAE for pool model '\(entry.modelSpec)': \(error)")
        }
      }
      zimageVariant = (entry.detectedInfo as? ZImageVariant) ?? .turbo
    }
    // K-FIX-1 / Codex I1: the activated pipeline is the authority on what is
    // applied. Reconciling here — BEFORE `publishHealth()` — is what stops
    // `/health.loras` (and the next render's default stack) from describing
    // the model that just left.
    let reconciledLoRAs = PoolAdapterState.reconciled(
      family: entry.family, activated: krea2Pipeline, coordinator: activeLoRAs)
    if reconciledLoRAs.map(\.source.displayName) != activeLoRAs.map(\.source.displayName) {
      let names = reconciledLoRAs.map { $0.source.displayName }.joined(separator: ", ")
      let line = "ModelPool: activation reconciled the LoRA stack from the pipeline read-back — "
        + "\(activeLoRAs.count) advertised → \(reconciledLoRAs.count) applied [\(names)]"
      logger.info("\(line)")
    }
    activeLoRAs = reconciledLoRAs

    revalidateLastRecipe()
    pipelinePrepared = true
    // Model/family/variant just changed — refresh the health snapshot (#217).
    publishHealth()

    return ModelActivateResponse(
      status: "activated",
      model: entry.modelSpec,
      family: entry.family.rawValue
    )
  }

  /// Unload a model from the pool.
  /// QUEUE-INTERNAL (K-FIX-1 / Codex C2). Call this ONLY from inside the
  /// process loop — a queued `.modelOperation`, the ComfyBridge switch's
  /// `enqueueModelSwitch` body, or a render's own #218 reload. A route or
  /// handler that reaches it directly reintroduces the race: actor isolation
  /// does not hold across an await, so the pool's eviction and
  /// `GPU.clearCache()` would be free to run under an active render. From
  /// outside the loop use `enqueueModelOperation` / `enqueueModelOperationDetached`.
  func poolUnload(modelId: String) async throws -> ModelUnloadResponse {
    // Find the entry to get the model spec before unloading.
    guard let entry = await modelPool.findEntry(for: modelId) else {
      throw ModelPoolError.modelNotInPool(modelId)
    }
    let freedMB = try await modelPool.unload(modelId: entry.id)
    return ModelUnloadResponse(
      status: "unloaded",
      model: entry.modelSpec,
      freedMB: freedMB,
      poolSize: await modelPool.count()
    )
  }

  /// List all models in the pool.
  func poolList() async -> ModelPoolListResponse {
    let entries = await modelPool.listPool()
    let activeId = await modelPool.activeModelId()
    let activeSpec: String?
    if let aid = activeId, let entry = await modelPool.findEntry(for: aid) {
      activeSpec = entry.modelSpec
    } else {
      activeSpec = nil
    }
    return ModelPoolListResponse(
      active: activeSpec,
      pool: entries,
      totalVramMB: await modelPool.totalVramMB(),
      budgetMB: await modelPool.budget(),
      // #282: the warm default is only visible from here — `/health.loras`
      // reports what is RESIDENT (the last job's stack), which since #282 is a
      // different question.
      warmDefaultStack: warmDefaultStack.map(LoRAState.init)
    )
  }

  func enqueueGenerate(
    _ payload: GeneratePayload,
    progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil,
    latentPreviewHandler: ZImagePipeline.LatentPreviewHandler? = nil,
    source: String = "api",
    rawBody: Data? = nil,
    jobId: String? = nil
  ) async throws -> GenerateResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    // comfybox#283/#217 (PR #370 review I6): resolve the id and the
    // enqueue timestamp BEFORE constructing the box, so the completion
    // handler (set at CONSTRUCTION — `ContinuationBox.onResume` is `let`)
    // can be built with them.
    let resolvedId = jobId ?? UUID().uuidString
    let enqueuedAt = Date()
    let onResume: @Sendable (Result<GenerateResponse, Error>) -> Void =
      lifecycleCompletionHandler(jobId: resolvedId, jobKind: QueueJobKind.generate.rawValue, source: source, enqueuedAt: enqueuedAt)
    return try await withCheckedThrowingContinuation { continuation in
      // AC-18 (WP-E10 "E9b"): `jobId` is the CLIENT-VISIBLE id when the caller
      // has one (`/v1/generate/async`'s tracker id, or a persisted job's own
      // id on replay), so the queue, its on-disk snapshot, a failed replay
      // and the status route all name the same job. nil → a fresh id (the
      // synchronous route, which never exposes one).
      let newJob = PendingJob(
        id: resolvedId, source: source,
        operation: .generate(payload, ContinuationBox(continuation, onResume: onResume), progressHandler, latentPreviewHandler),
        rawBody: rawBody)
      pending.append(newJob)
      // comfybox#283/#217: read-only — see QueueLifecycleLedger.swift.
      lifecycleLedger.record(jobId: newJob.id, kind: .enqueued, jobKind: QueueJobKind.generate.rawValue, source: source)
      startProcessingIfNeeded()
    }
  }

  func enqueueSwap(_ payload: LoRASwapPayload, rawBody: Data? = nil, jobId: String? = nil) async throws -> LoRASwapResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    let resolvedId = jobId ?? UUID().uuidString
    let enqueuedAt = Date()
    let onResume: @Sendable (Result<LoRASwapResponse, Error>) -> Void =
      lifecycleCompletionHandler(jobId: resolvedId, jobKind: QueueJobKind.loraSwap.rawValue, source: "api", enqueuedAt: enqueuedAt)
    return try await withCheckedThrowingContinuation { continuation in
      // #339 review r3, item 1b: `jobId` lets `recoverPersistedQueue` name
      // this pending job with its ORIGINAL persisted id (mirroring
      // `enqueueGenerate`'s AC-18 `jobId`), so the replay loop can poll for
      // that exact id becoming admitted instead of a fresh random one it
      // has no way to observe. nil (every live route) keeps the default
      // fresh UUID, unchanged.
      let newJob = PendingJob(id: resolvedId, operation: .swap(payload, ContinuationBox(continuation, onResume: onResume)), rawBody: rawBody)
      pending.append(newJob)
      lifecycleLedger.record(jobId: newJob.id, kind: .enqueued, jobKind: QueueJobKind.loraSwap.rawValue, source: newJob.source)
      startProcessingIfNeeded()
    }
  }

  func enqueueControlGenerate(_ request: ZImageControlGenerationRequest) async throws -> GenerateResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    let resolvedId = UUID().uuidString
    let enqueuedAt = Date()
    let onResume: @Sendable (Result<GenerateResponse, Error>) -> Void =
      lifecycleCompletionHandler(jobId: resolvedId, jobKind: QueueJobKind.controlnet.rawValue, source: "api", enqueuedAt: enqueuedAt)
    return try await withCheckedThrowingContinuation { continuation in
      let newJob = PendingJob(id: resolvedId, operation: .controlGenerate(request, ContinuationBox(continuation, onResume: onResume)))
      pending.append(newJob)
      lifecycleLedger.record(jobId: newJob.id, kind: .enqueued, jobKind: QueueJobKind.controlnet.rawValue, source: newJob.source)
      startProcessingIfNeeded()
    }
  }

  /// Execute one ``ModelOperation`` — called ONLY from the process loop, so
  /// the whole mutation (including `ModelPool.load`'s eviction and
  /// `GPU.clearCache()`) is serialized against renders (K-FIX-1 / C2).
  ///
  /// The `poolLoad`/`poolActivate`/`poolUnload` methods stay as the internal
  /// implementation because the loop's OWN jobs call them (a render that must
  /// restore an image model after a video eviction, #218): re-entering the
  /// queue from inside the queue would deadlock. What changed is that no
  /// route reaches them except through here.
  private func runModelOperation(_ op: ModelOperation) async throws -> ModelOperationResult {
    switch op {
    case .load(let modelSpec, let quantization, let activate):
      return .load(try await poolLoad(
        modelSpec: modelSpec, quantization: quantization, activate: activate))
    case .activate(let modelId):
      return .activate(try await poolActivate(modelId: modelId))
    case .unload(let modelId):
      return .unload(try await poolUnload(modelId: modelId))
    }
  }

  /// How many MUTATING pool operations are waiting. Counts only
  /// `.modelOperation` jobs, so parked renders never consume the model-op
  /// budget and vice versa.
  private var pendingModelOperationCount: Int {
    pending.reduce(into: 0) { count, job in
      if case .modelOperation = job.operation { count += 1 }
    }
  }

  /// The model-operation bound (review finding 1). Independent of
  /// `maxPendingRequests` in both directions.
  private func checkModelOperationCapacity() throws {
    if pendingModelOperationCount >= configuration.maxPendingModelOps {
      throw ServerError.modelOperationQueueFull(maxPending: configuration.maxPendingModelOps)
    }
  }

  /// Enqueue a mutating pool operation and wait for it (K-FIX-1 / C2).
  ///
  /// MUST NOT be called from inside the process loop — see `runModelOperation`.
  func enqueueModelOperation(_ op: ModelOperation) async throws -> ModelOperationResult {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    // Not the RENDER capacity gate (WP-E8 hygiene): that gate exists to bound
    // render backlog, and applied here it reproduced the pause wedge one
    // layer up — a paused queue holding `maxPendingRequests` parked renders
    // answered `/v1/model/unload` with `queueFull`, so the operator could not
    // free the GPU precisely when they needed to. Model operations get their
    // OWN cap instead (review finding 1): parked renders never crowd them
    // out, and an unauthenticated client cannot grow the FIFO without limit.
    try checkModelOperationCapacity()

    let resolvedId = UUID().uuidString
    let enqueuedAt = Date()
    let onResume: @Sendable (Result<ModelOperationResult, Error>) -> Void =
      lifecycleCompletionHandler(jobId: resolvedId, jobKind: op.kind, source: "api", enqueuedAt: enqueuedAt)
    return try await withCheckedThrowingContinuation { continuation in
      let newJob = PendingJob(id: resolvedId, operation: .modelOperation(op, ContinuationBox(continuation, onResume: onResume)))
      pending.append(newJob)
      lifecycleLedger.record(jobId: newJob.id, kind: .enqueued, jobKind: op.kind, source: newJob.source)
      startProcessingIfNeeded()
    }
  }

  /// Enqueue a mutating pool operation WITHOUT waiting, returning its FIFO job
  /// id (K-FIX-1 / C2).
  ///
  /// This replaces `/v1/model/load`'s `wait: false` arm, which used to start a
  /// detached `Task` that ran `poolLoad` outside the queue entirely — the
  /// worst version of the race, because nothing in the system knew it was
  /// running. Now it is an ordinary queue job: it appears in `/v1/queue`, it
  /// can be cancelled, and it cannot begin under a render.
  @discardableResult
  func enqueueModelOperationDetached(_ op: ModelOperation) throws -> String {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    // Same reasoning as `enqueueModelOperation` — the model-op cap, not the
    // render capacity gate.
    try checkModelOperationCapacity()
    let job = PendingJob(operation: .modelOperation(op, nil))
    pending.append(job)
    lifecycleLedger.record(jobId: job.id, kind: .enqueued, jobKind: op.kind, source: job.source)
    startProcessingIfNeeded()
    return job.id
  }

  /// Run a Krita model auto-switch through the FIFO render queue so the pool
  /// load/activate executes after any in-flight render finishes instead of
  /// mutating the active pipeline underneath it. The body performs the actual
  /// pool operations and returns whether a switch occurred.
  func enqueueModelSwitch(_ body: @escaping @Sendable () async throws -> Bool) async throws -> Bool {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    let resolvedId = UUID().uuidString
    let enqueuedAt = Date()
    let onResume: @Sendable (Result<Bool, Error>) -> Void =
      lifecycleCompletionHandler(jobId: resolvedId, jobKind: QueueJobKind.modelSwitch.rawValue, source: "api", enqueuedAt: enqueuedAt)
    return try await withCheckedThrowingContinuation { continuation in
      let newJob = PendingJob(id: resolvedId, operation: .modelSwitch(body, ContinuationBox(continuation, onResume: onResume)))
      pending.append(newJob)
      lifecycleLedger.record(jobId: newJob.id, kind: .enqueued, jobKind: QueueJobKind.modelSwitch.rawValue, source: newJob.source)
      startProcessingIfNeeded()
    }
  }

  /// Enqueue a local LTX-2 video generation through the FIFO render queue so
  /// it never runs the GPU concurrently with an image render.
  /// Enqueue a local LTX-2 render on the serial GPU queue. `body` receives a
  /// `report(percent)` callback (0-100) it should call from the generator's
  /// per-chunk/per-step progress hook; the coordinator wires it into the
  /// lock-based progress + health trackers so /health and /v1/queue reflect the
  /// live render without an actor hop (mirrors the image render path, #217).
  func enqueueLocalVideo(
    wantsAudio: Bool = false,
    videoJobId: String? = nil,
    _ body: @escaping @Sendable (@escaping @Sendable (Int) -> Void) throws -> LTX2RenderOutcome
  ) async throws -> LTX2VideoResult {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    let resolvedId = UUID().uuidString
    let enqueuedAt = Date()
    let onResume: @Sendable (Result<LTX2VideoResult, Error>) -> Void =
      lifecycleCompletionHandler(jobId: resolvedId, jobKind: QueueJobKind.video.rawValue, source: "api", enqueuedAt: enqueuedAt)
    return try await withCheckedThrowingContinuation { continuation in
      let newJob = PendingJob(
        id: resolvedId, operation: .localVideo(body, ContinuationBox(continuation, onResume: onResume), wantsAudio: wantsAudio, videoJobId: videoJobId))
      pending.append(newJob)
      // comfybox#283: `videoJobId` (the id `/v1/video/status/{id}` uses) can
      // differ from `newJob.id` (the id `/v1/queue` uses) — pre-existing, not
      // this instrument's concern to unify — so both are recorded when known,
      // via `originalJobId` as a correlation field, so a lifecycle query keyed
      // on either id can still be joined by a human reading both streams.
      lifecycleLedger.record(
        jobId: newJob.id, kind: .enqueued, jobKind: QueueJobKind.video.rawValue, source: newJob.source,
        originalJobId: videoJobId)
      startProcessingIfNeeded()
    }
  }

  func enqueueShutdown() async throws -> ShutdownResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }

    shuttingDown = true
    let resolvedId = UUID().uuidString
    let enqueuedAt = Date()
    let onResume: @Sendable (Result<ShutdownResponse, Error>) -> Void =
      lifecycleCompletionHandler(jobId: resolvedId, jobKind: QueueJobKind.shutdown.rawValue, source: "api", enqueuedAt: enqueuedAt)
    return try await withCheckedThrowingContinuation { continuation in
      let newJob = PendingJob(id: resolvedId, operation: .shutdown(ContinuationBox(continuation, onResume: onResume)))
      pending.append(newJob)
      lifecycleLedger.record(jobId: newJob.id, kind: .enqueued, jobKind: QueueJobKind.shutdown.rawValue, source: newJob.source)
      startProcessingIfNeeded()
    }
  }

  #if DEBUG
  /// 0.B-2 test seam: enqueue a synthetic loop-occupying operation. Mirrors
  /// `enqueueModelSwitch`'s FIFO handling exactly.
  func enqueueSynthetic(durationMs: Int, id: String = UUID().uuidString) async throws -> Bool {
    if shuttingDown { throw ServerError.shuttingDown }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }
    let enqueuedAt = Date()
    let onResume: @Sendable (Result<Bool, Error>) -> Void =
      lifecycleCompletionHandler(jobId: id, jobKind: QueueJobKind.synthetic.rawValue, source: "api", enqueuedAt: enqueuedAt)
    return try await withCheckedThrowingContinuation { continuation in
      let newJob = PendingJob(id: id, operation: .synthetic(durationMs: durationMs, ContinuationBox(continuation, onResume: onResume)))
      pending.append(newJob)
      lifecycleLedger.record(jobId: newJob.id, kind: .enqueued, jobKind: QueueJobKind.synthetic.rawValue, source: newJob.source)
      startProcessingIfNeeded()
    }
  }

  /// comfybox#308 (review r2, item 2b) test seam: drives `finishLocalVideo`
  /// directly — bypassing the queue, the memory-admission gate (real system
  /// probing + up to ~18s of drain sleeps, unsafe to run in a unit test) and
  /// the GPU render — and returns the counters + lastError afterward. Proves
  /// the SAME function all three real `.localVideo` exit points call moves
  /// them correctly for each outcome; deleting the call at one of those exit
  /// points removes that outcome from ever reaching `finishLocalVideo` in
  /// production without changing what this seam proves about the function
  /// itself, which is the most a private, weights/GPU-dependent actor can
  /// offer without a live engine (see intent.md: unit tests only).
  func testSeamFinishLocalVideo(
    _ outcome: LocalVideoCompletionOutcome, lastError message: String? = nil
  ) -> (successCount: Int, failedCount: Int, lastDurationMs: Int?, lastError: String?) {
    finishLocalVideo(outcome, lastError: message)
    return (successfulRenderCount, failedRenderCount, lastRenderDurationMs, lastError)
  }

  /// comfybox#308/#322 (review r3) test seam: drives `handleLocalVideoCatch`
  /// directly against a REAL coordinator — the exact function the
  /// production `.localVideo` generic `catch` calls — so a test can prove a
  /// WRAPPED cancellation (or any error `isRenderInterruption` recognises)
  /// leaves the counters and `lastError` untouched, while a genuine error
  /// still counts as a failed render.
  func testSeamHandleLocalVideoCatch(
    _ error: Error
  ) -> (successCount: Int, failedCount: Int, lastDurationMs: Int?, lastError: String?) {
    handleLocalVideoCatch(error)
    return (successfulRenderCount, failedRenderCount, lastRenderDurationMs, lastError)
  }
  #endif

  /// Publish the current health-relevant state into the lock-based
  /// ``LiveHealthState`` so GET /health reads it without hopping onto this
  /// actor (which blocks for a whole render). Call at every state transition:
  /// job start/end, enqueue, model/LoRA change, pause, shutdown, startup (#217).
  ///
  /// `isRendering` keys off `activeJobId` (set the instant a job is dequeued,
  /// before the render method's first await sets `activeRenderStartedAt`), and
  /// the render start time falls back to `currentJobStartedAt` so the age/stale
  /// signal is correct throughout the synchronous GPU section.
  private func publishHealth() {
    liveHealth.publish(makeHealthSnapshot())
  }

  /// The health snapshot, built but not published — split out (comfybox#362
  /// review r2, item 1) so `publishActiveRenderAndHealth` can hand it to
  /// `LiveHealthState` in the SAME lock acquisition as the interrupt triple.
  private func makeHealthSnapshot() -> HealthSnapshot {
    HealthSnapshot(
      shuttingDown: shuttingDown,
      model: activePoolModelSpec ?? configuration.modelSpec ?? ZImageRepository.id,
      modelFamily: currentModelFamily.rawValue,
      modelVariant: {
        switch currentModelFamily {
        case .fibo: return "fibo"
        case .flux1: return zimageVariant.rawValue
        case .flux2: return detectedFlux2Model?.variant
        case .krea2: return krea2Variant?.rawValue  // "turbo" | "raw" (WP-E5, AC-34b)
        case .chroma: return nil
        }
      }(),
      // WP-E10 "E9b": the declared alias beside the resolved path (AC-34b).
      modelAlias: currentModelFamily == .krea2
        ? (activePoolModelSpec ?? configuration.modelSpec).flatMap { Krea2ModelDetection.alias(forSpec: $0) }
        : nil,
      lastRecipe: lastRecipe,
      loaded: pipelinePrepared,
      loras: activeLoRAs.map(LoRAState.init),
      renderCount: successfulRenderCount,
      failedRenderCount: failedRenderCount,
      pendingCount: pending.count,
      isRendering: activeJobId != nil,
      activeRenderStartedAt: activeRenderStartedAt ?? currentJobStartedAt,
      activeJobId: activeJobId,
      lastRenderDurationMs: lastRenderDurationMs,
      lastError: lastError,
      // isPaused intentionally omitted (§3.1.5): LiveHealthState.read() overlays
      // the AUTHORITATIVE value, so publishHealth no longer writes it.
      activeSummary: activeJobSummary,
      activeSource: activeJobSource,
      pending: pending.map { job in
        QueueJobInfo(
          id: job.id,
          kind: Self.kind(of: job.operation),
          summary: Self.describe(job.operation),
          source: job.source,
          enqueuedAt: job.enqueuedAt
        )
      },
      maxPending: configuration.maxPendingRequests
    )
  }

  /// #339 review r1: set (or clear, with `[]`) the not-yet-admitted tail of
  /// an in-flight persisted-queue replay — see `recoveryUnadmittedTail`.
  /// Persists immediately so the merged snapshot reflects the new tail right
  /// away rather than waiting for the next unrelated mutation.
  func setRecoveryUnadmittedTail(_ tail: [PersistedQueueJob]) {
    recoveryUnadmittedTail = tail
    persistQueueState()
  }

  /// Mirror the recoverable slice of the queue (see QueuePersistence.swift)
  /// to disk so it survives a crash. Called at every mutation: enqueue,
  /// dequeue-into-active, job completion, cancel, reorder, clear. Cheap
  /// (small JSON, atomic write) relative to how rarely the queue actually
  /// changes compared to render duration.
  ///
  /// #339 review r1: merges in `recoveryUnadmittedTail` (the durability fix —
  /// see its doc comment and `RecoverySnapshotMerger`) so a replay in
  /// progress never narrows the file to just what has been admitted so far.
  private func persistQueueState() {
    let active: PersistedQueueJob? = {
      guard let rawBody = activeJobRawBody,
            let kind = activeJobKindForPersistence,
            let id = activeJobId else { return nil }
      return PersistedQueueJob(
        id: id, kind: kind, source: activeJobSource ?? "api",
        enqueuedAt: currentJobStartedAt ?? Date(), rawBody: rawBody)
    }()
    let pendingJobs: [PersistedQueueJob] = pending.compactMap { job in
      guard let rawBody = job.rawBody else { return nil }
      return PersistedQueueJob(
        id: job.id, kind: Self.kind(of: job.operation), source: job.source,
        enqueuedAt: job.enqueuedAt, rawBody: rawBody)
    }
    QueueStateStore.save(RecoverySnapshotMerger.merge(
      admittedActive: active, admittedPending: pendingJobs, unadmittedTail: recoveryUnadmittedTail))
  }

  /// Queue status for the ComfyUI bridge /queue endpoint.
  func queueStatus() -> ComfyBridgeQueueStatus {
    return ComfyBridgeQueueStatus(
      pendingCount: pending.count,
      maxPending: configuration.maxPendingRequests,
      isRendering: activeRenderStartedAt != nil,
      currentJobId: activeJobId,
      progressPercent: progressTracker.get(),
      renderCount: successfulRenderCount,
      failedCount: failedRenderCount
    )
  }

  /// Cancel the in-flight render, if any (ComfyUI /interrupt) — the ASYNC
  /// fallback path (`ControlPlaneSyncFlag` off). Pending jobs are unaffected.
  ///
  /// comfybox#362 review r1, finding 2: this delegates to
  /// `LiveHealthState.cancelActiveRender` — the SAME function the sync path
  /// runs, reading the SAME published triples — rather than re-deriving the
  /// answer from the actor's own `activeJobId`/`activeJobKindForPersistence`.
  /// Those fields are not a second source of truth: `runGenerate`'s `defer`
  /// nils `activeJobId` without republishing, so between a render finishing
  /// and the queue loop's `defer` running they disagreed with the publication,
  /// and the two arms of ONE route answered a job-id target differently. The
  /// actor hop this method still costs is the fallback's whole point (it
  /// queues behind the actor exactly as it did before the 0.B-2 carve-out);
  /// only the decision is shared.
  func cancelActiveRender(target: String? = nil) -> InterruptCancelOutcome {
    liveHealth.cancelActiveRender(target: target)
  }

  /// Clear all pending jobs from the queue. Active job continues.
  func clearPending() -> Int {
    let count = pending.count
    // Cancel all pending continuations with a queue-clear error (distinct
    // from shuttingDown — the server keeps running after a queue clear).
    for job in pending {
      Self.cancel(job.operation)
      // comfybox#283/#217: read-only. `Self.cancel` uses
      // `resumeIgnoringLifecycleHook` for exactly this reason — recorded
      // here explicitly, distinct from `.interrupted` (which stops a job
      // already admitted/running).
      lifecycleLedger.record(jobId: job.id, kind: .dropped, jobKind: Self.kind(of: job.operation), source: job.source, reason: "queue cleared")
    }
    pending.removeAll()
    publishHealth()
    persistQueueState()
    return count
  }

  /// Cancel one pending job by id. Returns false when the id isn't queued
  /// (already running or already finished).
  func cancelPending(id: String) -> Bool {
    guard let index = pending.firstIndex(where: { $0.id == id }) else { return false }
    Self.cancel(pending[index].operation)
    // comfybox#283/#217: read-only — see `clearPending`'s comment above.
    lifecycleLedger.record(
      jobId: id, kind: .dropped, jobKind: Self.kind(of: pending[index].operation), source: pending[index].source,
      reason: "cancelled while pending")
    pending.remove(at: index)
    publishHealth()
    persistQueueState()
    return true
  }

  /// comfybox#283/#217: uses `resumeIgnoringLifecycleHook` (not `resume`) —
  /// see that method's doc comment. `cancel` is called ONLY for jobs still
  /// in `pending` (`clearPending`/`cancelPending`/`drainQueueDeltas`'s
  /// `.cancel` branch), never for the active job, and each of those three
  /// call sites already records an explicit `.dropped` ledger event.
  private static func cancel(_ operation: QueuedOperation) {
    switch operation {
    case .generate(_, let cont, _, _):
      cont.resumeIgnoringLifecycleHook(throwing: ServerError.cancelled)
    case .controlGenerate(_, let cont):
      cont.resumeIgnoringLifecycleHook(throwing: ServerError.cancelled)
    case .swap(_, let cont):
      cont.resumeIgnoringLifecycleHook(throwing: ServerError.cancelled)
    case .modelSwitch(_, let cont):
      cont.resumeIgnoringLifecycleHook(throwing: ServerError.cancelled)
    case .modelOperation(_, let cont):
      // A `wait: false` load has no waiting caller — cancelling it is simply
      // dropping the job (C2).
      cont?.resumeIgnoringLifecycleHook(throwing: ServerError.cancelled)
    case .localVideo(_, let cont, _, _):
      cont.resumeIgnoringLifecycleHook(throwing: ServerError.cancelled)
    case .shutdown(let cont):
      cont.resumeIgnoringLifecycleHook(throwing: ServerError.cancelled)
    #if DEBUG
    case .synthetic(_, let cont):
      cont.resumeIgnoringLifecycleHook(throwing: ServerError.cancelled)
    #endif
    }
  }

  /// One line describing an operation for queue listings.
  private static func describe(_ operation: QueuedOperation) -> String {
    switch operation {
    case .generate(let payload, _, _, _):
      return "Render: \(payload.prompt.prefix(100))"
    case .controlGenerate(let request, _):
      return "ControlNet render: \(request.prompt.prefix(100))"
    case .swap(let payload, _):
      return "LoRA swap (\(payload.loras.count))"
    case .modelSwitch:
      return "Model switch"
    case .modelOperation(let op, _):
      return op.summary
    case .localVideo:
      return "LTX-2 video"
    case .shutdown:
      return "Shutdown"
    #if DEBUG
    case .synthetic(let durationMs, _):
      return "Synthetic op (\(durationMs)ms)"
    #endif
    }
  }

  // #339 review r1, item 2: every case reads from `QueueJobKind`
  // (QueueRecoveryGate.swift) — the single source of truth the recovery
  // gate's allowlist also reads from, so the two cannot silently drift.
  private static func kind(of operation: QueuedOperation) -> String {
    switch operation {
    case .generate: return QueueJobKind.generate.rawValue
    case .controlGenerate: return QueueJobKind.controlnet.rawValue
    case .swap: return QueueJobKind.loraSwap.rawValue
    case .modelSwitch: return QueueJobKind.modelSwitch.rawValue
    case .modelOperation(let op, _): return op.kind
    case .localVideo: return QueueJobKind.video.rawValue
    case .shutdown: return QueueJobKind.shutdown.rawValue
    #if DEBUG
    case .synthetic: return QueueJobKind.synthetic.rawValue
    #endif
    }
  }

  /// One pending entry in a /v1/queue listing.
  struct QueueJobInfo: Sendable {
    let id: String
    let kind: String
    let summary: String
    let source: String
    let enqueuedAt: Date
  }

  // MARK: - Queue controls (pause / resume / reorder)

  func setPaused(_ paused: Bool) {
    // Authority + persistence (the sentinel is the on-disk form of the flag)
    // live in the lock store now (§3.1.5). Idempotent, so this is safe whether or
    // not the sync `/v1/queue/pause` route already wrote it.
    liveHealth.setPaused(paused)
    // The wake (§3.1.4a point 1): resume must still cause a PARKED loop to pick
    // work up — the exact path v1's mailbox `resume` wedged. `resume` reaches
    // here fire-and-forget; only the caller's ACK is decoupled, not this effect.
    if !paused { startProcessingIfNeeded() }
    publishHealth()
  }

  /// Move a pending job within the queue. direction: up | down | top | bottom.
  /// Returns true if the job was found and moved.
  func movePending(id: String, direction: String) -> Bool {
    let moved = reorderPending(id: id, direction: direction)
    if moved { publishHealth(); persistQueueState() }
    return moved
  }

  /// Reorder `pending` in place. Shared by `movePending` (the flag-off async arm)
  /// and `drainQueueDeltas` (a `.move` delta from the sync path) so both use
  /// identical top/bottom/up/down semantics.
  @discardableResult
  private func reorderPending(id: String, direction: String) -> Bool {
    guard let idx = pending.firstIndex(where: { $0.id == id }) else { return false }
    let job = pending.remove(at: idx)
    let target: Int
    switch direction {
    case "top": target = 0
    case "bottom": target = pending.count
    case "up": target = max(0, idx - 1)
    case "down": target = min(pending.count, idx + 1)
    default: pending.insert(job, at: idx); return false
    }
    pending.insert(job, at: target)
    return true
  }

  /// 0.B-2: apply the lock store's undrained deltas against the live `pending`
  /// array at a scheduling point. Called at the top of every `processLoop`
  /// iteration AND from `startProcessingIfNeeded()`, so a delta lands whether the
  /// loop is running or parked (FDD §3.1.4a point 3). A `.cancel` here resumes the
  /// waiting continuation with `.cancelled`, exactly like `cancelPending`; a
  /// `.move` reorders. WAL ordering (F-2, adversarial review): PEEK the deltas,
  /// apply them, persist the canonical queue state, and only THEN commit (drop
  /// the applied deltas and shrink the sidecar). A kill anywhere in that window
  /// leaves the sidecar on disk, and replaying an already-applied cancel over
  /// the persisted state is a no-op — the old take-and-clear-first order let a
  /// kill between the clear and `persistQueueState()` resurrect a cancelled job.
  private func drainQueueDeltas() {
    var commands = liveHealth.peekDeltas()
    var degradedLiveness = false

    if commands.isEmpty {
      // comfybox#386 review round 3, item 1a: `peekDeltas` can come back
      // empty while deltas still exist — the durability marker is stuck
      // behind a previous failed write. Retry once before giving up: the
      // common case (one transient disk hiccup) resolves right here and the
      // drain proceeds normally in this same pass, instead of a cancel
      // silently vanishing from `/v1/queue` (via the ungated
      // `undrainedDeltas`) while never actually reaching the render loop.
      if liveHealth.retryPendingSidecarWrite() {
        commands = liveHealth.peekDeltas()
      }
    }

    if commands.isEmpty {
      // Item 1b: writes are STILL failing. Once they've failed continuously
      // past `LiveHealthState.degradedModeWindowSeconds` /
      // `degradedModeFailureCountThreshold`, apply the non-durable deltas
      // anyway rather than starve a cancel forever behind a broken disk —
      // pre-comfybox#386 had no durability gate at all, so this restores
      // that liveness guarantee, but makes the tradeoff OBSERVABLE via
      // `/health`'s `queue_delta_sidecar_degraded`/`queue_delta_non_durable_count`
      // instead of silent. This trades the WAL guarantee for liveness (a
      // kill right here could resurrect the drained delta) — acceptable only
      // because the sidecar has demonstrably been broken for a while, not on
      // the first failure.
      let status = liveHealth.deltaDurabilityStatus()
      guard status.isDegraded, status.nonDurableCount > 0 else { return }
      commands = liveHealth.undrainedDeltas()
      degradedLiveness = true
    }

    guard !commands.isEmpty else { return }

    for command in commands {
      // Structural guard against the F1 wedge (§3.1.4a): a mailbox delta must
      // never require a wake. `resume` is fire-and-forget, not a delta. (The type
      // makes `requiresWake: true` unconstructable; this asserts it at the drain
      // too, so a future factory that sets it fails here in test.)
      assert(!command.requiresWake,
             "queue delta must not require a wake — resume is fire-and-forget (FDD §3.1.4a)")
      switch command.kind {
      case .cancel(let id):
        if let index = pending.firstIndex(where: { $0.id == id }) {
          Self.cancel(pending[index].operation)
          // comfybox#283/#217: read-only — the sync-control-plane twin of
          // `cancelPending`'s own `.dropped` record above.
          lifecycleLedger.record(
            jobId: id, kind: .dropped, jobKind: Self.kind(of: pending[index].operation), source: pending[index].source,
            reason: degradedLiveness
              ? "cancelled while pending (sync control plane, degraded — sidecar unwritable)"
              : "cancelled while pending (sync control plane)")
          pending.remove(at: index)
        }
      case .move(let id, let direction):
        _ = reorderPending(id: id, direction: direction)
      }
    }
    publishHealth()
    persistQueueState()
    #if DEBUG
    QueueDeltaStore.drainCrashWindowHook?()
    #endif
    liveHealth.commitDrainedDeltas(commands.count)
  }

  /// Best-effort prompt drain nudged by the sync cancel/move/clear routes so a
  /// delta applies quickly when the pool is healthy (the actor is free during a
  /// render — the render runs in a child task). If the nudge cannot run (pool
  /// exhausted), the delta still applies at the next real scheduling point; the
  /// composed `GET /v1/queue` reflects it immediately regardless.
  func drainControlDeltas() { drainQueueDeltas() }

  private func startProcessingIfNeeded() {
    // 0.B-2 drain point 2/2 (FDD §3.1.4a point 3): apply undrained deltas here
    // too, so a delta lands even when the loop is PARKED (this is the only path
    // that restarts a parked loop). Runs before publishHealth so the published
    // pending count reflects the applied deltas.
    drainQueueDeltas()
    // Every enqueue routes through here, so this is the one spot that reflects a
    // just-changed pending count into the lock-based health snapshot (#217).
    publishHealth()
    persistQueueState()
    guard !isProcessing else { return }
    isProcessing = true
    Task {
      await processLoop()
    }
  }

  /// Whether a queued operation may run while the queue is PAUSED
  /// (K-FIX-1 round 2, New-1).
  ///
  /// "Pause" means *no renders* — it is what Todd uses to free the GPU for a
  /// deploy, and it persists across restarts via the sentinel. Mutating model
  /// operations are exactly what an operator needs during that window
  /// (swap the resident checkpoint, unload to free memory), and they were
  /// available before this wave because the routes called the pool directly.
  /// Routing them through the FIFO (C2) must not take that away: the FIFO's
  /// job is to serialise them against an in-flight render, which it still
  /// does — the loop runs one job at a time either way.
  ///
  /// Exhaustive on purpose (no `default`): a new queue kind must decide.
  ///
  /// `.shutdown` joins them (WP-E8 hygiene). `enqueueShutdown` sets
  /// `shuttingDown = true` BEFORE appending, so a shutdown parked behind the
  /// pause gate wedged the engine permanently: the caller's continuation
  /// never resumed and every later enqueue threw `.shuttingDown`, leaving
  /// SIGKILL as the only recovery — on exactly the paused engine an operator
  /// is trying to shut down. Its handler is a bare continuation resume with
  /// no GPU work, so nothing about "pause means no RENDERS" argues for
  /// parking it.
  private static func runsWhilePaused(_ operation: QueuedOperation) -> Bool {
    switch operation {
    case .modelOperation, .shutdown:
      return true
    case .generate, .controlGenerate, .swap, .modelSwitch, .localVideo:
      return false
    #if DEBUG
    case .synthetic:
      return false
    #endif
    }
  }

  /// comfybox#283/#217: the closure every `enqueue*` method passes to
  /// `ContinuationBox`'s `onResume` AT CONSTRUCTION time (PR #370 review
  /// I6 — `onResume` is `let`, so it must be supplied when the box is
  /// built, not mutated in afterward from `processLoop`), so every queued
  /// job kind's ACTUAL terminal outcome — completed, failed, or interrupted
  /// — is recorded exactly once, from the one place every job's
  /// continuation already resumes exactly once (`ContinuationBox.resume`).
  /// `durationMs` measures from ENQUEUE, not from admission/render-start —
  /// the box is built at enqueue time, before a job's `activeRenderStartedAt`
  /// exists — so it is queue-wait-plus-render latency, a different (also
  /// useful) number from `/health.last_render_duration_ms`.
  ///
  /// Interrupted vs failed uses the SAME `isRenderInterruption` classifier
  /// `VideoJobTracker.markFailed`/`localVideoCatchOutcome` already apply to
  /// the same errors (module-level func, this file) — this ledger never
  /// invents a second opinion about what counts as an operator interrupt —
  /// plus `ServerError.cancelled` (`clearPending`/`cancelPending`'s own
  /// cancellation error) for the render-already-admitted analogue of a
  /// pending job's `.dropped`.
  ///
  /// Generic over the continuation's `Value` so one function serves every
  /// queue-job kind's differently-typed `ContinuationBox<Value>`.
  private func lifecycleCompletionHandler<T>(
    jobId: String, jobKind: String, source: String, enqueuedAt: Date
  ) -> @Sendable (Result<T, Error>) -> Void {
    let ledger = lifecycleLedger
    return { result in
      let durationMs = Int(Date().timeIntervalSince(enqueuedAt) * 1000)
      switch result {
      case .success:
        ledger.record(jobId: jobId, kind: .completed, jobKind: jobKind, source: source, durationMs: durationMs)
      case .failure(let error):
        let isInterrupt: Bool
        if case ServerError.cancelled = error { isInterrupt = true } else { isInterrupt = isRenderInterruption(error) }
        ledger.record(
          jobId: jobId, kind: isInterrupt ? .interrupted : .failed, jobKind: jobKind, source: source,
          reason: error.localizedDescription)
      }
    }
  }

  private func processLoop() async {
    while true {
      // 0.B-2 drain point 1/2 (FDD §3.1.4a point 3): apply undrained deltas at
      // the top of every iteration — whether this iteration runs a job or parks,
      // and crucially BEFORE dequeuing, so a cancelled job never runs.
      drainQueueDeltas()
      // Paused: renders stay parked, but a model operation still runs (New-1).
      // Picking it out of the middle does not reorder anything that runs: the
      // jobs it passes are parked until resume, and they keep their relative
      // order for when it comes.
      let index: Int
      if liveHealth.isPausedAuthoritative() {
        guard let next = pending.firstIndex(where: { Self.runsWhilePaused($0.operation) }) else {
          isProcessing = false
          return
        }
        index = next
      } else {
        guard !pending.isEmpty else {
          isProcessing = false
          return
        }
        index = 0
      }

      let job = pending.remove(at: index)
      activeJobSummary = Self.describe(job.operation)
      activeJobSource = job.source
      // Keep the same id the job had while pending, so clients can correlate.
      activeJobId = job.id
      currentJobStartedAt = Date()
      // Move the job's recoverable data (if any) from "pending" to "active" in
      // the persisted queue snapshot, so a crash mid-render still recovers it
      // (replayed from scratch on restart — there's no way to resume a
      // partial diffusion render, so "at least once" is the correctness goal).
      activeJobRawBody = job.rawBody
      activeJobKindForPersistence = Self.kind(of: job.operation)
      // comfybox#283/#217: read-only lifecycle telemetry — dequeuing into the
      // active slot is exactly the "admitted" transition #283 finding 2 shows
      // no instrument reports (`queue_remaining` only ever counts `pending`).
      lifecycleLedger.record(jobId: job.id, kind: .admitted, jobKind: activeJobKindForPersistence, source: job.source)
      // Publish is_rendering + job id BEFORE the synchronous GPU section begins,
      // so /health reflects the render for its whole (actor-blocking) duration (#217).
      publishHealth()
      persistQueueState()
      defer {
        activeJobSummary = nil; activeJobSource = nil; activeJobId = nil; currentJobStartedAt = nil
        activeJobRawBody = nil; activeJobKindForPersistence = nil
        publishHealth()
        persistQueueState()
      }
      switch job.operation {
      case .generate(let payload, let continuation, let progressHandler, let latentPreviewHandler):
        // Run the render in a retained child task so /interrupt can cancel it
        // without cancelling the queue's processing loop.
        let renderTask = Task {
          await self.runGenerate(payload, continuation: continuation, progressHandler: progressHandler, latentPreviewHandler: latentPreviewHandler)
        }
        publishActiveRender(task: renderTask)  // 0.B-2: sync /interrupt handle
        await renderTask.value
        publishActiveRender(task: nil)
      case .controlGenerate(let request, let continuation):
        let renderTask = Task {
          await self.runControlGenerate(request, continuation: continuation)
        }
        publishActiveRender(task: renderTask)  // 0.B-2: sync /interrupt handle
        await renderTask.value
        publishActiveRender(task: nil)
      #if DEBUG
      case .synthetic(let durationMs, let continuation):
        // 0.B-2 test seam (FDD §4.1). Runs through the retained-task path (so
        // /interrupt can cancel it) and BLOCKS its thread for the duration —
        // occupying a worker exactly like a real synchronous render, which is
        // what makes the "sync control plane answers with zero cooperative
        // threads" test meaningful.
        // comfybox#283/#217: read-only — gives this DEBUG-only kind a real
        // `.started` event too, so a lifecycle test can drive a full
        // enqueued→admitted→started→completed cycle through the REAL
        // coordinator without any model weights.
        lifecycleLedger.record(jobId: job.id, kind: .started, jobKind: QueueJobKind.synthetic.rawValue, source: job.source)
        let renderTask = Task {
          if !Task.isCancelled {
            Thread.sleep(forTimeInterval: Double(durationMs) / 1000.0)
          }
          continuation.resume(returning: !Task.isCancelled)
        }
        publishActiveRender(task: renderTask)
        await renderTask.value
        publishActiveRender(task: nil)
      #endif
      case .swap(let payload, let continuation):
        await runSwap(payload, continuation: continuation)
      case .modelSwitch(let body, let continuation):
        do {
          continuation.resume(returning: try await body())
        } catch {
          continuation.resume(throwing: error)
        }
      case .modelOperation(let op, let continuation):
        // C2: the pool mutation runs HERE, with the loop holding the queue —
        // no render can be dequeued until it returns.
        do {
          let result = try await runModelOperation(op)
          continuation?.resume(returning: result)
          // comfybox#283/#217: a `wait: false` load has no `ContinuationBox`
          // at all (`enqueueModelOperationDetached` never constructs one) —
          // record the completion directly so THIS kind's success is not
          // silently invisible to the ledger the way it used to be to
          // /health.
          if continuation == nil {
            lifecycleLedger.record(jobId: job.id, kind: .completed, jobKind: op.kind, source: job.source)
          }
        } catch {
          if let continuation {
            continuation.resume(throwing: error)
          } else {
            // A `wait: false` load has no caller left to throw to; it is
            // recorded the same way a failed render is, so /health's
            // last_error names it instead of it vanishing into a log line.
            failedRenderCount += 1
            lastError = "\(op.summary) failed: \(error.localizedDescription)"
            logger.error("Queued model operation failed — \(op.summary): \(error.localizedDescription)")
            lifecycleLedger.record(
              jobId: job.id, kind: .failed, jobKind: op.kind, source: job.source, reason: error.localizedDescription)
          }
        }
      case .localVideo(let body, let continuation, let wantsAudio, let videoJobId):
        // Runs on the serial queue so LTX-2 never shares the GPU with a render.
        activeRenderStartedAt = Date()
        // comfybox#283/#217: read-only — the synchronous GPU section begins here.
        lifecycleLedger.record(jobId: job.id, kind: .started, jobKind: QueueJobKind.video.rawValue, source: job.source)
        // activeJobId is set from job.id at the top of the loop.
        defer { activeRenderStartedAt = nil; activeJobId = nil }
        // Stream render progress into the lock-based trackers /health + /queue
        // read, exactly like the image path. Both trackers are Sendable, so the
        // off-actor @Sendable report closure can update them without an actor
        // hop. Cleared on completion via the defer inside the task.
        let progress = self.progressTracker
        let health = self.liveHealth
        // comfybox#283/#217: read-only, bounded-rate progress ticks. Percent
        // only — the richer chunk/step numbers are collapsed into `pct`
        // before they reach this closure (`Self.localVideoProgressPercent`,
        // called by the route-level closures that wrap `body`); the video
        // job's own checkpoint/resume events (`runPreemptionEpisode`) carry
        // the real chunk/step/phase instead.
        let ledger = self.lifecycleLedger
        let videoJobIdForProgress = job.id
        let report: @Sendable (Int) -> Void = { pct in
          progress.set(pct)
          health.setProgress(pct)
          ledger.record(jobId: videoJobIdForProgress, kind: .progress, jobKind: QueueJobKind.video.rawValue, percent: pct)
        }
        // comfybox#322: run the render in a RETAINED child task and publish it,
        // exactly as `.generate` / `.controlGenerate` do above. Before this,
        // `.localVideo` was the one render case that never published a handle,
        // so `activeRenderTask` stayed nil for the whole 5-60 minute clip and
        // `/v1/queue/interrupt` answered `interrupted: false` — the interrupt
        // could only stop the NEXT queue item. The LTX-2 loops
        // (`LTX2LoopBoundary` / `Task.checkCancellation`) observe the
        // cancellation this handle delivers.
        //
        // The task inherits this actor's isolation, so the render is still
        // serialized on the coordinator exactly as before; `await
        // renderTask.value` keeps the queue loop parked until it finishes.
        //
        // comfybox#322 (review r1, Important): the #218 admission gate lives
        // INSIDE this task, not before it. Hoisted out, `await renderTask.value`
        // suspended the actor between the gate and the weight load — a
        // reentrancy window in which another actor-isolated caller (an image
        // route's `poolLoad`, a queued model operation) could change residency
        // after the gate had passed and before the ~65GB stack loaded, which is
        // the documented #218/#34 SIGKILL condition. Gate and load now sit in
        // one uninterrupted synchronous run of this task: nothing suspends
        // between `vacateImageModelsAndAdmitVideo` returning and `body(report)`
        // being entered.
        let renderTask = Task {
          // An interrupt that arrived while this job waited its turn: refuse
          // before evicting every image model for a render nobody wants.
          if Task.isCancelled {
            self.logger.info("LTX-2: render interrupted before admission — nothing loaded, nothing evicted.")
            continuation.resume(throwing: WarmServerError.renderInterrupted)
            return
          }
          // #218: single-heavy-model residency. Right before the ~65GB LTX-2
          // stack loads inside body(), vacate ALL image models (pool +
          // per-family pipelines), then verify there is enough physical RAM to
          // proceed — refuse cleanly instead of OOM-killing the whole process.
          // Doing this on the serial render queue guarantees no image render
          // can re-load between the eviction and the video load. Extracted into
          // `vacateImageModelsAndAdmitVideo` (#1479, review C2) so a preemption
          // resume can run the EXACT SAME gate before resuming — the preempting
          // image job loaded its own weights while the video was evicted, and
          // resuming into whatever memory is left without re-checking is the
          // documented SIGKILL condition this gate exists to prevent.
          let admission = await self.admitVideoForRender(wantsAudio: wantsAudio)
          guard admission.admitted else {
            let message = "Insufficient memory for LTX-2 video: only \(admission.availableMB)MB free after evicting image models (need ~\(admission.neededMB)MB)"
            // comfybox#308: this job DID reach the front of the queue and DID
            // fail to run — a real completion, not a queue-full rejection
            // (those throw before ever dequeuing, same as the image path).
            self.finishLocalVideo(.admissionRefused, lastError: message)
            continuation.resume(throwing: WarmServerError.invalidRequest(message: message))
            return
          }
          self.videoHolder.beginRender()
          progress.set(0)
          health.setProgress(0)
          defer {
            self.videoHolder.endRender()
            progress.set(nil)
            health.setProgress(nil)
            self.ltx2StepPosition.clear()
            // #1479 (review C1): covers EVERY non-yielding exit from this
            // render's whole execution (initial body() completing/throwing,
            // a storyboard `.generate()` shot that never checks the signal
            // at all, or the `.yielded` loop below ending in `.completed`/
            // throw) — if a preempt raised the signal but the render never
            // actually checkpointed on it, a stranded raise would otherwise
            // hit the NEXT render's pre-load unwind point instantly (near-
            // zero-cost false checkpoint, bogus evict/reload sample). Clear
            // unconditionally; clearing an unraised signal is a no-op.
            self.ltx2PreemptionSignal.clear()
          }
          do {
            // #1479: body() may hand back a checkpoint instead of a finished
            // clip (a `preempt: true` image job raised the signal). Loop
            // rather than recurse: a resume can itself yield again (a second,
            // later preemption), and each iteration is handled identically.
            var outcome = try body(report)
            while case .yielded(let state) = outcome {
              outcome = try await self.runPreemptionEpisode(state: state, videoJobId: videoJobId, wantsAudio: wantsAudio, report: report)
            }
            if case .completed(let result) = outcome {
              // comfybox#308: the `.localVideo` completion path never touched
              // `/health`'s render counters — only the six image `run*Generate`
              // methods did. Every video render (HQ two-pass included) was
              // invisible to `render_count`/`last_render_duration_ms` no
              // matter how many completed.
              self.finishLocalVideo(.succeeded(elapsedSeconds: result.elapsedSeconds), lastError: nil)
              continuation.resume(returning: result)
            }
          } catch is CancellationError {
            // comfybox#322: an operator interrupt, not a failure. Named so
            // the client sees why the render stopped instead of the opaque
            // "CancellationError()" Todd reported on the 2026-08-30 incident.
            // `VideoJobTracker.markFailed` recognises this case and reports
            // the job interrupted rather than failed. Deliberately NOT routed
            // through `recordRenderCompletion`/`failedRenderCount` — an
            // operator interrupt is not a render failure (comfybox#322's own
            // framing); there is no separate "interrupted" health counter.
            self.logger.info("LTX-2: render interrupted by /v1/queue/interrupt.")
            continuation.resume(throwing: WarmServerError.renderInterrupted)
          } catch {
            // comfybox#308/#322 (review r3): a WRAPPED cancellation (e.g.
            // ModelPoolError.loadFailed on a resume's model reload) lands
            // here — `is CancellationError` above only matches the bare
            // case. `handleLocalVideoCatch` re-classifies with the SAME
            // `isRenderInterruption` check `VideoJobTracker.markFailed`
            // uses on this same error, so an interrupt is never
            // double-counted as a failed render.
            self.handleLocalVideoCatch(error)
            continuation.resume(throwing: error)
          }
        }
        // comfybox#362 review r1, finding 3: publish the `/v1/video/status/{id}`
        // id alongside the queue id, so `target: <id>` matches whichever of
        // the two a client holds (comfybox#283: they differ).
        publishActiveRender(task: renderTask, statusJobId: videoJobId)
        await renderTask.value
        publishActiveRender(task: nil)
      case .shutdown(let continuation):
        continuation.resume(
          returning: ShutdownResponse(
            success: true,
            message: "Server shutdown requested"
          )
        )
      }
    }
  }

  private func runGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>, progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil, latentPreviewHandler: ZImagePipeline.LatentPreviewHandler? = nil) async {
    // #282: `loraStackOrigin` is stamped on the payload below, once the job's
    // own stack has been resolved, so every family's response can report it.
    var payload = payload
    // Queue telemetry: tag this render with a job id and stream denoising
    // progress into the tracker that queueStatus() reads. Cleared on return
    // (success or failure) via defer. flux1 forwards the wrapped handler so the
    // pipeline's per-step callback updates progress; other families currently
    // have no per-step callback, so they report only is_rendering + job id.
    // (activeJobId is set from job.id at the top of the process loop.)
    progressTracker.set(0)
    liveHealth.setProgress(0)
    let tracker = progressTracker
    // Feed the lock-based health snapshot too, so /health's progress_percent
    // advances live during the render without an actor hop (#217).
    let health = liveHealth
    // comfybox#283/#217: read-only, bounded-rate progress ticks — captured as
    // plain values (never `self`) since this closure is `@Sendable` and can
    // run off-actor. The ledger itself throttles `.progress` events, so this
    // call site does not need its own rate limit.
    let ledger = lifecycleLedger
    let jobId = activeJobId
    let trackedHandler: @Sendable (ZImagePipeline.GenerationProgress) -> Void = { progress in
      if progress.stage == .denoising {
        let pct = Int(progress.fractionCompleted * 100)
        tracker.set(pct)
        health.setProgress(pct)
        if let jobId {
          ledger.record(
            jobId: jobId, kind: .progress, jobKind: QueueJobKind.generate.rawValue,
            step: progress.stepIndex, totalSteps: progress.totalSteps, percent: pct)
        }
      }
      progressHandler?(progress)
    }

    // Live denoising preview (GH #216): approximate each step's latents as a
    // small JPEG and stash it for REST/polling clients (Desktop already
    // polls /health for progress_percent — this rides the same cadence).
    // Krita/ComfyUI still get their own frame pushed via the bridge
    // WebSocket, forwarded first so that behavior is unchanged.
    let preview = previewTracker
    let trackedPreviewHandler: ZImagePipeline.LatentPreviewHandler = { latents, step, total, latentH, latentW in
      latentPreviewHandler?(latents, step, total, latentH, latentW)
      #if canImport(CoreGraphics)
      guard step > 0, step % Self.previewStepInterval == 0, step < total else { return }
      guard let approx = LatentPreviewApproximator.latentsToRGBA(latents, latentHeight: latentH, latentWidth: latentW) else { return }
      guard let framed = ComfyBridgePreviewEncoder.encodePreviewFrame(
        fromRGBA: approx.data, width: approx.width, height: approx.height,
        maxDimension: Self.previewMaxDimension, jpegQuality: Self.previewJPEGQuality
      ) else { return }
      // Strip the 8-byte ComfyUI WebSocket binary-frame header — REST
      // clients just want the raw JPEG bytes.
      preview.set(framed.dropFirst(8))
      #endif
    }
    defer { activeJobId = nil; progressTracker.set(nil); liveHealth.setProgress(nil); previewTracker.set(nil) }

    // #218: if a prior LTX-2 video render evicted the image models, restore the
    // previously-active image model before this render can run. poolLoad also
    // releases any resident video stack, so image and video stay mutually
    // exclusive. Runs before the per-job model/LoRA application below.
    do {
      try await reloadImageModelIfEvicted(requestedModel: payload.model)
    } catch {
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
      return
    }

    // Per-job model/LoRA application (queue-submit race fix): a job's own
    // model+loras are applied right before it runs, instead of trusting
    // whatever the shared pool's "currently active" model/LoRAs happen to
    // be by the time it dequeues — a plain synchronous /v1/generate caller
    // activates the model right before calling generate, but async
    // queue-submit (POST /v1/generate/async) can dequeue well after a
    // different request has changed global state. nil model/loras preserve
    // the old caller-activates-first behavior exactly.
    if let modelSpec = payload.model, !modelSpec.isEmpty {
      let resolvedSpec = WarmServer.parseModelSpec(from: modelSpec)
      let currentSpec = activePoolModelSpec ?? configuration.modelSpec
      if resolvedSpec != currentSpec {
        let resolvedQuant = WarmServer.parseQuantization(from: modelSpec)
        do {
          _ = try await poolLoad(modelSpec: resolvedSpec, quantization: resolvedQuant, activate: true)
        } catch {
          lastError = error.localizedDescription
          continuation.resume(throwing: error)
          return
        }
      }
    }
    // #282: EVERY render applies its own resolved stack here — the request's
    // `loras`, else the named preset's expansion (#286 put it in the same
    // field), else the WARM DEFAULT that `/v1/lora/swap` publishes. Before
    // this, a request that carried neither rendered on whatever the previous
    // job or an hours-old swap had left resident, which is the crosstalk this
    // ticket exists to end. The application itself is unchanged
    // (`applyActiveLoRAs`, with #286's same-stack shortcut), so a run of jobs
    // that all want the same stack still re-binds nothing.
    do {
      let plan = try resolveJobLoRAStack(payload)
      payload.loraStackOrigin = plan.origin.rawValue
      payload.warmDefaultSkipped = plan.warmDefaultSkipped
      let reloaded = try await applyJobLoRAStack(plan)
      if reloaded { payload.loraReload = true }
    } catch {
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
      return
    }
    #if DEBUG
    // Review r1 (I4): with the recorder installed the job stops here — the
    // family render below would resolve (and download) model weights, which no
    // unit test may do on this machine.
    if testSeamStackRecorder != nil {
      continuation.resume(throwing: WarmServerError.invalidRequest(
        message: "#282 test seam: job stack recorded, render not dispatched"))
      return
    }
    #endif

    // #154 (review r2, minor 4): resolve the recipe NAMES once, HERE, and let
    // their own 400 fire before any gate reads them.
    //
    // `decodedGeneratePayload` already validated them, so in practice this
    // cannot throw — but a `try?` would silently WIDEN the shift gate below on
    // an unresolvable schedule name (nil is read as `.flow`, which honours the
    // shift), which is the exact failure these gates exist to prevent. The
    // family-capability gate further down reuses this value for the same
    // reason: it used to `try?` and SKIP itself on a name it could not resolve.
    let recipeNames: ResolvedRecipeNames
    do {
      recipeNames = try payload.validateRecipeNames()
    } catch {
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
      return
    }

    // D3 `shift`: refuse before dispatch so no family can silently ignore it.
    if let message = GeneratePayload.validateShift(payload.shift, family: currentModelFamily) {
      lastError = message
      continuation.resume(throwing: WarmServerError.invalidRequest(message: message))
      return
    }
    // #154: and refuse a shift the requested SCHEDULE would drop — otherwise
    // `applied_shift` would report a number that never reached the grid.
    if let message = GeneratePayload.validateShiftSchedule(
      payload.shift,
      sigmaSchedule: recipeNames.sigmaSchedule,
      family: currentModelFamily) {
      lastError = message
      continuation.resume(throwing: WarmServerError.invalidRequest(message: message))
      return
    }
    // WP-E10 "E9b": `vae` on a non-krea2 family is refused, never ignored.
    if let error = GeneratePayload.vaeGate(payload.vae, family: currentModelFamily) {
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
      return
    }
    // WP-E17 (§3.14, D4): `stage2` on a family with no second-stage seam, an
    // out-of-range `stage2.denoise`, and the tool schema's `detail_pass` /
    // `detail_denoise` spelling — all 400 here, never a silently single-stage
    // render.
    if let error = GeneratePayload.stage2Gate(payload, family: currentModelFamily) {
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
      return
    }

    // I5: the family capability gate. The name resolved at decode (unknown
    // names are already 400 by then); whether THIS family's loop can honour
    // it is decided here, from the one matrix (D18: family gates at dispatch).
    // Reads the `recipeNames` resolved above rather than re-running a `try?`
    // that would skip this gate entirely on a name it could not resolve.
    if let error = GeneratePayload.validateFamilyRecipe(recipeNames, family: currentModelFamily) {
      lastError = error.localizedDescription ?? "unsupported sampler"
      continuation.resume(throwing: error)
      return
    }

    switch currentModelFamily {
    case .chroma:
      await runChromaGenerate(payload, continuation: continuation)
    case .fibo:
      await runFiboGenerate(payload, continuation: continuation)
    case .krea2:
      await runKrea2Generate(payload, continuation: continuation)
    case .flux2:
      await runFlux2Generate(payload, continuation: continuation)
    case .flux1:
      await runFlux1Generate(payload, continuation: continuation, progressHandler: trackedHandler, latentPreviewHandler: trackedPreviewHandler)
    }
  }

  /// How often (in denoising steps) to refresh the live preview frame.
  private static let previewStepInterval = 2
  private static let previewMaxDimension = 256
  private static let previewJPEGQuality: CGFloat = 0.6

  /// The latest live-denoising preview JPEG, if a render is active and has
  /// produced at least one frame. Served by GET /v1/generate/preview.
  func latestPreviewFrame() -> Data? {
    previewTracker.get()
  }

  private func runFlux1Generate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>, progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil, latentPreviewHandler: ZImagePipeline.LatentPreviewHandler? = nil) async {
    activeRenderStartedAt = Date()
    // comfybox#283/#217: read-only — the synchronous GPU section begins here.
    if let activeJobId { lifecycleLedger.record(jobId: activeJobId, kind: .started, jobKind: QueueJobKind.generate.rawValue, source: activeJobSource) }
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runFlux1Generate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "Flux1 generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "Flux1 generation failed unexpectedly"))
      }
    }

    // When a pool model is active, override configuration.modelSpec so
    // that generateCore loads/validates the pool model, not the startup model.
    let effectiveConfig: WarmServerConfiguration
    if let poolSpec = activePoolModelSpec, poolSpec != configuration.modelSpec {
      var cfg = configuration
      cfg.modelSpec = poolSpec
      effectiveConfig = cfg
    } else {
      effectiveConfig = configuration
    }

    do {
      let outputURL: URL
      // #282: `activeLoRAs` here is THIS job's stack — `runGenerate` resolved
      // and applied it a few lines before dispatching, so the request carries
      // what the pipeline is holding rather than "whatever is resident".
      if payload.imagePath != nil {
        let img2imgRequest = try payload.makeImg2ImgRequest(
          configuration: effectiveConfig,
          activeLoRAs: activeLoRAs
        )
        outputURL = try await pipeline.generateImg2Img(img2imgRequest, progressHandler: progressHandler)
      } else {
        let request = try payload.makePipelineRequest(
          configuration: effectiveConfig,
          activeLoRAs: activeLoRAs
        )
        outputURL = try await pipeline.generateFromRequest(request, progressHandler: progressHandler, latentPreviewHandler: latentPreviewHandler)
      }
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      resumed = true
      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: outputURL.path,
          durationMs: durationMs,
          appliedLoras: appliedLoRAStates(),
          presetUnresolved: payload.presetUnresolved,
          presetUnresolvedReason: payload.presetUnresolvedReason,
          presetStackMismatch: payload.presetStackMismatch,
          memoryEstimateBytes: payload.memoryEstimateBytes, memoryAvailableBytes: payload.memoryAvailableBytes,
          loraStackOrigin: payload.loraStackOrigin,
          warmDefaultSkipped: payload.warmDefaultSkipped,
          loraReload: payload.loraReload,
          // #154: echo the explicit ModelSamplingAuraFlow shift this render
          // applied — from the request or from its named preset. nil (the
          // key absent) means the model's own schedule ran, as before #154.
          appliedShift: payload.shift
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  private func runFlux2Generate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    // comfybox#283/#217: read-only — the synchronous GPU section begins here.
    if let activeJobId { lifecycleLedger.record(jobId: activeJobId, kind: .started, jobKind: QueueJobKind.generate.rawValue, source: activeJobSource) }
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runFlux2Generate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "Flux2 generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "Flux2 generation failed unexpectedly"))
      }
    }

    do {
      guard let f2 = flux2Pipeline else {
        throw WarmServerError.flux2NotLoaded
      }

      let outputURL: URL
      outputURL = try payload.resolvedOutputURL(
        configuration: configuration,
        defaultFilename: ComfyBoxOutputNaming.defaultFilename(
          modelSpec: activePoolModelSpec ?? "flux2", presetId: payload.preset,
          contentMode: payload.contentMode, source: payload.source)
      )

      // Map GeneratePayload fields to Flux2GenerationRequest.
      // Base models: 50 steps, guidance configurable.
      // Distilled models: 4 steps, guidance 1.0.
      let defaultSteps = f2.defaultSteps
      let defaultGuidance: Float = f2.isDistilled ? 1.0 : 3.5
      // Resolve img2img parameters from payload.
      // imagePath takes priority; denoise defaults to 1.0 (txt2img).
      // imageStrength maps to denoise as (1.0 - strength), creativity maps directly.
      let inputImageURL: URL? = payload.imagePath.map { URL(fileURLWithPath: $0) }
      let resolvedDenoise: Float
      if inputImageURL != nil {
        if let creativity = payload.creativity {
          resolvedDenoise = max(0.01, min(1.0, creativity))
        } else if let strength = payload.imageStrength {
          resolvedDenoise = max(0.01, min(1.0, 1.0 - strength))
        } else if let d = payload.denoise {
          resolvedDenoise = max(0.01, min(1.0, d))
        } else {
          resolvedDenoise = 0.7  // sensible default for img2img
        }
      } else {
        resolvedDenoise = 1.0
      }

      // FDD §3.3, D3: config-layer render defaults for flux2. Only width/height
      // are seeded on first-run migration — `defaultSteps`/`defaultGuidance`
      // above are already checkpoint-dependent (base vs. distilled), so an
      // explicit config override is honoured when present but nothing is
      // frozen into config.json for them (see ServerConfigStore.engineSeed).
      let flux2ConfigDefaults = ServerConfigStore.shared.renderDefaults(family: "flux2")
      let flux2Request = Flux2GenerationRequest(
        prompt: payload.prompt,
        negativePrompt: payload.negativePrompt,
        width: payload.width ?? flux2ConfigDefaults.width ?? 1024,
        height: payload.height ?? flux2ConfigDefaults.height ?? 1024,
        steps: payload.steps ?? flux2ConfigDefaults.steps ?? defaultSteps,
        guidanceScale: payload.guidance ?? flux2ConfigDefaults.guidance.map(Float.init) ?? defaultGuidance,
        seed: payload.seed,
        outputPath: outputURL,
        levelsMin: payload.levelsMin ?? 0.0,
        levelsMax: payload.levelsMax ?? 1.0,
        maxSequenceLength: configuration.maxSequenceLength,
        inputImagePath: inputImageURL,
        denoise: resolvedDenoise,
        contentMode: payload.contentMode
      )

      let result = try await f2.generate(flux2Request, progressHandler: { progress in
        // Flux2Pipeline progress — not routed to ZImagePipeline progress handler
        // since the types differ. Logged internally by the pipeline.
      })

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      resumed = true
      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: result.path,
          durationMs: durationMs,
          appliedLoras: appliedLoRAStates(),
          presetUnresolved: payload.presetUnresolved,
          presetUnresolvedReason: payload.presetUnresolvedReason,
          presetStackMismatch: payload.presetStackMismatch,
          memoryEstimateBytes: payload.memoryEstimateBytes, memoryAvailableBytes: payload.memoryAvailableBytes,
          loraStackOrigin: payload.loraStackOrigin,
          warmDefaultSkipped: payload.warmDefaultSkipped,
          loraReload: payload.loraReload
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  private func runKrea2Generate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    // comfybox#283/#217: read-only — the synchronous GPU section begins here.
    if let activeJobId { lifecycleLedger.record(jobId: activeJobId, kind: .started, jobKind: QueueJobKind.generate.rawValue, source: activeJobSource) }
    let start = Date()
    var resumed = false
    defer {
      if !resumed {
        logger.error("runKrea2Generate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "Krea2 generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "Krea2 generation failed unexpectedly"))
      }
    }
    do {
      guard let k2 = krea2Pipeline else {
        throw WarmServerError.krea2NotLoaded
      }
      // WP-E4 (D18): krea2-only tier gates — a 400, never a silent downgrade.
      try payload.validateKrea2TierGates(try payload.validateRecipeNames())
      // WP-E3 (§3.3, D11, D25): the sampler, the sigma schedule and the shift
      // the caller asked for, forwarded into the request the loop dispatches on.
      let recipe = try payload.krea2RecipeFields()
      // WP-E17 (§3.14, D4): the second stage, resolved fail-loud — an unknown
      // sampler/schedule name on the STAGE throws here, before any model work,
      // exactly as it does for the render's own recipe.
      let stage2 = try payload.krea2Stage2Fields()
      // ClownsharK wire dials, validated where they are applied (same
      // fail-loud stance as an unknown sampler name): a non-finite or
      // out-of-range projector_scale and an unknown noise_type are 400s here,
      // before any model work — never a clamp, never a silent gaussian.
      let projectorScale = try payload.validatedProjectorScale()
      let noiseType = try payload.validatedNoiseType()
      let implicitSteps = try payload.validatedImplicitSteps()
      let c2 = try payload.validatedC2()
      let samplerAsked: String = recipe.samplerRequested ?? "-"
      let scheduleAsked: String = recipe.sigmaScheduleRequested ?? "-"
      let shiftLabel: String = recipe.shift.map { "\($0)" } ?? "dynamic"
      let recipeLine: String =
        "Krea2 recipe: sampler=\(recipe.sampler.rawValue) (requested \(samplerAsked)) "
        + "sigma_schedule=\(recipe.sigmaSchedule.rawValue) (requested \(scheduleAsked)) "
        + "shift=\(shiftLabel) eta=\(recipe.eta) bongmath=\(recipe.bongmath)"
      logger.info("\(recipeLine)")
      if let stage2 {
        let stage2Line: String =
          "Krea2 stage 2: sampler=\(stage2.sampler?.rawValue ?? recipe.sampler.rawValue) "
          + "sigma_schedule=\(stage2.sigmaSchedule?.rawValue ?? recipe.sigmaSchedule.rawValue) "
          + "steps=\(stage2.steps) denoise=\(stage2.denoise)"
        logger.info("\(stage2Line)")
      }
      // WP-E9 (§3.9, D16, D17): VAE selection — payload.vae → model dir. A
      // named file that is not on disk fails the render here (AC-56); a
      // different file than the resident one reloads the decoder IN PLACE on
      // the one Krea2VAE (never a pool eviction), and the selection is
      // recorded on the pipeline for the response record (WP-E10).
      let vaeChoice = try Krea2VAESelector.resolve(
        requested: payload.vae, paths: k2.paths, fromPreset: payload.presetVAEApplied == true)
      try k2.ensureVAE(path: vaeChoice.file, source: vaeChoice.source)
      logger.info("Krea2: VAE \(k2.currentVAE.layout.rawValue) \(k2.currentVAE.file.path) (source=\(k2.currentVAE.source.rawValue), reloads=\(k2.vaeReloadCount))")
      let outputURL = try payload.resolvedOutputURL(
        configuration: configuration,
        defaultFilename: ComfyBoxOutputNaming.defaultFilename(
          modelSpec: activePoolModelSpec ?? configuration.modelSpec ?? "krea2", presetId: payload.preset,
          contentMode: payload.contentMode, source: payload.source)
      )

      let seed = payload.seed ?? UInt64.random(in: 1..<UInt64(UInt32.max))
      // Variant defaults (WP-E5, AC-5b): turbo 9 / 1.0, raw 30 / 1.0 — never 3.5.
      // FDD §3.3, D3: the config layer slots BETWEEN the request and the
      // variant's own default — `resolvedSteps`/`resolvedGuidance` already do
      // `requested ?? variant.defaultX`, so passing a config default in place
      // of `nil` preserves that exact fallback chain one layer further out.
      // Only width/height are seeded on first-run migration (steps/guidance
      // are physical-variant-dependent, not a fixed engine constant — see
      // ServerConfigStore.engineSeed); an explicit override still applies.
      let krea2ConfigDefaults = ServerConfigStore.shared.renderDefaults(family: "krea2")
      let variant = k2.variant
      let merged = mergedKrea2RenderDefaults(
        requestWidth: payload.width, requestHeight: payload.height,
        requestSteps: payload.steps, requestGuidance: payload.guidance,
        configDefaults: krea2ConfigDefaults)
      let steps = variant.resolvedSteps(merged.steps)
      let guidance = variant.resolvedGuidance(merged.guidance)
      let width = merged.width
      let height = merged.height
      // Krea-2 builds its requests straight from the payload rather than going
      // through makePipelineRequest, so resolve DyPE explicitly here.
      let krea2DyPE = payload.resolvedDyPEConfig(width: width, height: height)

      // img2img fix (2026-07-19): Krea2Pipeline.generateImg2Img already
      // implements VAE-encode + partial-denoise; runKrea2Generate simply never
      // wired it, so an init image was silently ignored (txt2img). When an init
      // image is present, load+normalize it to NHWC [-1,1] and run img2img.
      // strength = 1 - denoise, matching flux1 makeImg2ImgRequest's convention.
      // Depth Control-LoRA: load control weights + encode control tokens when a control image is supplied.
      var controlPixels: MLXArray? = nil
      let resolvedControlData: Data? = payload.controlImageData ?? payload.controlImage.flatMap { try? Data(contentsOf: URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)) }
      if let controlData = resolvedControlData {
        let ccg = try InpaintUtilities.loadCGImage(from: controlData)
        let cpix = try QwenImageIO.resizedPixelArray(from: ccg, width: width, height: height)
        controlPixels = QwenImageIO.normalizeForEncoder(cpix).transposed(0, 2, 3, 1)
        let loraURL = URL(fileURLWithPath: "/Volumes/Bolt/Models/krea2-controlnet/depth-control-lora.safetensors")
        try await k2.setControlLoRA(loraURL, scale: payload.controlnetStrength ?? 1.0)
        logger.info("Krea2: depth Control-LoRA active (strength=\(payload.controlnetStrength ?? 1.0))")
      } else if k2.controlLoRAActive {
        try await k2.setControlLoRA(nil)
      }
      // Rewriter-proof trigger guarantee (Todd 2026-08-11): re-assert every
      // applied LoRA's library trigger on the FINAL prompt — the sealed
      // rewrite happens upstream and can drop activation tokens.
      let loraTriggers: [String] = (payload.loras ?? []).compactMap { entry in
        let filename = (entry.path as NSString).lastPathComponent
        return loraLibrary?.entry(for: filename)?.triggerwords.first
      }
      let guardedPrompt = LoRATriggerGuard.ensure(prompt: payload.prompt, triggers: loraTriggers)

      // WP-E10: publish per-step progress the way the Z-Image path does.
      // `/health.progress_percent` used to stay 0 for the whole of a Krea 2
      // render because this arm's callback only logged.
      let tracker = progressTracker
      let health = liveHealth
      // comfybox#283/#217: read-only, bounded-rate progress ticks for the
      // production image model — see `runGenerate`'s equivalent hook for why
      // these are captured as plain values rather than `self`.
      let ledger = lifecycleLedger
      let jobId = activeJobId
      let publishProgress: @Sendable (Int, Int) -> Void = { [logger] step, total in
        let pct = RenderProgressPercent.of(step: step, total: total)
        tracker.set(pct)
        health.setProgress(pct)
        logger.info("Krea2: step \(step)/\(total)")
        if let jobId {
          ledger.record(
            jobId: jobId, kind: .progress, jobKind: QueueJobKind.generate.rawValue,
            step: step, totalSteps: total, percent: pct)
        }
      }

      let image: MLXArray
      // WP-E17: one trace per stage that ran, in order. `traces[0]` is the
      // render's own — the geometry, seed and schedule shift every sink reads.
      let traces: [Krea2RunTrace]
      if let initPath = payload.imagePath {
        let imageData = try Data(contentsOf: URL(fileURLWithPath: initPath))
        let cg = try InpaintUtilities.loadCGImage(from: imageData)
        let pixNCHW = try QwenImageIO.resizedPixelArray(from: cg, width: width, height: height)
        let sourceNHWC = QwenImageIO.normalizeForEncoder(pixNCHW).transposed(0, 2, 3, 1)
        let strength: Float
        if let c = payload.creativity {
          strength = 1.0 - max(0.01, min(0.99, c))
        } else if let sVal = payload.imageStrength {
          strength = max(0.01, min(0.99, sVal))
        } else if let d = payload.denoise {
          strength = 1.0 - max(0.01, min(0.99, d))
        } else {
          strength = 0.3
        }
        logger.info("Krea2: img2img init=\(initPath) strength=\(strength)")
        // §3.14 leaves img2img alone on purpose: its `strength → startIndex`
        // rule is the established contract on this path, and stage 2 is a
        // different, differently-specified mechanism (AC-30 keeps them
        // distinguishable). Composing the two is unspecified, so it is refused
        // rather than silently resolved one way.
        guard stage2 == nil else {
          throw WarmServerError.mutuallyExclusive(
            "stage2 and image_path cannot be combined: the second stage re-noises the LATENT to "
              + "the stretched tail's first sigma (WP-E17), while img2img starts partway down the "
              + "grid from `strength` — two different mechanisms with no defined composition. "
              + "Send one of them")
        }
        let trace1: Krea2RunTrace
        (image, trace1) = try k2.generateImg2ImgWithRecipe(
          .init(prompt: guardedPrompt, negativePrompt: payload.negativePrompt,
                guidance: guidance,
                sourceImage: sourceNHWC, width: width, height: height,
                steps: steps, seed: seed, strength: strength, dyPE: krea2DyPE,
                shift: recipe.shift,
                sampler: recipe.sampler, sigmaSchedule: recipe.sigmaSchedule,
                sigmaScheduleRequested: recipe.sigmaScheduleRequested,
                eta: recipe.eta, bongmath: recipe.bongmath,
                c2: c2,
                projectorScale: projectorScale,
                noiseType: noiseType,
                noiseAlpha: payload.noiseAlpha ?? 0.0,
                implicitStepsFull: implicitSteps),
          progress: publishProgress)
        traces = [trace1]
      } else {
        // WP-E17: `generateStaged` IS `generateWithRecipe`'s body. Without
        // `stage2` it executes the same statements it did before this WP —
        // which is what makes the byte-identity gates hold by construction —
        // and with it, the second stage runs before the one `vae.decode`.
        (image, traces) = try k2.generateStaged(
          .init(prompt: guardedPrompt, negativePrompt: payload.negativePrompt,
                guidance: guidance,
                width: width, height: height, steps: steps, seed: seed,
                controlImagePixels: controlPixels, dyPE: krea2DyPE,
                shift: recipe.shift,
                sampler: recipe.sampler, sigmaSchedule: recipe.sigmaSchedule,
                sigmaScheduleRequested: recipe.sigmaScheduleRequested,
                eta: recipe.eta, bongmath: recipe.bongmath, c2: c2, stage2: stage2,
                projectorScale: projectorScale,
                noiseType: noiseType,
                noiseAlpha: payload.noiseAlpha ?? 0.0,
                implicitStepsFull: implicitSteps),
          progress: publishProgress)
      }
      let trace = traces[0]
      // WP-E10 (FDD §3.10, AC-60): the provenance record is READ BACK from
      // the pipeline — the variant and transformer file it loaded, the
      // quantization it applied, the VAE resident in its slot, its loaded
      // LoRA configs joined with their bind reports, and the run trace the
      // loop just counted. `steps`/`guidance` above are NOT consulted here.
      // Fail CLOSED on an incomplete read-back: a record naming two of three
      // adapters is worse than no record, because it reads as complete.
      let loraReadBacks = RenderRecipe.loRAReadBacks(
        configs: k2.loadedLoRAConfigs, reports: k2.loadedLoRAReports,
        // I6: the relativity the guard ENFORCED, so `relative_to` names what
        // was applied rather than what the request happened to declare.
        relativities: k2.loadedLoRARelativities)
      if loraReadBacks == nil {
        let mismatch = "Krea2 provenance: \(k2.loadedLoRAConfigs.count) loaded LoRA configs but "
          + "\(k2.loadedLoRAReports.count) bind reports — refusing to emit a partial `applied` for this render"
        logger.error("\(mismatch)")
      }
      // `base_model` is the declared alias when the active spec is one (or
      // resolves to one's directory — AC-34b), else the spec as loaded.
      let activeSpec = activePoolModelSpec ?? configuration.modelSpec ?? "krea2"
      let record: RenderRecipe? = loraReadBacks.map { readBacks in
        RenderRecipe.krea2(.init(
          baseModel: Krea2ModelDetection.alias(forSpec: activeSpec) ?? activeSpec,
          variant: k2.variant,
          transformerFile: k2.paths.transformerFile,
          quantizationBits: k2.transformerQuantBits,
          vae: k2.currentVAE,
          textEncoderFile: k2.paths.textEncoderFile,
          loras: readBacks,
          control: k2.controlLoRAActive ? k2.controlLoRAApplied : nil,
          // D4 / WP-E17: every stage that ran, so `applied.stages[]` and
          // `model_evals_total` describe the whole render rather than its
          // first half.
          traces: traces))
      }
      // Sink 2 — the PNG. The negative comes from the TRACE (K-FIX-1 / I4):
      // absent when CFG never ran (AC-61), and an applied `""` is written as
      // `""` rather than dropped, so the file records the second model pass
      // an omitted negative still paid for.
      let metadata = QwenImageIO.ImageMetadata.generation(
        prompt: guardedPrompt,
        negativePrompt: trace.negativePromptApplied,
        seed: trace.seed,
        // WP-E17: the flat scalar a human reads names the WHOLE render, not its
        // first stage — `applied.stages[]` is where the split lives. Identical
        // to `trace.stepsRequested` for a single-stage render.
        steps: traces.reduce(0) { $0 + $1.stepsRequested },
        guidance: trace.guidance,
        width: trace.width,
        height: trace.height,
        model: ComfyBoxOutputNaming.shortModelName(activePoolModelSpec ?? configuration.modelSpec),
        generatedBy: payload.source,
        contentMode: payload.contentMode,
        loras: k2.loadedLoRAConfigs,
        // The SLOT, so a refused record writes `"applied": null` in the file
        // rather than looking like a family that has no record (round 2, C4).
        appliedSlot: AppliedRecordSlot(record: record)
      )
      try QwenImageIO.saveImage(array: image.transposed(2, 0, 1), to: outputURL, metadata: metadata)

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      // Round 2 (C4): THIS render is Krea 2, so the slot is always present —
      // it carries the record, or a literal `null` when `loraReadBacks` refused
      // it above. An absent key must never be able to mean "engine-incomplete".
      let applied = AppliedRecordSlot(record: record)
      lastRecipe = applied  // sink 3 — /health.last_recipe
      activeRenderStartedAt = nil
      resumed = true
      // sink 1 — the response; sink 4 reads `applied` off this same value.
      continuation.resume(returning: GenerateResponse(
        success: true, outputPath: outputURL.path, durationMs: durationMs, applied: applied,
        appliedLoras: appliedLoRAStates(), presetUnresolved: payload.presetUnresolved,
        presetUnresolvedReason: payload.presetUnresolvedReason,
        presetStackMismatch: payload.presetStackMismatch,
        memoryEstimateBytes: payload.memoryEstimateBytes, memoryAvailableBytes: payload.memoryAvailableBytes,
        loraStackOrigin: payload.loraStackOrigin,
        warmDefaultSkipped: payload.warmDefaultSkipped,
        loraReload: payload.loraReload))
        // #154 deliberately sets no `applied_shift` here: Krea 2 already reads
        // back a full `applied` recipe (`shift`, `shift_source`, and
        // `stages[].shift_applied`, which is false for `bong_tangent`), and a
        // flat field duplicating it would be a second, weaker claim about the
        // same render — and would say "applied" for a schedule that ignored it.
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  private func runFiboGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    // comfybox#283/#217: read-only — the synchronous GPU section begins here.
    if let activeJobId { lifecycleLedger.record(jobId: activeJobId, kind: .started, jobKind: QueueJobKind.generate.rawValue, source: activeJobSource) }
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runFiboGenerate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "FIBO generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "FIBO generation failed unexpectedly"))
      }
    }

    do {
      guard let fp = fiboPipeline else {
        throw WarmServerError.fiboNotLoaded
      }

      let outputURL: URL
      outputURL = try payload.resolvedOutputURL(
        configuration: configuration,
        defaultFilename: ComfyBoxOutputNaming.defaultFilename(
          modelSpec: activePoolModelSpec ?? "fibo", presetId: payload.preset,
          contentMode: payload.contentMode, source: payload.source)
      )

      // FDD §3.3, D3: config-layer render defaults for fibo (the "steps ?? 30"
      // family-specific fallback the FDD's §2.5 finding cites).
      let fiboConfigDefaults = ServerConfigStore.shared.renderDefaults(family: "fibo")
      let fiboRequest = FiboGenerationRequest(
        prompt: payload.prompt,
        negativePrompt: payload.negativePrompt,
        width: payload.width ?? fiboConfigDefaults.width ?? 1024,
        height: payload.height ?? fiboConfigDefaults.height ?? 1024,
        steps: payload.steps ?? fiboConfigDefaults.steps ?? 30,
        guidanceScale: payload.guidance ?? fiboConfigDefaults.guidance.map(Float.init) ?? 4.0,
        seed: payload.seed,
        outputPath: outputURL,
        levelsMin: payload.levelsMin ?? 0.0,
        levelsMax: payload.levelsMax ?? 1.0,
        contentMode: payload.contentMode
      )

      let result = try await fp.generate(fiboRequest, progressHandler: nil)

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      resumed = true
      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: result.path,
          durationMs: durationMs,
          appliedLoras: appliedLoRAStates(),
          presetUnresolved: payload.presetUnresolved,
          presetUnresolvedReason: payload.presetUnresolvedReason,
          presetStackMismatch: payload.presetStackMismatch,
          memoryEstimateBytes: payload.memoryEstimateBytes, memoryAvailableBytes: payload.memoryAvailableBytes,
          loraStackOrigin: payload.loraStackOrigin,
          warmDefaultSkipped: payload.warmDefaultSkipped,
          loraReload: payload.loraReload
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  private func runChromaGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    // comfybox#283/#217: read-only — the synchronous GPU section begins here.
    if let activeJobId { lifecycleLedger.record(jobId: activeJobId, kind: .started, jobKind: QueueJobKind.generate.rawValue, source: activeJobSource) }
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runChromaGenerate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "Chroma generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "Chroma generation failed unexpectedly"))
      }
    }

    do {
      guard let pipeline = chromaPipeline else {
        throw WarmServerError.chromaNotLoaded
      }
      guard let tokenizer = chromaTokenizer else {
        throw WarmServerError.chromaNotLoaded
      }

      let outputURL: URL
      if let outputPath = payload.outputPath, !outputPath.isEmpty {
        outputURL = URL(fileURLWithPath: outputPath)
      } else {
        outputURL = URL(fileURLWithPath: NSTemporaryDirectory())
          .appendingPathComponent("zimage-chroma-\(UUID().uuidString).png")
      }

      // Run the synchronous Chroma render off the actor (the static helper is
      // nonisolated, so it executes on the global concurrent executor). This
      // mirrors the flux2/fibo paths, which await pipeline work without
      // blocking the actor — keeping /health, /queue, and progress telemetry
      // responsive for the duration of the render.
      // Review r1 (M2): Chroma has no LoRA application path — `ChromaPipeline
      // .generate` takes no adapters and `ModelPool` never forwards
      // `initialLoRAs` to it — so it renders with NONE, always. This argument
      // reaches only the PNG's `loras` list, and passing `activeLoRAs` made
      // every Chroma file claim adapters that took no part in it.
      try await Self.renderChroma(
        pipeline: pipeline,
        tokenizer: tokenizer,
        payload: payload,
        outputURL: outputURL,
        loras: []
      )

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      resumed = true
      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: outputURL.path,
          durationMs: durationMs,
          appliedLoras: appliedLoRAStates(),
          presetUnresolved: payload.presetUnresolved,
          presetUnresolvedReason: payload.presetUnresolvedReason,
          presetStackMismatch: payload.presetStackMismatch,
          memoryEstimateBytes: payload.memoryEstimateBytes, memoryAvailableBytes: payload.memoryAvailableBytes,
          loraStackOrigin: payload.loraStackOrigin,
          warmDefaultSkipped: payload.warmDefaultSkipped,
          loraReload: payload.loraReload
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  /// Perform the synchronous Chroma pipeline render. Static (hence nonisolated)
  /// and async, so it runs on the global concurrent executor rather than on
  /// the coordinator actor — a Chroma render would otherwise block /health,
  /// /queue, and progress telemetry for its full duration.
  private static func renderChroma(
    pipeline: ChromaPipeline,
    tokenizer: ChromaTokenizer,
    payload: GeneratePayload,
    outputURL: URL,
    loras: [LoRAConfiguration]
  ) async throws {
    // FDD §3.3, D3: config-layer render defaults for chroma.
    let chromaConfigDefaults = ServerConfigStore.shared.renderDefaults(family: "chroma")
    let width = payload.width ?? chromaConfigDefaults.width ?? 1024
    let height = payload.height ?? chromaConfigDefaults.height ?? 1024
    let steps = payload.steps ?? chromaConfigDefaults.steps ?? 28
    let guidance = payload.guidance ?? chromaConfigDefaults.guidance.map(Float.init) ?? 0.0
    let seed = payload.seed ?? UInt64.random(in: 0...UInt64.max)

    // Tokenize prompt (unpadded — matches Python behavior)
    let tokenIds = tokenizer.encodeUnpadded(prompt: payload.prompt)

    // Tokenize negative prompt for CFG (empty string = unconditional)
    let negTokenIds = tokenizer.encodeUnpadded(prompt: payload.negativePrompt ?? "")

    // CFG parameters (default: cfg=4.0, no warmup steps)
    let cfgScale = payload.cfg ?? 4.0
    let cfgWarmup = payload.firstNStepsWithoutCFG ?? 0

    // K-FIX-1 / Codex I5: Chroma HAS native heun and beta and this call used
    // to pass neither, so `scheduler: "heun"` rendered Euler pixels under the
    // name "heun". The pair is mapped through the one family matrix — which
    // has already refused (400) any pair it cannot map, so the fallback here
    // is the unreachable default, not a silent substitution.
    let names = try payload.validateRecipeNames()
    let chromaScheduler = FamilyRecipeMatrix.chromaSchedulerType(
      sampler: names.scheduler, schedule: names.sigmaSchedule) ?? .euler

    // Generate — returns MLXArray in [B, H, W, C] (NHWC, values [0,1])
    let result = try pipeline.generate(
      tokenIds: tokenIds,
      negativeTokenIds: negTokenIds,
      width: width,
      height: height,
      numSteps: steps,
      guidance: guidance,
      cfg: cfgScale,
      firstNStepsWithoutCFG: cfgWarmup,
      schedulerType: chromaScheduler,
      seed: seed,
      progressCallback: { step, total in
        // Progress logging
      }
    )

    // Transpose from NHWC [1, H, W, 3] to CHW [3, H, W] for QwenImageIO
    let imageArray = result.squeezed(axis: 0).transposed(2, 0, 1)

    // Save image (with embedded, Finder-readable generation metadata)
    try QwenImageIO.saveImage(array: imageArray, to: outputURL,
      // Non-Krea 2 families hand `ImageMetadata` a RAW payload value, where
      // `""` means "not given" — normalised through the one shared helper so
      // the I4 change to `ImageMetadata.generation` (an explicit `""` is
      // written) cannot add a `negative_prompt: ""` key to any other family's
      // PNG. Krea 2's APPLIED value deliberately does not go through it.
      metadata: .generation(prompt: payload.prompt,
        negativePrompt: QwenImageIO.ImageMetadata.requestNegative(payload.negativePrompt),
        seed: seed, steps: steps, guidance: guidance, width: width, height: height,
        generatedBy: payload.source, contentMode: payload.contentMode, loras: loras))
  }

  private func runControlGenerate(_ request: ZImageControlGenerationRequest, continuation: ContinuationBox<GenerateResponse>) async {
    if currentModelFamily == .flux2 || currentModelFamily == .fibo || currentModelFamily == .chroma || currentModelFamily == .krea2 {
      continuation.resume(throwing: WarmServerError.controlNetNotSupported)
      return
    }

    activeRenderStartedAt = Date()
    // comfybox#283/#217: read-only — the synchronous GPU section begins here.
    if let activeJobId { lifecycleLedger.record(jobId: activeJobId, kind: .started, jobKind: QueueJobKind.controlnet.rawValue, source: activeJobSource) }
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runControlGenerate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "ControlNet generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "ControlNet generation failed unexpectedly"))
      }
    }

    do {
      // Lazy-init the control pipeline on first ControlNet request
      if controlPipeline == nil {
        logger.info("Initializing ControlNet pipeline (first use)...")
        controlPipeline = ZImageControlPipeline(logger: logger)
      }

      let control = controlPipeline!
      let outputURL = try await control.generate(request)
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      resumed = true
      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: outputURL.path,
          durationMs: durationMs,
          // I4: the ControlNet pipeline is what rendered — read ITS adapters.
          appliedLoras: appliedLoRAStates(from: control),
          // #154: the ControlNet arm runs the same flow schedule.
          appliedShift: request.shift
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  /// Currently-loaded LoRA configs for whichever pipeline is active — used to
  /// resync `activeLoRAs` after a failed or crashed swap.
  private func loadedLoRAConfigs(for family: WarmModelFamily?) -> [LoRAConfiguration] {
    switch family {
    case .flux2: return flux2Pipeline?.loadedLoRAConfigs ?? []
    case .krea2: return krea2Pipeline?.loadedLoRAConfigs ?? []
    default: return pipeline.loadedLoRAConfigs
    }
  }

  /// #286: the resident LoRA stack for the `applied_loras` response field —
  /// READ BACK from the pipeline that actually rendered, never from the
  /// coordinator's intent, so a client can diff it against what
  /// `POST /v1/presets/resolve` reported and catch a wrong stack itself.
  ///
  /// nil (key absent) for FIBO and Chroma, which have no LoRA path at all
  /// (`/v1/lora/swap` refuses them), so an absent key can never read as
  /// "rendered bare".
  ///
  /// I4 (review round 1): the ControlNet arm renders through `controlPipeline`,
  /// a DIFFERENT instance from the family pipeline — reading the family's
  /// configs there reported unrelated resident state. `pipeline` names the one
  /// that rendered.
  private func appliedLoRAStates(from pipeline: ZImageControlPipeline) -> [LoRAState]? {
    pipeline.loadedLoRAConfigs.map(LoRAState.init)
  }

  private func appliedLoRAStates() -> [LoRAState]? {
    if currentModelFamily == .fibo || currentModelFamily == .chroma { return nil }
    return loadedLoRAConfigs(for: currentModelFamily).map(LoRAState.init)
  }

  /// #282 — the per-job stack, resolved from the payload the loop dequeued.
  ///
  /// The ONE place `runGenerate` decides what a render's adapters are, kept
  /// separate from the application so the decision can be driven from a unit
  /// test (`WarmServerQueueProbe.resolveJobStack`) without weights or a GPU.
  ///
  /// Throws only what `LoRAEntry.makeConfiguration()` has always thrown — an
  /// explicit `loras` entry that names an adapter the engine cannot resolve
  /// keeps its long-standing 400 at dequeue. The warm default is already a
  /// list of resolved configurations (it was resolved when the swap that set
  /// it was accepted), so a bare request cannot fail here.
  ///
  /// Review r1 (C1): a warm default published under another base is SKIPPED
  /// here — the job renders with no adapters and the response says why. The
  /// per-job model switch has already run by this point, so
  /// `currentModelFamily` and the active spec are THIS job's base, not the
  /// one the swap saw.
  private func resolveJobLoRAStack(_ payload: GeneratePayload) throws -> JobLoRAPlan {
    let carried: [LoRAConfiguration]? = try payload.loras.map {
      try $0.map { try $0.makeConfiguration() }
    }
    let presetOwned = payload.presetStackApplied == true
    let resolved = RequestStackResolver.resolve(
      requestLoras: presetOwned ? nil : carried,
      presetStack: presetOwned ? carried : nil,
      warmDefault: warmDefaultStack)

    guard resolved.origin == .warmDefault else {
      return JobLoRAPlan(stack: resolved.stack, origin: resolved.origin)
    }
    switch RequestStackResolver.admitWarmDefault(
      isEmpty: resolved.stack.isEmpty,
      tag: warmDefaultTag,
      requestFamily: currentModelFamily.rawValue,
      requestModelSpec: normalizedActiveModelSpec()
    ) {
    case .admit:
      return JobLoRAPlan(stack: resolved.stack, origin: .warmDefault)
    case .skip(let reason):
      let names = resolved.stack.map { $0.source.displayName }.joined(separator: ", ")
      let line = "Warm default NOT applied (\(reason)): it was published under "
        + "\(warmDefaultTag.family ?? "?")/\(warmDefaultTag.modelSpec ?? "?") and this job renders on "
        + "\(currentModelFamily.rawValue)/\(normalizedActiveModelSpec() ?? "?") — rendering with no "
        + "adapters rather than forcing [\(names)] onto a base they were not published for"
      logger.warning("\(line)")
      return JobLoRAPlan(stack: [], origin: .warmDefault, warmDefaultSkipped: reason)
    }
  }

  /// The active model spec, normalised the way ``PresetLoRAStack`` compares
  /// specs (alias → directory, `~` expanded on both sides) so a spelling
  /// difference is never read as a different base.
  private func normalizedActiveModelSpec() -> String? {
    guard let spec = activePoolModelSpec ?? configuration.modelSpec, !spec.isEmpty else {
      return nil
    }
    return PresetLoRAStack.normalizedModel(spec, WarmServer.parseModelSpec)
  }

  /// The provenance a warm default published right now would carry.
  private func currentWarmDefaultTag() -> RequestStackResolver.WarmDefaultTag {
    .init(family: currentModelFamily.rawValue, modelSpec: normalizedActiveModelSpec())
  }

  /// #282 — apply exactly this job's stack, at dequeue, for every family that
  /// has a LoRA path. Routes through ``applyActiveLoRAs``, so #286's
  /// same-stack shortcut still skips a reload when the resident stack already
  /// is this one — which is what keeps a 24/7 daemon rendering the same preset
  /// back to back from re-binding 5-10 adapters per render.
  ///
  /// Returns whether this job forced a real clear+reload, so the response can
  /// carry `lora_reload` (review r1, I1): alternating bare and preset renders
  /// legitimately thrash the adapter stack, and that cost must be VISIBLE
  /// rather than merely accepted.
  @discardableResult
  private func applyJobLoRAStack(_ plan: JobLoRAPlan) async throws -> Bool {
    let familyHasLoRAPath = !(currentModelFamily == .fibo || currentModelFamily == .chroma)
    guard RequestStackResolver.appliesAtDequeue(
      origin: plan.origin, familyHasLoRAPath: familyHasLoRAPath)
    else {
      // Review r1 (M1): no LoRA path means NOTHING is applied, whatever the
      // origin — loading a job's adapters into the Flux-1 pipeline these
      // families do not render through changed no pixels and made
      // `/health.loras` and the PNG name adapters that took no part.
      let skipped = "LoRA stack: \(currentModelFamily.rawValue) has no LoRA application path — "
        + "this render uses no adapters (origin \(plan.origin.rawValue), "
        + "\(plan.stack.count) requested)"
      logger.info("\(skipped)")
      return false
    }
    let described = plan.stack.isEmpty
      ? "(none)"
      : plan.stack.map { "\($0.source.displayName)@\(String(format: "%.4g", $0.scale))" }
        .joined(separator: ", ")

    #if DEBUG
    // Review r1 (I4): the load-bearing call, observable without weights. See
    // `WarmServerQueueProbe.setStackRecorder`.
    if let recorder = testSeamStackRecorder {
      recorder.record(
        origin: plan.origin.rawValue, names: plan.stack.map { $0.source.displayName },
        warmDefaultSkipped: plan.warmDefaultSkipped)
      activeLoRAs = plan.stack
      return false
    }
    #endif

    let previous = loadedLoRAConfigs(for: currentModelFamily)
    let application = try await applyActiveLoRAs(plan.stack)
    logger.info("LoRA stack for this job (origin: \(plan.origin.rawValue)): \(described)")
    if application.reloaded, !previous.isEmpty {
      let churn = "LoRA reload: this job cleared \(previous.count) resident adapter(s) and bound "
        + "\(application.stack.count) — origin \(plan.origin.rawValue). Alternating bare/preset "
        + "renders pay this per job (response carries lora_reload)"
      logger.warning("\(churn)")
    }
    return application.reloaded
  }

  /// #282 — publish the WARM DEFAULT stack: what a request that names neither
  /// `preset` nor `loras` renders with.
  ///
  /// Called by `runSwap` and nowhere else. Per-job application
  /// (``applyJobLoRAStack``) deliberately does NOT call it: a job's own stack
  /// is that job's business and must never become the next job's default —
  /// that inheritance is precisely what #282 retires.
  private func adoptWarmDefaultStack(
    _ stack: [LoRAConfiguration], tag: RequestStackResolver.WarmDefaultTag
  ) {
    warmDefaultStack = stack
    warmDefaultTag = tag
    let names = stack.isEmpty
      ? "(none)" : stack.map { $0.source.displayName }.joined(separator: ", ")
    let line = "/v1/lora/swap: warm default stack is now [\(names)] under "
      + "\(tag.family ?? "?")/\(tag.modelSpec ?? "?") — it applies ONLY to a request that carries "
      + "neither `preset` nor `loras`, and only on that same base"
    logger.info("\(line)")
  }

  #if DEBUG
  /// #282 review r1 (I4) — the recorder that stands in for the pipeline
  /// application inside ``applyJobLoRAStack``. nil in every real run.
  var testSeamStackRecorder: StackApplicationRecorder?
  func setStackRecorder(_ recorder: StackApplicationRecorder?) {
    testSeamStackRecorder = recorder
  }

  /// #282 test seam. `resolveJobLoRAStack` is the function `runGenerate` calls
  /// at dequeue; the coordinator is file-private and the application itself
  /// needs model weights (intent.md: agents run unit tests only), so this
  /// drives THE SAME function and reports what it decided.
  func testSeamResolveJobStack(
    _ payload: GeneratePayload
  ) throws -> (origin: String, names: [String], warmDefaultSkipped: String?) {
    let plan = try resolveJobLoRAStack(payload)
    return (plan.origin.rawValue, plan.stack.map { $0.source.displayName }, plan.warmDefaultSkipped)
  }

  /// #282 test seam: the warm-default adoption `runSwap` performs, so a test
  /// can set a non-empty default without loading weights (applying a real
  /// stack needs a model). `runSwap`'s own call to this is proved end-to-end
  /// by an EMPTY swap, which is the one swap that succeeds with no pipeline.
  func testSeamAdoptWarmDefaultStack(
    _ stack: [LoRAConfiguration], tag: RequestStackResolver.WarmDefaultTag? = nil
  ) {
    adoptWarmDefaultStack(stack, tag: tag ?? currentWarmDefaultTag())
  }

  /// The warm default as `/v1/model/pool` reports it.
  func testSeamWarmDefaultStackNames() -> [String] {
    warmDefaultStack.map { $0.source.displayName }
  }

  /// The tag the warm default was published under.
  func testSeamWarmDefaultTag() -> RequestStackResolver.WarmDefaultTag { warmDefaultTag }

  /// The family this coordinator would render on. `.flux1` on a fresh probe.
  func testSeamCurrentFamily() -> String { currentModelFamily.rawValue }
  #endif

  /// Apply LoRAs to whichever pipeline is active for `currentModelFamily`.
  /// Shared by POST /v1/lora/swap and per-job LoRA application at generate
  /// dequeue time (queue-submit race fix — see GeneratePayload.loras).
  @discardableResult
  private func applyActiveLoRAs(
    _ newLoRAs: [LoRAConfiguration]
  ) async throws -> LoRAApplication {
    if currentModelFamily == .flux2 {
      guard let f2 = flux2Pipeline else { throw WarmServerError.flux2NotLoaded }
      // I2 (review round 1): Flux 2 and Krea 2 clear and reload every adapter
      // on every call, unlike `ZImagePipeline.loadLoRAs`, which skips an
      // identical stack. Now that a preset render applies its stack on EVERY
      // request, a 5-10 adapter preset would otherwise reload the whole stack
      // per render — pure latency and unified-memory churn on a 24/7 daemon.
      if LoRAStackIdentity.isSameStack(f2.loadedLoRAConfigs, newLoRAs) {
        logger.info("LoRA stack already resident (Flux 2) — skipping reload of \(newLoRAs.count) adapter(s)")
        activeLoRAs = newLoRAs
        return LoRAApplication(stack: newLoRAs, reloaded: false)
      }
      try await f2.loadLoRAs(newLoRAs)
      activeLoRAs = newLoRAs
      publishHealth()
      return LoRAApplication(stack: newLoRAs, reloaded: true)
    } else if currentModelFamily == .krea2 {
      guard let k2 = krea2Pipeline else { throw WarmServerError.krea2NotLoaded }
      // WP-E6: fold the library's declared relativity into any config that
      // did not declare one itself (request > library > seed — never inferred).
      let declared = newLoRAs.map { cfg -> LoRAConfiguration in
        guard cfg.requiresBase == nil, case .local(let url) = cfg.source,
              let entry = loraLibrary?.entry(for: url.lastPathComponent),
              let relative = entry.krea2Relative
        else { return cfg }
        var out = cfg
        out.requiresBase = relative
        return out
      }
      // I2: compared AFTER relativity folding, so the comparison is against
      // what would actually be loaded.
      if LoRAStackIdentity.isSameStack(k2.loadedLoRAConfigs, declared) {
        logger.info("LoRA stack already resident (Krea 2) — skipping reload of \(declared.count) adapter(s)")
        activeLoRAs = declared
        return LoRAApplication(stack: declared, reloaded: false)
      }
      try await k2.loadLoRAs(declared)
      activeLoRAs = declared
      // Active LoRA set changed — refresh the health snapshot (#217).
      publishHealth()
      return LoRAApplication(stack: declared, reloaded: true)
    } else {
      // `ZImagePipeline.loadLoRAs` short-circuits an identical stack itself, so
      // the comparison here only decides what to REPORT (review r1, I1).
      let reloaded = pipeline.loadedLoRAConfigs != newLoRAs
      try await pipeline.swapLoRAs(newLoRAs)
      activeLoRAs = newLoRAs
      if reloaded { publishHealth() }
      return LoRAApplication(stack: newLoRAs, reloaded: reloaded)
    }
  }

  private func runSwap(_ payload: LoRASwapPayload, continuation: ContinuationBox<LoRASwapResponse>) async {
    if currentModelFamily == .fibo || currentModelFamily == .chroma {
      continuation.resume(throwing: WarmServerError.loraSwapNotSupported)
      return
    }

    var resumed = false

    defer {
      if !resumed {
        logger.error("runSwap: continuation was not resumed — likely a crash in LoRA application. Resuming with error.")
        activeLoRAs = loadedLoRAConfigs(for: currentModelFamily)
        lastError = "LoRA swap failed unexpectedly (continuation not resumed)"
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "LoRA swap failed unexpectedly"))
      }
    }

    do {
      // A swap-first client (kira-daemon: swap → generate) must not fail
      // because the image pipeline is not resident — video eviction (#218)
      // or a fresh boot leaves it nil, and the only restore path lived in
      // runGenerate, so a failed swap meant generate was never called and
      // image creation deadlocked until an out-of-band generate landed.
      let familyPipelineMissing =
        (currentModelFamily == .krea2 && krea2Pipeline == nil)
        || (currentModelFamily == .flux2 && flux2Pipeline == nil)
      let restoreSpec = activePoolModelSpec ?? lastActiveImageSpec ?? configuration.modelSpec
      switch SwapResidencyRestore.decide(
        imageModelsEvicted: imageModelsEvicted,
        familyPipelineMissing: familyPipelineMissing,
        restoreSpec: restoreSpec
      ) {
      case .none:
        break
      case .reloadEvicted:
        try await reloadImageModelIfEvicted(requestedModel: nil)
      case .load(let modelSpec):
        logger.info("Swap arrived with no resident image pipeline — loading '\(modelSpec)' before applying LoRAs")
        _ = try await poolLoad(
          modelSpec: WarmServer.parseModelSpec(from: modelSpec),
          quantization: WarmServer.parseQuantization(from: modelSpec),
          activate: true)
      }

      let newLoRAs = try payload.makeConfigurations()
      let application = try await applyActiveLoRAs(newLoRAs)
      // #282: this is now the WHOLE meaning of a swap — it publishes the
      // default for requests that name neither `preset` nor `loras`. The
      // application above stays (a swap-first client expects the pipeline to
      // be holding the stack when it returns, and `SwapResidencyRestore` above
      // exists precisely so that application can happen), but no later job
      // inherits it except by asking for nothing. Response JSON unchanged.
      // Review r1 (I3): the APPLIED stack, not the requested one — relativity
      // folding happens inside `applyActiveLoRAs`, and publishing the pre-fold
      // configs would make `warm_default_stack` and `/health.loras` describe
      // the same adapters differently.
      // Review r1 (C1): tagged with the base it was published under.
      adoptWarmDefaultStack(application.stack, tag: currentWarmDefaultTag())

      lastError = nil
      resumed = true
      continuation.resume(
        returning: LoRASwapResponse(
          success: true,
          loraCount: activeLoRAs.count,
          loras: activeLoRAs.map(LoRAState.init)
        )
      )
    } catch {
      activeLoRAs = loadedLoRAConfigs(for: currentModelFamily)
      lastError = error.localizedDescription
      resumed = true
      continuation.resume(throwing: error)
    }
  }
}

private final class ConnectionHandler {
  private static let headerDelimiter = Data("\r\n\r\n".utf8)
  /// 10 MB — raised from 1 MB to support ComfyUI image uploads via PUT /api/etn/image/.
  private static let maximumRequestBytes = 10_485_760

  private let connection: NWConnection
  private let queue: DispatchQueue
  private weak var server: WarmServer?
  private var buffer = Data()
  private var responseSent = false
  private var retainSelf: ConnectionHandler?

  init(connection: NWConnection, queue: DispatchQueue, server: WarmServer) {
    self.connection = connection
    self.queue = queue
    self.server = server
  }

  func start() {
    retainSelf = self
    connection.start(queue: queue)
    receiveNextChunk()
  }

  private func receiveNextChunk() {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, isComplete, error in
      guard let self else { return }

      if let data, !data.isEmpty {
        self.buffer.append(data)
      }

      if self.buffer.count > Self.maximumRequestBytes {
        self.finish(with: .error(status: 413, message: "Request too large"))
        return
      }

      switch self.parseRequest() {
      case .request(let request):
        self.handle(request: request)
        return
      case .error(let response):
        self.finish(with: response)
        return
      case .incomplete:
        break
      }

      if let error {
        self.finish(with: .error(status: 400, message: error.localizedDescription))
        return
      }

      if isComplete {
        self.finish(with: .error(status: 400, message: "Unexpected end of request"))
        return
      }

      self.receiveNextChunk()
    }
  }

  private func parseRequest() -> HTTPParseResult {
    guard let headerRange = buffer.range(of: Self.headerDelimiter) else {
      return .incomplete
    }

    let headerData = buffer.subdata(in: 0..<headerRange.lowerBound)
    guard let headerString = String(data: headerData, encoding: .utf8) else {
      return .error(.error(status: 400, message: "Invalid request headers"))
    }

    let lines = headerString.components(separatedBy: "\r\n")
    guard let requestLine = lines.first, !requestLine.isEmpty else {
      return .error(.error(status: 400, message: "Missing request line"))
    }

    let requestParts = requestLine.split(separator: " ", omittingEmptySubsequences: true)
    guard requestParts.count >= 2 else {
      return .error(.error(status: 400, message: "Malformed request line"))
    }

    var headers: [String: String] = [:]
    for line in lines.dropFirst() where !line.isEmpty {
      guard let separator = line.firstIndex(of: ":") else {
        return .error(.error(status: 400, message: "Malformed header"))
      }
      let name = line[..<separator].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      let value = line[line.index(after: separator)...].trimmingCharacters(in: .whitespacesAndNewlines)
      headers[name] = value
    }

    let contentLength = Int(headers["content-length"] ?? "0") ?? 0
    if contentLength < 0 || contentLength > Self.maximumRequestBytes {
      return .error(.error(status: 413, message: "Request body too large"))
    }

    let bodyStart = headerRange.upperBound
    let totalLength = bodyStart + contentLength
    guard buffer.count >= totalLength else {
      return .incomplete
    }

    let body = buffer.subdata(in: bodyStart..<totalLength)
    let rawPath = String(requestParts[1])
    let pathAndQuery = rawPath.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
    let path = pathAndQuery.first.map(String.init) ?? rawPath
    let queryString: String? = pathAndQuery.count > 1 ? String(pathAndQuery[1]) : nil

    return .request(
      HTTPRequest(
        method: String(requestParts[0]).uppercased(),
        path: path,
        queryString: queryString,
        headers: headers,
        body: body
      )
    )
  }

  private func handle(request: HTTPRequest) {
    guard let server else {
      finish(with: .error(status: 500, message: "Server unavailable"))
      return
    }

    // Check for WebSocket upgrade before entering the async router.
    if (request.path == "/ws" || request.path == "/api/ws"), request.method == "GET" {
      if let wsResponse = server.comfyBridge.handleWebSocketUpgrade(request: request, connection: connection, queue: queue) {
        // Send the upgrade response, then keep the connection alive for WebSocket framing.
        guard !responseSent else { return }
        responseSent = true
        connection.send(content: wsResponse, completion: .contentProcessed { [weak self] _ in
          guard let self, let server = self.server else { return }
          let clientId = request.queryParameters["clientId"] ?? UUID().uuidString
          server.comfyBridge.wsManager.registerConnection(
            clientId: clientId,
            connection: self.connection,
            queue: self.queue
          )
          // Release the ConnectionHandler — the WS manager now owns the NWConnection.
          // Do NOT cancel the connection; only release our retain cycle.
          self.retainSelf = nil
        })
      } else {
        // Invalid WebSocket upgrade request — send 400 and close.
        finish(with: .error(status: 400, message: "Invalid WebSocket upgrade request"))
      }
      return
    }

    // 0.B-2 (FDD §3.1.4): serve the sync-servable control set synchronously on
    // THIS connection's queue, before any Task — zero cooperative threads, no
    // actor hop, so these routes answer even if the pool is exhausted by a
    // render. Flag off (`COMFYBOX_CONTROL_PLANE_SYNC=0`) skips this entirely and
    // every route falls through to the async path below, byte-for-byte as before.
    if ControlPlaneSyncFlag.isEnabled,
       let response = server.serveControlPlaneSync(request) {
      finish(with: response)
      return
    }

    Task {
      let routed = await server.respond(to: request)
      switch routed {
      case .error(let response):
        self.finish(with: response)
      case .json(let response):
        self.finish(with: response)
      case .shutdown(let response):
        self.finish(with: response, shutdownAfterSend: true)
      case .websocketUpgrade:
        // Should not reach here — /ws is handled above before async dispatch.
        self.finish(with: .error(status: 400, message: "Invalid WebSocket upgrade request"))
      }
    }
  }

  private func finish(with response: HTTPResponse, shutdownAfterSend: Bool = false) {
    guard !responseSent else { return }
    responseSent = true

    connection.send(content: response.serialize(), completion: .contentProcessed { [weak self] _ in
      guard let self else { return }
      self.connection.cancel()
      if shutdownAfterSend {
        self.server?.requestShutdownAfterResponse()
      }
      self.retainSelf = nil
    })
  }
}

struct HTTPRequest {
  let method: String
  let path: String
  let queryString: String?
  let headers: [String: String]
  let body: Data

  /// Parse query parameters from the query string.
  ///
  /// comfybox#380: percent-decodes both keys and values. A custom model path
  /// (or any other free-text query value — a CivitAI search term, a prompt
  /// fragment) can contain a space, `#`, a literal `%`, or non-ASCII text,
  /// none of which can survive an HTTP request line unencoded — the caller
  /// percent-encodes, and this was returning the still-encoded literal, so
  /// e.g. `GET /v1/model/family?model=...` stat'ed a path that never existed
  /// on disk (`loadable: false`) for any such spec.
  ///
  /// Deliberately `removingPercentEncoding`, NOT `+` → space: this is a URI
  /// query per RFC 3986 (`%XX` triplets only), not `application/x-www-form-
  /// urlencoded` — a literal `+` in a model path or search term must stay a
  /// `+`, not become a space. `removingPercentEncoding` already has that
  /// exact behavior (it only touches `%XX`, never `+`), so no extra handling
  /// is needed to keep `+` literal.
  ///
  /// Falls back to the raw (still-encoded) substring on a malformed escape
  /// (e.g. a trailing `%` or `%` followed by non-hex) rather than dropping
  /// the parameter — `removingPercentEncoding` returns `nil` for those, and
  /// the pre-existing behavior of returning SOMETHING for every `key=value`
  /// pair is preserved.
  var queryParameters: [String: String] {
    guard let qs = queryString, !qs.isEmpty else { return [:] }
    var params: [String: String] = [:]
    for pair in qs.split(separator: "&") {
      let parts = pair.split(separator: "=", maxSplits: 1)
      guard parts.count == 1 || parts.count == 2 else { continue }
      let rawKey = String(parts[0])
      let rawValue = parts.count == 2 ? String(parts[1]) : ""
      let key = rawKey.removingPercentEncoding ?? rawKey
      let value = rawValue.removingPercentEncoding ?? rawValue
      params[key] = value
    }
    return params
  }
}

enum HTTPParseResult {
  case incomplete
  case request(HTTPRequest)
  case error(HTTPResponse)
}

struct HTTPResponse {
  let status: Int
  let reasonPhrase: String
  let contentType: String
  let body: Data
  /// Additional response headers (e.g. `ETag`, `Warning`) beyond the fixed
  /// Content-Type/Content-Length/Connection trio `serialize()` always writes.
  /// Empty for every pre-existing call site — additive, no behavior change
  /// (FDD §3.3: advisory `ETag`/`If-Match` on `/v1/config`).
  var extraHeaders: [String: String] = [:]

  static func json<T: Encodable>(status: Int, payload: T) -> HTTPResponse {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let body = (try? encoder.encode(payload)) ?? Data("{\"success\":false,\"error\":\"encoding failure\"}".utf8)
    return HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), contentType: "application/json", body: body)
  }

  /// Create a JSON response from pre-encoded Data (no snake_case conversion).
  static func rawJSON(status: Int, data: Data) -> HTTPResponse {
    HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), contentType: "application/json", body: data)
  }

  /// Create a binary response with a specified content type.
  static func binary(status: Int, contentType: String, data: Data) -> HTTPResponse {
    HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), contentType: contentType, body: data)
  }

  static func empty(status: Int) -> HTTPResponse {
    HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), contentType: "application/json", body: Data())
  }

  static func error(status: Int, message: String) -> HTTPResponse {
    json(status: status, payload: ErrorPayload(success: false, error: message))
  }

  func serialize() -> Data {
    var data = Data()
    // No CORS headers: all known clients (desktop app, Krita plugin, Telegram
    // bot, MCP) are native, so browser cross-origin access is intentionally
    // not enabled.
    var lines = [
      "HTTP/1.1 \(status) \(reasonPhrase)",
      "Content-Type: \(contentType)",
      "Content-Length: \(body.count)",
      "Connection: close",
    ]
    // Deterministic order (sorted) so header emission is stable across runs —
    // matters for tests asserting on exact serialized bytes.
    for (name, value) in extraHeaders.sorted(by: { $0.key < $1.key }) {
      lines.append("\(name): \(value)")
    }
    lines.append("")
    lines.append("")
    data.append(Data(lines.joined(separator: "\r\n").utf8))
    data.append(body)
    return data
  }

  static func reasonPhrase(for status: Int) -> String {
    switch status {
    case 204: return "No Content"
    case 200: return "OK"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 409: return "Conflict"
    case 413: return "Payload Too Large"
    case 429: return "Too Many Requests"
    case 500: return "Internal Server Error"
    case 502: return "Bad Gateway"
    case 503: return "Service Unavailable"
    case 507: return "Insufficient Storage"
    default: return "OK"
    }
  }
}

enum RoutedResponse {
  case json(HTTPResponse)
  case shutdown(HTTPResponse)
  case error(HTTPResponse)
  /// WebSocket upgrade — the bridge takes ownership of the connection.
  case websocketUpgrade

  static func json<T: Encodable>(status: Int, payload: T) -> RoutedResponse {
    .json(.json(status: status, payload: payload))
  }

  static func shutdown<T: Encodable>(status: Int, payload: T) -> RoutedResponse {
    .shutdown(.json(status: status, payload: payload))
  }
}

/// WP-E17 (FDD-krea2-raw-recipe §3.14, D4, D25): the `stage2` object on
/// `/v1/generate` — the detail pass, inside ONE render.
///
/// Additive: a payload without it is byte-identical to today. `steps` and
/// `denoise` are REQUIRED because they are the two fields that decide the
/// stretched grid, and a default for either would be an engine-invented recipe;
/// everything else absent means "the render's own value" (see
/// ``Krea2Pipeline/Stage2``). The family's published pairing lives in the
/// client's policy table (WP-C8) and arrives here spelled out.
struct Stage2Payload: Sendable, Decodable, Equatable {
  let steps: Int
  /// **`Double`** — every other float on `GeneratePayload` decodes as `Float`
  /// and this one must not: `total = int(steps/denoise)` is sensitive to which
  /// side of the integer the division lands on (§3.14, AC-31).
  let denoise: Double
  /// D25: the wire key is `scheduler`, with `sampler` as an accepted alias;
  /// both present and different is a 400, exactly as at the top level.
  let scheduler: String?
  let sigmaSchedule: String?
  let guidance: Float?
  let eta: Float?
  let bongmath: Bool?
  /// `null`/absent → the stage-1 seed `&+ 1`, recorded either way.
  let seed: UInt64?

  private enum CodingKeys: String, CodingKey {
    case steps, denoise, scheduler, sampler, sigmaSchedule, guidance, eta, bongmath, seed
  }

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    guard let steps = try c.decodeIfPresent(Int.self, forKey: .steps) else {
      throw WarmServerError.invalidRequest(
        message: "stage2.steps is required — it decides the stretched grid and has no default")
    }
    guard let denoise = try c.decodeIfPresent(Double.self, forKey: .denoise) else {
      throw WarmServerError.invalidRequest(
        message: "stage2.denoise is required — it decides the stretched grid and has no default")
    }
    self.steps = steps
    self.denoise = denoise
    let schedulerRaw = try c.decodeIfPresent(String.self, forKey: .scheduler)
    let samplerRaw = try c.decodeIfPresent(String.self, forKey: .sampler)
    if let schedulerRaw, let samplerRaw, schedulerRaw != samplerRaw {
      throw WarmServerError.mutuallyExclusive(
        "stage2.scheduler='\(schedulerRaw)' and stage2.sampler='\(samplerRaw)' disagree — "
          + "'sampler' is an alias of 'scheduler'; send one, or the same value in both")
    }
    self.scheduler = schedulerRaw ?? samplerRaw
    self.sigmaSchedule = try c.decodeIfPresent(String.self, forKey: .sigmaSchedule)
    self.guidance = try c.decodeIfPresent(Float.self, forKey: .guidance)
    self.eta = try c.decodeIfPresent(Float.self, forKey: .eta)
    self.bongmath = try c.decodeIfPresent(Bool.self, forKey: .bongmath)
    self.seed = try c.decodeIfPresent(UInt64.self, forKey: .seed)
  }

  /// For bridge/test construction; the wire always goes through `init(from:)`.
  init(
    steps: Int, denoise: Double, scheduler: String? = nil, sigmaSchedule: String? = nil,
    guidance: Float? = nil, eta: Float? = nil, bongmath: Bool? = nil, seed: UInt64? = nil
  ) {
    self.steps = steps
    self.denoise = denoise
    self.scheduler = scheduler
    self.sigmaSchedule = sigmaSchedule
    self.guidance = guidance
    self.eta = eta
    self.bongmath = bongmath
    self.seed = seed
  }
}

struct GeneratePayload: Sendable {
  let prompt: String
  let negativePrompt: String?
  let width: Int?
  let height: Int?
  /// `var` since #286: filled from the named `preset`'s DECLARED `steps` when
  /// the request omitted them (never from `ResolvedPreset`, whose default is 4).
  var steps: Int?
  /// `var` since #286, same rule as `steps`.
  var guidance: Float?
  let seed: UInt64?
  let outputPath: String?
  let levelsMin: Float?
  let levelsMax: Float?
  let scheduler: String?
  let sigmaSchedule: String?
  let eta: Float?
  /// RES4LYF `bongmath` (parity tier T3, WP-E16). Krea 2 + the RES4LYF
  /// samplers only; asked for with any other sampler it is a 400 naming the
  /// sampler (`validateKrea2TierGates`), never a silent drop. Absent/false is
  /// byte-identical to today.
  let bongmath: Bool?
  /// Explicit schedule shift. nil = the model's own resolution-dependent
  /// default; a value is validated by `validateShift(_:family:)` → 400, never
  /// clamped, and refused on a family that does not read it.
  ///
  /// The two families that honour it mean DIFFERENT things by it:
  ///
  /// - Krea 2 (FDD-krea2-raw-recipe D3, Addendum A.1): the value IS `mu`,
  ///   ComfyUI's `ModelSamplingFlux(shift=…)` **log**-shift (`1.15`
  ///   reproduces the published grid).
  /// - Z-Image / Flux 1 (comfybox#154, the family Zeta Chroma runs on): the
  ///   value is the LINEAR shift of ComfyUI's `ModelSamplingAuraFlow`,
  ///   `σ' = shift·σ / (1 + (shift − 1)·σ)`. `1.0` is the identity; Zeta
  ///   Chroma's author recommends `3.0`. It replaces the model's own shift and
  ///   the resolution-dependent `mu` for that render.
  ///
  /// `var` since #154, the same rule as `steps`/`guidance`/`vae`: filled from
  /// the named `preset`'s DECLARED `shift` when the request carried none.
  var shift: Float?
  let dype: String?
  // Phase 3: Inpainting data (set by bridge, not by HTTP API)
  let inpaintImageData: Data?
  let maskData: Data?
  let denoise: Float?
  let maskGrow: Int?
  let maskFeather: Int?
  let maskCropX: Int?
  let maskCropY: Int?

  // Chroma CFG parameters
  let cfg: Float?
  let firstNStepsWithoutCFG: Int?

  // Phase 4: Img2img (set via HTTP API)
  var imagePath: String?   // var: may be filled in from initImageData (bytes upload)
  /// Img2img init image sent as base64 (init_image_base64) — for remote clients
  /// that can't put a file on the server's filesystem. Decoded to a temp file.
  let initImageData: Data?
  let imageStrength: Float?
  let creativity: Float?

  /// Optional mask PNG file path for selective inpainting on the img2img path.
  /// White = inpaint region, black = keep. nil → standard full-frame img2img.
  let maskPath: String?

  /// Auto-generated mask region ("face" | "upper" | "lower") for img2img —
  /// mutually exclusive with maskPath. The named region is regenerated.
  let maskRegion: String?

  /// Flip the img2img mask (white ⇄ black): e.g. mask_region "face" +
  /// mask_invert = lock the face, regenerate everything else.
  let maskInvert: Bool?

  /// Submitting client/app (desktop, bree, api…) — for queue attribution.
  let source: String?

  /// Preset id for this request. Carried into the gallery filename + PNG
  /// metadata (Todd 2026-08-11), and — since #286 — EXPANDED into `model`,
  /// `loras` and the preset's declared `steps`/`guidance`, through the same
  /// `PresetStore` read that backs `POST /v1/presets/resolve`
  /// (``WarmServer/expandGeneratePayload(_:store:stageNearline:log:)``).
  ///
  /// It used to be a label only, which meant a preset-by-name render used
  /// whatever LoRA stack the warm pipeline happened to hold — stale adapters
  /// from an earlier swap, or none at all after a restart — on whatever base
  /// was active, and reported success either way.
  ///
  /// It is STILL a label whenever the engine cannot expand the preset
  /// (`presetUnresolved`), which is never an error; and when the request
  /// carries its own `loras`/`model`, those still win. The single hard failure
  /// is a request `model` that contradicts the preset's (409).
  let preset: String?

  /// Fruit mode (neutral | banana | avocado) — stamped into render metadata.
  let contentMode: String?

  /// Per-job model override (spec/CivitAI id/pool key). When set, the job's
  /// own model is loaded/activated at dequeue time instead of trusting
  /// whatever the shared pool's "currently active" model happens to be —
  /// required for queue-submit (POST /v1/generate/async) to be race-free,
  /// since a job's dequeue can happen well after another request changed
  /// the active model. nil preserves the old "caller activates first"
  /// behavior for direct /v1/generate callers.
  ///
  /// `var` since #286: filled from the named `preset`'s `model` when the
  /// request named none. A preset's adapters must never be applied to a
  /// different base — an explicit `model` that contradicts the preset's is a
  /// 409 (``WarmServerError/presetModelConflict(preset:presetModel:requestModel:)``).
  var model: String?
  /// Per-job LoRA override, applied the same way as `model` at dequeue time.
  ///
  /// `var` since #286: ``WarmServer/expandGeneratePayload(_:store:stageNearline:log:)``
  /// fills this in from the named `preset` when the request carried no `loras`
  /// of its own, so the ONE place that applies a per-request stack
  /// (`applyActiveLoRAs`, at dequeue) is also the one place a preset-by-name
  /// render goes through. See ``PresetLoRAStack``.
  ///
  /// Since #282 the dequeue applies a stack for EVERY render, not only when
  /// this field is non-nil: nil means "the warm default", which is what
  /// `/v1/lora/swap` publishes. Which of the two owns a non-nil value is
  /// recorded in ``presetStackApplied``; the whole decision is
  /// ``RequestStackResolver``.
  var loras: [LoRAEntry]?

  /// #286 (C2): set by the engine, never by the wire — the named preset could
  /// not be expanded (unknown, flagged invalid, a video preset, a non-local
  /// engine/provider, no model to load, a missing LoRA file, or a dial the
  /// engine has no application path for). The render behaves exactly as it did
  /// before #286 (the preset is a label) and the response says so via
  /// `preset_unresolved`.
  var presetUnresolved: String?
  /// #286 (round 2): the machine-readable reason code beside
  /// `presetUnresolved` — `engine:mflux`, `no_model`, `missing_lora:x`, … See
  /// ``PresetExpansion/Unresolved``. Reaches the wire as
  /// `preset_unresolved_reason`.
  var presetUnresolvedReason: String?
  /// #286 (I1): set by the engine — the request carried explicit `loras` that
  /// differ from what the named preset resolves to. The explicit list still
  /// wins; the response says so via `preset_stack_mismatch`.
  var presetStackMismatch: Bool?
  /// #22 (PR #363 review, C1b): set by `validateImageMemoryPreflight`, never
  /// by the wire — the render's estimated peak activation memory, in bytes.
  /// nil when the preflight was skipped (width/height both omitted) or never
  /// ran (replay, `gateSubmission: false`). ADVISORY: present even when the
  /// estimate exceeded budget, as long as `imageMemoryCaps.enforceMemoryEstimate`
  /// is false (the default) — that is the whole point of surfacing it, so an
  /// operator can compare this number against real `/health` samples before
  /// deciding whether to enforce.
  var memoryEstimateBytes: UInt64?
  /// #22: live free system memory at the moment the estimate above was
  /// computed, so the two numbers can be compared without a second `/health`
  /// call racing the render.
  var memoryAvailableBytes: UInt64?

  /// #282: set by the engine, never by the wire — `loras` above was filled
  /// from the named `preset`'s expansion rather than sent by the client. It is
  /// the only way ``RequestStackResolver`` can tell a preset-owned stack from
  /// a request-owned one once both live in the same field, and it is what
  /// makes `lora_stack_origin` on the response honest.
  ///
  /// A persisted-queue REPLAY reads the rewritten body, which carries the
  /// accepted stack as explicit `loras` (#286 I5) — so a replayed preset job
  /// reports origin `request`. The stack it applies is byte-identical to the
  /// one that was accepted; only the label differs, which is the price of
  /// replaying a frozen body instead of re-resolving a preset that may have
  /// been edited since.
  var presetStackApplied: Bool?

  /// #282: set at DEQUEUE by `runGenerate` — which of the three sources owned
  /// this job's stack (``RequestStackResolver/Origin``). Reaches the response
  /// as `lora_stack_origin`, so a daemon can confirm from the response alone
  /// that a render used the stack it asked for and not one it inherited.
  var loraStackOrigin: String?
  /// #282 review r1 (C1): set at dequeue when the warm default was DROPPED
  /// because it was published under a different base — `family_mismatch` or
  /// `model_mismatch`. The job rendered with no adapters. Never an error.
  var warmDefaultSkipped: String?
  /// #282 review r1 (I1): set at dequeue when this job forced a real
  /// clear+reload of the adapter stack. Reaches the wire as `lora_reload` so
  /// bare/preset alternation churn is measurable, not merely accepted.
  var loraReload: Bool?
  // Depth Control-LoRA (docs/FDD-krea2-depth-controlnet.md)
  let controlImageData: Data?
  let controlnetStrength: Float?
  let controlImage: String?  // Mac-side control map path (e.g. depth), read in place

  /// #1479: request an in-flight LTX-2 video render be checkpointed so this
  /// image job can run immediately, resuming the video afterward. Additive,
  /// default absent/false — omitting it is byte-identical to today.
  let preempt: Bool?

  /// WP-E9 (FDD §3.9, D16): path of the VAE file to decode (and encode)
  /// through — e.g. `Wan2_1_VAE_fp32.safetensors`. Tilde allowed. Absent →
  /// the model directory's VAE. A path that is not on disk FAILS the render
  /// (AC-56); the layout is sniffed from the file's keys, never its name.
  /// Krea 2 only today (the other families ignore it).
  ///
  /// `var` since #285: `WarmServer.expandGeneratePayload` fills this in from
  /// the named `preset`'s declared `vae` when the request carried none of its
  /// own — the same request > preset > model-dir precedence
  /// `RequestStackResolver` established for LoRAs. See ``PresetLoRAStack``.
  var vae: String?
  /// #285: set by the engine, never by the wire — `vae` above was filled from
  /// the named `preset`'s declared value rather than sent by the client. The
  /// only way `Krea2VAESelector.resolve` (called at dispatch, once `vae`'s
  /// two sources have collapsed onto this one field) can still tell them
  /// apart and record an honest `vae_source: "preset"` instead of
  /// `"payload"`. Mirrors ``presetStackApplied`` for LoRAs.
  var presetVAEApplied: Bool?

  /// WP-E17 (§3.14, D4): the second stage of this render. Krea 2 only —
  /// refused, never ignored, on any other family (``stage2Gate(_:family:)``).
  let stage2: Stage2Payload?

  /// The MCP tool schema's spelling of a detail pass (§3.17, AC-68a): the
  /// CLIENT expands `detail_pass` into `stage2` from its family policy table.
  /// Decoded here so the engine can REFUSE them by name — it has no policy
  /// table to expand a bare boolean into a sampler/schedule/step recipe, and
  /// inventing one is the silent substitution this programme exists to kill.
  let detailPass: Bool?
  /// `detail_denoise` without `detail_pass` is an orphan (Addendum A.2 → C3),
  /// NaN included. `Double` for the same reason `stage2.denoise` is.
  let detailDenoise: Double?

  /// Default memberwise init for bridge-created payloads.
  /// Projector-scale text-conditioning gain (wire: `projector_scale`). Krea 2
  /// only; 1.0/absent = neutral. Forwarded verbatim to Krea2Pipeline.Request.
  let projectorScale: Float?
  /// RES4LYF spatial noise generator (wire: `noise_type`: gaussian|fractal|
  /// pyramid). Krea 2 only; absent/`gaussian` = byte-identical to today.
  let noiseType: String?
  /// Fractal `alpha` exponent (wire: `noise_alpha`); only read for
  /// `noise_type: fractal`. Absent = 0.0 (fractal ≡ gaussian).
  let noiseAlpha: Float?
  /// RES4LYF implicit-RK refinement (wire: `implicit_steps`). Krea 2 + the
  /// RES4LYF explicit tableaus only; re-iterates the tableau this many extra
  /// times as a fixed point. Absent/0 = byte-identical to today. Mirrors
  /// `eta`/`bongmath`: decoded here, forwarded to Krea2Pipeline.Request.
  let implicitSteps: Int?
  /// RES4LYF `res_2s` / `res_3s` substep location (wire: `c2`). Krea 2
  /// only; absent = 0.5, preserving the existing scheduler recipe.
  let c2: Float?
  init(
    prompt: String, negativePrompt: String? = nil,
    width: Int? = nil, height: Int? = nil, steps: Int? = nil,
    guidance: Float? = nil, seed: UInt64? = nil, outputPath: String? = nil,
    levelsMin: Float? = nil, levelsMax: Float? = nil,
    scheduler: String? = nil, sigmaSchedule: String? = nil, eta: Float? = nil,
    bongmath: Bool? = nil,
    shift: Float? = nil,
    dype: String? = nil, inpaintImageData: Data? = nil, maskData: Data? = nil,
    denoise: Float? = nil, maskGrow: Int? = nil, maskFeather: Int? = nil,
    maskCropX: Int? = nil, maskCropY: Int? = nil,
    cfg: Float? = nil, firstNStepsWithoutCFG: Int? = nil,
    imagePath: String? = nil, imageStrength: Float? = nil, creativity: Float? = nil,
    maskPath: String? = nil, maskRegion: String? = nil, maskInvert: Bool? = nil,
    source: String? = nil, contentMode: String? = nil, initImageData: Data? = nil,
    model: String? = nil, loras: [LoRAEntry]? = nil,
    controlImageData: Data? = nil, controlnetStrength: Float? = nil, controlImage: String? = nil,
    preempt: Bool? = nil, vae: String? = nil,
    stage2: Stage2Payload? = nil, detailPass: Bool? = nil, detailDenoise: Double? = nil,
    projectorScale: Float? = nil,
    noiseType: String? = nil, noiseAlpha: Float? = nil,
    implicitSteps: Int? = nil, c2: Float? = nil
  ) {
    self.preempt = preempt
    self.vae = vae
    self.presetVAEApplied = nil
    self.stage2 = stage2
    self.detailPass = detailPass
    self.detailDenoise = detailDenoise
    self.projectorScale = projectorScale
    self.noiseType = noiseType
    self.noiseAlpha = noiseAlpha
    self.implicitSteps = implicitSteps
    self.c2 = c2
    self.source = source
    self.preset = nil
    self.contentMode = contentMode
    self.initImageData = initImageData
    self.model = model
    self.loras = loras
    self.presetUnresolved = nil
    self.presetUnresolvedReason = nil
    self.presetStackMismatch = nil
    self.memoryEstimateBytes = nil
    self.memoryAvailableBytes = nil
    self.presetStackApplied = nil
    self.loraStackOrigin = nil
    self.warmDefaultSkipped = nil
    self.loraReload = nil
    self.controlImageData = controlImageData; self.controlnetStrength = controlnetStrength; self.controlImage = controlImage
    self.prompt = prompt; self.negativePrompt = negativePrompt
    self.width = width; self.height = height; self.steps = steps
    self.guidance = guidance; self.seed = seed; self.outputPath = outputPath
    self.levelsMin = levelsMin; self.levelsMax = levelsMax
    self.scheduler = scheduler; self.sigmaSchedule = sigmaSchedule
    self.eta = eta; self.bongmath = bongmath; self.shift = shift; self.dype = dype
    self.inpaintImageData = inpaintImageData; self.maskData = maskData
    self.denoise = denoise; self.maskGrow = maskGrow; self.maskFeather = maskFeather
    self.maskCropX = maskCropX; self.maskCropY = maskCropY
    self.cfg = cfg; self.firstNStepsWithoutCFG = firstNStepsWithoutCFG
    self.imagePath = imagePath; self.imageStrength = imageStrength; self.creativity = creativity
    self.maskPath = maskPath
    self.maskRegion = maskRegion
    self.maskInvert = maskInvert
  }
}

extension GeneratePayload: Decodable {
  private enum CodingKeys: String, CodingKey {
    case prompt, negativePrompt, width, height, steps, guidance, seed
    case outputPath, levelsMin, levelsMax, scheduler, sigmaSchedule, eta, bongmath, shift, dype
    /// D25: `sampler` is an accepted ALIAS of the wire key `scheduler` (so a
    /// value pasted out of a ComfyUI/RES4LYF UI works). The request key stays
    /// `scheduler` — two live senders post that spelling — while the response
    /// record reports `applied.stages[].sampler` (WP-E10); the asymmetry is
    /// deliberate. Both present and different → `mutuallyExclusive` (400).
    case sampler
    case preset
    case denoise, maskGrow, maskFeather
    // NOTE: the /v1/generate decoder uses .convertFromSnakeCase, which rewrites
    // incoming keys to camelCase BEFORE matching CodingKey stringValues. So the
    // wire keys inpaint_image_base64 / mask_base64 arrive as these camelCase
    // forms — the rawValues MUST be the post-conversion spelling, not snake_case.
    case inpaintImageData = "inpaintImageBase64"
    case maskImageData = "maskBase64"
    case cfg, firstNStepsWithoutCFG
    case imagePath, imageStrength, creativity
    case maskPath
    case maskRegion
    case maskInvert
    case source
    case contentMode
    // Wire key init_image_base64 arrives as this camelCase form after
    // .convertFromSnakeCase (same gotcha as the inpaint keys).
    case initImageData = "initImageBase64"
    case controlImageData
    case controlnetStrength
    case controlImage
    case model, loras
    case preempt
    case vae
    // WP-E17. `stage2` has no underscore, so `.convertFromSnakeCase` leaves it
    // alone; `detail_pass` / `detail_denoise` arrive as these camelCase forms.
    case stage2
    case detailPass
    case detailDenoise
    case projectorScale
    // `noise_type` / `noise_alpha` arrive as these camelCase forms after
    // `.convertFromSnakeCase`.
    case noiseType
    case noiseAlpha
    // `implicit_steps` arrives as this camelCase form after .convertFromSnakeCase.
    case implicitSteps
    case c2
  }

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    prompt = try c.decode(String.self, forKey: .prompt)
    negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
    width = try c.decodeIfPresent(Int.self, forKey: .width)
    height = try c.decodeIfPresent(Int.self, forKey: .height)
    steps = try c.decodeIfPresent(Int.self, forKey: .steps)
    guidance = try c.decodeIfPresent(Float.self, forKey: .guidance)
    seed = try c.decodeIfPresent(UInt64.self, forKey: .seed)
    outputPath = try c.decodeIfPresent(String.self, forKey: .outputPath)
    levelsMin = try c.decodeIfPresent(Float.self, forKey: .levelsMin)
    levelsMax = try c.decodeIfPresent(Float.self, forKey: .levelsMax)
    let schedulerRaw = try c.decodeIfPresent(String.self, forKey: .scheduler)
    let samplerRaw = try c.decodeIfPresent(String.self, forKey: .sampler)
    if let schedulerRaw, let samplerRaw, schedulerRaw != samplerRaw {
      throw WarmServerError.mutuallyExclusive(
        "scheduler='\(schedulerRaw)' and sampler='\(samplerRaw)' disagree — 'sampler' is an alias of 'scheduler'; send one, or the same value in both")
    }
    scheduler = schedulerRaw ?? samplerRaw
    sigmaSchedule = try c.decodeIfPresent(String.self, forKey: .sigmaSchedule)
    eta = try c.decodeIfPresent(Float.self, forKey: .eta)
    bongmath = try c.decodeIfPresent(Bool.self, forKey: .bongmath)
    projectorScale = try c.decodeIfPresent(Float.self, forKey: .projectorScale)
    shift = try c.decodeIfPresent(Float.self, forKey: .shift)
    dype = try c.decodeIfPresent(String.self, forKey: .dype)
    // Inpaint image + mask arrive as base64 strings from the HTTP API.
    inpaintImageData = (try c.decodeIfPresent(String.self, forKey: .inpaintImageData))
        .flatMap { Data(base64Encoded: $0) }
    maskData = (try c.decodeIfPresent(String.self, forKey: .maskImageData))
        .flatMap { Data(base64Encoded: $0) }
    denoise = try c.decodeIfPresent(Float.self, forKey: .denoise)
    maskGrow = try c.decodeIfPresent(Int.self, forKey: .maskGrow)
    maskFeather = try c.decodeIfPresent(Int.self, forKey: .maskFeather)
    maskCropX = nil
    maskCropY = nil
    cfg = try c.decodeIfPresent(Float.self, forKey: .cfg)
    firstNStepsWithoutCFG = try c.decodeIfPresent(Int.self, forKey: .firstNStepsWithoutCFG)
    imagePath = try c.decodeIfPresent(String.self, forKey: .imagePath)
    initImageData = (try c.decodeIfPresent(String.self, forKey: .initImageData))
        .flatMap { Data(base64Encoded: $0) }
    imageStrength = try c.decodeIfPresent(Float.self, forKey: .imageStrength)
    creativity = try c.decodeIfPresent(Float.self, forKey: .creativity)
    maskPath = try c.decodeIfPresent(String.self, forKey: .maskPath)
    maskRegion = try c.decodeIfPresent(String.self, forKey: .maskRegion)
    maskInvert = try c.decodeIfPresent(Bool.self, forKey: .maskInvert)
    source = try c.decodeIfPresent(String.self, forKey: .source)
    preset = try c.decodeIfPresent(String.self, forKey: .preset)
    contentMode = try c.decodeIfPresent(String.self, forKey: .contentMode)
    model = try c.decodeIfPresent(String.self, forKey: .model)
    loras = try c.decodeIfPresent([LoRAEntry].self, forKey: .loras)
    // #286: engine-set, never decoded from the wire.
    presetUnresolved = nil
    presetUnresolvedReason = nil
    presetStackMismatch = nil
    // #22: engine-set (validateImageMemoryPreflight), never decoded from the wire.
    memoryEstimateBytes = nil
    memoryAvailableBytes = nil
    // #282: engine-set, never decoded from the wire. A replayed persisted body
    // carries its accepted stack as explicit `loras`, so it decodes here as a
    // request-owned stack — the same adapters, under the honest label.
    presetStackApplied = nil
    loraStackOrigin = nil
    warmDefaultSkipped = nil
    loraReload = nil
    controlImageData = (try c.decodeIfPresent(String.self, forKey: .controlImageData)).flatMap { Data(base64Encoded: $0) }
    controlnetStrength = try c.decodeIfPresent(Float.self, forKey: .controlnetStrength)
    controlImage = try c.decodeIfPresent(String.self, forKey: .controlImage)
    preempt = try c.decodeIfPresent(Bool.self, forKey: .preempt)
    vae = try c.decodeIfPresent(String.self, forKey: .vae)
    // #285: engine-set, never decoded from the wire — see its doc comment.
    presetVAEApplied = nil
    stage2 = try c.decodeIfPresent(Stage2Payload.self, forKey: .stage2)
    detailPass = try c.decodeIfPresent(Bool.self, forKey: .detailPass)
    detailDenoise = try c.decodeIfPresent(Double.self, forKey: .detailDenoise)
    noiseType = try c.decodeIfPresent(String.self, forKey: .noiseType)
    noiseAlpha = try c.decodeIfPresent(Float.self, forKey: .noiseAlpha)
    implicitSteps = try c.decodeIfPresent(Int.self, forKey: .implicitSteps)
    c2 = try c.decodeIfPresent(Float.self, forKey: .c2)
  }

  /// Validate the `shift` field for the family that will render it.
  ///
  /// Returns a 400 message, or nil when the request is acceptable. `shift` is
  /// always a positive finite number, and it is refused — not ignored — on a
  /// family whose schedules do not read it ("fail loud, never silently
  /// substitute").
  ///
  /// Two families honour it, and they mean DIFFERENT things by it — which is
  /// why this gate is per-family and not a decoder check:
  ///
  /// - `.krea2` (FDD-krea2-raw-recipe D3, Addendum A.1): the value IS `mu`,
  ///   ComfyUI's `ModelSamplingFlux(shift=…)` **log**-shift. `1.15` reproduces
  ///   the published grid.
  /// - `.flux1` — the Z-Image / Flux-1 family, and the one Zeta Chroma runs on
  ///   (comfybox#154): the value is the LINEAR shift of ComfyUI's
  ///   `ModelSamplingAuraFlow` / `ModelSamplingSD3`,
  ///   `σ' = shift·σ / (1 + (shift − 1)·σ)`. `1.0` is the identity; Zeta
  ///   Chroma's author recommends `3.0`. It replaces the model's own shift and
  ///   the resolution-dependent `mu` for that render
  ///   (``ZImageSchedulerConfig/applyingExplicitShift(_:)``).
  ///
  /// The remaining families (`.flux2`, `.fibo`, `.chroma`) run fixed schedules
  /// that never consult it, so they still refuse it by name.
  static func validateShift(_ shift: Float?, family: WarmModelFamily) -> String? {
    guard let shift else { return nil }
    guard shift.isFinite, shift > 0 else {
      return "shift must be a positive number (got \(shift)); omit it for the resolution-dependent default"
    }
    guard family == .krea2 || family == .flux1 else {
      return "shift is a schedule field for the krea2 and flux1 (Z-Image) families and is not "
        + "honoured by model family '\(family.rawValue)'; remove it"
    }
    return nil
  }

  /// comfybox#154 — the SIGMA-SCHEDULE half of the shift gate: refuse a shift
  /// the requested schedule would drop on the floor.
  ///
  /// `validateShift` above answers "does this family read `shift` at all?".
  /// This answers the narrower question the family cannot: three schedules are
  /// defined by something other than `config.shift` and ignore it completely —
  /// `krea2` (which is `mu`), `bong_tangent` (model-free by construction, D6),
  /// and, under Krea 2's `.flux` model sampling, the table-backed
  /// `simple`/`beta`/`beta57`/`karras`/`exponential`. Accepting a shift into
  /// one of those and reporting success is precisely the silent substitution
  /// the recipe gates exist to stop, and it is what would make the response's
  /// `applied_shift` a lie.
  ///
  /// Scoped to `.flux1`, whose scheduler configs are always `.discreteFlow`
  /// (the `scheduler_config.json` decoder hardcodes it), so the model sampling
  /// is known here without loading anything. Krea 2 has its own `shift`
  /// semantics and its own `applied.stages[].shift_applied` record and is left
  /// alone.
  ///
  /// Returns the 400 message, or nil.
  static func validateShiftSchedule(
    _ shift: Float?, sigmaSchedule: SigmaScheduleKind?, family: WarmModelFamily
  ) -> String? {
    guard shift != nil, family == .flux1 else { return nil }
    let schedule = sigmaSchedule ?? .flow
    guard !SchedulerFactory.honoursExplicitShift(
      schedule: schedule, modelSampling: .discreteFlow)
    else { return nil }
    return "shift is not read by sigma schedule '\(schedule.rawValue)' "
      + "(it is defined by mu / by index arithmetic, not by the model's shift) — "
      + "the render would silently ignore it; drop `shift`, or ask for a schedule that "
      + "honours it (flow/normal, simple, beta, beta57, karras, exponential)"
  }

  /// comfybox#154 — the BRIDGE's gate on a `ModelSamplingAuraFlow` node.
  ///
  /// The node's `shift` is a LINEAR warp (`σ' = shift·σ / (1 + (shift−1)·σ)`).
  /// The engine's `shift` request field means that on `.flux1` and something
  /// else entirely on Krea 2 — `mu`, a LOG-shift feeding
  /// `ModelSamplingFlux`'s `e^mu / (e^mu + 1/σ − 1)` — so forwarding the node's
  /// number into a Krea 2 render would apply a wildly different grid under the
  /// caller's number (`shift 3.0` as mu is `e³ ≈ 20`, not 3).
  ///
  /// Refused, not ignored: a Krita workflow that PLACED the node asked for
  /// something, and rendering as if it had not is the failure mode this repo
  /// treats as worse than an error. The 400 names the node and the family.
  static func auraFlowNodeGate(_ shift: Float?, family: WarmModelFamily) -> WarmServerError? {
    guard shift != nil, family != .flux1 else { return nil }
    return .unsupportedRecipeField(
      field: "ModelSamplingAuraFlow.shift",
      value: shift.map { "\($0)" } ?? "",
      family: family.rawValue,
      reason: family == .krea2
        ? "this workflow's ModelSamplingAuraFlow node sets a LINEAR schedule shift, but the "
          + "resident krea2 model is sampled by ModelSamplingFlux, whose shift is a LOG-shift "
          + "(mu) — the same number means a different grid. Remove the node, or send `shift` on "
          + "/v1/generate where the krea2 meaning is documented"
        : "this workflow's ModelSamplingAuraFlow node sets a schedule shift, and the resident "
          + "'\(family.rawValue)' model runs a fixed schedule that does not read one. Remove the node")
  }

  /// WP-E10 "E9b" (Addendum A.2, E9 review MAJOR): `vae` is a Krea 2 request
  /// field. On any other family it used to be silently ignored — the caller
  /// named a decoder and got the family's default with no error. Refused
  /// with the D18 shape (`unsupportedRecipeField` → 400) naming the field and
  /// the family. nil when the request carries no `vae` or the family honours it.
  static func vaeGate(_ vae: String?, family: WarmModelFamily) -> WarmServerError? {
    guard let vae, family != .krea2 else { return nil }
    return .unsupportedRecipeField(
      field: "vae", value: vae, family: family.rawValue,
      reason: "VAE selection is a Krea 2 request field (WP-E9); this family decodes through its own VAE and does not honour it — remove it")
  }

  /// K-FIX-1 / Codex I5: refuse a sampler / sigma schedule the ACTIVE FAMILY
  /// cannot honour, from the one family capability matrix
  /// (``FamilyRecipeMatrix``).
  ///
  /// Supersedes WP-E13's `validateTableauSampler`, which is now one row of
  /// that table: N-row tableaus (`ralston_2s/3s/4s`, `res_3s`) are dispatched
  /// only by `Krea2DenoiseLoop`. The table also closes the silences E13's
  /// single row left open — Chroma ignoring its own native `heun`/`beta`, and
  /// Flux 2 / FIBO accepting any name into a fixed Euler loop.
  ///
  /// Names stay accepted and advertised globally (E4: advertised == accepted
  /// as a NAME); what is family-scoped is whether the render can honour them.
  /// Returns the 400 to throw, or nil. Runs at the one dispatch point in
  /// `generate` and in the bridge's family arm, beside
  /// `validateShift(_:family:)` (D18: family gates live at dispatch, not at
  /// the decoder).
  static func validateFamilyRecipe(
    _ names: ResolvedRecipeNames, family: WarmModelFamily
  ) -> WarmServerError? {
    FamilyRecipeMatrix.validate(names, family: family)
  }

  /// The DyPE configuration this payload implies at the given resolution.
  ///
  /// An explicit `dype` always wins, including "none". Otherwise DyPE
  /// auto-enables above the model's base resolution — the branch that matters
  /// most, since the callers that need it (Kira's HQ 2K rerender, the Krita
  /// bridge) send no `dype` at all.
  ///
  /// `.ntk` is deliberately the ceiling: `.yarn` is an unimplemented stub that
  /// warns and falls back to NTK, so selecting it would only add log noise.
  func resolvedDyPEConfig(width resolvedWidth: Int, height resolvedHeight: Int) -> DyPEConfig {
    if let raw = dype?.lowercased() {
      switch raw {
      case "ntk": return .ntk
      case "yarn": return .yarn
      default: return .disabled
      }
    }
    return max(resolvedWidth, resolvedHeight) > 1024 ? .ntk : .disabled
  }

  /// #22: memory/resolution preflight, run at `decodedGeneratePayload` — the
  /// ONE decode+validate choke point both `/v1/generate` and
  /// `/v1/generate/async` share — so it runs before any model load, exactly
  /// like `validateOutputPath`/`validateRecipeNames` beside it. A SEPARATE
  /// pure-ish method (rather than inlined into `decodedGeneratePayload`) so
  /// it stays a single, obviously-reviewable diff independent of whatever
  /// else that call site is doing.
  ///
  /// Skipped when either dimension is omitted: an omitted width/height either
  /// resolves to the small 1024×1024 engine default (never DyPE territory —
  /// see `resolvedDyPEConfig`'s own `> 1024` gate) or, for img2img, is
  /// derived from the source image later in the pipeline — a case this
  /// decode-time gate cannot see (`makeImg2ImgRequest`'s own comment on why
  /// it does not inject a config-default width/height either).
  ///
  /// `caps`/`availableBytes` default to LIVE reads (the real config store,
  /// the real machine's free memory) — evaluated fresh at each call with no
  /// argument, exactly like `StatsProvider.uptimeSeconds(now: Date = Date())`
  /// elsewhere in this file. Tests inject deterministic values instead of
  /// depending on whatever config/memory happens to be live on the runner.
  ///
  /// `warmFamily` — I4 (PR #363 review): the WARM/active family, when the
  /// caller knows it (`await coordinator.modelFamily`), used only when the
  /// request carries no explicit `model`. `log` — C1b: called with a warning
  /// line when the live-memory budget would have refused but
  /// `enforceMemoryEstimate` is false (the default), so the refusal is
  /// visible in the server log even though the request proceeds. Mutates
  /// `memoryEstimateBytes`/`memoryAvailableBytes` on success (or advisory
  /// pass) so the caller can stamp them onto the eventual response.
  @discardableResult
  mutating func validateImageMemoryPreflight(
    warmFamily: WarmModelFamily? = nil,
    caps: ImageMemoryCapsConfig = ServerConfigStore.shared.imageMemoryCaps(),
    availableBytes: UInt64 = MemoryProbe.systemAvailableMemoryBytes(),
    log: (String) -> Void = { _ in }
  ) throws -> ImageMemoryPreflight.Outcome? {
    guard let width, let height else { return nil }
    let family = ImageMemoryPreflight.resolvedFamily(model: model, warmFamily: warmFamily)
    let dype = resolvedDyPEConfig(width: width, height: height).enabled
    let outcome = try ImageMemoryPreflight.validate(
      width: width, height: height, family: family, dype: dype,
      caps: caps, availableBytes: availableBytes)
    memoryEstimateBytes = outcome.estimateBytes
    memoryAvailableBytes = outcome.availableBytes
    if !outcome.withinBudget {
      log("advisory (enforceMemoryEstimate=false, request proceeding) — \(outcome.reason)")
    }
    return outcome
  }

  /// F3 (comfybox#324, adversarial review of Phase 3 config): `configDefaults`
  /// defaults to a LIVE read of `ServerConfigStore.shared` — evaluated fresh
  /// at each call with no argument, exactly like `validateImageMemoryPreflight`'s
  /// `caps`/`availableBytes` above. Tests inject an explicit `RenderDefaultValues`
  /// instead of depending on whatever config happens to be live on the runner
  /// (or reaching into `.shared`, which a unit test must never mutate — see
  /// `ComfyBoxStateDirectoryIsolation.swift`), so the `??` resolution chain
  /// below is exercised at THIS real call site rather than re-implemented
  /// inline in a test file.
  func makePipelineRequest(
    configuration: WarmServerConfiguration,
    activeLoRAs: [LoRAConfiguration],
    configDefaults: RenderDefaultValues = ServerConfigStore.shared.renderDefaults(family: "flux1")
  ) throws -> ZImageGenerationRequest {
    let outputURL = try resolvedOutputURL(
      configuration: configuration,
      defaultFilename: ComfyBoxOutputNaming.defaultFilename(
        modelSpec: configuration.modelSpec ?? "z-image", presetId: preset,
        contentMode: contentMode, source: source)
    )

    let names = try validateRecipeNames()
    let schedulerKind = names.scheduler ?? .euler
    let sigmaScheduleKind = names.sigmaSchedule ?? .flow

    // FDD §3.3, D3: config-layer render defaults for the base Z-Image family
    // ("flux1" internally — WarmModelFamily's default case), resolved fresh
    // (lock, no disk I/O) and slotted BELOW request/preset, ABOVE the engine's
    // own hardcoded fallback. An unmigrated/empty config resolves every field
    // to nil, so `?? ZImageModelMetadata.recommendedX` below is unchanged.

    // Build DyPE config — auto-enable for high-res requests
    let resolvedWidth = width ?? configDefaults.width ?? ZImageModelMetadata.recommendedWidth
    let resolvedHeight = height ?? configDefaults.height ?? ZImageModelMetadata.recommendedHeight
    let dyPEConfig = resolvedDyPEConfig(width: resolvedWidth, height: resolvedHeight)

    return ZImageGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: resolvedWidth,
      height: resolvedHeight,
      steps: steps ?? configDefaults.steps ?? ZImageModelMetadata.recommendedInferenceSteps,
      guidanceScale: guidance ?? configDefaults.guidance.map(Float.init) ?? ZImageModelMetadata.recommendedGuidanceScale,
      seed: seed,
      outputPath: outputURL,
      levelsMin: levelsMin ?? 0.0,
      levelsMax: levelsMax ?? 1.0,
      model: configuration.modelSpec,
      source: source,
      contentMode: contentMode,
      textEncoderPath: configuration.textEncoderPath,
      maxSequenceLength: configuration.maxSequenceLength,
      loras: activeLoRAs,
      enhancePrompt: false,
      enhanceMaxTokens: 512,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly,
      schedulerKind: schedulerKind,
      sigmaSchedule: sigmaScheduleKind,
      eta: eta,
      // #154: the explicit ModelSamplingAuraFlow shift, already validated for
      // this family by `validateShift` at dispatch. nil = the model's own
      // resolution-dependent schedule, unchanged.
      shift: shift,
      dyPE: dyPEConfig,
      inpaintImageData: inpaintImageData,
      maskData: maskData,
      denoise: denoise ?? 1.0,
      maskGrow: maskGrow ?? 0,
      maskFeather: maskFeather ?? 0,
      maskCropX: maskCropX ?? 0,
      maskCropY: maskCropY ?? 0
    )
  }

  /// F3 (comfybox#324): `configDefaults` is injectable — see
  /// `makePipelineRequest`'s doc comment for why.
  func makeImg2ImgRequest(
    configuration: WarmServerConfiguration,
    activeLoRAs: [LoRAConfiguration],
    configDefaults: RenderDefaultValues = ServerConfigStore.shared.renderDefaults(family: "flux1")
  ) throws -> Img2ImgRequest {
    guard let imagePath else {
      fatalError("makeImg2ImgRequest called without imagePath")
    }

    if imageStrength != nil && creativity != nil {
      throw Img2ImgValidationError.mutuallyExclusive("imageStrength and creativity cannot both be specified")
    }

    let resolvedStrength: Float
    let specifiedAs: Img2ImgRequest.Img2ImgSpecifier
    if let creativity {
      resolvedStrength = 1.0 - max(0.01, min(0.99, creativity))
      specifiedAs = .creativity
    } else if let imageStrength {
      resolvedStrength = imageStrength
      specifiedAs = .strength
    } else if let denoise {
      resolvedStrength = 1.0 - max(0.01, min(0.99, denoise))
      specifiedAs = .denoise
    } else {
      resolvedStrength = 0.3
      specifiedAs = .strength
    }

    let names = try validateRecipeNames()
    let schedulerKind = names.scheduler ?? .euler
    let sigmaScheduleKind = names.sigmaSchedule ?? .flow

    // FDD §3.3, D3: same config-layer defaults as makePipelineRequest. Note
    // width/height are passed through UNRESOLVED below (`width`/`height`, not
    // `resolvedWidth`/`resolvedHeight`) — img2img's pipeline derives the actual
    // output size from the source image when the request omits them, so
    // injecting a config default there would silently override that behavior.
    // Only the DyPE heuristic (which only ever affects an internal auto-enable
    // decision, never the output size) and steps/guidance are config-aware here.
    let resolvedWidth = width ?? configDefaults.width ?? ZImageModelMetadata.recommendedWidth
    let resolvedHeight = height ?? configDefaults.height ?? ZImageModelMetadata.recommendedHeight
    let dyPEConfig = resolvedDyPEConfig(width: resolvedWidth, height: resolvedHeight)

    let outputURL = try resolvedOutputURL(
      configuration: configuration,
      defaultFilename: ComfyBoxOutputNaming.defaultFilename(
        modelSpec: configuration.modelSpec ?? "z-image", presetId: preset,
        contentMode: contentMode, source: source)
    )

    return Img2ImgRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: width,
      height: height,
      steps: steps ?? configDefaults.steps ?? ZImageModelMetadata.recommendedInferenceSteps,
      guidanceScale: guidance ?? configDefaults.guidance.map(Float.init) ?? ZImageModelMetadata.recommendedGuidanceScale,
      seed: seed,
      outputPath: outputURL,
      levelsMin: levelsMin ?? 0.0,
      levelsMax: levelsMax ?? 1.0,
      model: configuration.modelSpec,
      textEncoderPath: configuration.textEncoderPath,
      maxSequenceLength: configuration.maxSequenceLength,
      loras: activeLoRAs,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly,
      schedulerKind: schedulerKind,
      sigmaSchedule: sigmaScheduleKind,
      eta: eta,
      // #154: the explicit ModelSamplingAuraFlow shift, forwarded on the
      // img2img path too — the schedule is the same one.
      shift: shift,
      dyPE: dyPEConfig,
      sourceImagePath: imagePath,
      strength: resolvedStrength,
      specifiedAs: specifiedAs,
      contentMode: contentMode,
      source: source,
      maskPath: maskPath,
      maskRegion: maskRegion,
      maskInvert: maskInvert ?? false,
      maskGrow: maskGrow ?? 0,
      maskFeather: maskFeather ?? 0
    )
  }

  enum Img2ImgValidationError: Error, LocalizedError {
    case mutuallyExclusive(String)
    var errorDescription: String? {
      switch self {
      case .mutuallyExclusive(let msg): return msg
      }
    }
  }

  /// WP-E4 (FDD-krea2-raw-recipe §3.4, D22, D25): resolve the sampler and
  /// sigma-schedule names fail-loud. Returns the resolved kinds AND the raw
  /// strings so the record can carry `sigma_schedule_requested`. Absent names
  /// come back nil — the request builders apply euler / flow as today.
  /// Family-agnostic; safe before the family is known.
  func validateRecipeNames() throws -> ResolvedRecipeNames {
    try RecipeNameResolver.resolve(scheduler: scheduler, sigmaSchedule: sigmaSchedule)
  }

  /// WP-E4 (D18, §3.4): Krea 2 tier / capability gates. Runs inside the
  /// Krea 2 generate path and the bridge's `.krea2` arm ONLY — `eta` on the
  /// Z-Image path is a different, shipped parameter (DDIM η / DPM++ 2S-A η)
  /// and keeps working (AC-28).
  ///
  /// The sampler and the sigma-schedule arms went with WP-E3: the Krea 2 loop
  /// dispatches on both, so every name `RecipeNameResolver` accepts is honoured
  /// rather than refused. **The `eta` arm went with WP-E15**: tier T2 has
  /// landed, so a non-zero `eta` is now either applied (the RES4LYF samplers)
  /// or refused BY SAMPLER at `Krea2Pipeline.makeSDEInjector` — it is never
  /// ignored, and the refusal names the sampler rather than the tier.
  ///
  /// `names` is still taken — the unknown-name failure happens in
  /// `validateRecipeNames()`, which the caller runs to produce it, and the
  /// parameter keeps that ordering explicit at every call site.
  ///
  /// **The `bongmath` arm arrived with WP-E16**, and it is the eta arm's twin:
  /// tier T3 has landed, `bongmath` now has a wire key, and it is applied on
  /// the RES4LYF samplers or refused BY SAMPLER — never ignored. Mirrors
  /// `Krea2Pipeline.makeBongMath(bongmath:sampler:sigmaSchedule:shift:)`,
  /// which refuses the same request for a non-server caller.
  ///
  /// What remains is the SAMPLER boundary the SDE has: RES4LYF's `eta` splits
  /// a step against RES4LYF's own prepared grid and re-noises the non-final
  /// rows of its tableau, so it is defined for the RES4LYF ports and for
  /// nothing else. Asked for with `euler` — the Krea 2 default — or with
  /// `ddim` / `dpmpp-2s-a`, where the same wire key already means a different
  /// stochasticity parameter on the Z-Image path, it is a 400 naming the
  /// sampler, never a silent drop. Mirrors
  /// `Krea2Pipeline.makeSDEInjector(eta:sampler:stageSeed:layout:)`, which
  /// refuses the same request for a non-server caller.
  func validateKrea2TierGates(_ names: ResolvedRecipeNames) throws {
    let sampler = names.scheduler ?? .euler
    let res4lyfList =
      "res_2s / res_3s / ralston_2s / ralston_3s / ralston_4s / deis_2m / deis_3m / deis_4m"
    if let eta, eta != 0, !sampler.isRES4LYFFamily {
      throw WarmServerError.unsupportedRecipeField(
        field: "eta", value: "\(eta)", family: "krea2",
        reason: "eta is RES4LYF's SDE (parity tier T2) and applies to the RES4LYF samplers only; "
          + "'\(sampler.rawValue)' is not one of them. Send eta 0, or a sampler from "
          + res4lyfList)
    }
    if bongmath == true, !sampler.isRES4LYFFamily {
      throw WarmServerError.unsupportedRecipeField(
        field: "bongmath", value: "true", family: "krea2",
        reason: "bongmath is RES4LYF's fixed point (parity tier T3) over its own tableau rows "
          + "and applies to the RES4LYF samplers only; '\(sampler.rawValue)' is not one of "
          + "them. Send bongmath false, or a sampler from " + res4lyfList)
    }
  }

  /// The range the Desktop dial clamps to (`GenerationView`'s Projector Scale
  /// slider, 0…3, 1.0 = neutral). The wire must not accept what the UI cannot
  /// express: a NaN/inf or out-of-range scale multiplied into the projector's
  /// text conditioning would render garbage (or something the caller did not
  /// ask for) under a well-formed-looking record.
  static let projectorScaleRange: ClosedRange<Float> = 0.0...3.0
  static let implicitStepsRange: ClosedRange<Int> = 0...8
  static let c2Pole: Float = 2.0 / 3.0

  /// `projector_scale`, validated at the point of application: absent → the
  /// neutral 1.0; present → finite and inside ``projectorScaleRange``, else a
  /// 400 naming the value (never clamped, same fail-loud stance as an unknown
  /// sampler name).
  func validatedProjectorScale() throws -> Float {
    guard let projectorScale else { return 1.0 }
    guard projectorScale.isFinite, Self.projectorScaleRange.contains(projectorScale) else {
      throw WarmServerError.projectorScaleOutOfRange(value: "\(projectorScale)")
    }
    return projectorScale
  }

  /// `noise_type`, validated at the point of application: absent → gaussian
  /// (the default, not a coercion); present → a `RES4LYFNoiseType` raw value,
  /// else a 400 naming the value and the valid set. The old
  /// `RES4LYFNoiseType(rawValue:) ?? .gaussian` silently rendered gaussian
  /// under whatever name the caller sent.
  func validatedNoiseType() throws -> RES4LYFNoiseType {
    guard let noiseType else { return .gaussian }
    guard let kind = RES4LYFNoiseType(rawValue: noiseType) else {
      throw WarmServerError.unknownNoiseType(
        name: noiseType, valid: RES4LYFNoiseType.allCases.map(\.rawValue))
    }
    return kind
  }

  /// `implicit_steps`, validated at the point of application (implicit-RK
  /// batch-2 review F1): absent -> 0 (today's explicit render, byte-identical).
  /// Negative would trap the denoise loop's precondition and abort the warm
  /// server; unbounded would hang a render (model evals scale with passes).
  /// 0...8 covers every practical RES4LYF full_iter setting.
  func validatedImplicitSteps() throws -> Int {
    guard let implicitSteps else { return 0 }
    guard Self.implicitStepsRange.contains(implicitSteps) else {
      throw WarmServerError.implicitStepsOutOfRange(value: "\(implicitSteps)")
    }
    return implicitSteps
  }

  /// `c2`, validated before constructing a scheduler whose RES3S tableau has
  /// a pole at 2/3. Absent keeps the established midpoint substep (0.5).
  func validatedC2() throws -> Float {
    guard let c2 else { return 0.5 }
    guard c2.isFinite, c2 > 0, c2 <= 1,
          abs(c2 - Self.c2Pole) >= 1e-6 else {
      throw WarmServerError.c2OutOfRange(value: "\(c2)")
    }
    return c2
  }

  /// WP-E3 (§3.3, D11, D22, D25): the recipe fields a Krea 2 request carries,
  /// resolved. A pure function of the payload, so the forwarding is asserted
  /// without a server or weights (`Krea2RecipeForwardingTests`).
  ///
  /// The defaults ARE today's render: euler over the family's native `krea2`
  /// warp, no explicit shift, no SDE. An unknown name throws (it does not
  /// become euler) because `validateRecipeNames()` throws.
  func krea2RecipeFields() throws -> Krea2RecipeFields {
    let names = try validateRecipeNames()
    return Krea2RecipeFields(
      sampler: names.scheduler ?? .euler,
      sigmaSchedule: names.sigmaSchedule ?? .krea2,
      shift: shift,
      eta: eta ?? 0,
      bongmath: bongmath ?? false,
      samplerRequested: names.schedulerRequested,
      sigmaScheduleRequested: names.sigmaScheduleRequested)
  }

  /// WP-E17 (§3.14, D4, D22, D25): the second stage, resolved into what the
  /// pipeline runs — or `nil` when the request has no second stage.
  ///
  /// Pure, like `krea2RecipeFields()`, so the forwarding is asserted without a
  /// server or weights. An unknown sampler / schedule name THROWS here (it does
  /// not become euler), and an unstated field stays `nil` so
  /// `Krea2Pipeline.Stage2.resolved(against:)` fills it from the render's own
  /// recipe rather than from an engine default.
  func krea2Stage2Fields() throws -> Krea2Pipeline.Stage2? {
    guard let stage2 else { return nil }
    let names = try RecipeNameResolver.resolve(
      scheduler: stage2.scheduler, sigmaSchedule: stage2.sigmaSchedule)
    return Krea2Pipeline.Stage2(
      steps: stage2.steps,
      denoise: stage2.denoise,
      sampler: names.scheduler,
      sigmaSchedule: names.sigmaSchedule,
      sigmaScheduleRequested: names.sigmaScheduleRequested,
      guidance: stage2.guidance,
      eta: stage2.eta,
      bongmath: stage2.bongmath,
      seed: stage2.seed)
  }

  /// WP-E17 (§3.14, D18; Addendum A.2 → C3): the `stage2` family + range gate,
  /// and the refusal of the tool schema's `detail_pass` / `detail_denoise`
  /// spelling. Returns the 400 to throw, or `nil`.
  ///
  /// Runs at the same dispatch point as `vaeGate` and `validateFamilyRecipe`
  /// (D18: family gates live at dispatch, not at the decoder), for EVERY
  /// family — the detail-pass keys are wrong everywhere, and `stage2` is a
  /// Krea 2 field that no other family's loop could honour.
  static func stage2Gate(_ payload: GeneratePayload, family: WarmModelFamily) -> WarmServerError? {
    // The tool-schema keys first: they are wrong on every family, and a request
    // carrying both them and `stage2` should be told about the spelling rather
    // than about the family.
    if payload.detailPass != nil {
      return .unsupportedRecipeField(
        field: "detail_pass", value: "\(payload.detailPass ?? false)", family: family.rawValue,
        reason: "`detail_pass` is the MCP tool schema's spelling; the client expands it into the "
          + "engine's `stage2` object from its family policy table (AC-68a). The engine holds no "
          + "such table and will not invent a sampler, schedule or step count — send "
          + "`stage2: {steps, denoise, scheduler, sigma_schedule, …}`")
    }
    if let detailDenoise = payload.detailDenoise {
      return .orphanField(
        field: "detail_denoise", requires: "detail_pass",
        reason: "`detail_denoise` = \(detailDenoise) names the denoise of a detail pass that was "
          + "never requested. It used to be dropped silently; send `stage2.denoise` instead")
    }

    guard let stage2 = payload.stage2 else { return nil }

    guard family == .krea2 else {
      return .unsupportedRecipeField(
        field: "stage2", value: "{steps: \(stage2.steps), denoise: \(stage2.denoise)}",
        family: family.rawValue,
        reason: "a second stage inside one render is a Krea 2 mechanism (WP-E17): it re-noises the "
          + "LATENT to the stretched tail's first sigma and solves again with no VAE round-trip. "
          + "This family's loop has no such seam and would have rendered one stage under a "
          + "two-stage record — load a krea2 model, or remove `stage2`")
    }

    guard stage2.steps > 0 else {
      return .unsupportedRecipeField(
        field: "stage2.steps", value: "\(stage2.steps)", family: family.rawValue,
        reason: "stage2.steps must be positive")
    }
    guard stage2.denoise.isFinite, stage2.denoise > 0, stage2.denoise <= 1 else {
      return .unsupportedRecipeField(
        field: "stage2.denoise", value: "\(stage2.denoise)", family: family.rawValue,
        reason: "stage2.denoise is the fraction of the schedule the stage runs and must be in "
          + "(0, 1]; `denoise <= 0` has no schedule to stretch and there is nothing to "
          + "substitute (§3.14)")
    }

    // The stage's own sampler / schedule against the family's capability
    // matrix — the same gate the render's own recipe goes through. Unknown
    // NAMES already threw at `krea2Stage2Fields()`; this is whether the family
    // can honour a known one.
    guard let names = try? RecipeNameResolver.resolve(
      scheduler: stage2.scheduler, sigmaSchedule: stage2.sigmaSchedule),
      let renderNames = try? RecipeNameResolver.resolve(
        scheduler: payload.scheduler, sigmaSchedule: payload.sigmaSchedule)
    else { return nil }
    if let error = FamilyRecipeMatrix.validate(names, family: family) { return error }

    // The stage's TIER gates, evaluated on the values the stage will actually
    // run with — an unstated field inherits the render's, so the pairing that
    // matters is the resolved one (a stage that names `euler` inherits the
    // render's `eta: 0.5` and would be an SDE on a sampler RES4LYF's SDE is not
    // defined against). `Krea2StagedRender.preflight` refuses these too, before
    // any model work; refusing them HERE is what makes them a 400 rather than
    // the 500 an unmapped pipeline error would become.
    if stage2.bongmath ?? false {
      return .unsupportedRecipeField(
        field: "stage2.bongmath", value: "true", family: family.rawValue,
        reason: "bongmath is parity tier T3 (WP-E16) and is not implemented yet; omit it or send false")
    }
    let effectiveEta = stage2.eta ?? payload.eta ?? 0
    let effectiveSampler = names.scheduler ?? renderNames.scheduler ?? .euler
    if effectiveEta != 0, !effectiveSampler.isRES4LYFFamily {
      return .unsupportedRecipeField(
        field: "stage2.eta", value: "\(effectiveEta)", family: family.rawValue,
        reason: "eta is RES4LYF's SDE (parity tier T2) and applies to the RES4LYF samplers only; "
          + "stage 2 runs '\(effectiveSampler.rawValue)', which is not one of them. Send "
          + "stage2.eta 0, or a stage2 sampler from res_2s / res_3s / ralston_2s / ralston_3s / "
          + "ralston_4s / deis_2m / deis_3m / deis_4m")
    }
    return nil
  }

  /// What `krea2RecipeFields()` resolved: the kinds the pipeline runs, plus
  /// the raw names the caller sent so an alias is visible in the record and
  /// the log rather than silently applied (D22).
  struct Krea2RecipeFields: Sendable, Equatable {
    let sampler: SchedulerKind
    let sigmaSchedule: SigmaScheduleKind
    /// `nil` = the resolution-dependent mu (D3/A.1); a value IS mu.
    let shift: Float?
    /// RES4LYF SDE eta (T2, WP-E15). Forwarded whatever the sampler;
    /// `validateKrea2TierGates` is what decides whether the render may have
    /// it, and it decides by SAMPLER.
    let eta: Float
    /// RES4LYF bongmath (T3, WP-E16). Forwarded on the same terms as `eta`,
    /// and gated the same way — by sampler, in `validateKrea2TierGates`.
    let bongmath: Bool
    let samplerRequested: String?
    let sigmaScheduleRequested: String?
  }

  func validateOutputPath(configuration: WarmServerConfiguration) throws {
    guard let outputPath, !outputPath.isEmpty else { return }
    _ = try WarmServerOutputPathValidator.resolveOutputPath(
      outputPath,
      allowedOutputDirectory: configuration.allowedOutputDirectory
    )
  }

  func resolvedOutputURL(
    configuration: WarmServerConfiguration,
    defaultFilename: String
  ) throws -> URL {
    guard let outputPath, !outputPath.isEmpty else {
      // Default to the gallery folder, NOT temp — otherwise renders from clients
      // that omit outputPath (e.g. HTTP/MCP pipelines) land in /var/folders/T and
      // are silently purged by macOS. Fall back to temp only if the gallery dir
      // can't be created.
      let dir = (configuration.allowedOutputDirectory as NSString).expandingTildeInPath
      let created = (try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)) != nil
      if created || FileManager.default.fileExists(atPath: dir) {
        return URL(fileURLWithPath: dir).appendingPathComponent(defaultFilename)
      }
      return URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(defaultFilename)
    }

    return try WarmServerOutputPathValidator.resolveOutputPath(
      outputPath,
      allowedOutputDirectory: configuration.allowedOutputDirectory
    )
  }
}

/// Internal (not private) so the async-job seam is unit-testable
/// (`AsyncJobIdTests`); still `Encodable` only — nothing decodes into it (AC-64).
struct GenerateResponse: Encodable, Sendable {
  let success: Bool
  let outputPath: String
  let durationMs: Int
  /// #1479: set on the SYNC `/v1/generate` path when this job asked to
  /// preempt an in-flight video but the refusal guard declined (finishing
  /// beats preempting) — the job still ran normally, just not preempting.
  /// Additive: defaulted in this explicit init (a property-level default
  /// would DROP the parameter from the synthesized memberwise init entirely,
  /// making it impossible to ever construct a `true`/non-nil one), so every
  /// pre-#1479 construction site is unaffected.
  let preemptRefused: Bool
  let etaSec: Double?
  /// WP-E10 sink 1 (FDD §3.10, D8): the provenance record — what APPLIED,
  /// read back from the pipeline. Krea 2 only (D12).
  ///
  /// Tri-state via ``AppliedRecordSlot`` (round 2, C4): key ABSENT for another
  /// family, literal `null` for a Krea 2 render whose record was refused
  /// (engine-incomplete), the object otherwise. Defaulted in this explicit init
  /// for the same reason as `preemptRefused`: a property-level default would
  /// drop it from the synthesized memberwise init.
  let applied: AppliedRecordSlot?

  /// #286: `applied_loras` — the LoRA stack that was actually resident for
  /// this render, READ BACK from the pipeline that rendered it: `name`, `path`
  /// and `scale` per adapter. Additive; no existing field is renamed.
  ///
  /// `applied` above answers the same question far more fully, but only for
  /// Krea 2 (D12). This one is a flat list a client can diff against what
  /// `POST /v1/presets/resolve` reported for the preset it asked for — which
  /// is what nobody could do while a preset-by-name render silently used
  /// residency. The key is ABSENT for FIBO and Chroma, which have no LoRA path
  /// at all, so it can never read as "rendered bare".
  let appliedLoras: [LoRAState]?

  /// #286 (C2): the named preset could not be expanded, so it behaved as the
  /// provenance label it has always been and this render used the request's own
  /// settings plus the resident stack. Names the preset. Absent = the preset
  /// was expanded (or none was named). NOT an error — the pre-#286 contract is
  /// preserved deliberately; this field is how it stops being silent.
  let presetUnresolved: String?

  /// #286 (round 2): `preset_unresolved_reason` — the machine-readable code
  /// beside the name, so a daemon can branch on WHY: `unknown_preset`,
  /// `invalid_preset`, `media_kind:video`, `engine:<x>`, `provider:<x>`,
  /// `no_model`, `bypass_declared`, `missing_lora:<name>`.
  let presetUnresolvedReason: String?

  /// #286 (I1): the request carried explicit `loras` that differ from what its
  /// named `preset` resolves to. The explicit list won, as it always has; this
  /// says the two disagreed. The production async client sends a FLAT `loras`
  /// list that has already dropped `bypass`/`role`, which is exactly the case
  /// this makes visible.
  let presetStackMismatch: Bool?

  /// #22 (PR #363 review, C1b): the render's estimated peak activation
  /// memory and the live free memory at the moment it was estimated —
  /// present whenever the preflight ran (width/height both given, and not a
  /// replay), regardless of whether the estimate was within budget. See
  /// `GeneratePayload.memoryEstimateBytes`.
  let memoryEstimateBytes: UInt64?
  let memoryAvailableBytes: UInt64?
  /// #282: `lora_stack_origin` — WHERE this render's LoRA stack came from:
  /// `"request"` (the request's own `loras`), `"preset"` (the named preset's
  /// expansion) or `"warm_default"` (the stack `POST /v1/lora/swap` published,
  /// which applies only to a request carrying neither).
  ///
  /// Additive, and the field that makes "did this job render with the stack it
  /// asked for?" answerable from the response alone. Absent on the ControlNet
  /// arm, which has always rendered its request's own stack through its own
  /// pipeline instance and has neither a `preset` nor a warm default.
  let loraStackOrigin: String?

  /// #282 review r1 (C1): `warm_default_skipped` — the warm default was NOT
  /// applied to this render because it was published under a different base
  /// (`family_mismatch` / `model_mismatch`). The render used no adapters and
  /// still succeeded; forcing another base's stack could have thrown, turning
  /// an always-rendering request into a 500. Absent when nothing was skipped.
  let warmDefaultSkipped: String?

  /// #282 review r1 (I1): `lora_reload` — this job cleared the resident
  /// adapters and bound a different stack, rather than taking the same-stack
  /// shortcut. Present only when true, so alternating bare/preset callers can
  /// MEASURE the churn that per-request correctness costs.
  let loraReload: Bool?

  /// #154: `applied_shift` — the explicit schedule shift this render actually
  /// applied, whether it came from the request or from its named preset.
  ///
  /// ABSENT means the render used the model's own schedule (the
  /// resolution-dependent `mu`, or the `scheduler_config.json` shift) — which
  /// is every render made before #154 and every render that names no shift, so
  /// the key's absence is the old contract, unchanged.
  ///
  /// A flat field rather than a second `applied` block: `applied`
  /// (``RenderRecipe``) is Krea 2 only by D12, and the family this ticket is
  /// about (`.flux1`, where Zeta Chroma runs) emits none.
  ///
  /// **Set on the `.flux1` and ControlNet arms only.** Krea 2 reads back the
  /// full recipe instead (`applied.shift`, `applied.shift_source`, and
  /// `applied.stages[].shift_applied`, which is false for a schedule like
  /// `bong_tangent` that ignores it); a flat field beside that would be a
  /// second, weaker claim about the same render.
  ///
  /// The number here always REACHED THE SIGMA GRID: a shift the requested
  /// schedule would drop is refused up front by
  /// ``GeneratePayload/validateShiftSchedule(_:sigmaSchedule:family:)``, so
  /// this is never "what you asked for" — it is what applied.
  ///
  /// There is deliberately no `applied_shift_source` twin: on the flat path
  /// the server cannot see the model's `scheduler_config.json` without a new
  /// return channel from the pipeline, and a guessed `"dynamic"` label would
  /// be exactly the confident-but-unverified claim this codebase keeps out of
  /// its records. `applied.shift_source` remains the place that question is
  /// answered, for the family that reads back a full recipe.
  let appliedShift: Float?

  /// The record itself, for Swift readers that do not care about the
  /// absent-vs-null distinction.
  var appliedRecord: RenderRecipe? { applied?.record }

  init(success: Bool, outputPath: String, durationMs: Int, preemptRefused: Bool = false, etaSec: Double? = nil,
       applied: AppliedRecordSlot? = nil, appliedLoras: [LoRAState]? = nil,
       presetUnresolved: String? = nil, presetUnresolvedReason: String? = nil,
       presetStackMismatch: Bool? = nil,
       memoryEstimateBytes: UInt64? = nil, memoryAvailableBytes: UInt64? = nil,
       loraStackOrigin: String? = nil,
       warmDefaultSkipped: String? = nil, loraReload: Bool? = nil,
       appliedShift: Float? = nil) {
    self.success = success
    self.outputPath = outputPath
    self.durationMs = durationMs
    self.preemptRefused = preemptRefused
    self.etaSec = etaSec
    self.applied = applied
    self.appliedLoras = appliedLoras
    self.presetUnresolved = presetUnresolved
    self.presetUnresolvedReason = presetUnresolvedReason
    self.presetStackMismatch = presetStackMismatch
    self.memoryEstimateBytes = memoryEstimateBytes
    self.memoryAvailableBytes = memoryAvailableBytes
    self.loraStackOrigin = loraStackOrigin
    self.warmDefaultSkipped = warmDefaultSkipped
    self.loraReload = loraReload
    self.appliedShift = appliedShift
  }
}

// MARK: - Upscale Payload & Response

struct UpscalePayload: Decodable, Sendable {
  let imagePath: String
  let targetResolution: Int?
  let seed: Int?
  let softness: Float?
  let outputPath: String?
  let model: String?   // "seedvr2-3b" or "seedvr2-7b"

  /// Validate target resolution. Returns an error message if invalid, nil if valid.
  static func validateResolution(_ resolution: Int) -> String? {
    guard resolution >= 256 && resolution <= 2048 else {
      return "target_resolution must be between 256 and 2048"
    }
    return nil
  }

  /// Validate softness. Returns an error message if invalid, nil if valid.
  static func validateSoftness(_ softness: Float) -> String? {
    guard softness >= 0.0 && softness <= 1.0 else {
      return "softness must be between 0.0 and 1.0"
    }
    return nil
  }

  /// Validate model variant. Returns an error message if invalid, nil if valid.
  static func validateModel(_ model: String?) -> String? {
    guard let model = model else { return nil }
    guard model == "seedvr2-3b" || model == "seedvr2-7b" else {
      return "Invalid model: '\(model)'. Must be 'seedvr2-3b' or 'seedvr2-7b'."
    }
    return nil
  }

  /// Return a warning string if resolution is experimental (>1024), nil otherwise.
  static func resolutionWarning(for resolution: Int) -> String? {
    resolution > 1024
      ? "target_resolution \(resolution) is experimental and may cause OOM errors. Safe maximum is 1024."
      : nil
  }
}

struct UpscaleResponse: Encodable, Sendable {
  let success: Bool
  let outputPath: String
  let durationMs: Int
  let inputResolution: String     // e.g. "512x512"
  let outputResolution: String    // e.g. "1024x1024"
  let model: String               // "seedvr2-3b" or "seedvr2-7b"
  let warning: String?            // non-nil if target_resolution > 1024
}

private struct LoRASwapPayload: Decodable, Sendable {
  let loras: [LoRAEntry]

  func makeConfigurations() throws -> [LoRAConfiguration] {
    try loras.map { try $0.makeConfiguration() }
  }
}

private struct LoRASwapResponse: Encodable, Sendable {
  let success: Bool
  let loraCount: Int
  let loras: [LoRAState]
}

struct LoRAEntry: Codable, Sendable {
  let path: String
  let scale: Float?
  /// WP-E10 (FDD §3.10 `Applied.role`): the configuration SLOT this adapter
  /// fills — `kroma` | `accel` | `bypass` | `control`. Declared by the sender
  /// that expanded the preset — since #286 that can be the engine itself,
  /// which carries `LoraReference.role` through and labels the structured
  /// kroma `"kroma"`; it is never inferred from a filename. Stored on the
  /// `LoRAConfiguration` the pipeline applies and READ BACK from there into
  /// `applied.loras[].role`. An unknown label is a 400, never stored.
  let role: String?

  /// The declared roles, in one place.
  static let roles: [String] = ["kroma", "accel", "bypass", "control"]

  init(path: String, scale: Float?, role: String? = nil) {
    self.path = path
    self.scale = scale
    self.role = role
  }

  private enum CodingKeys: String, CodingKey { case path, scale, role }

  /// Allowed range for LoRA scales — finite values outside are clamped.
  private static let scaleRange: ClosedRange<Float> = -10.0...10.0

  /// The declared role, validated. nil when absent.
  private func resolvedRole() throws -> String? {
    guard let role else { return nil }
    guard Self.roles.contains(role) else {
      throw WarmServerError.invalidRequest(
        message: "Invalid LoRA role '\(role)' for '\(path)': expected one of \(Self.roles.joined(separator: ", "))")
    }
    return role
  }

  /// Validate the requested scale: reject non-finite values, clamp finite
  /// values to `scaleRange`. Defaults to 1.0 when absent.
  private func resolvedScale() throws -> Float {
    guard let scale else { return 1.0 }
    guard scale.isFinite else {
      throw WarmServerError.invalidRequest(
        message: "Invalid LoRA scale for '\(path)': must be a finite number"
      )
    }
    return min(max(scale, Self.scaleRange.lowerBound), Self.scaleRange.upperBound)
  }

  /// Directories searched, in order, when a LoRA is named by bare filename
  /// (no path — typically reconstructed from embedded PNG metadata, which
  /// only ever stores a display name, never the original absolute path).
  /// COMFYBOX_MODELS matches the same env var LoRALibrary itself resolves
  /// against — this used to hardcode a stale, unrelated "~/Models/loras"
  /// path that nothing actually writes to, so bare-filename resolution
  /// against the real library silently never worked.
  private static var bareFilenameSearchRoots: [String] {
    var roots: [String] = []
    if let envRoot = ProcessInfo.processInfo.environment["COMFYBOX_MODELS"], !envRoot.isEmpty {
      roots.append((envRoot as NSString).expandingTildeInPath)
    }
    roots.append(("~/.comfybox/loras" as NSString).expandingTildeInPath)
    roots.append("/Volumes/Bolt/Models/loras")
    // Ad-hoc/test LoRAs commonly land in Downloads before being filed into
    // the library proper — worth checking before giving up.
    roots.append(("~/Downloads" as NSString).expandingTildeInPath)
    var seen = Set<String>()
    return roots.filter { seen.insert($0).inserted }
  }

  func makeConfiguration() throws -> LoRAConfiguration {
    var configuration = try resolveSource()
    configuration.role = try resolvedRole()
    return configuration
  }

  private func resolveSource() throws -> LoRAConfiguration {
    let clampedScale = try resolvedScale()
    let expanded = (path as NSString).expandingTildeInPath

    // Direct path (absolute, relative, tilde-expanded)
    if path.hasPrefix("/") || path.hasPrefix("./") || path.hasPrefix("../") || path.hasPrefix("~")
       || FileManager.default.fileExists(atPath: expanded) {
      return .local(expanded, scale: clampedScale)
    }

    // Library resolution: search known local LoRA locations for the bare
    // filename before assuming it's a remote reference.
    let fm = FileManager.default
    for root in Self.bareFilenameSearchRoots {
      guard fm.fileExists(atPath: root) else { continue }
      guard let enumerator = fm.enumerator(
        at: URL(fileURLWithPath: root), includingPropertiesForKeys: [.isRegularFileKey]
      ) else { continue }
      for case let fileURL as URL in enumerator {
        if fileURL.lastPathComponent == path {
          return .local(fileURL.path, scale: clampedScale)
        }
      }
    }

    // A string shaped like a local filename (ends in a known weight
    // extension, no "/") is never a valid HuggingFace repo id — don't
    // attempt a network download that's certain to fail with a confusing
    // "Model not found" error. This is almost always a stale reference
    // (e.g. reconstructed from a PNG's embedded metadata, which only ever
    // stores a display name, not the original path) — say so plainly.
    let looksLikeLocalFilename = !path.contains("/") &&
      [".safetensors", ".ckpt", ".pt", ".bin"].contains { path.hasSuffix($0) }
    if looksLikeLocalFilename {
      throw WarmServerError.invalidRequest(
        message: "LoRA '\(path)' not found. Searched: \(Self.bareFilenameSearchRoots.joined(separator: ", "))."
      )
    }

    // HuggingFace fallback — only for strings actually shaped like a repo id.
    return .huggingFace(path, scale: clampedScale)
  }
}

private struct ShutdownResponse: Encodable, Sendable {
  let success: Bool
  let message: String
}

/// Internal (not private) so the sink shape is unit-testable (`HealthSinkTests`).
struct HealthResponse: Encodable, Sendable {
  let status: String
  let model: String
  let modelFamily: String
  let modelVariant: String?
  /// WP-E10 "E9b" (AC-34b): `model_alias` — the declared alias beside the
  /// resolved `model` path for the krea2 family; null otherwise.
  let modelAlias: String?
  /// WP-E10 (FDD §7.3 smoke e): git short sha stamped at build time, or
  /// `"unknown"` for a build that did not run `scripts/gen-build-info.sh`.
  let buildSha: String
  let textEncoderPath: String?
  let loaded: Bool
  let loras: [LoRAState]
  let uptimeSeconds: Int
  let renderCount: Int
  let failedRenderCount: Int
  let pendingCount: Int
  let maxPending: Int
  let isRendering: Bool
  /// Queue pause gate (`is_paused` on the wire) — surfaced in /health so every
  /// client (desktop toolbar, daemons, MCP) sees the same creation state
  /// without an extra request.
  let isPaused: Bool
  let activeRequestAgeMs: Int?
  /// Synthetic id of the currently-rendering job — `current_job_id` on the wire.
  let currentJobId: String?
  /// Live progress (0-100) of the active render — `progress_percent` on the wire.
  let progressPercent: Int?
  let memoryUsageBytes: UInt64
  let memoryUsageMB: UInt64
  let lastRenderDurationMs: Int?
  let lastError: String?
  /// WP-E10 sink 3: `last_recipe` — the record of the last successful Krea 2
  /// render, identical to that render's `applied` (AC-62). Tri-state via
  /// ``AppliedRecordSlot`` (round 2, C4): ABSENT until a Krea 2 render has run
  /// (and for other families, D12), literal `null` when the last Krea 2
  /// render's record was refused, the object otherwise.
  let lastRecipe: AppliedRecordSlot?
  /// comfybox#386 review round 3, item 1c: additive telemetry for the
  /// liveness-over-durability tradeoff in `WarmServerCoordinator.drainQueueDeltas`
  /// — `true` once the undrained-delta sidecar has failed continuously long
  /// enough that the drain has started (or will start) applying deltas it
  /// could not confirm durable on disk. `var` (not `let`) with a default is
  /// what makes it an OPTIONAL parameter on the synthesized memberwise init
  /// (Swift excludes a defaulted `let` from the init parameter list
  /// entirely), so every existing `HealthResponse(...)` call site keeps
  /// compiling unchanged.
  var queueDeltaSidecarDegraded: Bool = false
  /// How many currently-undrained deltas are not yet confirmed durable —
  /// `queue_delta_non_durable_count` on the wire. Nonzero without
  /// `queue_delta_sidecar_degraded` just means a write is in flight or
  /// recently failed but hasn't crossed the degraded-mode threshold yet.
  var queueDeltaNonDurableCount: Int = 0
}

/// #282 — what one dequeued job renders with, and why.
struct JobLoRAPlan {
  /// The adapters to apply. Empty is a real answer, not "nothing decided".
  let stack: [LoRAConfiguration]
  /// Which of the three sources owned it.
  let origin: RequestStackResolver.Origin
  /// Review r1 (C1): set when the warm default was DROPPED because it was
  /// published under a different base. Reaches the wire as
  /// `warm_default_skipped`; the job renders with no adapters and never errors.
  var warmDefaultSkipped: String?

  init(
    stack: [LoRAConfiguration], origin: RequestStackResolver.Origin,
    warmDefaultSkipped: String? = nil
  ) {
    self.stack = stack
    self.origin = origin
    self.warmDefaultSkipped = warmDefaultSkipped
  }
}

/// #282 — the outcome of one `applyActiveLoRAs` call.
struct LoRAApplication {
  /// The stack AS APPLIED — Krea-2 relativity already folded in (review r1,
  /// I3), which is what `/health.loras` and `warm_default_stack` must agree on.
  let stack: [LoRAConfiguration]
  /// Whether adapters were actually cleared and re-bound, as opposed to the
  /// same-stack shortcut skipping the work (review r1, I1).
  let reloaded: Bool
}

#if DEBUG
/// #282 review r1 (I4) — records what ``applyJobLoRAStack`` was asked to apply,
/// INSTEAD of touching a pipeline.
///
/// A real application needs model weights and a GPU, so the load-bearing call
/// could not otherwise be covered: deleting `try await applyJobLoRAStack(plan)`
/// from `runGenerate` left every test green. With a recorder installed the
/// coordinator applies nothing, records the decision, and `runGenerate` fails
/// the job before dispatching to a family — so no model resolution, no
/// download, no GPU, and a deleted call shows up as an empty recorder.
final class StackApplicationRecorder: @unchecked Sendable {
  struct Call: Equatable {
    let origin: String
    let names: [String]
    let warmDefaultSkipped: String?
  }
  private let lock = NSLock()
  private var storage: [Call] = []

  init() {}

  func record(origin: String, names: [String], warmDefaultSkipped: String?) {
    lock.lock()
    storage.append(Call(origin: origin, names: names, warmDefaultSkipped: warmDefaultSkipped))
    lock.unlock()
  }

  var calls: [Call] {
    lock.lock()
    defer { lock.unlock() }
    return storage
  }
}
#endif

public struct LoRAState: Codable, Sendable {
  /// Unchanged since before #286: the absolute local path, or `repo/file` for
  /// a HuggingFace reference. Kept as-is — `/health.loras` and
  /// `/v1/lora/swap`'s response have always carried this spelling.
  public let source: String
  public let scale: Float
  /// #286 (minor review point): the adapter's NAME — the last path component,
  /// which is what a preset's `loras[].filename` and `/v1/presets/resolve`
  /// carry. Diffing an applied stack against a resolved preset needs this, not
  /// a machine-specific absolute path. Additive.
  public let name: String
  /// #286: the same value as `source`, under the spelling a caller reading
  /// "name + path" expects. Additive; `source` is untouched for compatibility.
  public let path: String
  /// #286: the declared configuration slot (`kroma`/`accel`/`bypass`/
  /// `control`), read back from the applied `LoRAConfiguration`. nil when the
  /// sender declared none.
  public let role: String?

  public init(_ configuration: LoRAConfiguration) {
    switch configuration.source {
    case .local(let url):
      self.source = url.path
    case .huggingFace(let modelId, let filename):
      self.source = filename.map { "\(modelId)/\($0)" } ?? modelId
    }
    self.scale = configuration.scale
    self.name = configuration.source.displayName
    self.path = self.source
    self.role = configuration.role
  }

  /// Tolerant decode: `name`/`path`/`role` postdate #286, so a persisted
  /// pre-#286 job status still round-trips (`name`/`path` fall back to
  /// `source`).
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    source = try c.decode(String.self, forKey: .source)
    scale = try c.decode(Float.self, forKey: .scale)
    name = try c.decodeIfPresent(String.self, forKey: .name)
      ?? (source as NSString).lastPathComponent
    path = try c.decodeIfPresent(String.self, forKey: .path) ?? source
    role = try c.decodeIfPresent(String.self, forKey: .role)
  }
}

struct ErrorPayload: Encodable {
  let success: Bool
  let error: String
  /// Additive (#22): a machine-readable refusal code alongside the human
  /// `error` string, so a client can branch without string-matching it.
  /// `nil` for every pre-existing refusal — the synthesized `Encodable`
  /// conformance calls `encodeIfPresent` for `Optional` properties, so a
  /// `nil` field is OMITTED from the JSON body, not encoded as `null`.
  /// Existing error responses are therefore byte-identical to before.
  var errorCode: String? = nil
  /// #22 image-memory-preflight numbers: the render's estimated peak
  /// activation bytes, live free system bytes at decision time, and the
  /// byte cap it was compared against. Present only on
  /// `imageMemoryPreflightRefused`.
  var estimateBytes: UInt64? = nil
  var availableBytes: UInt64? = nil
  var capBytes: UInt64? = nil
}

private enum QueuedOperation: Sendable {
  case generate(GeneratePayload, ContinuationBox<GenerateResponse>, (@Sendable (ZImagePipeline.GenerationProgress) -> Void)?, ZImagePipeline.LatentPreviewHandler?)
  case controlGenerate(ZImageControlGenerationRequest, ContinuationBox<GenerateResponse>)
  case swap(LoRASwapPayload, ContinuationBox<LoRASwapResponse>)
  case modelSwitch(@Sendable () async throws -> Bool, ContinuationBox<Bool>)
  /// K-FIX-1 / Codex C2: a MUTATING pool operation (`/v1/model/load`,
  /// `/v1/model/activate`, `/v1/model/unload`) run on the SAME FIFO as
  /// renders, LoRA swaps and the ComfyBridge model switch — so a load,
  /// eviction or `GPU.clearCache()` can never begin under an active render.
  /// The continuation is nil for a `wait: false` load, which is tracked by
  /// its queue job id instead of by a waiting caller (never a detached Task).
  case modelOperation(ModelOperation, ContinuationBox<ModelOperationResult>?)
  /// Local LTX-2 video generation, run through the queue so it serializes with
  /// image renders on the shared GPU. The closure captures the generator+request.
  /// #1479: the closure returns `LTX2RenderOutcome` (not the bare result) so
  /// the process loop can observe a `.yielded` checkpoint and run a
  /// preemption episode before finally resolving `continuation` with the
  /// completed `LTX2VideoResult`. `videoJobId` is the async tracker's id
  /// (nil for the synchronous `/v1/video/generate` path), used to mark the
  /// job paused-for-preemption / resumed in `VideoJobTracker`.
  case localVideo(@Sendable (@escaping @Sendable (Int) -> Void) throws -> LTX2RenderOutcome, ContinuationBox<LTX2VideoResult>, wantsAudio: Bool, videoJobId: String?)
  case shutdown(ContinuationBox<ShutdownResponse>)
  #if DEBUG
  /// 0.B-2 test seam (FDD §4.1): occupies the processing loop for a controlled
  /// duration with no GPU. Its body BLOCKS its thread (Thread.sleep, not
  /// Task.sleep), so it exercises the exact pool-exhaustion mechanism 0.B-1 fixes
  /// — a render that holds a worker without suspending. DEBUG-only; the release
  /// deploy never compiles it.
  case synthetic(durationMs: Int, ContinuationBox<Bool>)
  #endif
}

private final class ContinuationBox<Value>: @unchecked Sendable {
  private let continuation: CheckedContinuation<Value, Error>
  /// comfybox#283/#217: fired exactly once, right when the wrapped
  /// continuation resumes, with the ACTUAL terminal outcome — every
  /// `enqueue*` method on `WarmServerCoordinator` passes this at
  /// CONSTRUCTION time (see `lifecycleCompletionHandler(jobId:jobKind:
  /// source:enqueuedAt:)`), so every queued job kind's true
  /// completed/failed/interrupted result is observable from one place.
  /// Read-only: never influences what `resume` does, only observes it —
  /// this line runs, then the real resume runs, unconditionally, exactly as
  /// before this hook existed.
  ///
  /// PR #370 review I6: `let`, set once at construction, never mutated —
  /// the prior version was a `var` mutated AFTER construction (from
  /// `processLoop`, once a job was admitted), which is a real data race on
  /// an `@unchecked Sendable` type: nothing enforced that the mutation
  /// happened-before any concurrent read from `resume`. It happened to be
  /// safe in practice (the write always preceded the render `Task` that
  /// could call `resume`, and Swift's task-creation boundary is itself a
  /// synchronization point) but that safety was implicit and fragile.
  /// Setting it here, at construction, removes the race by construction
  /// instead of by argument.
  private let onResume: (@Sendable (Result<Value, Error>) -> Void)?

  init(_ continuation: CheckedContinuation<Value, Error>, onResume: (@Sendable (Result<Value, Error>) -> Void)? = nil) {
    self.continuation = continuation
    self.onResume = onResume
  }

  func resume(returning value: Value) {
    onResume?(.success(value))
    continuation.resume(returning: value)
  }

  func resume(throwing error: Error) {
    onResume?(.failure(error))
    continuation.resume(throwing: error)
  }

  /// Resume WITHOUT firing `onResume`'s lifecycle-completion hook. The ONLY
  /// caller is `WarmServerCoordinator.cancel(_:)`, which cancels a job still
  /// sitting in `pending` (never admitted) — its lifecycle already gets an
  /// explicit `.dropped` event from `clearPending`/`cancelPending`/the sync
  /// cancel path (`drainQueueDeltas`), and firing `onResume` too would
  /// double-record the same cancellation as BOTH `.dropped` and
  /// `.interrupted` (since `resume`'s hook classifies `ServerError.cancelled`
  /// as an interrupt).
  func resumeIgnoringLifecycleHook(throwing error: Error) {
    continuation.resume(throwing: error)
  }
}

private final class SyncResult<Value> {
  private let semaphore = DispatchSemaphore(value: 0)
  private let lock = NSLock()
  private var result: Result<Value, Error>?

  func succeed(_ value: Value) {
    store(.success(value))
  }

  func fail(_ error: Error) {
    store(.failure(error))
  }

  func wait() throws -> Value {
    semaphore.wait()
    lock.lock()
    defer { lock.unlock() }
    return try result!.get()
  }

  private func store(_ result: Result<Value, Error>) {
    lock.lock()
    defer { lock.unlock() }
    guard self.result == nil else { return }
    self.result = result
    semaphore.signal()
  }
}

public enum WarmServerError: Error, LocalizedError {
  case invalidPort(UInt16)
  case invalidOutputPath(path: String, allowedDirectory: String)
  case invalidRequest(message: String)
  case flux2DetectionFailed(String)
  case flux2NotLoaded
  case fiboDetectionFailed(String)
  case fiboNotLoaded
  case chromaDetectionFailed(String)
  case chromaNotLoaded
  case krea2NotLoaded
  /// WP-E19: the krea2 family is resident but the coordinator holds no
  /// `Krea2Variant` for it. The bridge arm refuses rather than assuming turbo.
  case krea2VariantUnknown
  case loraSwapNotSupported
  case controlNetNotSupported
  // WP-E4 (FDD-krea2-raw-recipe §3.4, D22, D25, D18): fail-loud recipe names.
  /// A sampler name that is neither a `SchedulerKind` raw value nor a declared
  /// alias. `valid` is the full accepted set, listed in the message.
  case unknownSampler(name: String, valid: [String])
  /// A sigma-schedule name that is neither a `SigmaScheduleKind` raw value nor
  /// a declared alias (`normal`/`simple`/`sgm_uniform`/`ddim_uniform`/`beta57`).
  case unknownSigmaSchedule(name: String, valid: [String])
  /// Two request keys that name the same thing carry different values
  /// (`scheduler` vs its `sampler` alias, D25).
  case mutuallyExclusive(String)
  /// A recipe field the named family cannot honour yet — an unimplemented
  /// tier is a 400, never a downgrade (D18).
  case unsupportedRecipeField(field: String, value: String, family: String, reason: String)
  /// WP-E13: an N-row tableau sampler (`ralston_2s/3s/4s`, `res_3s`) asked for
  /// on a family whose denoise loop takes one model evaluation per step. It
  /// would render first-order Euler under the sampler's name, so it is a 400.
  case unsupportedSampler(name: String, family: String, reason: String)
  /// WP-E17 / Addendum A.2 → C3: a field that only means something beside
  /// another field, sent without it. Silently dropping it made a request that
  /// asked for something render as if it had not.
  case orphanField(field: String, requires: String, reason: String)
  /// A `projector_scale` the projector cannot honour: non-finite (NaN/inf) or
  /// outside the Desktop dial's clamp range (`GenerationView`'s 0…3 slider).
  /// Refused by value rather than clamped — a clamp would render something the
  /// caller did not ask for under the number they sent.
  case projectorScaleOutOfRange(value: String)
  case implicitStepsOutOfRange(value: String)
  case c2OutOfRange(value: String)
  /// A `noise_type` that is not a `RES4LYFNoiseType` raw value. An unknown
  /// name is a 400 naming the valid set — it must never silently degrade to
  /// gaussian (absent stays gaussian; that is the default, not a coercion).
  case unknownNoiseType(name: String, valid: [String])
  /// #286: the request named a `preset` AND an explicit `model` that resolve
  /// to different bases. Applying the preset's adapters to the requested base,
  /// or the requested base under the preset's name, are both wrong — so it is
  /// a 409 naming all three rather than a silent pick.
  case presetModelConflict(preset: String, presetModel: String, requestModel: String)
  /// comfybox#322: the in-flight render was cancelled by
  /// `/v1/queue/interrupt`. Distinct from every failure above — nothing went
  /// wrong, an operator asked for the box back. `VideoJobTracker` and the
  /// video routes recognise it and report the job interrupted, not failed.
  case renderInterrupted
  /// #339 review r1: a non-recoverable-kind submission (ControlNet, a Krita
  /// model switch) refused because `recoverPersistedQueue`'s replay is in
  /// flight — see `QueueRecoveryGate`. `retryAfterSeconds` is this
  /// throw-site's own estimate, mirrored into the 503 body + header.
  case queueRecoveryInProgress(retryAfterSeconds: Int)
  /// #22: an image request `ImageMemoryPreflight.validate` refused before any
  /// model load — either it exceeds the configured resolution cap
  /// (`code: "resolution_cap"`, no memory numbers — refused before probing)
  /// or its estimated peak activation memory does not fit the live headroom
  /// budget (`code: "insufficient_memory"`). `errorResponse(for:)` maps this
  /// to 413 with the numbers as additive JSON fields (`ErrorPayload`).
  case imageMemoryPreflightRefused(
    code: String, reason: String,
    estimateBytes: UInt64?, availableBytes: UInt64?, capBytes: UInt64?)

  public var errorDescription: String? {
    switch self {
    case .invalidPort(let port):
      return "Invalid server port: \(port)"
    case .invalidOutputPath(let path, let allowedDirectory):
      return "Output path '\(path)' must be under allowed output directory '\(allowedDirectory)'"
    case .invalidRequest(let message):
      return message
    case .presetModelConflict(let preset, let presetModel, let requestModel):
      return "Preset '\(preset)' declares model '\(presetModel)' but the request asked for "
        + "'\(requestModel)'. A preset's LoRA stack is only valid on its own base — send one or "
        + "the other, or send the LoRAs explicitly in `loras` without the preset."

    case .flux2DetectionFailed(let model):
      return "Model '\(model)' was identified as Flux 2 but detection failed at the snapshot directory"
    case .flux2NotLoaded:
      return "Flux 2 pipeline is not loaded"
    case .fiboDetectionFailed(let model):
      return "Model '\(model)' was identified as FIBO but detection failed at the snapshot directory"
    case .fiboNotLoaded:
      return "FIBO pipeline is not loaded"
    case .chromaDetectionFailed(let model):
      return "Model '\(model)' was identified as Chroma but detection failed at the snapshot directory"
    case .chromaNotLoaded:
      return "Chroma pipeline is not loaded"
    case .krea2NotLoaded:
      return "Krea-2 pipeline is not loaded"
    case .krea2VariantUnknown:
      return "Krea-2 pipeline is resident but its variant (turbo|raw) is unknown — refusing to assume turbo"
    case .loraSwapNotSupported:
      return "LoRA swap is not supported for this model family"
    case .controlNetNotSupported:
      return "ControlNet is not supported for this model family"
    case .unknownSampler(let name, let valid):
      return "Unknown sampler '\(name)'. Valid samplers: \(valid.joined(separator: ", "))"
    case .unknownSigmaSchedule(let name, let valid):
      return "Unknown sigma schedule '\(name)'. Valid schedules: \(valid.joined(separator: ", "))"
    case .mutuallyExclusive(let message):
      return message
    case .unsupportedRecipeField(let field, let value, let family, let reason):
      return "'\(field)' = '\(value)' is not supported on the \(family) family: \(reason)"
    case .unsupportedSampler(let name, let family, let reason):
      return "sampler '\(name)' is not supported on the \(family) family: \(reason)"
    case .orphanField(let field, let requires, let reason):
      return "'\(field)' has no meaning without '\(requires)': \(reason)"
    case .projectorScaleOutOfRange(let value):
      return "projector_scale must be a finite number in 0.0...3.0 (got \(value)); "
        + "1.0 is neutral — omit it for the default"
    case .implicitStepsOutOfRange(let value):
      return "implicit_steps must be an integer in 0...8 (got \(value)); "
        + "0 is the explicit render — omit it for the default"
    case .c2OutOfRange(let value):
      return "c2 must be a finite number in (0, 1] other than 2/3 (got \(value)); "
        + "0.5 is the default"
    case .unknownNoiseType(let name, let valid):
      return "Unknown noise_type '\(name)'. Valid noise types: \(valid.joined(separator: ", ")); "
        + "omit it for gaussian"
    case .renderInterrupted:
      return "Render interrupted by /v1/queue/interrupt"
    case .queueRecoveryInProgress:
      return QueueRecoveryGate.reason
    case .imageMemoryPreflightRefused(let code, let reason, _, _, _):
      return "[\(code)] \(reason)"
    }
  }
}

/// F3 (comfybox#324, adversarial review of Phase 3 config): the exact
/// config-layer `??` merge `WarmServerCoordinator.runKrea2Generate` applies
/// to `payload.width/height/steps/guidance` BEFORE handing steps/guidance to
/// `Krea2Variant.resolvedSteps/resolvedGuidance` — pulled out to a top-level,
/// weight-free func (mirrors `isRenderInterruption`/`localVideoCatchOutcome`
/// below: `WarmServerCoordinator` is `private` to this file, so a pure helper
/// its methods call has to live at file scope to be reachable from a unit
/// test) so a test can drive this REAL merge directly. `runKrea2Generate`
/// itself needs a loaded `Krea2Pipeline` to run past this point, so it cannot
/// be exercised end-to-end in a unit test — this is the testable slice of it.
/// The prior test coverage (`ServerConfigStoreTests`) re-implemented this
/// same `??` chain inline instead of calling production code; this is the
/// call-site guard that closes that gap.
func mergedKrea2RenderDefaults(
  requestWidth: Int?, requestHeight: Int?, requestSteps: Int?, requestGuidance: Float?,
  configDefaults: RenderDefaultValues
) -> (width: Int, height: Int, steps: Int?, guidance: Float?) {
  (
    width: requestWidth ?? configDefaults.width ?? 1024,
    height: requestHeight ?? configDefaults.height ?? 1024,
    steps: requestSteps ?? configDefaults.steps,
    guidance: requestGuidance ?? configDefaults.guidance.map(Float.init)
  )
}

/// comfybox#322: is this error an operator interrupt rather than a failure?
///
/// Both spellings reach the trackers: `CancellationError` straight out of a
/// pipeline loop (the image path's #304 contract — propagate unmodified), and
/// `WarmServerError.renderInterrupted`, the named form the video queue case
/// substitutes so the client sees a sentence instead of "CancellationError()".
/// A wrapped cancellation counts (review r1): `ltx2IsCancellation` unwraps the
/// `case x(String, Error)` wrappers this codebase uses on the load/render paths,
/// so a `CancellationError` that arrives inside one is still reported as an
/// interrupt rather than as a render failure.
func isRenderInterruption(_ error: Error) -> Bool {
  if ltx2IsCancellation(error) { return true }
  if case WarmServerError.renderInterrupted = error { return true }
  return false
}

/// comfybox#308/#322 (review r3): what the `.localVideo` case's generic
/// `catch` should do with a caught error, as a pure decision. nil means "an
/// operator interrupt — do not touch the health counters", using the SAME
/// `isRenderInterruption` classification `VideoJobTracker.markFailed`
/// already applies so the two never disagree about the same error.
///
/// This exists because the sibling `catch is CancellationError` branch (in
/// `WarmServerCoordinator`'s process loop) only catches a BARE
/// `CancellationError` thrown straight out of a pipeline loop — a WRAPPED
/// one (e.g. `ModelPoolError.loadFailed("…", CancellationError())`, which a
/// resume's model reload can throw) doesn't match `is CancellationError` and
/// used to fall through to the generic catch, where it was counted as a
/// failed render even though `VideoJobTracker.markFailed` (fed the same
/// error via the continuation) correctly reported it as interrupted —
/// `/health.failed_count` and the job status disagreed about the same event.
func localVideoCatchOutcome(for error: Error) -> LocalVideoCompletionOutcome? {
  isRenderInterruption(error) ? nil : .threw
}

#if DEBUG
/// K-FIX-1 / Codex C2 test seam — drives the coordinator's FIFO directly.
///
/// `WarmServerCoordinator` and the queue types are file-private (deliberately:
/// nothing outside this file should hold the queue), so the barrier test that
/// proves "no pool load or eviction begins until an active render exits" gets
/// this narrow probe instead of a widened actor. It exposes exactly three
/// things: a way to occupy the queue with a controllable job, the two model
/// operation enqueue seams the REST routes now use, and the pending count.
///
/// Construct it with `COMFYBOX_STATE_DIR` pointed at a temp directory — the
/// coordinator persists its queue snapshot and reads its pause sentinel from
/// that directory, and the LIVE engine's are not the test's to touch.
final class WarmServerQueueProbe: @unchecked Sendable {
  private let coordinator: WarmServerCoordinator
  private let liveHealth = LiveHealthState()
  /// comfybox#283/#217: same `COMFYBOX_STATE_DIR` override every other piece
  /// of this probe's state honors (see `stateDirectory`'s doc comment on
  /// `QueueStateStore`) — a test that points it at a temp directory before
  /// constructing this probe gets an isolated `queue-lifecycle.jsonl` too.
  private let lifecycleLedger = QueueLifecycleLedger()
  /// comfybox#362 review r2, item 3: the audit log the production sync route
  /// records to. Default path, which honours `COMFYBOX_STATE_DIR` — so a test
  /// that called `isolateComfyBoxStateDirectory()` never appends to the live
  /// `~/.comfybox/audit-log.jsonl`.
  private let probeAuditLog = AuditLog()
  private var liveHealthSnapshot: HealthSnapshot { liveHealth.read().0 }

  /// `modelSpec` is stored on the configuration only — nothing is resolved or
  /// loaded at construction (`prepare()` is a separate call this probe never
  /// makes), so it is safe in a unit test and is what lets the #282 review-r1
  /// `model_mismatch` case be driven for real.
  /// comfybox#217: the coordinator's OWN configuration value (PR #384 review,
  /// item 7 — the probe used to build a second one), so the probe's `/health`
  /// payload can never describe different limits than the queue it reads.
  private let configuration: WarmServerConfiguration

  init(maxPendingRequests: Int = 10, maxPendingModelOps: Int = 8, modelSpec: String? = nil) {
    let configuration = WarmServerConfiguration(
      modelSpec: modelSpec,
      maxPendingRequests: maxPendingRequests, maxPendingModelOps: maxPendingModelOps)
    self.configuration = configuration
    self.coordinator = WarmServerCoordinator(
      configuration: configuration,
      logger: Logger(label: "z-image.queue-probe"),
      videoHolder: VideoGeneratorHolder(),
      liveHealth: liveHealth,
      videoJobTracker: VideoJobTracker(),
      ltx2Telemetry: LTX2PhaseTelemetry(),
      ltx2PreemptionSignal: PreemptionSignal(),
      ltx2StepPosition: LTX2StepPosition(),
      ltx2EvictMean: RollingMeanSec(),
      ltx2ReloadMean: RollingMeanSec(),
      preemptionInFlight: LockedFlag(),
      pendingPreemptorBox: PendingPreemptorBox(),
      lifecycleLedger: lifecycleLedger)
  }

  /// comfybox#283/#217: the lifecycle events recorded for one job id, for a
  /// test to assert an enqueue→admit→start→complete cycle through the REAL
  /// coordinator produced the expected sequence.
  func lifecycleEvents(jobId: String) -> [QueueLifecycleEvent] {
    lifecycleLedger.events(jobId: jobId)
  }

  /// comfybox#283/#217: the lifecycle events recorded for one queue-job
  /// KIND, for a test whose job kind (e.g. `.modelSwitch` via
  /// `enqueueFakeRender`) never surfaces its own internal id to the caller.
  /// Fine for a fresh probe running one job of that kind at a time (every
  /// caller in this file's tests) — not a substitute for `jobId` filtering
  /// when several jobs of the same kind could be in flight together.
  func lifecycleEvents(jobKind: String) -> [QueueLifecycleEvent] {
    lifecycleLedger.events().filter { $0.jobKind == jobKind }
  }

  /// Occupy the queue with an arbitrary body — the stand-in for a render.
  /// `.modelSwitch` is used because it is the one existing queue kind whose
  /// work is a caller-supplied closure; the loop treats it exactly like any
  /// other job, one at a time.
  func enqueueFakeRender(_ body: @escaping @Sendable () async throws -> Bool) async throws -> Bool {
    try await coordinator.enqueueModelSwitch(body)
  }

  /// #339 review r3, item 3: the seam `recoverPersistedQueue`'s "generate"
  /// replay uses. Callers must PAUSE the probe (`setPaused(true)`) before
  /// calling this if they don't want the render to actually attempt to run
  /// — `.generate` never runs while paused (`runsWhilePaused`), so the job
  /// sits durably in `pending` without ever touching a real pipeline/model
  /// weights, which is what makes it safe to drive from a unit test.
  func enqueueGenerate(
    _ payload: GeneratePayload, source: String = "api", rawBody: Data? = nil, jobId: String? = nil
  ) async throws -> GenerateResponse {
    try await coordinator.enqueueGenerate(payload, source: source, rawBody: rawBody, jobId: jobId)
  }

  /// #339 review r3, item 3: the seam `recoverPersistedQueue` uses to
  /// publish the not-yet-admitted tail (`RecoverySnapshotMerger`) so a
  /// second restart mid-replay can't lose it.
  func setRecoveryUnadmittedTail(_ tail: [PersistedQueueJob]) async {
    await coordinator.setRecoveryUnadmittedTail(tail)
  }

  /// Cancel a pending job by id — the `DELETE /v1/queue/{id}` seam. Used by
  /// the r3 probe test to clean up a `.generate` job admitted (then never
  /// run, paused) via `enqueueGenerate` above, so the probe drains for
  /// `makeQueueProbe`'s teardown guard without ever letting the render
  /// actually attempt to start.
  @discardableResult
  func cancelPending(id: String) async -> Bool {
    await coordinator.cancelPending(id: id)
  }

  /// The seam `/v1/model/load` (wait: true), `/v1/model/activate` and
  /// `/v1/model/unload` use.
  func enqueueModelOperation(_ op: ModelOperation) async throws -> ModelOperationResult {
    try await coordinator.enqueueModelOperation(op)
  }

  /// The seam `/v1/model/load` (wait: false) uses — returns the queue job id.
  func enqueueModelOperationDetached(_ op: ModelOperation) async throws -> String {
    try await coordinator.enqueueModelOperationDetached(op)
  }

  /// `ServerError` is nested in the file-private coordinator, so a test
  /// cannot pattern-match it. This is the one predicate the WP-E8 hygiene
  /// tests need: was this refusal the capacity gate?
  static func isQueueFull(_ error: Error) -> Bool {
    if case WarmServerCoordinator.ServerError.queueFull = error { return true }
    return false
  }

  // MARK: - #282 per-request preset stacks

  /// What `runGenerate` would apply for this payload, and where it came from.
  /// Drives `WarmServerCoordinator.resolveJobLoRAStack` — the same function
  /// the dequeue path calls.
  func resolveJobStack(
    _ payload: GeneratePayload
  ) async throws -> (origin: String, names: [String], warmDefaultSkipped: String?) {
    try await coordinator.testSeamResolveJobStack(payload)
  }

  /// Publish a warm default without a swap (applying a real stack needs model
  /// weights). Runs `runSwap`'s own `adoptWarmDefaultStack`. `tag` defaults to
  /// the coordinator's CURRENT base, i.e. the ordinary "swapped here, rendered
  /// here" case; pass one explicitly to stage a cross-base mismatch (C1).
  func adoptWarmDefaultStack(
    _ stack: [LoRAConfiguration], tag: RequestStackResolver.WarmDefaultTag? = nil
  ) async {
    await coordinator.testSeamAdoptWarmDefaultStack(stack, tag: tag)
  }

  /// The provenance tag the warm default carries.
  func warmDefaultTag() async -> RequestStackResolver.WarmDefaultTag {
    await coordinator.testSeamWarmDefaultTag()
  }

  /// The family this coordinator renders on (`flux1` on a fresh probe).
  func currentFamily() async -> String {
    await coordinator.testSeamCurrentFamily()
  }

  /// Review r1 (I4): install a recorder so `applyJobLoRAStack`'s pipeline
  /// application is replaced by a recording and `runGenerate` stops before
  /// dispatching to a family. Lets a REAL queued `.generate` job prove the
  /// apply call exists, with no weights and no GPU.
  func setStackRecorder(_ recorder: StackApplicationRecorder?) async {
    await coordinator.setStackRecorder(recorder)
  }

  /// The warm default the coordinator holds.
  func warmDefaultStackNames() async -> [String] {
    await coordinator.testSeamWarmDefaultStackNames()
  }

  /// The warm default as `GET /v1/model/pool` reports it (`warm_default_stack`).
  func poolWarmDefaultStack() async -> [LoRAState]? {
    await coordinator.poolList().warmDefaultStack
  }

  /// `POST /v1/lora/swap`, through the real queue (`LoRASwapPayload` is
  /// file-private, so the probe builds it). Safe in a unit test ONLY with an
  /// empty stack: `applyActiveLoRAs([])` unloads rather than loads, so no
  /// model weights are touched — which is exactly what makes it a real
  /// end-to-end proof that `runSwap` publishes the warm default.
  @discardableResult
  func enqueueSwap(loras: [LoRAEntry]) async throws -> Int {
    try await coordinator.enqueueSwap(LoRASwapPayload(loras: loras), rawBody: nil).loraCount
  }

  /// The model-operation cap, distinct from the render queue's.
  static func isModelOperationQueueFull(_ error: Error) -> Bool {
    if case WarmServerCoordinator.ServerError.modelOperationQueueFull = error { return true }
    return false
  }

  /// The seam `/v1/shutdown` uses. Returns the response's `success` flag —
  /// the response type is file-private, and the only thing a test needs to
  /// know is that the call RETURNED rather than parking forever under a
  /// pause (WP-E8 hygiene).
  func enqueueShutdown() async throws -> Bool {
    try await coordinator.enqueueShutdown().success
  }

  /// The kinds of the jobs still waiting, read through the same
  /// lock-based snapshot `/health` and `/v1/queue` publish.
  func pendingJobKinds() -> [String] {
    liveHealthSnapshot.pending.map { $0.kind }
  }

  /// The pause sentinel the coordinator would read, exposed so a test can
  /// assert it is redirected away from the LIVE `~/.comfybox/queue-paused`
  /// before any coordinator is constructed.
  static var pauseSentinelPath: String { WarmServerCoordinator.pauseSentinelPath }

  /// Pause / resume the queue, exactly as `/v1/queue/pause` does.
  func setPaused(_ paused: Bool) async {
    await coordinator.setPaused(paused)
  }

  /// The summary of the job the loop is running, or nil when idle.
  var activeJobSummary: String? { liveHealthSnapshot.activeSummary }

  /// comfybox#283/#217: the id of the job the loop is running, or nil when
  /// idle — for a test that needs to query `lifecycleEvents(jobId:)` for a
  /// job kind (e.g. `.modelSwitch` via `enqueueFakeRender`) that has no
  /// caller-supplied id to hand.
  var activeJobId: String? { liveHealthSnapshot.activeJobId }

  /// Nothing running and nothing waiting.
  ///
  /// The drain guard reads this: a test must not tear down its isolated state
  /// directory while the coordinator still has work, because the loop's
  /// per-job `persistQueueState()` would then resolve — and `removeItem` —
  /// the LIVE snapshot path (K-FIX-1 round 2, New-2).
  var isDrained: Bool {
    let snapshot = liveHealthSnapshot
    return snapshot.activeJobId == nil && snapshot.pending.isEmpty
  }

  var pendingCount: Int { liveHealthSnapshot.pending.count }

  var isPaused: Bool { liveHealthSnapshot.isPaused }

  // MARK: - 0.B-2 control-plane probe surface

  /// Occupy the loop with a synthetic op that BLOCKS its thread for `durationMs`.
  func enqueueSynthetic(durationMs: Int, id: String = UUID().uuidString) async throws -> Bool {
    try await coordinator.enqueueSynthetic(durationMs: durationMs, id: id)
  }

  /// comfybox#308 (review r2, item 2b): the `.localVideo` completion
  /// bookkeeping seam — see `WarmServerCoordinator.testSeamFinishLocalVideo`.
  func finishLocalVideo(
    _ outcome: LocalVideoCompletionOutcome, lastError message: String? = nil
  ) async -> (successCount: Int, failedCount: Int, lastDurationMs: Int?, lastError: String?) {
    await coordinator.testSeamFinishLocalVideo(outcome, lastError: message)
  }

  /// comfybox#308/#322 (review r3): the `.localVideo` generic-catch seam —
  /// see `WarmServerCoordinator.testSeamHandleLocalVideoCatch`.
  func handleLocalVideoCatch(
    _ error: Error
  ) async -> (successCount: Int, failedCount: Int, lastDurationMs: Int?, lastError: String?) {
    await coordinator.testSeamHandleLocalVideoCatch(error)
  }

  /// The sync `/v1/queue/pause` path: authoritative lock-store write.
  func controlPause() { liveHealth.setPaused(true) }

  /// The sync `/v1/queue/resume` path: authoritative write + fire-and-forget wake
  /// (never a mailbox command — the F1 wedge guard).
  func controlResume() {
    liveHealth.setPaused(false)
    Task { await coordinator.setPaused(false) }
  }

  /// Whether the AUTHORITATIVE (lock-store) pause flag is set — the value the
  /// between-items gate and the read path both see.
  var lockStorePaused: Bool { liveHealth.isPausedAuthoritative() }

  /// The sync `DELETE /v1/queue/{id}` path: record a cancel delta + drain nudge.
  func controlCancel(id: String) {
    liveHealth.recordDelta(.cancel(id))
    Task { await coordinator.drainControlDeltas() }
  }

  /// The sync `POST /v1/queue/{id}/move` path.
  func controlMove(id: String, direction: String) {
    liveHealth.recordDelta(.move(id, direction: direction))
    Task { await coordinator.drainControlDeltas() }
  }

  /// The sync `/v1/queue/interrupt` path, default target (whatever health
  /// shows as active).
  @discardableResult
  func controlInterrupt() -> Bool {
    if case .cancelled = liveHealth.cancelActiveRender() { return true }
    return false
  }

  /// comfybox#362: the sync `/v1/queue/interrupt` path with an explicit
  /// `target` — returns enough for a test to assert the response body's
  /// additive `interrupted_job_id`/`interrupted_kind` agree with what was
  /// actually cancelled, and to distinguish "nothing running there" from "no
  /// such target" (the 404 case).
  func controlInterrupt(
    target: String?
  ) -> (interrupted: Bool, jobId: String?, kind: String?, unknownTarget: Bool) {
    switch liveHealth.cancelActiveRender(target: target) {
    case .cancelled(let jobId, let kind): return (true, jobId, kind, false)
    case .nothingToCancel: return (false, nil, nil, false)
    case .unknownTarget: return (false, nil, nil, true)
    }
  }

  /// Record a cancel delta WITHOUT the drain nudge — lets tests hold a delta
  /// in the undrained window deterministically (F-2 crash-window test).
  func recordCancelDeltaOnly(id: String) { liveHealth.recordDelta(.cancel(id)) }

  /// comfybox#386 review round 2, item 3: the exact call `recoverPersistedQueue`
  /// makes after folding undrained deltas into the recovered queue — exposed
  /// so a test can drive it concurrently with an in-flight older sidecar
  /// write and prove the clear can't be resurrected.
  func clearAllDeltas() { liveHealth.clearDeltas() }

  /// Deterministically run one drain on the actor (the same
  /// `drainControlDeltas` the sync routes nudge fire-and-forget).
  func drainNow() async { await coordinator.drainControlDeltas() }

  var undrainedDeltaCount: Int { liveHealth.undrainedDeltas().count }

  /// comfybox#386 review round 2, item 1: the DURABLE-only view `peekDeltas`
  /// exposes to the drain — distinct from `undrainedDeltaCount`, which shows
  /// every recorded delta regardless of whether its sidecar write landed.
  var peekedDrainableDeltaCount: Int { liveHealth.peekDeltas().count }

  /// comfybox#386 review round 3, item 1: re-attempt the sidecar write for
  /// whatever's currently undrained, without adding a new delta — the same
  /// primitive `drainQueueDeltas`'s liveness retry (item 1a) uses.
  @discardableResult
  func retrySidecarWrite() -> Bool { liveHealth.retryPendingSidecarWrite() }

  /// Whether the sidecar has failed long/often enough that the drain applies
  /// non-durable deltas anyway (item 1b).
  var isDeltaSidecarDegraded: Bool { liveHealth.deltaDurabilityStatus().isDegraded }

  /// comfybox#386 review round 4, item 2 test seam: same as
  /// `isDeltaSidecarDegraded`, but with an injectable clock so a test can
  /// simulate the degraded-mode time window elapsing without an actual sleep.
  func isDeltaSidecarDegraded(asOf now: Date) -> Bool { liveHealth.deltaDurabilityStatus(now: now).isDegraded }

  /// How many currently-undrained deltas are not yet confirmed durable —
  /// what `/health`'s additive `queue_delta_non_durable_count` reports.
  var nonDurableDeltaCount: Int { liveHealth.deltaDurabilityStatus().nonDurableCount }

  /// The consecutive-failure count that trips degraded mode — exposed so a
  /// test can drive exactly to the threshold without duplicating the constant.
  static var degradedModeFailureCountThreshold: Int { LiveHealthState.degradedModeFailureCountThreshold }

  /// The time-based half of the degraded-mode window — comfybox#386 review
  /// round 4, item 2's test seam for `isDeltaSidecarDegraded(asOf:)`.
  static var degradedModeWindowSeconds: TimeInterval { LiveHealthState.degradedModeWindowSeconds }

  /// comfybox#386 review round 4, item 3: the background self-heal's retry
  /// delay and attempt cap, exposed so a test can bound its own wait without
  /// duplicating the constants. Round 5, item 2: settable — a test driving
  /// the heal through several real failures shrinks this so the run doesn't
  /// burn several real seconds; nothing resets it automatically, so a test
  /// that lowers it MUST restore it (`defer`).
  static var backgroundHealRetryDelaySeconds: TimeInterval {
    get { LiveHealthState.backgroundHealRetryDelaySeconds }
    set { LiveHealthState.backgroundHealRetryDelaySeconds = newValue }
  }
  static var backgroundHealMaxAttempts: Int { LiveHealthState.backgroundHealMaxAttempts }

  #if DEBUG
  /// comfybox#386 review round 5, item 3 test seam: directly stamp the
  /// failure-streak state (see `LiveHealthState.testSeamStampFailureStreak`).
  func stampFailureStreak(first: Date, last: Date, count: Int) {
    liveHealth.testSeamStampFailureStreak(first: first, last: last, count: count)
  }
  #endif

  /// Pending ids as `GET /v1/queue` composes them (snapshot + undrained deltas) —
  /// a just-cancelled job is already absent here.
  var composedPendingIds: [String] {
    let (snap, _) = liveHealth.read()
    return QueueDeltaApplier.apply(liveHealth.undrainedDeltas(), to: snap.pending, id: { $0.id }).map { $0.id }
  }

  /// Pending ids in the raw actor-published snapshot (no delta compose).
  var snapshotPendingIds: [String] { liveHealth.read().0.pending.map { $0.id } }

  // MARK: - comfybox#322 video-interrupt probe surface

  /// Skip the #218 admission gate for `.localVideo` jobs (DEBUG only).
  ///
  /// The gate needs ~65-80GB of real free RAM. These tests exist to prove the
  /// `.localVideo` queue case publishes a cancellable render task and that an
  /// interrupt aimed at the video does not cancel a preemptor — not to re-test
  /// admission, which `HeavyModelAdmissionTests` already covers with injected
  /// byte figures.
  ///
  /// comfybox#362: takes an explicit `enabled` (default `true`, so every
  /// existing `bypassVideoAdmission()` call site is unchanged) so a test can
  /// also flip it back to `false` — each `WarmServerQueueProbe` already owns
  /// a fresh coordinator per test (`makeQueueProbe`), so this never leaked
  /// ACROSS tests, but nothing previously let a single test turn admission
  /// back on for a later assertion, and `makeQueueProbe`'s teardown now
  /// resets it defensively for the same reason.
  func bypassVideoAdmission(_ enabled: Bool = true) async {
    await coordinator.setBypassVideoAdmission(enabled)
  }

  /// Enqueue a `.localVideo` job through the REAL queue case — the same
  /// `enqueueLocalVideo` seam `/v1/video/generate` uses. `body` receives the
  /// progress reporter and runs on the coordinator's render task, so
  /// `Task.isCancelled` inside it is what a real LTX-2 render observes.
  func enqueueLocalVideo(
    wantsAudio: Bool = false,
    _ body: @escaping @Sendable (@escaping @Sendable (Int) -> Void) throws -> LTX2RenderOutcome
  ) async throws -> LTX2VideoResult {
    try await coordinator.enqueueLocalVideo(wantsAudio: wantsAudio, body)
  }

  /// Did this error come back as the named interrupt? `WarmServerError` is
  /// public, but this keeps the test reading like the route does.
  static func isInterrupted(_ error: Error) -> Bool { isRenderInterruption(error) }

  /// The PRODUCTION shield the preemption episode wraps its image job in
  /// (`runShieldedFromCancellation`) — the same function, not a copy — so a
  /// test can cancel the caller and prove the shielded work neither observes
  /// the cancellation nor stops early.
  func runShieldedFromCancellation(_ work: @escaping @Sendable () async -> Void) async {
    await coordinator.runShieldedFromCancellation(work)
  }

  // MARK: - comfybox#362 interrupt-target probe surface

  /// The PRODUCTION function `runPreemptionEpisode` wraps its preempting
  /// image job in (`runAsPublishedActiveRender`) — the same function, not a
  /// copy — so a test can drive the exact publish/restore sequence that makes
  /// `/health` and `/v1/queue/interrupt`'s default target agree during an
  /// episode, using fake tasks instead of a real checkpoint/render.
  func runAsPublishedActiveRender(
    restoringTo restoreTo: PublishedRender,
    preemptorIdentity: (jobId: String, kind: String)? = nil,
    restoredIdentity: (jobId: String?, kind: String?)? = nil,
    _ work: @escaping @Sendable () async -> Void
  ) async {
    await coordinator.runAsPublishedActiveRenderForTest(
      restoringTo: restoreTo, preemptorIdentity: preemptorIdentity,
      restoredIdentity: restoredIdentity, work)
  }

  /// comfybox#362 review r2, item 2: the PUBLISHED interrupt triple, read
  /// non-destructively from `LiveHealthState` (no actor hop, no cancel) — so
  /// a test can wait for a publication instead of polling a destructive
  /// interrupt until one lands.
  var liveActiveRender: PublishedRender { liveHealth.activeRenderPublication() }

  /// comfybox#362 review r2, item 1: install/remove the publication observer.
  /// See `LiveHealthState.publicationObserver`.
  static func observePublications(
    _ observer: (@Sendable (String?, String?, String?) -> Void)?
  ) {
    LiveHealthState.publicationObserver = observer
  }

  /// comfybox#362 test seam: stage a fake "checkpointed video" exactly as
  /// `runPreemptionEpisode` publishes it for the span of a real episode,
  /// without needing a real LTX-2 checkpoint. `statusJobId` is the
  /// `/v1/video/status/{id}` id (comfybox#283: it differs from the queue id).
  func setCheckpointedVideo(
    task: Task<Void, Never>?, jobId: String?, statusJobId: String? = nil
  ) async {
    await coordinator.setCheckpointedVideoForTest(
      task: task, jobId: jobId, statusJobId: statusJobId)
  }

  /// comfybox#362 test seam: publish an active render directly, as the queue
  /// loop does around every job.
  func setActiveRender(
    task: Task<Void, Never>?, jobId: String?, statusJobId: String? = nil, kind: String?
  ) async {
    await coordinator.setActiveRenderForTest(
      task: task, jobId: jobId, statusJobId: statusJobId, kind: kind)
  }

  /// comfybox#362 test seam: the currently published active render — what
  /// `runPreemptionEpisode` captures and hands back as `restoringTo`.
  var publishedActiveRender: PublishedRender {
    get async { await coordinator.activeRenderForTest }
  }

  /// comfybox#362 review r1, finding 5: the EXACT chain
  /// `WarmServer.syncInterruptResponse` runs — `InterruptRoute.decodeTarget`,
  /// `LiveHealthState.cancelActiveRender`, `InterruptRoute.response` — so a
  /// test can drive the sync route from request BYTES to response BYTES. Only
  /// the audit-log line (which needs a `WarmServer`) is left out.
  func syncInterruptRoute(body: Data) -> HTTPResponse {
    syncInterruptRoute(
      request: HTTPRequest(
        method: "POST", path: "/v1/queue/interrupt", queryString: nil, headers: [:], body: body))
  }

  /// The PRODUCTION sync route (`WarmServer.interruptRouteResponse`, which
  /// `WarmServer.syncInterruptResponse` is a one-line call to) driven with a
  /// full `HTTPRequest` — review r2, item 3. The audit log is a scratch one
  /// under `COMFYBOX_STATE_DIR`, so nothing touches the live file.
  func syncInterruptRoute(request: HTTPRequest) -> HTTPResponse {
    WarmServer.interruptRouteResponse(
      request: request, liveHealth: liveHealth, auditLog: probeAuditLog)
  }

  /// comfybox#380: the PRODUCTION sync `GET /v1/model/family` route
  /// (`WarmServer.syncModelFamilyResponse`, which `WarmServer.modelFamilyRouteResponse`
  /// is a one-line call to) driven with a full `HTTPRequest`, so a test
  /// exercises the real `request.queryParameters` percent-decoding rather
  /// than calling `ModelFamilyDetector.detect` directly.
  func syncModelFamilyRoute(request: HTTPRequest) -> HTTPResponse {
    WarmServer.modelFamilyRouteResponse(request: request)
  }

  /// Audit events this probe's route calls recorded, so a route test can
  /// assert the route logged what it did.
  func recordedAuditEvents() -> [(kind: String, metadata: [String: String]?)] {
    probeAuditLog.recent(limit: 50).map { ($0.kind.rawValue, $0.metadata) }
  }

  /// The ASYNC `/v1/queue/interrupt` fallback path (`ControlPlaneSyncFlag`
  /// off) — see `controlInterrupt(target:)` for the sync path this must
  /// agree with.
  func coordinatorInterrupt(
    target: String? = nil
  ) async -> (interrupted: Bool, jobId: String?, kind: String?, unknownTarget: Bool) {
    switch await coordinator.cancelActiveRenderForTest(target: target) {
    case .cancelled(let jobId, let kind): return (true, jobId, kind, false)
    case .nothingToCancel: return (false, nil, nil, false)
    case .unknownTarget: return (false, nil, nil, true)
    }
  }

  /// Read back the DEBUG-only admission bypass — see
  /// `WarmServerCoordinator.bypassVideoAdmissionForTestsValue`.
  func bypassVideoAdmissionValueForTest() async -> Bool {
    await coordinator.bypassVideoAdmissionForTestsValue
  }

  // MARK: - comfybox#217 /health sync-servability probe surface

  /// The PRODUCTION payload assembly `GET /health` runs
  /// (`WarmServer.liveHealthPayload`) — the same function, not a copy — driven
  /// against THIS probe's lock store while its coordinator is busy. Calling it
  /// is synchronous by construction: a test that gets a value back here has
  /// proved the route needs no actor hop and no cooperative thread.
  func liveHealthPayload(memoryBytes: UInt64 = 0) -> HealthResponse {
    WarmServer.liveHealthPayload(
      liveHealth: liveHealth,
      configuration: configuration,
      serverStartTime: Date(), memoryBytes: memoryBytes)
  }

  /// The serialized `/health` body the route emits, from the same payload —
  /// so a test can assert the progress-adjacent fields the Desktop UI polls
  /// (`is_rendering`, `progress_percent`, `pending_count`, `current_job_id`)
  /// are present and correct mid-render.
  func liveHealthJSON() -> [String: Any] {
    guard let data = WarmServer.healthJSON(
      liveHealthPayload(), videoAvailable: false, activeVideoJobs: 0),
      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return [:] }
    return object
  }

  /// Publish live denoising progress exactly as the off-actor progress callback
  /// does, so a test can assert `/health` reports it without the actor.
  func publishProgress(_ percent: Int) { liveHealth.setProgress(percent) }

  /// comfybox#362 review r1, finding 2: reproduce the `runGenerate`-defer
  /// window (actor `activeJobId` nil, published snapshot still naming the
  /// job). See `WarmServerCoordinator.clearActiveJobIdWithoutPublishingForTest`.
  func clearActiveJobIdWithoutPublishing() async {
    await coordinator.clearActiveJobIdWithoutPublishingForTest()
  }
}
#endif
