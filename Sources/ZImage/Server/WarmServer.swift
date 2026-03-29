import Foundation
import Dispatch
import Logging
import Network
import Darwin

public struct WarmServerConfiguration: Sendable {
  public var port: UInt16
  public var modelSpec: String?
  public var textEncoderPath: String?
  public var initialLoRAs: [LoRAConfiguration]
  public var forceTransformerOverrideOnly: Bool
  public var maxSequenceLength: Int
  public var maxPendingRequests: Int

  public init(
    port: UInt16 = 7862,
    modelSpec: String? = nil,
    textEncoderPath: String? = nil,
    initialLoRAs: [LoRAConfiguration] = [],
    forceTransformerOverrideOnly: Bool = false,
    maxSequenceLength: Int = 512,
    maxPendingRequests: Int = 10
  ) {
    self.port = port
    self.modelSpec = modelSpec
    self.textEncoderPath = textEncoderPath
    self.initialLoRAs = initialLoRAs
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
    self.maxSequenceLength = maxSequenceLength
    self.maxPendingRequests = max(1, maxPendingRequests)
  }
}

public final class WarmServer {
  private let configuration: WarmServerConfiguration
  private let logger: Logger
  private let coordinator: WarmServerCoordinator
  private let listenerQueue = DispatchQueue(label: "z-image.warm-server.listener")
  private let shutdownSemaphore = DispatchSemaphore(value: 0)
  private let lifecycleLock = NSLock()
  private var listener: NWListener?
  private var terminalError: Error?
  private var shutdownSignalled = false

  public init(configuration: WarmServerConfiguration, logger: Logger = Logger(label: "z-image.warm-server")) {
    self.configuration = configuration
    self.logger = logger
    self.coordinator = WarmServerCoordinator(configuration: configuration, logger: logger)
  }

  public func run() throws {
    try preparePipeline()

    guard let port = NWEndpoint.Port(rawValue: configuration.port) else {
      throw WarmServerError.invalidPort(configuration.port)
    }

    let listener = try NWListener(using: .tcp, on: port)
    self.listener = listener

    listener.stateUpdateHandler = { [weak self] state in
      self?.handleListenerState(state)
    }
    listener.newConnectionHandler = { [weak self] connection in
      self?.accept(connection: connection)
    }

    listener.start(queue: listenerQueue)
    shutdownSemaphore.wait()

    if let terminalError {
      throw terminalError
    }
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
      logger.info("Warm server listening on http://127.0.0.1:\(self.configuration.port)")
    case .failed(let error):
      logger.error("Warm server listener failed: \(error.localizedDescription)")
      signalShutdown(error: error)
    case .cancelled:
      signalShutdown()
    default:
      break
    }
  }

  private func accept(connection: NWConnection) {
    let handler = ConnectionHandler(
      connection: connection,
      queue: DispatchQueue(label: "z-image.warm-server.connection.\(UUID().uuidString)"),
      server: self
    )
    handler.start()
  }

  fileprivate func respond(to request: HTTPRequest) async -> RoutedResponse {
    switch (request.method, request.path) {
    case ("GET", "/health"):
      let memoryBytes = Self.currentMemoryFootprintBytes()
      let health = await coordinator.health(memoryBytes: memoryBytes)
      return .json(status: 200, payload: health)

    case ("POST", "/v1/generate"):
      do {
        let payload = try decode(GeneratePayload.self, from: request.body)
        let result = try await coordinator.enqueueGenerate(payload)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/lora/swap"):
      do {
        let payload = try decode(LoRASwapPayload.self, from: request.body)
        let result = try await coordinator.enqueueSwap(payload)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/shutdown"):
      do {
        let result = try await coordinator.enqueueShutdown()
        return .shutdown(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    default:
      if ["/v1/generate", "/v1/lora/swap", "/v1/shutdown", "/health"].contains(request.path) {
        return .error(.error(status: 405, message: "Method not allowed"))
      }
      return .error(.error(status: 404, message: "Not found"))
    }
  }

  fileprivate func requestShutdownAfterResponse() {
    lifecycleLock.lock()
    defer { lifecycleLock.unlock() }

    guard !shutdownSignalled else { return }
    shutdownSignalled = true
    listener?.cancel()
    shutdownSemaphore.signal()
  }

  private func signalShutdown(error: Error? = nil) {
    lifecycleLock.lock()
    defer { lifecycleLock.unlock() }

    if let error, terminalError == nil {
      terminalError = error
    }

    guard !shutdownSignalled else { return }
    shutdownSignalled = true
    shutdownSemaphore.signal()
  }

  private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: data)
  }

  private func response(for error: Error) -> HTTPResponse {
    switch error {
    case let error as WarmServerCoordinator.ServerError:
      switch error {
      case .queueFull(let maxPending):
        return .error(status: 429, message: "Queue full (\(maxPending) pending max)")
      case .shuttingDown:
        return .error(status: 503, message: "Server is shutting down")
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

    case let error as DecodingError:
      return .error(status: 400, message: "Invalid JSON body: \(describe(decodingError: error))")

    default:
      return .error(status: 500, message: error.localizedDescription)
    }
  }

  private func describe(decodingError: DecodingError) -> String {
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
}

private actor WarmServerCoordinator {
  enum ServerError: Error {
    case queueFull(maxPending: Int)
    case shuttingDown
  }

  private let configuration: WarmServerConfiguration
  private let logger: Logger
  private let pipeline: ZImagePipeline
  private let startTime = Date()
  private var activeLoRAs: [LoRAConfiguration]
  private var pending: [QueuedOperation] = []
  private var isProcessing = false
  private var shuttingDown = false
  private var successfulRenderCount = 0
  private var failedRenderCount = 0
  private var lastRenderDurationMs: Int?
  private var lastError: String?
  private var activeRenderStartedAt: Date?
  private var pipelinePrepared = false

  init(configuration: WarmServerConfiguration, logger: Logger) {
    self.configuration = configuration
    self.logger = logger
    self.pipeline = ZImagePipeline(logger: logger, retentionPolicy: .keepLoaded)
    self.activeLoRAs = configuration.initialLoRAs
  }

  func prepare() async throws {
    logger.info("Preloading warm server pipeline")
    try await pipeline.prepare(
      modelSpec: configuration.modelSpec,
      textEncoderPath: configuration.textEncoderPath,
      loras: activeLoRAs,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly
    )
    pipelinePrepared = true
    logger.info("Warm server pipeline ready")
  }

  func enqueueGenerate(_ payload: GeneratePayload) async throws -> GenerateResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(.generate(payload, ContinuationBox(continuation)))
      startProcessingIfNeeded()
    }
  }

  func enqueueSwap(_ payload: LoRASwapPayload) async throws -> LoRASwapResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(.swap(payload, ContinuationBox(continuation)))
      startProcessingIfNeeded()
    }
  }

  func enqueueShutdown() async throws -> ShutdownResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }

    shuttingDown = true
    return try await withCheckedThrowingContinuation { continuation in
      pending.append(.shutdown(ContinuationBox(continuation)))
      startProcessingIfNeeded()
    }
  }

  func health(memoryBytes: UInt64) -> HealthResponse {
    let uptimeSeconds = Int(Date().timeIntervalSince(startTime))
    let activeAgeMs = activeRenderStartedAt.map { Int(Date().timeIntervalSince($0) * 1000.0) }

    return HealthResponse(
      status: shuttingDown ? "shutting_down" : "ok",
      model: configuration.modelSpec ?? ZImageRepository.id,
      textEncoderPath: configuration.textEncoderPath,
      loaded: pipelinePrepared,
      loras: activeLoRAs.map(LoRAState.init),
      uptimeSeconds: uptimeSeconds,
      renderCount: successfulRenderCount,
      failedRenderCount: failedRenderCount,
      pendingCount: pending.count,
      maxPending: configuration.maxPendingRequests,
      isRendering: activeRenderStartedAt != nil,
      activeRequestAgeMs: activeAgeMs,
      memoryUsageBytes: memoryBytes,
      memoryUsageMB: memoryBytes / (1024 * 1024),
      lastRenderDurationMs: lastRenderDurationMs,
      lastError: lastError
    )
  }

  private func startProcessingIfNeeded() {
    guard !isProcessing else { return }
    isProcessing = true
    Task {
      await processLoop()
    }
  }

  private func processLoop() async {
    while true {
      guard !pending.isEmpty else {
        isProcessing = false
        return
      }

      let operation = pending.removeFirst()
      switch operation {
      case .generate(let payload, let continuation):
        await runGenerate(payload, continuation: continuation)
      case .swap(let payload, let continuation):
        await runSwap(payload, continuation: continuation)
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

  private func runGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    let start = Date()

    do {
      let request = try payload.makePipelineRequest(
        configuration: configuration,
        activeLoRAs: activeLoRAs
      )
      let outputURL = try await pipeline.generateFromRequest(request)
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: outputURL.path,
          durationMs: durationMs
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      continuation.resume(throwing: error)
    }
  }

  private func runSwap(_ payload: LoRASwapPayload, continuation: ContinuationBox<LoRASwapResponse>) async {
    do {
      let newLoRAs = try payload.makeConfigurations()
      try await pipeline.swapLoRAs(newLoRAs)
      activeLoRAs = newLoRAs
      lastError = nil
      continuation.resume(
        returning: LoRASwapResponse(
          success: true,
          loraCount: activeLoRAs.count,
          loras: activeLoRAs.map(LoRAState.init)
        )
      )
    } catch {
      activeLoRAs = pipeline.loadedLoRAConfigs
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
    }
  }
}

private final class ConnectionHandler {
  private static let headerDelimiter = Data("\r\n\r\n".utf8)
  private static let maximumRequestBytes = 1_048_576

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
    let path = rawPath.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? rawPath

    return .request(
      HTTPRequest(
        method: String(requestParts[0]).uppercased(),
        path: path,
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

    Task {
      let routed = await server.respond(to: request)
      switch routed {
      case .error(let response):
        self.finish(with: response)
      case .json(let response):
        self.finish(with: response)
      case .shutdown(let response):
        self.finish(with: response, shutdownAfterSend: true)
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

private struct HTTPRequest {
  let method: String
  let path: String
  let headers: [String: String]
  let body: Data
}

private enum HTTPParseResult {
  case incomplete
  case request(HTTPRequest)
  case error(HTTPResponse)
}

private struct HTTPResponse {
  let status: Int
  let reasonPhrase: String
  let body: Data

  static func json<T: Encodable>(status: Int, payload: T) -> HTTPResponse {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let body = (try? encoder.encode(payload)) ?? Data("{\"success\":false,\"error\":\"encoding failure\"}".utf8)
    return HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), body: body)
  }

  static func error(status: Int, message: String) -> HTTPResponse {
    json(status: status, payload: ErrorPayload(success: false, error: message))
  }

  func serialize() -> Data {
    var data = Data()
    let header = [
      "HTTP/1.1 \(status) \(reasonPhrase)",
      "Content-Type: application/json",
      "Content-Length: \(body.count)",
      "Connection: close",
      "",
      ""
    ].joined(separator: "\r\n")
    data.append(Data(header.utf8))
    data.append(body)
    return data
  }

  private static func reasonPhrase(for status: Int) -> String {
    switch status {
    case 200: return "OK"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 413: return "Payload Too Large"
    case 429: return "Too Many Requests"
    case 500: return "Internal Server Error"
    case 503: return "Service Unavailable"
    default: return "OK"
    }
  }
}

private enum RoutedResponse {
  case json(HTTPResponse)
  case shutdown(HTTPResponse)
  case error(HTTPResponse)

  static func json<T: Encodable>(status: Int, payload: T) -> RoutedResponse {
    .json(.json(status: status, payload: payload))
  }

  static func shutdown<T: Encodable>(status: Int, payload: T) -> RoutedResponse {
    .shutdown(.json(status: status, payload: payload))
  }
}

private struct GeneratePayload: Decodable, Sendable {
  let prompt: String
  let negativePrompt: String?
  let width: Int?
  let height: Int?
  let steps: Int?
  let guidance: Float?
  let seed: UInt64?
  let outputPath: String?

  func makePipelineRequest(
    configuration: WarmServerConfiguration,
    activeLoRAs: [LoRAConfiguration]
  ) throws -> ZImageGenerationRequest {
    let outputURL: URL
    if let outputPath, !outputPath.isEmpty {
      outputURL = URL(fileURLWithPath: outputPath)
    } else {
      outputURL = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("zimage-\(UUID().uuidString).png")
    }

    return ZImageGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: width ?? ZImageModelMetadata.recommendedWidth,
      height: height ?? ZImageModelMetadata.recommendedHeight,
      steps: steps ?? ZImageModelMetadata.recommendedInferenceSteps,
      guidanceScale: guidance ?? ZImageModelMetadata.recommendedGuidanceScale,
      seed: seed,
      outputPath: outputURL,
      model: configuration.modelSpec,
      textEncoderPath: configuration.textEncoderPath,
      maxSequenceLength: configuration.maxSequenceLength,
      loras: activeLoRAs,
      enhancePrompt: false,
      enhanceMaxTokens: 512,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly
    )
  }
}

private struct GenerateResponse: Encodable, Sendable {
  let success: Bool
  let outputPath: String
  let durationMs: Int
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

private struct LoRAEntry: Codable, Sendable {
  let path: String
  let scale: Float?

  func makeConfiguration() throws -> LoRAConfiguration {
    let expanded = (path as NSString).expandingTildeInPath
    if path.hasPrefix("/") || path.hasPrefix("./") || path.hasPrefix("../") || path.hasPrefix("~") || FileManager.default.fileExists(atPath: expanded) {
      return .local(expanded, scale: scale ?? 1.0)
    }
    return .huggingFace(path, scale: scale ?? 1.0)
  }
}

private struct ShutdownResponse: Encodable, Sendable {
  let success: Bool
  let message: String
}

private struct HealthResponse: Encodable, Sendable {
  let status: String
  let model: String
  let textEncoderPath: String?
  let loaded: Bool
  let loras: [LoRAState]
  let uptimeSeconds: Int
  let renderCount: Int
  let failedRenderCount: Int
  let pendingCount: Int
  let maxPending: Int
  let isRendering: Bool
  let activeRequestAgeMs: Int?
  let memoryUsageBytes: UInt64
  let memoryUsageMB: UInt64
  let lastRenderDurationMs: Int?
  let lastError: String?
}

private struct LoRAState: Encodable, Sendable {
  let source: String
  let scale: Float

  init(_ configuration: LoRAConfiguration) {
    switch configuration.source {
    case .local(let url):
      self.source = url.path
    case .huggingFace(let modelId, let filename):
      self.source = filename.map { "\(modelId)/\($0)" } ?? modelId
    }
    self.scale = configuration.scale
  }
}

private struct ErrorPayload: Encodable {
  let success: Bool
  let error: String
}

private enum QueuedOperation: Sendable {
  case generate(GeneratePayload, ContinuationBox<GenerateResponse>)
  case swap(LoRASwapPayload, ContinuationBox<LoRASwapResponse>)
  case shutdown(ContinuationBox<ShutdownResponse>)
}

private final class ContinuationBox<Value>: @unchecked Sendable {
  private let continuation: CheckedContinuation<Value, Error>

  init(_ continuation: CheckedContinuation<Value, Error>) {
    self.continuation = continuation
  }

  func resume(returning value: Value) {
    continuation.resume(returning: value)
  }

  func resume(throwing error: Error) {
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

  public var errorDescription: String? {
    switch self {
    case .invalidPort(let port):
      return "Invalid server port: \(port)"
    }
  }
}
