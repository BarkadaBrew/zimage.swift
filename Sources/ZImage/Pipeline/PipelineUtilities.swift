import Foundation
import Logging
import MLX
import MLXNN

public enum PipelineUtilities {
    public enum UtilityError: Error {
        case textEncoderFailed
        case emptyEmbeddings
    }
    public static func encodePrompt(
        _ prompt: String,
        tokenizer: QwenTokenizer,
        textEncoder: QwenTextEncoder,
        maxLength: Int
    ) throws -> (embeddings: MLXArray, mask: MLXArray) {
        // Z-Image expects raw prompt tokenization here. Chat-template wrapping
        // changes the sequence format and broke prompt encoding for the bf16 fork checkpoints.
        let encoded = tokenizer.encodePlain(prompts: [prompt], maxLength: maxLength)
        let embeddingsList = textEncoder.encodeForZImage(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask
        )

        guard let firstEmbeds = embeddingsList.first else {
            throw UtilityError.emptyEmbeddings
        }

        let embedsBatch = firstEmbeds.expandedDimensions(axis: 0)
        let mask = MLX.ones([1, firstEmbeds.dim(0)], dtype: .int32)

        return (embedsBatch, mask)
    }

    public static func resolveTextEncoderSelection(
        for snapshot: URL,
        overridePath: String?,
        logger: Logger? = nil
    ) -> TextEncoderSelection {
        let selection = ZImageFiles.resolveTextEncoderSelection(at: snapshot, overridePath: overridePath)
        if let logger {
            let sourceDescription: String
            switch selection.source {
            case .overridePath:
                sourceDescription = "CLI override"
            case .environment:
                sourceDescription = "ZIMAGE_ENCODER_PATH"
            case .autoDetectedPreferred:
                sourceDescription = "auto-detected preferred encoder"
            case .defaultDirectory:
                sourceDescription = "default text_encoder directory"
            }
            logger.info("Using text encoder directory: \(selection.directory.path) (\(sourceDescription))")
        }
        return selection
    }

    public static func alignNegativeEmbeddingsIfNeeded(
        promptEmbeds: MLXArray,
        negativeEmbeds: MLXArray
    ) -> (prompt: MLXArray, negative: MLXArray) {
        let promptSequenceLength = promptEmbeds.dim(1)
        let negativeSequenceLength = negativeEmbeds.dim(1)
        guard promptSequenceLength != negativeSequenceLength else {
            return (promptEmbeds, negativeEmbeds)
        }

        let targetLength = max(promptSequenceLength, negativeSequenceLength)
        let hiddenDim = promptEmbeds.dim(2)
        var alignedPrompt = promptEmbeds
        var alignedNegative = negativeEmbeds

        if promptSequenceLength < targetLength {
            let padding = MLX.zeros([1, targetLength - promptSequenceLength, hiddenDim], dtype: promptEmbeds.dtype)
            alignedPrompt = MLX.concatenated([promptEmbeds, padding], axis: 1)
        }

        if negativeSequenceLength < targetLength {
            let padding = MLX.zeros([1, targetLength - negativeSequenceLength, hiddenDim], dtype: negativeEmbeds.dtype)
            alignedNegative = MLX.concatenated([negativeEmbeds, padding], axis: 1)
        }

        return (alignedPrompt, alignedNegative)
    }

    public static func decodeLatents(
        _ latents: MLXArray,
        vae: VAEImageDecoding,
        height: Int,
        width: Int
    ) -> MLXArray {
        let input: MLXArray
        if latents.dtype == .bfloat16 {
            input = latents
        } else {
            input = latents.asType(.bfloat16)
        }

        let (decoded, _) = vae.decode(input, return_dict: false)
        var image = decoded
        if height != decoded.dim(2) || width != decoded.dim(3) {
            var nhwc = image.transposed(0, 2, 3, 1)
            let hScale = Float(height) / Float(decoded.dim(2))
            let wScale = Float(width) / Float(decoded.dim(3))
            nhwc = MLXNN.Upsample(scaleFactor: .array([hScale, wScale]), mode: .nearest)(nhwc)
            image = nhwc.transposed(0, 3, 1, 2)
        }

        image = QwenImageIO.denormalizeFromDecoder(image)
        return MLX.clip(image, min: 0, max: 1)
    }

    public static func calculateShift(
        imageSeqLen: Int,
        baseSeqLen: Int,
        maxSeqLen: Int,
        baseShift: Float,
        maxShift: Float
    ) -> Float {
        let m = (maxShift - baseShift) / Float(maxSeqLen - baseSeqLen)
        let b = baseShift - m * Float(baseSeqLen)
        return Float(imageSeqLen) * m + b
    }

    public static func prepareSnapshot(
        model: String?,
        defaultModelId: String,
        defaultRevision: String,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let resolvedURL = try await ModelResolution.resolveOrDefault(
            modelSpec: model,
            defaultModelId: defaultModelId,
            defaultRevision: defaultRevision,
            progressHandler: progressHandler
        )

        return resolvedURL
    }
    public static func validateDimensions(
        width: Int,
        height: Int,
        vaeScale: Int = 16
    ) throws {
        if width % vaeScale != 0 {
            throw DimensionError.widthNotDivisible(width: width, scale: vaeScale)
        }
        if height % vaeScale != 0 {
            throw DimensionError.heightNotDivisible(height: height, scale: vaeScale)
        }
    }
    public enum DimensionError: Error, LocalizedError {
        case widthNotDivisible(width: Int, scale: Int)
        case heightNotDivisible(height: Int, scale: Int)

        public var errorDescription: String? {
            switch self {
            case .widthNotDivisible(let width, let scale):
                return "Width must be divisible by \(scale) (got \(width)). Please adjust to a multiple of \(scale)."
            case .heightNotDivisible(let height, let scale):
                return "Height must be divisible by \(scale) (got \(height)). Please adjust to a multiple of \(scale)."
            }
        }
    }
}
