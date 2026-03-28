import XCTest
import MLX
@testable import ZImage

final class LoRAApplicatorTests: XCTestCase {

  func testMergeWeightsNormalizesFullyTransposedLoRAPair() {
    let key = "layers.0.attention.to_q.weight"
    let baseWeights = [key: MLXArray([Float](repeating: 0.0, count: 4), [2, 2]).asType(.float32)]
    let loraWeights = LoRAWeights(
      weights: [
        key: (
          down: MLXArray([Float(1.0), 2.0], [2, 1]).asType(.float32),
          up: MLXArray([Float(3.0), 4.0], [1, 2]).asType(.float32)
        )
      ],
      rank: 1
    )

    let merged = LoRAApplicator.mergeWeights(
      baseWeights: baseWeights,
      loraWeights: loraWeights,
      scale: 1.0
    )

    XCTAssertEqual(merged[key]?.shape, [2, 2])
    XCTAssertEqual(merged[key]?.asArray(Float.self), [3.0, 6.0, 4.0, 8.0])
  }

  func testMergeWeightsNormalizesSingleTransposedUpTensor() {
    let key = "layers.0.attention.to_q.weight"
    let baseWeights = [key: MLXArray([Float](repeating: 0.0, count: 4), [2, 2]).asType(.float32)]
    let loraWeights = LoRAWeights(
      weights: [
        key: (
          down: MLXArray([Float(1.0), 2.0], [1, 2]).asType(.float32),
          up: MLXArray([Float(3.0), 4.0], [1, 2]).asType(.float32)
        )
      ],
      rank: 1
    )

    let merged = LoRAApplicator.mergeWeights(
      baseWeights: baseWeights,
      loraWeights: loraWeights,
      scale: 1.0
    )

    XCTAssertEqual(merged[key]?.shape, [2, 2])
    XCTAssertEqual(merged[key]?.asArray(Float.self), [3.0, 6.0, 4.0, 8.0])
  }
}
