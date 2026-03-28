import XCTest
import MLX
@testable import ZImage

final class LoRALinearTests: XCTestCase {

  override func setUpWithError() throws {
    throw XCTSkip("MLX metallib is unavailable in the SwiftPM test runner on this machine")
  }

  func testMultipleAdaptersAccumulateIncludingNegativeScales() {
    let layer = LoRALinear(
      weight: MLXArray([1 as Float, 0, 0, 1], [2, 2]).asType(.float32),
      bias: nil
    )

    layer.addLoRA(
      down: MLXArray([1 as Float, 0, 0, 1], [2, 2]).asType(.float32),
      up: MLXArray([1 as Float, 0, 0, 1], [2, 2]).asType(.float32),
      scale: 0.5
    )
    layer.addLoRA(
      down: MLXArray([1 as Float, 0, 0, 1], [2, 2]).asType(.float32),
      up: MLXArray([1 as Float, 0, 0, 1], [2, 2]).asType(.float32),
      scale: -0.25
    )

    XCTAssertEqual(layer.loraAdapters.count, 2)
    XCTAssertEqual(layer.loraAdapters[0].scale, 0.5)
    XCTAssertEqual(layer.loraAdapters[1].scale, -0.25)
    XCTAssertTrue(layer.hasLoRA)
  }

  func testClearLoRARemovesEveryAdapter() {
    let layer = LoRALinear(
      weight: MLXArray([1 as Float, 0, 0, 1], [2, 2]).asType(.float32),
      bias: nil
    )

    layer.addLoRA(
      down: MLXArray([1 as Float, 0, 0, 1], [2, 2]).asType(.float32),
      up: MLXArray([1 as Float, 0, 0, 1], [2, 2]).asType(.float32),
      scale: 1.0
    )
    layer.clearLoRA()

    XCTAssertTrue(layer.loraAdapters.isEmpty)
    XCTAssertFalse(layer.hasLoRA)
  }
}
