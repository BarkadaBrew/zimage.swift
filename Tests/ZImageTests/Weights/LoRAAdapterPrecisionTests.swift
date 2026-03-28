import XCTest
import MLX
@testable import ZImage

final class LoRAAdapterPrecisionTests: XCTestCase {

  func testPromotesLowPrecisionAdapterWeightsToFloat32() {
    let down = MLXArray([Float(1.0), 2.0], [1, 2]).asType(.bfloat16)
    let up = MLXArray([Float(3.0), 4.0], [2, 1]).asType(.bfloat16)

    let adapter = LoRAAdapter(down: down, up: up, scale: 0.5)

    XCTAssertEqual(adapter.computeDType, .float32)
    XCTAssertEqual(adapter.down.dtype, .float32)
    XCTAssertEqual(adapter.up.dtype, .float32)
  }

  func testKeepsFloat32AdapterWeightsInFloat32() {
    let down = MLXArray([Float(1.0), 2.0], [1, 2]).asType(.float32)
    let up = MLXArray([Float(3.0), 4.0], [2, 1]).asType(.float32)

    let adapter = LoRAAdapter(down: down, up: up, scale: 1.0)

    XCTAssertEqual(adapter.computeDType, .float32)
    XCTAssertEqual(adapter.down.dtype, .float32)
    XCTAssertEqual(adapter.up.dtype, .float32)
  }
}
