import XCTest
@testable import ZImage

final class VAEWeightLayoutTests: XCTestCase {

  func testDoesNotRequestTransposeForAlreadyAlignedVAEConvWeights() {
    XCTAssertFalse(
      ZImageWeightsMapping.needsTransposeToMatchShape([2, 3, 3, 4], expectedShape: [2, 3, 3, 4])
    )
  }

  func testRequestsTransposeForPyTorchStyleVAEConvWeights() {
    XCTAssertTrue(
      ZImageWeightsMapping.needsTransposeToMatchShape([2, 4, 3, 3], expectedShape: [2, 3, 3, 4])
    )
  }

  func testDoesNotRequestTransposeForUnrelatedShapeMismatch() {
    XCTAssertFalse(
      ZImageWeightsMapping.needsTransposeToMatchShape([2, 5, 3, 3], expectedShape: [2, 3, 3, 4])
    )
  }
}
