import Foundation
import MLX
import MLXNN

public struct LoRAAdapter {
    public let down: MLXArray
    public let up: MLXArray
    public let scale: Float
    public let computeDType: DType

    public init(down: MLXArray, up: MLXArray, scale: Float) {
        let preferredType = LoRAAdapter.preferredComputeDType(down: down, up: up)
        self.down = down.dtype == preferredType ? down : down.asType(preferredType)
        self.up = up.dtype == preferredType ? up : up.asType(preferredType)
        self.scale = scale
        self.computeDType = preferredType
    }

    private static func preferredComputeDType(down: MLXArray, up: MLXArray) -> DType {
        if down.dtype == .float32 || up.dtype == .float32 {
            return .float32
        }
        if down.dtype == .bfloat16 || down.dtype == .float16 || up.dtype == .bfloat16 || up.dtype == .float16 {
            return .float32
        }
        return down.dtype == up.dtype ? down.dtype : .float32
    }
}

public protocol DynamicLoRACapable: AnyObject {
    var loraAdapters: [LoRAAdapter] { get set }
}

extension DynamicLoRACapable {

    public func setLoRA(down: MLXArray, up: MLXArray, scale: Float) {
        self.loraAdapters = [LoRAAdapter(down: down, up: up, scale: scale)]
    }
    public func addLoRA(down: MLXArray, up: MLXArray, scale: Float) {
        self.loraAdapters.append(LoRAAdapter(down: down, up: up, scale: scale))
    }
    public func clearLoRA() {
        self.loraAdapters = []
    }
    public var hasLoRA: Bool {
        loraAdapters.contains { $0.scale != 0 }
    }
    public func computeLoRAContribution(_ x: MLXArray) -> MLXArray? {
        let activeAdapters = loraAdapters.filter { $0.scale != 0 }
        guard !activeAdapters.isEmpty else {
            return nil
        }
        var total: MLXArray?
        for adapter in activeAdapters {
            let input = x.dtype == adapter.computeDType ? x : x.asType(adapter.computeDType)
            let scale = MLXArray(adapter.scale).asType(adapter.computeDType)
            let loraHidden = MLX.matmul(input, adapter.down.T)
            let loraOut = MLX.matmul(loraHidden, adapter.up.T) * scale
            total = total.map { $0 + loraOut } ?? loraOut
        }
        return total
    }
}
public class LoRALinear: Linear, DynamicLoRACapable {
    public var loraAdapters: [LoRAAdapter] = []
    public convenience init(from linear: Linear) {
        self.init(weight: linear.weight, bias: linear.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {

        var result: MLXArray
        if let bias = bias {
            result = MLX.addMM(bias, x, weight.T)
        } else {
            result = MLX.matmul(x, weight.T)
        }
        if let loraContribution = computeLoRAContribution(x) {
            result = result + loraContribution.asType(result.dtype)
        }

        return result
    }
}
public class LoRAQuantizedLinear: QuantizedLinear, DynamicLoRACapable {
    public var loraAdapters: [LoRAAdapter] = []
    public convenience init(from quantizedLinear: QuantizedLinear) {
        self.init(
            weight: quantizedLinear.weight,
            bias: quantizedLinear.bias,
            scales: quantizedLinear.scales,
            biases: quantizedLinear.biases,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {

        var result = MLX.quantizedMatmul(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits
        )

        if let bias = bias {
            result = result + bias
        }
        if let loraContribution = computeLoRAContribution(x) {
            result = result + loraContribution.asType(result.dtype)
        }

        return result
    }
}
